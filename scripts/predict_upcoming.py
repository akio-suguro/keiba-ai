from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

if __package__:
    from .database import (
        connect_db,
        execute,
        read_sql_query,
        table_columns,
        transaction,
    )
    from .model_features import make_prediction_feature_rows
    from .train_model import simulate_payout_returns
    from .train_upset_model import add_race_features
else:
    from database import connect_db, execute, read_sql_query, table_columns, transaction
    from model_features import make_prediction_feature_rows
    from train_model import simulate_payout_returns
    from train_upset_model import add_race_features


ROOT = Path(__file__).resolve().parents[1]
TOP3_MODEL_PATH = ROOT / "data" / "model" / "top3_model.joblib"
UPSET_MODEL_PATH = ROOT / "data" / "model" / "upset" / "upset_model.joblib"
UPSET_METRICS_PATH = ROOT / "data" / "model" / "upset" / "metrics.json"
JST = ZoneInfo("Asia/Tokyo")


def _json_value(value):
    if value is None or pd.isna(value):
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


def _model_version() -> str:
    digest = hashlib.sha256()
    for path in (TOP3_MODEL_PATH, UPSET_MODEL_PATH):
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _load_inputs(connection):
    today = datetime.now(JST).date()
    history_start = (
        pd.Timestamp(today) - pd.DateOffset(years=10)
    ).date().isoformat()
    races = read_sql_query(
        "SELECT * FROM races WHERE race_date >= ?",
        connection,
        params=(history_start,),
    )
    horses = read_sql_query(
        """
        SELECT h.*
        FROM horses h
        JOIN races r ON r.id = h.race_id
        WHERE r.race_date >= ?
        """,
        connection,
        params=(history_start,),
    )
    if races.empty or horses.empty:
        return races, horses, pd.DataFrame()

    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races["id"] = races["id"].astype(str)
    horses["race_id"] = horses["race_id"].astype(str)
    upcoming = races.loc[
        races["race_date"].ge(pd.Timestamp(today))
        & races["id"].isin(horses["race_id"])
    ].copy()
    if upcoming.empty:
        return races, horses, upcoming

    field_sizes = horses.groupby("race_id").size()
    upcoming["race_id"] = upcoming["id"]
    upcoming["field_size"] = (
        upcoming["race_id"].map(field_sizes).fillna(0).astype(int)
    )
    upcoming = upcoming.loc[upcoming["field_size"].ge(3)].copy()
    return races, horses, upcoming


def _upset_factors(model, features: pd.DataFrame, probability: float, threshold: float):
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(features)
    values = (
        transformed.toarray().ravel()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed).ravel()
    )
    contributions = values * classifier.coef_[0]
    names = preprocessor.get_feature_names_out()
    high_probability = probability >= threshold
    ranked = sorted(
        zip(names, contributions),
        key=lambda item: item[1],
        reverse=high_probability,
    )
    selected = [
        {
            "feature": str(name),
            "contribution": float(value),
        }
        for name, value in ranked
        if (value > 0 if high_probability else value < 0)
    ][:3]
    return selected


def evaluate_saved_predictions(connection) -> list[dict]:
    """Evaluate matured weekly predictions once their race results are stored."""
    saved = read_sql_query(
        """
        SELECT
            p.prediction_id, p.race_id, p.model_version,
            p.top3_predictions, p.chaos_probability, r.race_date
        FROM predictions p
        JOIN races r ON r.id = p.race_id
        WHERE p.top3_predictions IS NOT NULL
          AND p.evaluated_at IS NULL
          AND r.race_date < ?
        ORDER BY p.predicted_at
        """,
        connection,
        params=(datetime.now(JST).date().isoformat(),),
    )
    if saved.empty:
        return []

    race_ids = saved["race_id"].drop_duplicates().tolist()
    placeholders = ", ".join("?" for _ in race_ids)
    horses = read_sql_query(
        f"""
        SELECT race_id, horse_number, finish, popularity
        FROM horses
        WHERE race_id IN ({placeholders})
        """,
        connection,
        params=race_ids,
    )
    horses["finish"] = pd.to_numeric(horses["finish"], errors="coerce")
    complete_races = set()
    actual_upsets = {}
    for race_id, group in horses.groupby("race_id", sort=False):
        finish = group["finish"]
        if all(finish.eq(place).sum() == 1 for place in (1, 2, 3)):
            complete_races.add(str(race_id))
            placed = group.loc[finish.isin([1, 2, 3])]
            popularity = pd.to_numeric(
                placed["popularity"],
                errors="coerce",
            )
            if (
                popularity.notna().all()
                and np.isfinite(popularity).all()
                and popularity.ge(1).all()
                and popularity.le(len(group)).all()
                and popularity.mod(1).eq(0).all()
            ):
                actual_upsets[str(race_id)] = int(popularity.ge(7).any())
    saved = saved[saved["race_id"].astype(str).isin(complete_races)].copy()
    if saved.empty:
        return []

    completed_ids = saved["race_id"].drop_duplicates().tolist()
    payouts = read_sql_query(
        f"""
        SELECT race_id, bet_type, combination, amount
        FROM payouts
        WHERE race_id IN ({", ".join("?" for _ in completed_ids)})
        """,
        connection,
        params=completed_ids,
    )

    recorded = []
    for model_version, version_rows in saved.groupby("model_version", sort=False):
        prediction_rows = []
        hits = []
        used_races = []
        prediction_ids = []
        upset_actuals = []
        upset_probabilities = []
        for row in version_rows.itertuples(index=False):
            race_id = str(row.race_id)
            actual = horses.loc[
                (horses["race_id"].astype(str) == race_id)
                & horses["finish"].isin([1, 2, 3]),
                "horse_number",
            ].dropna().astype(int).tolist()
            ranked = json.loads(row.top3_predictions)
            ranked = sorted(ranked, key=lambda item: item["rank"])
            selected = ranked[:3]
            if len(selected) != 3:
                continue
            actual_set = set(actual)
            hits.append(len({int(item["horse_number"]) for item in selected} & actual_set))
            used_races.append(row.race_date)
            prediction_ids.append(int(row.prediction_id))
            if race_id in actual_upsets and pd.notna(row.chaos_probability):
                upset_actuals.append(actual_upsets[race_id])
                upset_probabilities.append(float(row.chaos_probability))
            for item in ranked:
                prediction_rows.append(
                    {
                        "race_id": row.race_id,
                        "horse_number": item["horse_number"],
                        "predicted_rank_in_race": item["rank"],
                    }
                )
        if not hits:
            continue
        metrics = simulate_payout_returns(
            pd.DataFrame(prediction_rows),
            payouts,
        )
        upset_metrics = {
            "races": len(upset_actuals),
            "brier_score": None,
            "top_10pct_upset_rate": None,
        }
        if upset_actuals:
            upset_metrics["brier_score"] = float(
                brier_score_loss(upset_actuals, upset_probabilities)
            )
            top_count = max(
                1,
                int(np.ceil(len(upset_actuals) * 0.1)),
            )
            upset_order = np.argsort(upset_probabilities)[::-1][:top_count]
            upset_metrics["top_10pct_upset_rate"] = float(
                np.asarray(upset_actuals)[upset_order].mean()
            )
        test_dates = pd.to_datetime(used_races, errors="coerce")
        values = (
            str(model_version),
            datetime.now(JST).isoformat(timespec="seconds"),
            str(test_dates.min().date()),
            str(test_dates.max().date()),
            len(hits),
            float(np.mean(hits)),
            float(np.sum(hits) / (3 * len(hits))),
            (
                metrics["単勝"]["return_rate"]
                if metrics["単勝"]["status"] == "computed"
                else None
            ),
            (
                metrics["馬連"]["return_rate"]
                if metrics["馬連"]["status"] == "computed"
                else None
            ),
            (
                metrics["3連複"]["return_rate"]
                if metrics["3連複"]["status"] == "computed"
                else None
            ),
            upset_metrics["races"],
            upset_metrics["brier_score"],
            upset_metrics["top_10pct_upset_rate"],
        )
        with transaction(connection):
            execute(
                connection,
                """
                INSERT INTO prediction_evaluations (
                    model_version, evaluated_at, test_from, test_until,
                    evaluated_races, mean_correct_picks, precision_at_3,
                    win_return_rate, quinella_return_rate, trio_return_rate,
                    upset_races, upset_brier_score, upset_top_10pct_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                values,
            )
            if prediction_ids:
                execute(
                    connection,
                    f"""
                    UPDATE predictions
                    SET evaluated_at = ?
                    WHERE prediction_id IN (
                        {", ".join("?" for _ in prediction_ids)}
                    )
                    """,
                    [
                        datetime.now(JST).isoformat(timespec="seconds"),
                        *prediction_ids,
                    ],
                )
        recorded.append(
            {
                "model_version": str(model_version),
                "evaluated_races": len(hits),
                "mean_correct_picks": float(np.mean(hits)),
                "precision_at_3": float(np.sum(hits) / (3 * len(hits))),
                "payout_simulation": metrics,
                "upset": upset_metrics,
            }
        )
    return recorded


def main() -> int:
    for path in (TOP3_MODEL_PATH, UPSET_MODEL_PATH, UPSET_METRICS_PATH):
        if not path.exists():
            raise FileNotFoundError(f"予測モデルが見つかりません: {path}")

    top3_saved = joblib.load(TOP3_MODEL_PATH)
    top3_model = top3_saved["model"]
    top3_columns = list(top3_saved["feature_columns"])
    upset_model = joblib.load(UPSET_MODEL_PATH)
    upset_metrics = json.loads(UPSET_METRICS_PATH.read_text(encoding="utf-8"))
    upset_threshold = float(
        upset_metrics["classification_threshold_selected_on_validation"]
    )
    model_version = _model_version()
    predicted_at = datetime.now(JST).isoformat(timespec="seconds")

    with connect_db() as connection:
        required_prediction_columns = {
            "race_id",
            "predicted_at",
            "model_version",
            "top3_predictions",
            "evaluated_at",
        }
        missing = required_prediction_columns - table_columns(
            connection,
            "predictions",
        )
        if missing:
            raise RuntimeError(
                "predictions テーブルに必要な列がありません: "
                f"{sorted(missing)}。週次クロールのDB初期化を確認してください。"
            )
        evaluations = evaluate_saved_predictions(connection)
        races, horses, upcoming = _load_inputs(connection)
        if upcoming.empty:
            print("[PREDICTION] 予測対象となる今後の出走表はありません")
            return 0

        historical_rows = horses.merge(
            races[
                ["id", "race_date", "race_name", "venue", "surface", "distance_m"]
            ].rename(columns={"id": "race_id"}),
            on="race_id",
            how="left",
            suffixes=("", "_race"),
        )
        race_features, _, _ = add_race_features(
            upcoming,
            races,
            horses,
        )
        upset_columns = list(upset_model.feature_names_in_)
        predictions = []
        for race in upcoming.to_dict("records"):
            race_horses = horses.loc[horses["race_id"] == str(race["id"])].copy()
            feature_rows = make_prediction_feature_rows(
                race,
                race_horses,
                historical_rows,
            )
            for column in top3_columns:
                if column not in feature_rows.columns:
                    feature_rows[column] = np.nan
            scores = np.asarray(
                top3_model.predict(feature_rows[top3_columns]),
                dtype=float,
            )
            ordered_indices = np.lexsort(
                (
                    pd.to_numeric(
                        feature_rows["horse_number"],
                        errors="coerce",
                    ).fillna(np.inf).to_numpy(),
                    -scores,
                )
            )
            ranked_rows = []
            for rank, index in enumerate(ordered_indices, start=1):
                row = feature_rows.iloc[index]
                ranked_rows.append(
                    {
                        "rank": rank,
                        "horse_number": _json_value(row.get("horse_number")),
                        "horse_name": _json_value(row.get("horse_name")),
                        "score": float(
                            100 * (len(ordered_indices) - rank + 1)
                            / len(ordered_indices)
                        ),
                        "frame": _json_value(row.get("frame")),
                        "jockey": _json_value(row.get("jockey")),
                        "popularity": _json_value(row.get("popularity")),
                        "odds": _json_value(row.get("odds")),
                    }
                )

            upset_row = race_features.loc[
                race_features["race_id"].astype(str) == str(race["id"])
            ].reindex(columns=upset_columns)
            upset_probability = float(upset_model.predict_proba(upset_row)[0, 1])
            factors = _upset_factors(
                upset_model,
                upset_row,
                upset_probability,
                upset_threshold,
            )
            predictions.append(
                {
                    "race_id": int(race["id"]),
                    "race_date": str(race["race_date"].date()),
                    "start_time": race.get("start_time"),
                    "top3": ranked_rows,
                    "upset_probability": upset_probability,
                    "upset_decision": (
                        "upset"
                        if upset_probability >= upset_threshold
                        else "normal"
                    ),
                    "upset_factors": factors,
                }
            )

        with transaction(connection):
            for prediction in predictions:
                execute(
                    connection,
                    """
                    DELETE FROM predictions
                    WHERE race_id = ? AND model_version = ?
                    """,
                    (prediction["race_id"], model_version),
                )
                execute(
                    connection,
                    """
                    INSERT INTO predictions (
                        race_id, predicted_at, cutoff_at, chaos_probability,
                        uncertainty_score, decision, model_version,
                        top3_predictions, upset_factors
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prediction["race_id"],
                        predicted_at,
                        (
                            f"{prediction['race_date']}T"
                            f"{prediction['start_time']}:00+09:00"
                            if prediction["start_time"]
                            else None
                        ),
                        prediction["upset_probability"],
                        None,
                        prediction["upset_decision"],
                        model_version,
                        json.dumps(
                            prediction["top3"],
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                        json.dumps(
                            prediction["upset_factors"],
                            ensure_ascii=False,
                            allow_nan=False,
                        ),
                    ),
                )

    print(
        f"[PREDICTION] {len(predictions)}レースをDBに保存 "
        f"| model={model_version}"
    )
    if evaluations:
        print(f"[WALK-FORWARD] 実測済み週次予測: {evaluations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
