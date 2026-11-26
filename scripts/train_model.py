from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sqlalchemy.engine import Connection
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import ndcg_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __package__:
    from .model_features import (
        HISTORICAL_FEATURE_COLUMNS,
        add_race_name_history_features,
        add_race_relative_features,
        add_historical_performance_features,
        add_sex_age_features,
    )
else:
    from model_features import (
        HISTORICAL_FEATURE_COLUMNS,
        add_race_name_history_features,
        add_race_relative_features,
        add_historical_performance_features,
        add_sex_age_features,
    )

if __package__:
    from .database import (
        connect_db,
        read_sql_query,
        table_columns,
    )
else:
    from database import (
        connect_db,
        read_sql_query,
        table_columns,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "keiba.db"
OUTPUT_DIR = PROJECT_ROOT / "data" / "model"

RANDOM_STATE = 42
TRAINING_WINDOW_YEARS = 5
FEATURE_HISTORY_YEARS = 10
WALK_FORWARD_FOLDS = 3

RACE_FEATURES = (
    "race_date",
    "venue",
    "distance_m",
    "surface",
    "direction",
    "race_name",
    "race_number",
    "start_time",
    "weather",
    "turf_condition",
    "dirt_condition",
    "race_grade",
    "race_conditions",
)
HORSE_FEATURES = (
    "finish",
    "frame",
    "horse_number",
    "horse_name",
    "sex_age",
    "weight",
    "time",
    "last_3f",
    "jockey",
    "trainer",
    "body_weight",
    "weight_change",
    "popularity",
    "odds",
    "horse_mark",
)


def _table_columns(
    conn: Connection | sqlite3.Connection,
    table: str,
) -> set[str]:
    return table_columns(conn, table)


def _connect_database() -> Connection | sqlite3.Connection:
    if os.environ.get("DATABASE_URL", "").strip():
        return connect_db()
    return sqlite3.connect(DB_PATH)


def load_data():
    """Load race entries and use only information available before the start."""
    if not os.environ.get("DATABASE_URL", "").strip() and not DB_PATH.exists():
        raise FileNotFoundError(f"DBが見つかりません: {DB_PATH}")

    with _connect_database() as conn:
        race_columns = _table_columns(conn, "races")
        horse_columns = _table_columns(conn, "horses")
        required_races = {"id", "race_date"}
        required_horses = {
            "race_id",
            "finish",
            "horse_number",
            "horse_name",
        }
        missing_races = required_races - race_columns
        missing_horses = required_horses - horse_columns
        if missing_races or missing_horses:
            raise ValueError(
                "DB schema is missing required columns: "
                f"races={sorted(missing_races)}, "
                f"horses={sorted(missing_horses)}"
            )
        latest_date_frame = read_sql_query(
            "SELECT MAX(race_date) AS latest_date FROM races",
            conn,
        )
        latest_date = pd.to_datetime(
            latest_date_frame.loc[0, "latest_date"],
            errors="coerce",
        )
        if pd.isna(latest_date):
            raise ValueError("レース日が有効な学習データがありません。")
        history_start = (
            latest_date - pd.DateOffset(years=FEATURE_HISTORY_YEARS)
        ).date().isoformat()

        race_select = [
            f'r."{column}" AS "{column}"'
            if column in race_columns
            else f'NULL AS "{column}"'
            for column in RACE_FEATURES
        ]
        horse_select = [
            f'h."{column}" AS "{column}"'
            if column in horse_columns
            else f'NULL AS "{column}"'
            for column in HORSE_FEATURES
        ]
        query = f"""
            SELECT
                r.id AS race_id,
                {", ".join(race_select)},
                {", ".join(horse_select)}
            FROM horses h
            JOIN races r ON r.id = h.race_id
            WHERE h.finish IS NOT NULL
              AND r.race_date IS NOT NULL
              AND r.race_date >= ?
        """
        df = read_sql_query(query, conn, params=(history_start,))

    if df.empty:
        raise ValueError("学習対象のデータがありません。")

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")
    df["horse_number"] = pd.to_numeric(
        df["horse_number"],
        errors="coerce",
    )
    df = df.dropna(
        subset=["race_date", "finish", "horse_number", "horse_name"]
    ).copy()
    df = df[
        df["finish"].ge(1)
        & df["finish"].mod(1).eq(0)
        & df["horse_number"].ge(1)
    ].copy()

    if df.empty:
        raise ValueError("有効な着順・馬番データがありません。")

    # Train only on races with one unambiguous first, second and third place.
    valid_races = []
    for race_id, group in df.groupby("race_id", sort=False):
        placed = group.loc[group["finish"].isin([1, 2, 3])]
        if (
            placed["finish"].value_counts().reindex([1, 2, 3], fill_value=0).eq(1).all()
            and group["horse_number"].is_unique
        ):
            valid_races.append(race_id)
    df = df[df["race_id"].isin(valid_races)].copy()
    if df.empty:
        raise ValueError("1〜3着が確定した有効なレースがありません。")

    for column in (
        "distance_m",
        "race_number",
        "frame",
        "weight",
        "body_weight",
        "weight_change",
        "popularity",
        "odds",
    ):
        df[column] = pd.to_numeric(df[column], errors="coerce")
    if "start_time" in df.columns:
        time_parts = df["start_time"].astype("string").str.extract(
            r"(\d{1,2})[:：](\d{2})"
        )
        df["start_minutes"] = (
            pd.to_numeric(time_parts[0], errors="coerce") * 60
            + pd.to_numeric(time_parts[1], errors="coerce")
        )

    field_size = df.groupby("race_id")["horse_number"].transform("size")
    invalid_popularity = (
        df["popularity"].notna()
        & (
            df["popularity"].lt(1)
            | df["popularity"].gt(field_size)
            | df["popularity"].mod(1).ne(0)
        )
    )
    df.loc[invalid_popularity, "popularity"] = pd.NA
    df.loc[df["odds"].le(0), "odds"] = pd.NA

    df = add_sex_age_features(df)
    df = add_historical_performance_features(df)
    df = add_race_name_history_features(df)
    df = add_race_relative_features(
        df,
        [
            "popularity",
            "odds",
            "horse_prior_starts",
            "horse_prior_win_rate",
            "horse_prior_top3_rate",
            "horse_recent_win_rate_5",
            "horse_recent_top3_rate_5",
            "horse_surface_prior_top3_rate",
            "horse_course_distance_prior_top3_rate",
            "jockey_horse_prior_top3_rate",
        ],
    )
    df["target_top3"] = df["finish"].le(3).astype(int)

    categorical_candidates = [
        "venue",
        "surface",
        "direction",
        "weather",
        "turf_condition",
        "dirt_condition",
        "race_grade",
        "race_conditions",
        "sex",
        "jockey",
        "trainer",
        "horse_mark",
    ]
    numeric_candidates = [
        "distance_m",
        "race_number",
        "start_minutes",
        "frame",
        "horse_number",
        "age",
        "weight",
        "body_weight",
        "weight_change",
        "popularity",
        "odds",
        *HISTORICAL_FEATURE_COLUMNS,
        *[
            f"{column}_race_percentile"
            for column in (
                "popularity",
                "odds",
                "horse_prior_starts",
                "horse_prior_win_rate",
                "horse_prior_top3_rate",
                "horse_recent_win_rate_5",
                "horse_recent_top3_rate_5",
                "horse_surface_prior_top3_rate",
                "horse_course_distance_prior_top3_rate",
                "jockey_horse_prior_top3_rate",
            )
        ],
    ]
    categorical_features = [
        column
        for column in categorical_candidates
        if column in df.columns and df[column].notna().any()
    ]
    numeric_features = [
        column
        for column in numeric_candidates
        if column in df.columns and pd.to_numeric(
            df[column], errors="coerce"
        ).notna().any()
    ]

    for column in categorical_features:
        df[column] = (
            df[column]
            .fillna("不明")
            .astype(str)
            .replace("", "不明")
        )
    for column in numeric_features:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    feature_columns = categorical_features + numeric_features
    if not feature_columns:
        raise ValueError("使用可能な特徴量がありません。")

    return df, feature_columns, categorical_features, numeric_features


def chronological_split(df):
    """Create chronological train/validation/test sets by whole race dates."""
    dates = sorted(df["race_date"].dt.normalize().dropna().unique())
    if len(dates) < 3:
        raise ValueError(
            "時系列の学習・検証・テストには異なる開催日が3日以上必要です。"
        )

    train_cutoff = max(1, min(int(len(dates) * 0.70), len(dates) - 2))
    validation_cutoff = max(
        train_cutoff + 1,
        min(int(len(dates) * 0.85), len(dates) - 1),
    )
    validation_start = pd.Timestamp(dates[train_cutoff])
    test_start = pd.Timestamp(dates[validation_cutoff])

    train_df = df[df["race_date"] < validation_start].copy()
    validation_df = df[
        (df["race_date"] >= validation_start)
        & (df["race_date"] < test_start)
    ].copy()
    test_df = df[df["race_date"] >= test_start].copy()
    if train_df.empty or validation_df.empty or test_df.empty:
        raise ValueError(
            "学習・検証・テストのいずれかが空です。"
        )
    if (
        set(train_df["race_id"]) & set(validation_df["race_id"])
        or set(train_df["race_id"]) & set(test_df["race_id"])
        or set(validation_df["race_id"]) & set(test_df["race_id"])
    ):
        raise ValueError("同一レースが複数のデータ区分に含まれています。")
    if train_df["target_top3"].nunique() < 2:
        raise ValueError("学習データの正解ラベルが1種類しかありません。")

    return train_df, validation_df, test_df, validation_start, test_start


def build_model(
    categorical_features,
    numeric_features,
):
    """Build a race-grouped LambdaRank model."""
    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="不明",
                            ),
                        ),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                min_frequency=10,
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )
    if not transformers:
        raise ValueError("モデルに使用できる特徴量がありません。")

    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=transformers,
                    sparse_threshold=1.0,
                ),
            ),
            (
                "ranker",
                lgb.LGBMRanker(
                    objective="lambdarank",
                    metric="ndcg",
                    n_estimators=180,
                    learning_rate=0.05,
                    num_leaves=20,
                    min_child_samples=30,
                    lambdarank_truncation_level=3,
                    random_state=RANDOM_STATE,
                    verbosity=-1,
                ),
            ),
        ]
    )


def _fit_ranker(model, data: pd.DataFrame, feature_columns: list[str]):
    """Sort rows by race before passing LightGBM's contiguous group sizes."""
    ordered = data.sort_values(
        ["race_id", "horse_number"],
        kind="stable",
    )
    group_sizes = ordered.groupby("race_id", sort=False).size().tolist()
    model.fit(
        ordered[feature_columns],
        ordered["target_top3"].astype(int),
        ranker__group=group_sizes,
        ranker__eval_at=(3,),
    )
    return model


def _top_three_metrics(
    frame: pd.DataFrame,
    score_column: str,
) -> dict[str, float | int]:
    race_results = []
    excluded_races = 0
    for _, group in frame.groupby("race_id", sort=False):
        finish = pd.to_numeric(group["finish"], errors="coerce")
        placed = finish.isin([1, 2, 3])
        if (
            group["horse_number"].isna().any()
            or not finish.eq(1).sum() == 1
            or not finish.eq(2).sum() == 1
            or not finish.eq(3).sum() == 1
            or int(placed.sum()) != 3
        ):
            excluded_races += 1
            continue

        ordered = group.sort_values(
            [score_column, "horse_number"],
            ascending=[False, True],
            kind="stable",
        )
        predicted = set(ordered.head(3)["horse_number"])
        actual = set(group.loc[placed, "horse_number"])
        hits = len(predicted & actual)
        race_results.append(
            {
                "hits": hits,
                "winner_hit": int(
                    group.loc[finish.eq(1), "horse_number"].iloc[0]
                    in predicted
                ),
                "exact_top3": int(predicted == actual),
                "ndcg_at_3": float(
                    ndcg_score(
                        [ordered["finish"].isin([1, 2, 3]).astype(int).to_numpy()],
                        [ordered[score_column].to_numpy()],
                        k=3,
                    )
                ),
            }
        )

    if not race_results:
        raise ValueError("上位3頭を評価できる有効なレースがありません。")

    results = pd.DataFrame(race_results)
    return {
        "races": int(len(results)),
        "excluded_races": excluded_races,
        "mean_correct_picks": float(results["hits"].mean()),
        "precision_at_3": float(results["hits"].sum() / (3 * len(results))),
        "recall_at_3": float(results["hits"].sum() / (3 * len(results))),
        "race_with_any_top3_hit_rate": float(results["hits"].gt(0).mean()),
        "winner_hit_rate": float(results["winner_hit"].mean()),
        "exact_top3_set_rate": float(results["exact_top3"].mean()),
        "ndcg_at_3": float(results["ndcg_at_3"].mean()),
    }


def _race_ranking_scores(
    frame: pd.DataFrame,
    probabilities,
) -> pd.DataFrame:
    scores = frame[["race_id", "horse_number", "finish"]].copy()
    scores["model_score"] = pd.Series(
        probabilities,
        index=frame.index,
    ).groupby(frame["race_id"]).rank(pct=True)
    scores["ranking_score"] = scores["model_score"]
    return scores


def evaluate_model(
    model,
    test_df,
    feature_columns,
    payouts: pd.DataFrame | None = None,
):
    """Report race-ranking quality, market baselines and payout returns."""
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(test_df[feature_columns])[:, 1]
    else:
        scores = np.asarray(model.predict(test_df[feature_columns])).reshape(-1)
    y_test = test_df["target_top3"].astype(int)

    metrics = {
        "model": "LightGBM LambdaRank",
        "objective": "race-grouped NDCG@3",
        "sklearn_version": sklearn.__version__,
        "test_horses": int(len(test_df)),
        "test_races": int(test_df["race_id"].nunique()),
        "test_start_date": str(test_df["race_date"].min().date()),
        "test_end_date": str(test_df["race_date"].max().date()),
        "features": feature_columns,
    }

    prediction_columns = [
        "race_id",
        "race_date",
        "venue",
        "race_name",
        "horse_number",
        "finish",
        "target_top3",
        "popularity",
        "odds",
    ]
    predictions_df = test_df[
        [column for column in prediction_columns if column in test_df.columns]
    ].copy()
    rank_scores = _race_ranking_scores(
        test_df,
        scores,
    )
    predictions_df["predicted_score"] = scores
    predictions_df["model_rank_score"] = rank_scores["model_score"].to_numpy()
    predictions_df["ranking_score"] = rank_scores["ranking_score"].to_numpy()
    ordered_predictions = predictions_df.sort_values(
        ["race_id", "ranking_score", "horse_number"],
        ascending=[True, False, True],
        kind="stable",
    )
    ordered_predictions["predicted_rank_in_race"] = (
        ordered_predictions.groupby("race_id").cumcount() + 1
    )
    predictions_df["predicted_rank_in_race"] = (
        ordered_predictions["predicted_rank_in_race"]
        .reindex(predictions_df.index)
        .astype(int)
    )
    predictions_df["predicted_top3"] = (
        predictions_df["predicted_rank_in_race"].le(3).astype(int)
    )
    metrics["race_level_top3"] = _top_three_metrics(
        predictions_df,
        "ranking_score",
    )
    metrics["target_top3_prevalence"] = float(y_test.mean())
    if "popularity" in test_df.columns:
        market = test_df.copy()
        market["popularity"] = pd.to_numeric(
            market["popularity"],
            errors="coerce",
        )
        complete_market_races = market.groupby(
            "race_id",
            sort=False,
        ).filter(
            lambda group: group["popularity"].notna().all()
            and group["popularity"].is_unique
        )
        if not complete_market_races.empty:
            market_race_ids = complete_market_races["race_id"].unique()
            complete_market_races["_market_score"] = (
                -complete_market_races["popularity"]
            )
            metrics["race_level_top3_on_market_coverage"] = (
                _top_three_metrics(
                    predictions_df[
                        predictions_df["race_id"].isin(market_race_ids)
                    ],
                    "ranking_score",
                )
            )
            popularity_metrics = _top_three_metrics(
                complete_market_races,
                "_market_score",
            )
            popularity_metrics["excluded_incomplete_market_races"] = int(
                test_df["race_id"].nunique()
                - complete_market_races["race_id"].nunique()
            )
            metrics["popularity_baseline"] = popularity_metrics
    if payouts is not None:
        metrics["payout_simulation"] = simulate_payout_returns(
            predictions_df,
            payouts,
        )

    return metrics, predictions_df


def _normalize_combination(value: str, *, ordered: bool) -> str | None:
    parts = str(value).strip().split("-")
    if not parts or any(not part.isdigit() for part in parts):
        return None
    parts = [str(int(part)) for part in parts]
    if not ordered:
        parts.sort(key=int)
    return "-".join(parts)


def simulate_payout_returns(
    predictions: pd.DataFrame,
    payouts: pd.DataFrame,
    stake_per_bet: int = 100,
) -> dict[str, dict[str, float | int | str]]:
    """Simulate one win, quinella and trio bet per complete test race."""
    bet_specs = {
        "単勝": ("単勝", 1, False),
        "馬連": ("馬連", 2, False),
        "3連複": ("3連複", 3, False),
    }
    results = {}
    prediction_race_count = predictions["race_id"].nunique()
    for label, (bet_type, selection_count, ordered) in bet_specs.items():
        returns = 0
        race_count = 0
        covered_races = set(
            payouts.loc[payouts["bet_type"] == bet_type, "race_id"]
            .astype(str)
        )
        for race_id, group in predictions.groupby("race_id", sort=False):
            if str(race_id) not in covered_races:
                continue
            selected = (
                group.sort_values(
                    ["predicted_rank_in_race", "horse_number"],
                    kind="stable",
                )
                .head(selection_count)["horse_number"]
                .dropna()
                .astype(int)
                .tolist()
            )
            if len(selected) != selection_count:
                continue
            race_count += 1
            if not ordered:
                selected.sort()
            combination = "-".join(str(number) for number in selected)
            actual = payouts.loc[
                (payouts["race_id"].astype(str) == str(race_id))
                & (payouts["bet_type"] == bet_type)
            ]
            actual = actual.loc[
                actual["combination"].map(
                    lambda value: _normalize_combination(value, ordered=ordered)
                )
                == combination
            ]
            if not actual.empty:
                returns += int(pd.to_numeric(actual["amount"]).sum())
        stake = race_count * stake_per_bet
        results[label] = {
            "races": race_count,
            "stake_yen": stake,
            "return_yen": returns,
            "net_yen": returns - stake,
            "return_rate": float(returns / stake) if stake else 0.0,
            "coverage": (
                float(race_count / prediction_race_count)
                if prediction_race_count
                else 0.0
            ),
            "status": "computed" if race_count else "no_payout_coverage",
        }
    return results


def load_payouts_for_races(race_ids) -> pd.DataFrame:
    race_ids = list(pd.unique(race_ids))
    if not race_ids:
        return pd.DataFrame(columns=["race_id", "bet_type", "combination", "amount"])
    placeholders = ", ".join("?" for _ in race_ids)
    with _connect_database() as conn:
        return read_sql_query(
            f"""
            SELECT race_id, bet_type, combination, amount
            FROM payouts
            WHERE race_id IN ({placeholders})
            """,
            conn,
            params=race_ids,
        )


def walk_forward_evaluate(
    data: pd.DataFrame,
    categorical_features: list[str],
    numeric_features: list[str],
    folds: int = WALK_FORWARD_FOLDS,
) -> list[dict]:
    """Evaluate sequential held-out calendar years using a rolling five-year window."""
    years = sorted(data["race_date"].dt.year.dropna().unique())
    test_years = years[-folds:]
    feature_columns = categorical_features + numeric_features
    results = []
    for year in test_years:
        test = data.loc[data["race_date"].dt.year.eq(year)].copy()
        if test.empty:
            continue
        start = pd.Timestamp(year=year, month=1, day=1)
        train = data.loc[
            data["race_date"].lt(start)
            & data["race_date"].ge(
                start - pd.DateOffset(years=TRAINING_WINDOW_YEARS)
            )
        ].copy()
        if train.empty or train["target_top3"].nunique() < 2:
            continue
        model = build_model(categorical_features, numeric_features)
        _fit_ranker(model, train, feature_columns)
        metrics, _ = evaluate_model(model, test, feature_columns)
        results.append(
            {
                "train_until": str(train["race_date"].max().date()),
                "test_year": int(year),
                **metrics["race_level_top3"],
            }
        )
    return results


def main():
    print("===== データ読み込み =====")
    df, feature_columns, categorical_features, numeric_features = load_data()
    latest_date = df["race_date"].max().normalize()
    training_start = latest_date - pd.DateOffset(
        years=TRAINING_WINDOW_YEARS
    )
    df = df.loc[df["race_date"].ge(training_start)].copy()
    (
        train_df,
        validation_df,
        test_df,
        validation_start,
        test_start,
    ) = chronological_split(df)

    print(f"全データ: {len(df):,}頭 / {df['race_id'].nunique():,}レース")
    print(
        f"学習: {len(train_df):,}頭 / "
        f"{train_df['race_id'].nunique():,}レース"
    )
    print(
        f"検証: {len(validation_df):,}頭 / "
        f"{validation_df['race_id'].nunique():,}レース"
    )
    print(
        f"テスト: {len(test_df):,}頭 / "
        f"{test_df['race_id'].nunique():,}レース"
    )
    print(f"検証開始日: {validation_start.date()}")
    print(f"テスト開始日: {test_start.date()}")
    print(f"特徴量: {feature_columns}")
    print(
        "予測時点: 馬体重・市場人気が確定したレース発走前。"
        "着順や走破内容などの事後情報は特徴量から除外。"
    )

    model = build_model(categorical_features, numeric_features)
    _fit_ranker(model, train_df, feature_columns)

    test_payouts = load_payouts_for_races(test_df["race_id"])
    validation_metrics, _ = evaluate_model(
        model,
        validation_df,
        feature_columns,
    )
    metrics, predictions_df = evaluate_model(
        model,
        test_df,
        feature_columns,
        test_payouts,
    )
    market_feature_names = {"popularity", "odds"}
    marketless_categorical = [
        name for name in categorical_features
        if name not in market_feature_names
    ]
    marketless_numeric = [
        name for name in numeric_features
        if name not in market_feature_names
        and not name.startswith(("popularity_", "odds_"))
    ]
    marketless_features = marketless_categorical + marketless_numeric
    marketless_model = build_model(
        marketless_categorical,
        marketless_numeric,
    )
    _fit_ranker(marketless_model, train_df, marketless_features)
    marketless_metrics, _ = evaluate_model(
        marketless_model,
        test_df,
        marketless_features,
        test_payouts,
    )
    walk_forward = walk_forward_evaluate(
        df,
        marketless_categorical,
        marketless_numeric,
    )
    deployment_start = df["race_date"].max().normalize() - pd.DateOffset(
        years=TRAINING_WINDOW_YEARS
    )
    deployment_df = df.loc[df["race_date"].ge(deployment_start)].copy()
    market_model = build_model(categorical_features, numeric_features)
    _fit_ranker(market_model, deployment_df, feature_columns)
    deployment_model = build_model(
        marketless_categorical,
        marketless_numeric,
    )
    _fit_ranker(deployment_model, deployment_df, marketless_features)
    metrics["validation"] = validation_metrics
    metrics["marketless_model"] = marketless_metrics
    metrics["walk_forward"] = walk_forward
    metrics["split"] = {
        "train_end": str(train_df["race_date"].max().date()),
        "validation_start": str(validation_start.date()),
        "validation_end": str(validation_df["race_date"].max().date()),
        "test_start": str(test_start.date()),
    }
    metrics["training_window"] = {
        "years": TRAINING_WINDOW_YEARS,
        "start": str(training_start.date()),
        "end": str(latest_date.date()),
    }
    metrics["deployment_model_trained_through"] = str(latest_date.date())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "top3_model.joblib"
    metrics_path = OUTPUT_DIR / "metrics.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    joblib.dump(
        {
            "model": deployment_model,
            "feature_columns": marketless_features,
            "categorical_features": marketless_categorical,
            "numeric_features": marketless_numeric,
            "target": "race-grouped top-three LambdaRank",
            "prediction_time": "before race start after weigh-in and market ranking",
            "market_features_excluded": True,
            "excluded_post_event_features": [
                "finish",
                "time",
                "margin",
                "last_3f",
                "payouts",
            ],
        },
        model_path,
    )
    joblib.dump(
        {
            "model": market_model,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "target": "race-grouped top-three LambdaRank",
            "market_features_excluded": False,
        },
        OUTPUT_DIR / "top3_market_model.joblib",
    )
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== レース単位の3頭選出評価 =====")
    print(
        json.dumps(
            metrics["race_level_top3"],
            ensure_ascii=False,
            indent=2,
        )
    )
    if "popularity_baseline" in metrics:
        print("\n===== 市場人気が揃ったレースでのモデル評価 =====")
        print(
            json.dumps(
                metrics["race_level_top3_on_market_coverage"],
                ensure_ascii=False,
                indent=2,
            )
        )
        print("\n===== 人気順ベースライン =====")
        print(
            json.dumps(
                metrics["popularity_baseline"],
                ensure_ascii=False,
                indent=2,
            )
        )
    print(f"\n学習済みモデル: {model_path}")
    print(f"評価結果: {metrics_path}")
    print(f"テスト予測: {predictions_path}")
    print("学習完了")


if __name__ == "__main__":
    main()
