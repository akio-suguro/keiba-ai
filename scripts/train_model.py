from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sqlalchemy.engine import Connection
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

if __package__:
    from .model_features import (
        HISTORICAL_FEATURE_COLUMNS,
        add_historical_performance_features,
        add_sex_age_features,
    )
else:
    from model_features import (
        HISTORICAL_FEATURE_COLUMNS,
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


DB_PATH = Path("/app/data/keiba.db")
OUTPUT_DIR = Path("/app/data/model")

RANDOM_STATE = 42

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
        """
        df = read_sql_query(query, conn)

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
    df["target_top3"] = df["finish"].le(3).astype(int)

    categorical_candidates = [
        "venue",
        "surface",
        "direction",
        "race_name",
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
    """Build an imputation, encoding and ranking-score classification pipeline."""
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
                    sparse_threshold=0.0,
                ),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    max_iter=140,
                    learning_rate=0.06,
                    max_leaf_nodes=15,
                    l2_regularization=3.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


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

        predicted = set(
            group.sort_values(
                [score_column, "horse_number"],
                ascending=[False, True],
                kind="stable",
            )
            .head(3)["horse_number"]
        )
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
):
    """Report classification scores and race-level top-three selection quality."""
    probabilities = model.predict_proba(test_df[feature_columns])[:, 1]
    y_test = test_df["target_top3"].astype(int)
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "HistGradientBoostingClassifier",
        "target": "finish <= 3",
        "sklearn_version": sklearn.__version__,
        "test_horses": int(len(test_df)),
        "test_races": int(test_df["race_id"].nunique()),
        "test_start_date": str(test_df["race_date"].min().date()),
        "test_end_date": str(test_df["race_date"].max().date()),
        "features": feature_columns,
        "positive_rate_test": float(y_test.mean()),
        "accuracy_at_0_5": float(accuracy_score(y_test, predictions)),
        "precision_at_0_5": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall_at_0_5": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "classification_report_at_0_5": classification_report(
            y_test,
            predictions,
            labels=[0, 1],
            target_names=["4着以下", "3着以内"],
            zero_division=0,
            output_dict=True,
        ),
    }
    if y_test.nunique() == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_test, probabilities)
        )
        metrics["average_precision"] = float(
            average_precision_score(y_test, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    predictions_df = test_df[
        [
            "race_id",
            "race_date",
            "venue",
            "race_name",
            "horse_number",
            "finish",
            "target_top3",
        ]
    ].copy()
    rank_scores = _race_ranking_scores(
        test_df,
        probabilities,
    )
    predictions_df["predicted_probability_top3"] = probabilities
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

    return metrics, predictions_df


def main():
    print("===== データ読み込み =====")
    df, feature_columns, categorical_features, numeric_features = load_data()
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
    model.fit(
        train_df[feature_columns],
        train_df["target_top3"].astype(int),
    )

    validation_metrics, _ = evaluate_model(
        model,
        validation_df,
        feature_columns,
    )
    metrics, predictions_df = evaluate_model(
        model,
        test_df,
        feature_columns,
    )
    metrics["validation"] = validation_metrics
    metrics["split"] = {
        "train_end": str(train_df["race_date"].max().date()),
        "validation_start": str(validation_start.date()),
        "validation_end": str(validation_df["race_date"].max().date()),
        "test_start": str(test_start.date()),
    }
    metrics["deployment_model_trained_through"] = str(
        df["race_date"].max().date()
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "top3_model.joblib"
    metrics_path = OUTPUT_DIR / "metrics.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"
    deployment_model = build_model(
        categorical_features,
        numeric_features,
    )
    deployment_model.fit(
        df[feature_columns],
        df["target_top3"].astype(int),
    )
    joblib.dump(
        {
            "model": deployment_model,
            "feature_columns": feature_columns,
            "feature_columns": feature_columns,
            "categorical_features": categorical_features,
            "numeric_features": numeric_features,
            "target": "finish <= 3",
            "prediction_time": "before race start after weigh-in and market ranking",
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
