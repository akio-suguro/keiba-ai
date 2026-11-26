
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

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
    from .database import connect_db, read_sql_query
else:
    from database import connect_db, read_sql_query


# ============================================================
# 設定
# ============================================================

# Local SQLite fallback path; DATABASE_URL selects PostgreSQL.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "data" / "keiba.db"

# 学習結果の保存先
OUTPUT_DIR = PROJECT_ROOT / "data" / "model" / "upset"

# 時系列分割
TRAIN_START = "2002-01-01"
TRAIN_END = "2022-12-31"

VALID_START = "2023-01-01"
VALID_END = "2024-12-31"

TEST_START = "2025-01-01"

# 波乱レースの定義
# 人気順位が7以上の馬が1〜3着に1頭以上入ったレース
UPSET_POPULARITY_THRESHOLD = 7

RANDOM_STATE = 42
TRAINING_WINDOW_YEARS = 5
FEATURE_HISTORY_YEARS = 10


class SigmoidCalibratedModel:
    """Apply a chronological Platt calibration to a fitted estimator."""

    def __init__(self, estimator, calibrator):
        self.estimator = estimator
        self.calibrator = calibrator

    @property
    def named_steps(self):
        return self.estimator.named_steps

    @property
    def feature_names_in_(self):
        return self.estimator.feature_names_in_

    @property
    def classes_(self):
        return self.estimator.classes_

    def predict_proba(self, features):
        raw_probability = self.estimator.predict_proba(features)[:, 1]
        raw_probability = np.clip(raw_probability, 1e-6, 1 - 1e-6)
        logits = np.log(raw_probability / (1 - raw_probability)).reshape(-1, 1)
        calibrated = self.calibrator.predict_proba(logits)[:, 1]
        return np.column_stack((1 - calibrated, calibrated))


def fit_probability_calibrator(
    labels: pd.Series,
    probabilities: np.ndarray,
) -> LogisticRegression:
    """Fit sigmoid probability calibration on chronological validation data."""
    y = np.asarray(labels, dtype=int)
    if len(np.unique(y)) != 2:
        raise ValueError(
            "確率校正には両クラスを含む時系列検証データが必要です。"
        )
    probabilities = np.clip(
        np.asarray(probabilities, dtype=float),
        1e-6,
        1 - 1e-6,
    )
    logits = np.log(probabilities / (1 - probabilities)).reshape(-1, 1)
    calibrator = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    calibrator.fit(logits, y)
    return calibrator


# ============================================================
# DB 読み込み
# ============================================================

def _connect_database():
    if os.environ.get("DATABASE_URL", "").strip():
        return connect_db()
    return sqlite3.connect(DB_PATH)


def load_database() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Configured databaseからレースと出走馬のデータを読み込む。"""

    if not os.environ.get("DATABASE_URL", "").strip() and not DB_PATH.exists():
        raise FileNotFoundError(
            f"DB が見つかりません: {DB_PATH}\n"
            "Docker Compose の volume 設定を確認してください。"
        )

    with _connect_database() as conn:
        latest_date_frame = read_sql_query(
            "SELECT MAX(race_date) AS latest_date FROM races",
            conn,
        )
        latest_date = pd.to_datetime(
            latest_date_frame.loc[0, "latest_date"],
            errors="coerce",
        )
        if pd.isna(latest_date):
            raise ValueError("レース日が有効なデータがありません。")
        history_start = (
            latest_date - pd.DateOffset(years=FEATURE_HISTORY_YEARS)
        ).date().isoformat()
        races = read_sql_query(
            "SELECT * FROM races WHERE race_date >= ?",
            conn,
            params=(history_start,),
        )
        horses = read_sql_query(
            """
            SELECT h.*
            FROM horses h
            JOIN races r ON r.id = h.race_id
            WHERE r.race_date >= ?
            """,
            conn,
            params=(history_start,),
        )

    required_races = {
        "id",
        "race_date",
        "venue",
        "distance_m",
        "surface",
    }

    required_horses = {
        "race_id",
        "finish",
        "popularity",
    }

    missing_races = required_races - set(races.columns)
    if missing_races:
        raise ValueError(
            "races テーブルに必要な列がありません: "
            f"{sorted(missing_races)}"
        )

    missing_horses = required_horses - set(horses.columns)
    if missing_horses:
        raise ValueError(
            "horses テーブルに必要な列がありません: "
            f"{sorted(missing_horses)}"
        )

    # 日付を datetime に変換
    races["race_date"] = pd.to_datetime(
        races["race_date"],
        errors="coerce",
    )

    races = races.dropna(subset=["race_date"]).copy()

    # 数値列を数値型に変換
    for col in ("finish", "popularity"):
        horses[col] = pd.to_numeric(
            horses[col],
            errors="coerce",
        )

    horses["race_id"] = horses["race_id"].astype(str)
    races["id"] = races["id"].astype(str)

    # 重複・不正データを診断できるよう、ここでは
    # 着順・人気順位の欠損行を一括削除しない。
    return races, horses


# ============================================================
# 波乱レースのラベル作成
# ============================================================

def build_race_labels(
    races: pd.DataFrame,
    horses: pd.DataFrame,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    7番人気以下の馬が1〜3着に入ったレースを波乱とする。

    target_race_upset:
        1 = 人気順位7以上の馬が1〜3着にいる
        0 = 1〜3着の馬がすべて6番人気以内

    正解ラベルを確実に作成できないレースは除外する。
    """

    race_rows: list[dict[str, Any]] = []
    excluded_incomplete = 0
    excluded_invalid_popularity = 0

    for race_id, group in horses.groupby("race_id", sort=False):
        valid = group.copy()

        valid["finish"] = pd.to_numeric(
            valid["finish"],
            errors="coerce",
        )
        valid["popularity"] = pd.to_numeric(
            valid["popularity"],
            errors="coerce",
        )

        # 出走頭数。対象馬の重複なども監査しやすいよう、
        # 元データの行数を使用する。
        field_size = len(valid)

        if field_size < 3:
            excluded_incomplete += 1
            continue

        # 着順は整数であり、1着・2着・3着がそれぞれ
        # 1頭ずつ存在するレースだけを対象とする。
        finish = valid["finish"]
        finish_is_integer = (
            finish.notna()
            & np.isfinite(finish)
            & (finish % 1 == 0)
        )

        valid_finish = valid.loc[finish_is_integer].copy()

        placed = valid_finish[
            valid_finish["finish"].isin([1, 2, 3])
        ].copy()

        finish_counts = placed["finish"].value_counts()

        if (
            set(finish_counts.index.astype(int)) != {1, 2, 3}
            or not finish_counts.eq(1).all()
        ):
            excluded_incomplete += 1
            continue

        # 1〜3着の人気順位がすべて有効であることを確認
        popularity = placed["popularity"]

        popularity_is_valid = (
            popularity.notna()
            & np.isfinite(popularity)
            & (popularity >= 1)
            & (popularity <= field_size)
            & (popularity % 1 == 0)
        )

        if not popularity_is_valid.all():
            excluded_invalid_popularity += 1
            continue

        # 7番人気以下の馬が3着以内にいるか
        upset = bool(
            placed["popularity"].ge(
                UPSET_POPULARITY_THRESHOLD
            ).any()
        )

        race_rows.append(
            {
                "race_id": str(race_id),
                "target_race_upset": int(upset),
                "field_size": field_size,
            }
        )

    labels = pd.DataFrame(
        race_rows,
        columns=[
            "race_id",
            "target_race_upset",
            "field_size",
        ],
    )

    if labels.empty:
        raise ValueError(
            "波乱ラベルを作成できるレースがありません。"
            "DB の race_id、finish、popularity を確認してください。"
        )

    data = races.merge(
        labels,
        left_on="id",
        right_on="race_id",
        how="inner",
        validate="one_to_one",
    )

    data["year"] = data["race_date"].dt.year

    # ラベル監査用ファイル。
    # 人気順位はラベル作成に使うが、モデル特徴量には使わない。
    audit_columns = [
        "id",
        "race_date",
        "venue",
        "race_name",
        "distance_m",
        "field_size",
        "target_race_upset",
    ]

    audit = data[
        [col for col in audit_columns if col in data.columns]
    ].copy()

    audit = audit.rename(
        columns={"target_race_upset": "actual_upset"}
    )

    if verbose:
        print("\n=== ラベル作成結果 ===")
        print(f"ラベル作成対象レース数: {len(data)}")
        print(
            "波乱レース数: "
            f"{int(data['target_race_upset'].sum())}"
        )
        print(
            "通常レース数: "
            f"{int((data['target_race_upset'] == 0).sum())}"
        )
        print(
            "波乱率: "
            f"{data['target_race_upset'].mean():.2%}"
        )
        print(f"3着以内の着順不備で除外: {excluded_incomplete}")
        print(
            "3着以内の人気順位不備で除外: "
            f"{excluded_invalid_popularity}"
        )
        print("1〜3着が確定できないレースは除外")

    return data, audit


# ============================================================
# 特徴量作成
# ============================================================

def add_race_features(
    data: pd.DataFrame,
    races: pd.DataFrame,
    horses: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    レース前の出走馬情報と過去実績だけからレース特徴量を作る。

    現レースの着順・人気・オッズ・タイム・払戻金は使わない。
    """
    race_frame = data.copy()
    for column in (
        "venue",
        "surface",
        "direction",
        "race_name",
        "distance_m",
        "race_number",
        "start_time",
        "weather",
        "turf_condition",
        "dirt_condition",
        "race_grade",
        "race_conditions",
    ):
        if column not in race_frame.columns:
            race_frame[column] = np.nan

    race_frame["distance_m"] = pd.to_numeric(
        race_frame["distance_m"],
        errors="coerce",
    )
    race_frame["distance_band"] = pd.cut(
        race_frame["distance_m"],
        bins=[0, 1400, 1800, 2200, 10000],
        labels=["short", "mile_middle", "middle_long", "long"],
        include_lowest=True,
    ).astype("object")
    race_frame["race_number"] = pd.to_numeric(
        race_frame["race_number"],
        errors="coerce",
    )
    start_parts = race_frame["start_time"].astype("string").str.extract(
        r"(\d{1,2})[:：](\d{2})"
    )
    race_frame["start_minutes"] = (
        pd.to_numeric(start_parts[0], errors="coerce") * 60
        + pd.to_numeric(start_parts[1], errors="coerce")
    )

    if "target_race_upset" in race_frame.columns:
        label_history = race_frame[
            ["race_id", "race_date", "race_name", "target_race_upset"]
        ].copy()
    else:
        label_history, _ = build_race_labels(
            races,
            horses,
            verbose=False,
        )
        label_history = label_history[
            ["race_id", "race_date", "race_name", "target_race_upset"]
        ]
    label_history["race_date"] = pd.to_datetime(
        label_history["race_date"],
        errors="coerce",
    ).dt.normalize()
    label_history = label_history.dropna(
        subset=["race_date", "race_name", "target_race_upset"]
    )
    if not label_history.empty:
        daily_name_rates = label_history.groupby(
            ["race_name", "race_date"],
            as_index=False,
            sort=True,
            observed=True,
        ).agg(
            __races=("target_race_upset", "size"),
            __upsets=("target_race_upset", "sum"),
        )
        grouped_name = daily_name_rates.groupby(
            "race_name",
            sort=False,
            observed=True,
        )
        daily_name_rates["race_name_prior_races"] = (
            grouped_name["__races"].cumsum() - daily_name_rates["__races"]
        )
        prior_upsets = (
            grouped_name["__upsets"].cumsum() - daily_name_rates["__upsets"]
        )
        daily_name_rates["race_name_prior_upset_rate"] = prior_upsets.div(
            daily_name_rates["race_name_prior_races"].replace(0, np.nan)
        )
        race_frame = race_frame.merge(
            daily_name_rates[
                [
                    "race_name",
                    "race_date",
                    "race_name_prior_races",
                    "race_name_prior_upset_rate",
                ]
            ],
            on=["race_name", "race_date"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        race_frame["race_name_prior_races"] = (
            race_frame["race_name_prior_races"].fillna(0)
        )
    else:
        race_frame["race_name_prior_races"] = 0
        race_frame["race_name_prior_upset_rate"] = np.nan

    date_by_race = races[["id", "race_date"]].rename(
        columns={"id": "race_id"}
    )
    horse_context = horses.drop(
        columns=["race_date"],
        errors="ignore",
    ).merge(
        date_by_race,
        on="race_id",
        how="left",
        validate="many_to_one",
    )
    horse_context = add_sex_age_features(horse_context)
    horse_context = add_historical_performance_features(horse_context)

    numeric_horse_features = [
        "frame",
        "horse_number",
        "weight",
        "body_weight",
        "weight_change",
        "age",
        *HISTORICAL_FEATURE_COLUMNS,
    ]
    available_horse_features = []
    for column in numeric_horse_features:
        if column not in horse_context.columns:
            continue
        horse_context[column] = pd.to_numeric(
            horse_context[column],
            errors="coerce",
        )
        if horse_context[column].notna().any():
            available_horse_features.append(column)

    scoped_horses = horse_context[
        horse_context["race_id"].isin(race_frame["race_id"])
    ]
    if available_horse_features:
        grouped = scoped_horses.groupby("race_id")[
            available_horse_features
        ]
        aggregate = grouped.agg(["mean", "std", "min", "max"])
        aggregate.columns = [
            f"field_{column}_{statistic}"
            for column, statistic in aggregate.columns
        ]
        race_frame = race_frame.merge(
            aggregate.reset_index(),
            on="race_id",
            how="left",
            validate="one_to_one",
        )

    if "horse_mark" in horse_context.columns:
        marks = horse_context.dropna(
            subset=["horse_mark"]
        ).copy()
        if not marks.empty:
            mark_counts = pd.crosstab(
                marks["race_id"],
                marks["horse_mark"].astype(str),
            )
            mark_counts.columns = [
                f"field_horse_mark_{column}"
                for column in mark_counts.columns
            ]
            race_frame = race_frame.merge(
                mark_counts.reset_index(),
                on="race_id",
                how="left",
                validate="one_to_one",
            )

    race_categorical_candidates = [
        "venue",
        "surface",
        "direction",
        "distance_band",
        "weather",
        "turf_condition",
        "dirt_condition",
        "race_grade",
        "race_conditions",
    ]
    categorical = [
        column
        for column in race_categorical_candidates
        if column in race_frame.columns
        and race_frame[column].notna().any()
    ]
    numeric = [
        "distance_m",
        "race_number",
        "start_minutes",
        "field_size",
        "race_name_prior_races",
        "race_name_prior_upset_rate",
        *[
            column
            for column in race_frame.columns
            if column.startswith("field_")
            and column not in categorical
        ],
    ]
    numeric = list(dict.fromkeys(numeric))
    numeric = [
        column
        for column in numeric
        if column in race_frame.columns
        and pd.to_numeric(race_frame[column], errors="coerce").notna().any()
    ]

    for column in categorical:
        race_frame[column] = (
            race_frame[column]
            .fillna("unknown")
            .astype(str)
            .replace("", "unknown")
        )
    for column in numeric:
        race_frame[column] = pd.to_numeric(
            race_frame[column],
            errors="coerce",
        )

    return race_frame, categorical, numeric


def chronological_split(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    """Use configured date ranges, falling back to date-based proportions."""
    date_key = data["race_date"].dt.normalize()
    train_mask = date_key.between(
        pd.Timestamp(TRAIN_START),
        pd.Timestamp(TRAIN_END),
    )
    valid_mask = date_key.between(
        pd.Timestamp(VALID_START),
        pd.Timestamp(VALID_END),
    )
    test_mask = date_key.ge(pd.Timestamp(TEST_START))

    train = data.loc[train_mask].copy()
    valid = data.loc[valid_mask].copy()
    test = data.loc[test_mask].copy()
    if not train.empty and not valid.empty and not test.empty:
        return (
            train,
            valid,
            test,
            {
                "strategy": "configured_date_ranges",
                "train": f"{TRAIN_START} to {TRAIN_END}",
                "validation": f"{VALID_START} to {VALID_END}",
                "test": f"{TEST_START} onward",
            },
        )

    dates = pd.DatetimeIndex(date_key.dropna().unique()).sort_values()
    if len(dates) < 3:
        raise ValueError(
            "時系列の学習・検証・テストには"
            "異なる開催日が3日以上必要です。"
        )

    train_end_index = max(1, min(int(len(dates) * 0.70), len(dates) - 2))
    validation_end_index = max(
        train_end_index + 1,
        min(int(len(dates) * 0.85), len(dates) - 1),
    )
    validation_start = dates[train_end_index]
    test_start = dates[validation_end_index]

    train_mask = date_key < validation_start
    valid_mask = (date_key >= validation_start) & (date_key < test_start)
    test_mask = date_key >= test_start
    train = data.loc[train_mask].copy()
    valid = data.loc[valid_mask].copy()
    test = data.loc[test_mask].copy()

    split = {
        "strategy": "chronological_date_proportions",
        "train": (
            f"{date_key.loc[train_mask].min().date()} to "
            f"{date_key.loc[train_mask].max().date()}"
        ),
        "validation": (
            f"{date_key.loc[valid_mask].min().date()} to "
            f"{date_key.loc[valid_mask].max().date()}"
        ),
        "test": f"{test_start.date()} onward",
    }
    return train, valid, test, split


# ============================================================
# モデル構築
# ============================================================

def make_pipeline(
    categorical: list[str],
    numeric: list[str],
) -> Pipeline:
    """カテゴリ・数値特徴量を前処理してロジスティック回帰を学習する。"""

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scale",
                StandardScaler(),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                categorical_pipeline,
                categorical,
            ),
            (
                "numeric",
                numeric_pipeline,
                numeric,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=3000,
                    class_weight=None,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# 評価指標
# ============================================================

def evaluate(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    """二値分類モデルを複数の指標で評価する。"""

    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (probabilities >= threshold).astype(int)

    result: dict[str, Any] = {
        "n_races": int(len(y)),
        "actual_upset_rate": float(y.mean()),
        "classification_threshold": float(threshold),
        "predicted_upset_rate": float(
            predictions.mean()
        ),
        "accuracy": float(
            accuracy_score(y, predictions)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(y, predictions)
        ),
        "precision": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "brier_score": float(
            brier_score_loss(y, probabilities)
        ),
        "log_loss": float(
            log_loss(
                y,
                probabilities,
                labels=[0, 1],
            )
        ),
        "confusion_matrix_labels_0_1": confusion_matrix(
            y,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    if len(np.unique(y)) == 2:
        result["roc_auc"] = float(
            roc_auc_score(y, probabilities)
        )
        result["pr_auc"] = float(
            average_precision_score(y, probabilities)
        )
    else:
        result["roc_auc"] = None
        result["pr_auc"] = None

    # 波乱確率が高いと予測した上位10%の実際の波乱率
    ranked = pd.DataFrame(
        {
            "actual": y,
            "probability": probabilities,
        }
    ).sort_values(
        "probability",
        ascending=False,
    )

    n_tail = max(
        1,
        int(np.ceil(len(ranked) * 0.10)),
    )

    result["top_10pct_upset_rate"] = float(
        ranked.head(n_tail)["actual"].mean()
    )

    result["bottom_10pct_upset_rate"] = float(
        ranked.tail(n_tail)["actual"].mean()
    )

    calibration_error = 0.0
    for lower in np.linspace(0, 0.9, 10):
        upper = lower + 0.1
        in_bin = (
            (probabilities >= lower)
            & (
                (probabilities < upper)
                | ((upper >= 1.0) & (probabilities <= upper))
            )
        )
        if in_bin.any():
            calibration_error += float(in_bin.mean()) * abs(
                float(y[in_bin].mean())
                - float(probabilities[in_bin].mean())
            )
    result["expected_calibration_error_10_bins"] = calibration_error

    return result


def choose_balanced_accuracy_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> float:
    """Choose an equal-error-cost threshold on validation data only."""
    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    if len(np.unique(y)) < 2:
        return 0.5

    false_positive_rate, true_positive_rate, thresholds = roc_curve(
        y,
        probabilities,
    )
    balanced_accuracy = (
        true_positive_rate + (1 - false_positive_rate)
    ) / 2
    best_index = int(np.argmax(balanced_accuracy))
    threshold = float(thresholds[best_index])
    return threshold if np.isfinite(threshold) else 0.5


# ============================================================
# 学習・評価・保存
# ============================================================

def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=== 波乱レース予測モデル ===")
    print(f"DB: {DB_PATH}")
    print(
        "波乱の定義: "
        f"{UPSET_POPULARITY_THRESHOLD}番人気以下の馬が3着以内"
    )

    races, horses = load_database()

    print(f"\nDB内レース数: {len(races)}")
    print(f"DB内出走馬データ数: {len(horses)}")
    print(
        "現レースの人気順位・オッズは波乱ラベルの定義と重なるため"
        "特徴量から除外します。"
    )

    data, audit = build_race_labels(
        races,
        horses,
    )

    data, categorical, numeric = add_race_features(
        data,
        races,
        horses,
    )

    latest_date = data["race_date"].max().normalize()
    training_start = latest_date - pd.DateOffset(
        years=TRAINING_WINDOW_YEARS
    )
    data = data.loc[data["race_date"].ge(training_start)].copy()

    target = "target_race_upset"
    feature_cols = categorical + numeric

    train, valid, test, split = chronological_split(data)

    if train[target].nunique() < 2:
        raise ValueError(
            "学習期間のラベルが1種類しかありません。"
            "波乱レースの定義と対象データを確認してください。"
        )

    print("\n=== データ分割 ===")
    print(f"分割方式: {split['strategy']}")
    print(
        f"学習: {split['train']} "
        f"({len(train)}レース)"
    )
    print(
        f"検証: {split['validation']} "
        f"({len(valid)}レース)"
    )
    print(
        f"テスト: {split['test']} "
        f"({len(test)}レース)"
    )

    print("\n学習データの波乱率: "
          f"{train[target].mean():.2%}")
    print("検証データの波乱率: "
          f"{valid[target].mean():.2%}")
    print("テストデータの波乱率: "
          f"{test[target].mean():.2%}")

    # 学習期間だけでモデルを学習
    model = make_pipeline(
        categorical,
        numeric,
    )

    model.fit(
        train[feature_cols],
        train[target].astype(int),
    )

    # Calibrate on the chronological validation interval only.
    valid_raw_probability = model.predict_proba(
        valid[feature_cols]
    )[:, 1]
    probability_calibrator = fit_probability_calibrator(
        valid[target],
        valid_raw_probability,
    )
    valid_model = SigmoidCalibratedModel(model, probability_calibrator)
    valid_prob = valid_model.predict_proba(
        valid[feature_cols]
    )[:, 1]
    test_prob = valid_model.predict_proba(
        test[feature_cols]
    )[:, 1]

    validation_metrics = evaluate(
        valid[target],
        valid_prob,
    )

    classification_threshold = choose_balanced_accuracy_threshold(
        valid[target],
        valid_prob,
    )
    test_metrics = evaluate(
        test[target],
        test_prob,
        threshold=classification_threshold,
    )

    metrics = {
        "model": "standardized_logistic_regression_with_sigmoid_calibration",
        "target": target,
        "definition": (
            "At least one horse with popularity rank >= 7 "
            "finishes in positions 1, 2, or 3"
        ),
        "popularity_threshold": UPSET_POPULARITY_THRESHOLD,
        "split": {
            **split,
            "counts": {
                "train": int(len(train)),
                "validation": int(len(valid)),
                "test": int(len(test)),
            },
        },
        "counts": {
            "all_labeled_races": int(len(data)),
            "train_races": int(len(train)),
            "validation_races": int(len(valid)),
            "test_races": int(len(test)),
        },
        "deployment_model_trained_through": str(
            data["race_date"].max().date()
        ),
        "training_window": {
            "years": TRAINING_WINDOW_YEARS,
            "start": str(training_start.date()),
            "end": str(latest_date.date()),
        },
        "class_rates": {
            "train_upset_rate": float(train[target].mean()),
            "validation_upset_rate": float(valid[target].mean()),
            "test_upset_rate": float(test[target].mean()),
        },
        "validation": validation_metrics,
        "test": test_metrics,
        "classification_threshold_selected_on_validation": (
            classification_threshold
        ),
        "probability_calibration": {
            "method": "Platt sigmoid calibration fitted on chronological validation data",
            "intercept": float(probability_calibrator.intercept_[0]),
            "coefficient": float(probability_calibrator.coef_[0, 0]),
            "validation": validation_metrics,
            "test": test_metrics,
        },
        "categorical_features": categorical,
        "numeric_features": numeric,
        "features": feature_cols,
        "excluded_post_event_fields": [
            "finish",
            "popularity",
            "odds",
            "time",
            "margin",
            "last_3f",
            "payouts",
            "max_exotic_payout",
            "target_race_upset",
        ],
        "notes": [
            "Popularity and odds from the target race are excluded to prevent label leakage.",
            "Horse, jockey and trainer form features use only results from earlier dates.",
            "Race-entry and weigh-in fields are aggregated from the current field.",
            "Races with incomplete top-three results are excluded.",
            "Races with invalid top-three popularity ranks are excluded.",
            "The test period is not used for model fitting.",
            "The classification threshold maximizes balanced accuracy on validation data only.",
        ],
    }

    # モデル保存
    model_path = OUTPUT_DIR / "upset_model.joblib"
    deployment_estimator = make_pipeline(
        categorical,
        numeric,
    )
    deployment_estimator.fit(
        data[feature_cols],
        data[target].astype(int),
    )
    deployment_model = SigmoidCalibratedModel(
        deployment_estimator,
        probability_calibrator,
    )
    joblib.dump(deployment_model, model_path)

    # 指標保存
    metrics_path = OUTPUT_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    # テスト期間の予測結果
    predictions = test[
        [
            "id",
            "race_date",
            "venue",
            "race_name",
            "surface",
            "distance_m",
            "field_size",
        ]
    ].copy()

    predictions["actual_upset"] = (
        test[target].astype(int).to_numpy()
    )

    predictions["upset_probability"] = test_prob
    predictions["predicted_upset"] = (
        predictions["upset_probability"]
        .ge(classification_threshold)
        .astype(int)
    )

    predictions["risk_rank"] = (
        predictions["upset_probability"]
        .rank(
            method="first",
            ascending=False,
        )
        .astype(int)
    )

    predictions = predictions.sort_values(
        "upset_probability",
        ascending=False,
    )

    predictions_path = OUTPUT_DIR / "race_predictions.csv"

    predictions.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    # ラベルを監査するための一覧
    audit_path = OUTPUT_DIR / "label_audit.csv"

    audit.to_csv(
        audit_path,
        index=False,
        encoding="utf-8-sig",
    )

    # 結果表示
    print("\n=== 検証指標 ===")
    print(
        json.dumps(
            validation_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== 最終テスト指標 ===")
    print(
        json.dumps(
            test_metrics,
            ensure_ascii=False,
            indent=2,
        )
    )

    print("\n=== 保存完了 ===")
    print(f"モデル: {model_path}")
    print(f"指標: {metrics_path}")
    print(f"テスト予測一覧: {predictions_path}")
    print(f"ラベル監査一覧: {audit_path}")


if __name__ == "__main__":
    main()
