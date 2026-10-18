
from __future__ import annotations

import json
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
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# 設定
# ============================================================

# Docker コンテナ内の SQLite DB
DB_PATH = Path("/app/data/keiba.db")

# 学習結果の保存先
OUTPUT_DIR = Path("/app/data/model/upset")

# 時系列分割
TRAIN_START = "2015-01-01"
TRAIN_END = "2022-12-31"

VALID_START = "2023-01-01"
VALID_END = "2024-12-31"

TEST_START = "2025-01-01"

# 波乱レースの定義
# 人気順位が7以上の馬が1〜3着に1頭以上入ったレース
UPSET_POPULARITY_THRESHOLD = 7

RANDOM_STATE = 42


# ============================================================
# DB 読み込み
# ============================================================

def load_database() -> tuple[pd.DataFrame, pd.DataFrame]:
    """SQLite DB からレースと出走馬のデータを読み込む。"""

    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DB が見つかりません: {DB_PATH}\n"
            "Docker Compose の volume 設定を確認してください。"
        )

    with sqlite3.connect(DB_PATH) as conn:
        races = pd.read_sql_query(
            "SELECT * FROM races",
            conn,
        )
        horses = pd.read_sql_query(
            "SELECT * FROM horses",
            conn,
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
    excluded_invalid_finish = 0

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
    print(f"不正な着順行を含むレースの扱い: 着順検証で判定")

    return data, audit


# ============================================================
# 特徴量作成
# ============================================================

def add_race_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    レース前に分かる情報だけで特徴量を作る。

    finish、popularity、odds、タイム、払戻金は使わない。
    """

    df = data.copy()

    # 収集元によって一部の列が存在しない場合にも対応
    for col in (
        "venue",
        "surface",
        "direction",
        "race_name",
        "distance_m",
    ):
        if col not in df.columns:
            df[col] = np.nan

    df["distance_m"] = pd.to_numeric(
        df["distance_m"],
        errors="coerce",
    )

    # 距離をカテゴリ化
    df["distance_band"] = pd.cut(
        df["distance_m"],
        bins=[0, 1400, 1800, 2200, 10000],
        labels=[
            "short",
            "mile_middle",
            "middle_long",
            "long",
        ],
        include_lowest=True,
    ).astype("object")

    categorical = [
        "venue",
        "surface",
        "direction",
        "race_name",
        "distance_band",
    ]

    numeric = [
        "distance_m",
        "field_size",
    ]

    for col in categorical:
        df[col] = (
            df[col]
            .fillna("unknown")
            .astype(str)
        )

    for col in numeric:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    return df, categorical, numeric


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
) -> dict[str, Any]:
    """二値分類モデルを複数の指標で評価する。"""

    y = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predictions = (probabilities >= 0.5).astype(int)

    result: dict[str, Any] = {
        "n_races": int(len(y)),
        "actual_upset_rate": float(y.mean()),
        "predicted_upset_rate_at_0_5": float(
            predictions.mean()
        ),
        "accuracy_at_0_5": float(
            accuracy_score(y, predictions)
        ),
        "precision_at_0_5": float(
            precision_score(
                y,
                predictions,
                zero_division=0,
            )
        ),
        "recall_at_0_5": float(
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

    return result


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
        "オッズ列は学習に使用しません。"
        "NULLのままで問題ありません。"
    )

    data, audit = build_race_labels(
        races,
        horses,
    )

    data, categorical, numeric = add_race_features(
        data
    )

    target = "target_race_upset"
    feature_cols = categorical + numeric

    # 時系列で分割する。未来のレースを学習に混ぜない。
    train_mask = data["race_date"].between(
        pd.Timestamp(TRAIN_START),
        pd.Timestamp(TRAIN_END),
    )

    valid_mask = data["race_date"].between(
        pd.Timestamp(VALID_START),
        pd.Timestamp(VALID_END),
    )

    test_mask = data["race_date"].ge(
        pd.Timestamp(TEST_START)
    )

    train = data.loc[train_mask].copy()
    valid = data.loc[valid_mask].copy()
    test = data.loc[test_mask].copy()

    if train.empty or valid.empty or test.empty:
        raise ValueError(
            "学習・検証・テストのいずれかが空です。\n"
            f"train={len(train)}, "
            f"validation={len(valid)}, "
            f"test={len(test)}\n"
            "race_date の収録範囲を確認してください。"
        )

    if train[target].nunique() < 2:
        raise ValueError(
            "学習期間のラベルが1種類しかありません。"
            "波乱レースの定義と対象データを確認してください。"
        )

    print("\n=== データ分割 ===")
    print(
        f"学習: {TRAIN_START} ～ {TRAIN_END} "
        f"({len(train)}レース)"
    )
    print(
        f"検証: {VALID_START} ～ {VALID_END} "
        f"({len(valid)}レース)"
    )
    print(
        f"テスト: {TEST_START} 以降 "
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

    # 検証・テストデータの波乱確率
    valid_prob = model.predict_proba(
        valid[feature_cols]
    )[:, 1]

    test_prob = model.predict_proba(
        test[feature_cols]
    )[:, 1]

    validation_metrics = evaluate(
        valid[target],
        valid_prob,
    )

    test_metrics = evaluate(
        test[target],
        test_prob,
    )

    metrics = {
        "model": "standardized_unweighted_logistic_regression",
        "target": target,
        "definition": (
            "At least one horse with popularity rank >= 7 "
            "finishes in positions 1, 2, or 3"
        ),
        "popularity_threshold": UPSET_POPULARITY_THRESHOLD,
        "split": {
            "train": f"{TRAIN_START} to {TRAIN_END}",
            "validation": f"{VALID_START} to {VALID_END}",
            "test": f"{TEST_START} onward",
        },
        "counts": {
            "all_labeled_races": int(len(data)),
            "train_races": int(len(train)),
            "validation_races": int(len(valid)),
            "test_races": int(len(test)),
        },
        "class_rates": {
            "train_upset_rate": float(train[target].mean()),
            "validation_upset_rate": float(valid[target].mean()),
            "test_upset_rate": float(test[target].mean()),
        },
        "validation": validation_metrics,
        "test": test_metrics,
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
            "Odds are not used as a feature or target.",
            "Races with incomplete top-three results are excluded.",
            "Races with invalid top-three popularity ranks are excluded.",
            "The test period is not used for model fitting.",
        ],
    }

    # モデル保存
    model_path = OUTPUT_DIR / "upset_model.joblib"
    joblib.dump(model, model_path)

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
