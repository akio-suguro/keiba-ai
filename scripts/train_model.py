import json
import sqlite3
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
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


DB_PATH = Path("/app/data/keiba.db")
OUTPUT_DIR = Path("/app/data/model")

TEST_RATIO = 0.20
RANDOM_STATE = 42
MIN_FINISH = 1


def load_data():
    """SQLiteから学習データを取得し、特徴量を整える。"""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DBが見つかりません: {DB_PATH}")

    query = """
        SELECT
            r.id AS race_id,
            r.race_date,
            r.venue,
            r.distance_m,
            r.surface,
            r.direction,
            r.race_name,
            h.finish,
            h.frame,
            h.horse_number,
            h.sex_age,
            h.weight AS carried_weight,
            h.jockey,
            h.trainer
        FROM horses h
        JOIN races r ON r.id = h.race_id
        WHERE h.finish IS NOT NULL
          AND r.race_date IS NOT NULL
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError("学習対象のデータがありません。")

    df["race_date"] = pd.to_datetime(
        df["race_date"], errors="coerce"
    )
    df["finish"] = pd.to_numeric(
        df["finish"], errors="coerce"
    )

    df = df.dropna(subset=["race_date", "finish"]).copy()
    df = df[df["finish"] >= MIN_FINISH].copy()

    if df.empty:
        raise ValueError("有効な着順データがありません。")

    # 3着以内を正例とする二値分類
    df["target_top3"] = (df["finish"] <= 3).astype(int)

    numeric_features = [
        "distance_m",
        "frame",
        "horse_number",
        "carried_weight",
    ]

    categorical_features = [
        "venue",
        "surface",
        "direction",
        "race_name",
        "sex_age",
        "jockey",
        "trainer",
    ]

    # 数値型へ変換
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # 全て欠損の数値列は除外
    numeric_features = [
        col for col in numeric_features
        if df[col].notna().any()
    ]

    # カテゴリ列は欠損値を明示的に扱う
    for col in categorical_features:
        df[col] = (
            df[col]
            .fillna("不明")
            .astype(str)
            .replace("", "不明")
        )

    feature_columns = categorical_features + numeric_features

    if not feature_columns:
        raise ValueError("使用可能な特徴量がありません。")

    # 人気順位、確定オッズ、着差、走破タイムなどの
    # レース結果確定後の情報は使用しない。
    return (
        df,
        feature_columns,
        categorical_features,
        numeric_features,
    )


def chronological_split(df):
    """開催日単位で時系列分割する。"""
    dates = sorted(
        df["race_date"].dt.normalize().dropna().unique()
    )

    if len(dates) < 2:
        raise ValueError(
            "時系列評価には異なる開催日が2日以上必要です。"
        )

    split_index = int(len(dates) * (1 - TEST_RATIO))
    split_index = max(1, min(split_index, len(dates) - 1))

    test_start_date = pd.Timestamp(dates[split_index])

    train_df = df[
        df["race_date"] < test_start_date
    ].copy()

    test_df = df[
        df["race_date"] >= test_start_date
    ].copy()

    if train_df.empty or test_df.empty:
        raise ValueError(
            "学習データまたはテストデータが空です。"
        )

    # 学習側とテスト側に同じレースIDが混入していないか
    overlap = set(train_df["race_id"]) & set(test_df["race_id"])
    if overlap:
        raise ValueError(
            f"学習・テスト間でレースIDが重複しています: {len(overlap)}"
        )

    if train_df["target_top3"].nunique() < 2:
        raise ValueError(
            "学習データの正解ラベルが1種類しかありません。"
        )

    return train_df, test_df, test_start_date


def build_model(
    categorical_features,
    numeric_features,
):
    """前処理と分類器を一つのPipelineにまとめる。"""
    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
            ]
        )

        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        categorical_pipeline = Pipeline(
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
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    if not transformers:
        raise ValueError("モデルに使用できる特徴量がありません。")

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=3,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return model


def evaluate_model(model, test_df, feature_columns):
    """テストデータを使って予測性能を評価する。"""
    X_test = test_df[feature_columns]
    y_test = test_df["target_top3"].astype(int)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "RandomForestClassifier",
        "target": "finish <= 3",
        "sklearn_version": sklearn.__version__,
        "test_horses": int(len(test_df)),
        "test_races": int(test_df["race_id"].nunique()),
        "test_start_date": str(
            test_df["race_date"].min().date()
        ),
        "test_end_date": str(
            test_df["race_date"].max().date()
        ),
        "features": feature_columns,
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(
            accuracy_score(y_test, predictions)
        ),
        "precision": float(
            precision_score(
                y_test, predictions, zero_division=0
            )
        ),
        "recall": float(
            recall_score(
                y_test, predictions, zero_division=0
            )
        ),
        "classification_report": classification_report(
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
        ["race_id", "race_date", "venue", "race_name",
         "horse_number", "finish", "target_top3"]
    ].copy()

    predictions_df["predicted_probability_top3"] = probabilities
    predictions_df["predicted_top3"] = predictions

    return metrics, predictions_df


def main():
    print("===== データ読み込み =====")

    (
        df,
        feature_columns,
        categorical_features,
        numeric_features,
    ) = load_data()

    train_df, test_df, test_start_date = chronological_split(df)

    print(f"全データ: {len(df):,}頭")
    print(f"全レース数: {df['race_id'].nunique():,}")
    print(f"学習データ: {len(train_df):,}頭")
    print(f"学習レース数: {train_df['race_id'].nunique():,}")
    print(f"テストデータ: {len(test_df):,}頭")
    print(f"テストレース数: {test_df['race_id'].nunique():,}")
    print(f"テスト開始日: {test_start_date.date()}")
    print(f"学習期間: {train_df['race_date'].min().date()} "
          f"〜 {train_df['race_date'].max().date()}")
    print(f"テスト期間: {test_df['race_date'].min().date()} "
          f"〜 {test_df['race_date'].max().date()}")
    print(f"特徴量: {feature_columns}")

    X_train = train_df[feature_columns]
    y_train = train_df["target_top3"].astype(int)

    model = build_model(
        categorical_features,
        numeric_features,
    )

    print("\n===== 学習開始 =====")
    model.fit(X_train, y_train)

    print("\n===== テスト評価 =====")
    metrics, predictions_df = evaluate_model(
        model, test_df, feature_columns
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "top3_model.joblib"
    metrics_path = OUTPUT_DIR / "metrics.json"
    predictions_path = OUTPUT_DIR / "test_predictions.csv"

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": "finish <= 3",
            "excluded_features": [
                "popularity",
                "odds",
                "time",
                "margin",
                "last_3f",
                "body_weight",
                "weight_change",
            ],
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            ensure_ascii=False,
            indent=2,
        )

    predictions_df.to_csv(
        predictions_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n===== 評価結果 =====")
    print(f"正解率 (Accuracy): {metrics['accuracy']:.4f}")
    print(f"適合率 (Precision): {metrics['precision']:.4f}")
    print(f"再現率 (Recall): {metrics['recall']:.4f}")

    if metrics["roc_auc"] is not None:
        print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
        print(
            "Average Precision: "
            f"{metrics['average_precision']:.4f}"
        )
    else:
        print("テストデータに正例または負例がないため、"
              "ROC-AUCとAverage Precisionは計算できません。")

    print("\n===== 分類レポート =====")
    print(
        classification_report(
            test_df["target_top3"].astype(int),
            (
                model.predict_proba(test_df[feature_columns])[:, 1]
                >= 0.5
            ).astype(int),
            labels=[0, 1],
            target_names=["4着以下", "3着以内"],
            zero_division=0,
        )
    )

    print(f"学習済みモデル: {model_path}")
    print(f"評価結果: {metrics_path}")
    print(f"テスト予測: {predictions_path}")
    print("\n学習完了")


if __name__ == "__main__":
    main()
