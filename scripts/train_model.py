
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
TEST_RATIO = 0.2
RANDOM_STATE = 42


def load_data():
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
            h.trainer,
            h.popularity
        FROM horses h
        JOIN races r ON r.id = h.race_id
        WHERE h.finish IS NOT NULL
          AND r.race_date IS NOT NULL
    """

    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query(query, conn)

    if df.empty:
        raise ValueError("学習対象のデータがありません。")

    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")
    df["finish"] = pd.to_numeric(df["finish"], errors="coerce")

    df = df.dropna(subset=["race_date", "finish"]).copy()
    df = df[df["finish"] >= 1].copy()

    if df.empty:
        raise ValueError("有効な着順データがありません。")

    # 3着以内なら1、それ以外なら0
    df["target_top3"] = (df["finish"] <= 3).astype(int)

    # 数値列は数値に変換。不正値は欠損値として扱う
    numeric_features = [
        "distance_m",
        "frame",
        "horse_number",
        "carried_weight",
        "popularity",
    ]

    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    categorical_features = [
        "venue",
        "surface",
        "direction",
        "race_name",
        "sex_age",
        "jockey",
        "trainer",
    ]

    for col in categorical_features:
        df[col] = df[col].fillna("不明").astype(str)

    # 全て欠損の列を除外
    numeric_features = [
        col for col in numeric_features if df[col].notna().any()
    ]

    feature_columns = categorical_features + numeric_features

    # popularity はレース結果ページに記録された最終人気。
    # 発走前にも利用可能な情報として扱うが、別途取得元の確認が必要。
    return df, feature_columns, categorical_features, numeric_features


def main():
    print("===== データ読み込み =====")
    df, feature_columns, categorical_features, numeric_features = load_data()

    # 同じレースの馬が学習・テスト両方に入らないよう、
    # 日付単位で時系列分割する
    dates = sorted(df["race_date"].dt.normalize().unique())

    if len(dates) < 2:
        raise ValueError(
            "時系列評価には異なる開催日が2日以上必要です。"
        )

    split_index = int(len(dates) * (1 - TEST_RATIO))
    split_index = max(1, min(split_index, len(dates) - 1))
    test_start_date = pd.Timestamp(dates[split_index])

    train_df = df[df["race_date"] < test_start_date].copy()
    test_df = df[df["race_date"] >= test_start_date].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("学習データまたはテストデータが空です。")

    X_train = train_df[feature_columns]
    y_train = train_df["target_top3"]
    X_test = test_df[feature_columns]
    y_test = test_df["target_top3"]

    if y_train.nunique() < 2:
        raise ValueError(
            "学習データの目的変数が1種類しかありません。"
            "データ量や分割期間を確認してください。"
        )

    print(f"全データ: {len(df)}頭")
    print(f"学習データ: {len(train_df)}頭")
    print(f"テストデータ: {len(test_df)}頭")
    print(f"テスト開始日: {test_start_date.date()}")
    print(f"学習対象のレース数: {train_df['race_id'].nunique()}")
    print(f"テスト対象のレース数: {test_df['race_id'].nunique()}")
    print(f"特徴量: {feature_columns}")

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

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
                OneHotEncoder(handle_unknown="ignore"),
            ),
        ]
    )

    transformers = []

    if numeric_features:
        transformers.append(
            ("numeric", numeric_pipeline, numeric_features)
        )

    if categorical_features:
        transformers.append(
            ("categorical", categorical_pipeline, categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

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

    print("\n===== 学習開始 =====")
    model.fit(X_train, y_train)

    print("===== 評価 =====")
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "model": "RandomForestClassifier",
        "target": "finish <= 3",
        "sklearn_version": sklearn.__version__,
        "total_horses": int(len(df)),
        "train_horses": int(len(train_df)),
        "test_horses": int(len(test_df)),
        "train_races": int(train_df["race_id"].nunique()),
        "test_races": int(test_df["race_id"].nunique()),
        "train_start_date": str(train_df["race_date"].min().date()),
        "train_end_date": str(train_df["race_date"].max().date()),
        "test_start_date": str(test_df["race_date"].min().date()),
        "test_end_date": str(test_df["race_date"].max().date()),
        "features": feature_columns,
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(
            precision_score(y_test, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y_test, predictions, zero_division=0)
        ),
        "classification_report": classification_report(
            y_test,
            predictions,
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
        print(
            "注意: テストデータの正解ラベルが1種類だけのため、"
            "ROC-AUCとAverage Precisionは計算できません。"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = OUTPUT_DIR / "top3_model.joblib"
    metrics_path = OUTPUT_DIR / "metrics.json"

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": "finish <= 3",
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

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

    print("\n===== 分類レポート =====")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["4着以下", "3着以内"],
            zero_division=0,
        )
    )

    print(f"学習済みモデル: {model_path}")
    print(f"評価結果: {metrics_path}")
    print("\n学習完了")


if __name__ == "__main__":
    main()
