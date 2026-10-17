from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
)
from sklearn.calibration import CalibratedClassifierCV

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "keiba.db"
MODEL_PATH = ROOT / "models" / "chaos_model.joblib"

FEATURES = [
    "field_size",
    "log_fav_odds",
    "log_third_odds",
    "top3_implied_share",
    "odds_log_std",
    "popularity_spread",
]

CHAOS_PAYOUT_YEN = 100_000


def build_dataset() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as con:
        df = pd.read_sql_query(
            """
            SELECT
                r.race_id,
                r.race_date,
                r.field_size,
                r.trifecta_payout_yen,
                e.win_odds,
                e.popularity
            FROM races r
            JOIN entries e ON e.race_id = r.race_id
            """,
            con,
        )

    df["race_date"] = pd.to_datetime(df["race_date"])

    records = []

    for race_id, group in df.groupby("race_id"):
        odds = pd.to_numeric(group["win_odds"], errors="coerce")
        odds = odds.replace([np.inf, -np.inf], np.nan).dropna()
        odds = odds[odds > 0].sort_values()

        popularity = pd.to_numeric(
            group["popularity"], errors="coerce"
        ).dropna()

        payout = pd.to_numeric(
            group["trifecta_payout_yen"], errors="coerce"
        ).dropna()

        field_size = pd.to_numeric(
            group["field_size"], errors="coerce"
        ).dropna()

        # 初期版ではオッズ・人気順位が揃わないレースを除外する。
        if len(odds) < 3 or len(popularity) < 3:
            continue

        if len(payout) == 0 or len(field_size) == 0:
            continue

        implied = 1.0 / odds.to_numpy()
        implied_share = implied / implied.sum()

        date = group["race_date"].iloc[0]

        records.append({
            "race_id": race_id,
            "race_date": date,
            "field_size": float(field_size.iloc[0]),
            "log_fav_odds": float(np.log(odds.iloc[0])),
            "log_third_odds": float(np.log(odds.iloc[2])),
            "top3_implied_share": float(
                implied_share[:3].sum()
            ),
            "odds_log_std": float(np.log(odds).std()),
            "popularity_spread": float(
                popularity.max() - popularity.min()
            ),
            "chaos": int(payout.iloc[0] >= CHAOS_PAYOUT_YEN),
        })

    return pd.DataFrame(records).sort_values("race_date")


def main():
    data = build_dataset()

    if len(data) < 200:
        raise ValueError(
            f"学習用レースが不足しています: {len(data)}件。"
            "データを追加してください。"
        )

    # 年代順に分割。未来のデータを学習に混ぜない。
    split = int(len(data) * 0.8)
    train = data.iloc[:split]
    test = data.iloc[split:]

    if train["chaos"].nunique() < 2 or test["chaos"].nunique() < 2:
        raise ValueError(
            "学習または評価データにクラスが1種類しかありません。"
            "データ量・期間・荒れレースの定義を見直してください。"
        )

    model = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.06,
        max_leaf_nodes=15,
        l2_regularization=1.0,
        random_state=42,
    )

    model.fit(train[FEATURES], train["chaos"])

    # 校正は学習データ内の時系列検証で行うべき。
    # v1では確率を未校正のスコアとして扱い、
    # 確率の正確さは別途評価する。
    probabilities = model.predict_proba(test[FEATURES])[:, 1]

    print("評価レース数:", len(test))
    print(
        "荒れレース比率:",
        round(float(test["chaos"].mean()), 3),
    )
    print(
        "Average precision:",
        round(average_precision_score(
            test["chaos"], probabilities
        ), 3),
    )
    print(
        "Brier score:",
        round(brier_score_loss(
            test["chaos"], probabilities
        ), 3),
    )

    # 仮に上位30%を見送り候補とする。
    threshold = float(np.quantile(probabilities, 0.70))
    skip = probabilities >= threshold

    print("見送り率:", round(float(skip.mean()), 3))
    print(
        "見送り候補の荒れレース捕捉率:",
        round(float(recall_score(
            test["chaos"], skip, zero_division=0
        )), 3),
    )
    print(
        "見送り候補の荒れレース適合率:",
        round(float(precision_score(
            test["chaos"], skip, zero_division=0
        )), 3),
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": FEATURES,
        "chaos_payout_yen": CHAOS_PAYOUT_YEN,
        "model_version": "v1",
        "threshold": threshold,
    }, MODEL_PATH)

    print("モデル保存:", MODEL_PATH)


if __name__ == "__main__":
    main()
