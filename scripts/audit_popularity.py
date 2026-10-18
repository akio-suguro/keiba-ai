
from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("/app/data/keiba.db")
OUTPUT_DIR = Path("/app/data/model/upset")


def main():
    with sqlite3.connect(DB_PATH) as conn:
        races = pd.read_sql_query(
            "SELECT id, race_date, venue, race_name FROM races",
            conn,
        )
        horses = pd.read_sql_query(
            """
            SELECT race_id, finish, popularity, horse_name,
                   horse_number
            FROM horses
            """,
            conn,
        )

    horses["finish"] = pd.to_numeric(
        horses["finish"], errors="coerce"
    )
    horses["popularity"] = pd.to_numeric(
        horses["popularity"], errors="coerce"
    )
    races["race_date"] = pd.to_datetime(
        races["race_date"], errors="coerce"
    )

    data = horses.merge(
        races[["id", "race_date", "venue", "race_name"]],
        left_on="race_id",
        right_on="id",
        how="left",
    )

    # 人気順位が欠損・0以下・小数のレコードを抽出
    data["popularity_issue"] = (
        data["popularity"].isna()
        | (data["popularity"] < 1)
        | (data["popularity"] % 1 != 0)
    )

    # 年別の人気順位欠損率
    data["year"] = data["race_date"].dt.year

    yearly = (
        data.groupby("year", dropna=False)
        .agg(
            horse_rows=("race_id", "size"),
            popularity_issue_rows=("popularity_issue", "sum"),
        )
        .reset_index()
    )
    yearly["issue_rate_pct"] = (
        yearly["popularity_issue_rows"]
        / yearly["horse_rows"] * 100
    ).round(2)

    # 3着以内の人気順位に問題があるレースを抽出
    podium = data[data["finish"].isin([1, 2, 3])].copy()

    race_audit = (
        podium.groupby("race_id")
        .agg(
            podium_rows=("finish", "size"),
            podium_popularity_issues=("popularity_issue", "sum"),
        )
        .reset_index()
    )

    race_audit = race_audit.merge(
        races,
        left_on="race_id",
        right_on="id",
        how="left",
    )

    affected = race_audit[
        race_audit["podium_popularity_issues"] > 0
    ].copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    yearly.to_csv(
        OUTPUT_DIR / "popularity_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    affected.to_csv(
        OUTPUT_DIR / "popularity_affected_races.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # 欠損レコードの例を表示
    bad = data[data["popularity_issue"]].copy()

    print("=== 人気順位の監査 ===")
    print(f"全出走馬レコード: {len(data)}")
    print(f"人気順位に問題がある馬: {len(bad)}")
    print(
        f"人気順位の問題率: "
        f"{len(bad) / len(data) * 100:.2f}%"
    )

    print("\n=== 年別の問題率（先頭・末尾） ===")
    print(pd.concat([yearly.head(10), yearly.tail(10)]).to_string(
        index=False
    ))

    print("\n=== 人気順位に問題がある馬の例 ===")
    print(
        bad[
            [
                "race_date", "venue", "race_name",
                "horse_name", "horse_number",
                "finish", "popularity",
            ]
        ].head(20).to_string(index=False)
    )

    print("\n出力:")
    print(OUTPUT_DIR / "popularity_by_year.csv")
    print(OUTPUT_DIR / "popularity_affected_races.csv")


if __name__ == "__main__":
    main()
