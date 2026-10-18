
from pathlib import Path
import sqlite3

import pandas as pd


DB_PATH = Path("/app/data/keiba.db")
OUTPUT_DIR = Path("/app/data/model/upset")


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DBが見つかりません: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        races = pd.read_sql_query(
            """
            SELECT id, race_date, venue, race_name
            FROM races
            """,
            conn,
        )
        horses = pd.read_sql_query(
            """
            SELECT race_id, finish, popularity
            FROM horses
            """,
            conn,
        )

    # 数値に変換できない値は欠損として扱う
    horses["finish"] = pd.to_numeric(
        horses["finish"], errors="coerce"
    )
    horses["popularity"] = pd.to_numeric(
        horses["popularity"], errors="coerce"
    )

    # 監査対象を1レースずつ分類する
    horse_groups = {
        race_id: group
        for race_id, group in horses.groupby("race_id")
    }

    results = []

    for _, race in races.iterrows():
        race_id = race["id"]
        group = horse_groups.get(race_id)

        reason = "included"
        field_size = 0
        podium_popularity = None
        actual_upset = None

        if group is None or group.empty:
            reason = "no_horse_records"
        else:
            field_size = len(group)

            # 1・2・3着の各行が1件ずつ存在するか
            podium = group[group["finish"].isin([1, 2, 3])]

            podium_counts = podium["finish"].value_counts()
            valid_podium = (
                len(podium) == 3
                and all(podium_counts.get(pos, 0) == 1
                        for pos in [1, 2, 3])
            )

            if not valid_podium:
                reason = "invalid_top3_finish"
            else:
                popularity = podium["popularity"]

                valid_popularity = (
                    popularity.notna().all()
                    and (popularity >= 1).all()
                    and (popularity <= field_size).all()
                    and (popularity % 1 == 0).all()
                )

                if not valid_popularity:
                    reason = "invalid_top3_popularity"
                else:
                    podium_popularity = ",".join(
                        str(int(v))
                        for v in podium.sort_values("finish")[
                            "popularity"
                        ]
                    )
                    actual_upset = int(
                        (popularity >= 7).any()
                    )

        results.append({
            "race_id": race_id,
            "race_date": race["race_date"],
            "venue": race["venue"],
            "race_name": race["race_name"],
            "field_size": field_size,
            "top3_popularity": podium_popularity,
            "actual_upset": actual_upset,
            "audit_status": reason,
        })

    audit = pd.DataFrame(results)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    detail_path = OUTPUT_DIR / "label_audit_detail.csv"
    summary_path = OUTPUT_DIR / "label_audit_summary.csv"

    audit.to_csv(
        detail_path, index=False, encoding="utf-8-sig"
    )

    summary = (
        audit.groupby(
            ["audit_status"], dropna=False
        )
        .size()
        .rename("race_count")
        .reset_index()
    )
    summary["percentage"] = (
        summary["race_count"] / len(audit) * 100
    ).round(2)

    summary.to_csv(
        summary_path, index=False, encoding="utf-8-sig"
    )

    print("=== ラベル監査結果 ===")
    print(f"レース総数: {len(audit)}")
    print(f"出走馬レコード数: {len(horses)}")
    print()
    print(summary.to_string(index=False))
    print()
    print(f"詳細: {detail_path}")
    print(f"集計: {summary_path}")


if __name__ == "__main__":
    main()
