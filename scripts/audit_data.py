
import sqlite3
from pathlib import Path
from collections import Counter

DB_PATH = Path("/app/data/processed/keiba.db")


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DB not found: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row

        print("=" * 60)
        print("KEIBA-AI DATA AUDIT")
        print("=" * 60)

        # 1. レース基本情報
        total_races = conn.execute(
            "SELECT COUNT(*) FROM races"
        ).fetchone()[0]

        print(f"\n[1] Total races: {total_races}")

        print("\nVenue distribution:")
        for row in conn.execute("""
            SELECT venue, COUNT(*) AS n
            FROM races
            GROUP BY venue
            ORDER BY n DESC
        """):
            print(f"  {row['venue']}: {row['n']}")

        print("\nSurface distribution:")
        for row in conn.execute("""
            SELECT surface, COUNT(*) AS n
            FROM races
            GROUP BY surface
            ORDER BY n DESC
        """):
            print(f"  {row['surface']}: {row['n']}")

        print("\nDate range:")
        print(conn.execute("""
            SELECT MIN(race_date), MAX(race_date)
            FROM races
        """).fetchone()[:])

        # 2. 馬データの欠損
        print("\n[2] Horse data quality")

        horse_stats = conn.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN finish_position IS NULL THEN 1 ELSE 0 END)
                    AS missing_finish,
                SUM(CASE WHEN horse_name IS NULL OR horse_name = ''
                    THEN 1 ELSE 0 END) AS missing_name,
                SUM(CASE WHEN popularity IS NULL THEN 1 ELSE 0 END)
                    AS missing_popularity,
                SUM(CASE WHEN win_odds IS NULL THEN 1 ELSE 0 END)
                    AS missing_odds
            FROM horses
        """).fetchone()

        for key in horse_stats.keys():
            print(f"  {key}: {horse_stats[key]}")

        # 3. レースごとの馬データ件数
        print("\n[3] Horses per race (distribution)")
        for row in conn.execute("""
            SELECT horse_count, COUNT(*) AS race_count
            FROM (
                SELECT race_id, COUNT(*) AS horse_count
                FROM horses
                GROUP BY race_id
            )
            GROUP BY horse_count
            ORDER BY horse_count
        """):
            print(
                f"  {row['horse_count']} horses: "
                f"{row['race_count']} races"
            )

        # 4. 払戻データ
        print("\n[4] Payout types")
        for row in conn.execute("""
            SELECT bet_type, COUNT(*) AS n
            FROM payouts
            GROUP BY bet_type
            ORDER BY n DESC
        """):
            print(f"  {row['bet_type']}: {row['n']}")

        payout_races = conn.execute("""
            SELECT COUNT(DISTINCT race_id)
            FROM payouts
        """).fetchone()[0]

        print(f"  Races with payouts: {payout_races}")
        print(f"  Races without payouts: {total_races - payout_races}")

        print("\nHigh payout examples (top 10):")
        for row in conn.execute("""
            SELECT r.race_date, r.race_name,
                   p.bet_type, p.combination, p.payout_yen
            FROM payouts p
            JOIN races r ON r.race_id = p.race_id
            WHERE p.bet_type IN ('三連単', '三連複')
            ORDER BY p.payout_yen DESC
            LIMIT 10
        """):
            print(
                f"  {row['race_date']} | {row['race_name']} | "
                f"{row['bet_type']} {row['combination']} | "
                f"{row['payout_yen']:,} yen"
            )

        # 5. 収集エラーの種類を要約
        print("\n[5] Crawl error summary")
        errors = conn.execute("""
            SELECT error
            FROM crawl_errors
        """).fetchall()

        print(f"  Total error records: {len(errors)}")

        error_types = Counter()
        for row in errors:
            message = row["error"] or ""
            if "FileNotFoundError" in message:
                error_types["FileNotFoundError"] += 1
            elif "Timeout" in message:
                error_types["Timeout"] += 1
            elif "HTTPError" in message:
                error_types["HTTPError"] += 1
            else:
                error_types[message.split(":", 1)[0][:100]] += 1

        for error_type, count in error_types.most_common(10):
            print(f"  {error_type}: {count}")

        # 6. レース名と競馬場の確認用サンプル
        print("\n[6] Venue / race name samples")
        for row in conn.execute("""
            SELECT race_date, venue, race_name, source_url
            FROM races
            ORDER BY race_date
            LIMIT 15
        """):
            print(
                f"  {row['race_date']} | {row['venue']} | "
                f"{row['race_name']} | {row['source_url']}"
            )

        print("\nAudit finished.")


if __name__ == "__main__":
    main()
