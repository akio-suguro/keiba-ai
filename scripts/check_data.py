import sqlite3

DB_PATH = "data/keiba.db"

with sqlite3.connect(DB_PATH) as conn:
    print("===== レースの重複 =====")
    rows = conn.execute("""
        SELECT url, COUNT(*) AS cnt
        FROM races
        GROUP BY url
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
    """).fetchall()

    if rows:
        for url, cnt in rows:
            print(f"{cnt}件 | {url}")
    else:
        print("レースの重複なし")

    print("\n===== レースごとのデータ件数 =====")
    rows = conn.execute("""
        SELECT
            r.race_date,
            r.venue,
            r.race_name,
            COUNT(DISTINCT h.id) AS horse_rows,
            COUNT(DISTINCT p.id) AS payout_rows
        FROM races r
        LEFT JOIN horses h ON h.race_id = r.id
        LEFT JOIN payouts p ON p.race_id = r.id
        GROUP BY r.id
        ORDER BY r.race_date DESC
        LIMIT 20
    """).fetchall()

    for date, venue, name, horses, payouts in rows:
        print(
            f"{date} {venue} {name}"
            f" | 馬 {horses}件 | 払戻 {payouts}件"
        )
