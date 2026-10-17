from pathlib import Path
import sqlite3

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / "data" / "db" / "keiba.db"
CSV_PATH = ROOT / "data" / "processed" / "race_entries.csv"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")

REQUIRED = {
    "race_id", "race_date", "venue", "race_number",
    "surface", "distance_m", "field_size",
    "horse_id", "horse_name", "win_odds", "popularity",
    "finish_position", "trifecta_payout_yen",
}


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSVがありません: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH, dtype={"race_id": str, "horse_id": str})

    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"必須列が不足しています: {sorted(missing)}")

    if df["race_id"].isna().any() or df["horse_id"].isna().any():
        raise ValueError("race_id / horse_id に空欄があります")

    if df.duplicated(["race_id", "horse_id"]).any():
        raise ValueError("同一レース内で horse_id が重複しています")

    df["race_date"] = pd.to_datetime(
        df["race_date"], errors="raise"
    ).dt.strftime("%Y-%m-%d")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as con:
        con.execute("PRAGMA foreign_keys = ON")
        con.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))

        # 同じレースに複数の馬がいるため、レース情報を集約する。
        races = df.drop_duplicates("race_id").copy()

        for col in ["race_name", "going"]:
            if col not in races.columns:
                races[col] = None

        if "source_name" not in races.columns:
            races["source_name"] = "manual_official_import"

        race_columns = [
            "race_id", "race_date", "venue", "race_number",
            "race_name", "surface", "distance_m", "going",
            "field_size", "trifecta_payout_yen", "source_name",
        ]

        # 初期版では同じIDの再インポートは更新する。
        con.executemany(
            """
            INSERT INTO races (
                race_id, race_date, venue, race_number,
                race_name, surface, distance_m, going,
                field_size, trifecta_payout_yen, source_name
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_id) DO UPDATE SET
                race_date=excluded.race_date,
                venue=excluded.venue,
                race_number=excluded.race_number,
                race_name=excluded.race_name,
                surface=excluded.surface,
                distance_m=excluded.distance_m,
                going=excluded.going,
                field_size=excluded.field_size,
                trifecta_payout_yen=excluded.trifecta_payout_yen,
                source_name=excluded.source_name
            """,
            list(races[race_columns].itertuples(index=False, name=None)),
        )

        con.executemany(
            """
            INSERT INTO horses (horse_id, horse_name)
            VALUES (?, ?)
            ON CONFLICT(horse_id) DO UPDATE SET
                horse_name=excluded.horse_name
            """,
            list(
                df[["horse_id", "horse_name"]]
                .drop_duplicates("horse_id")
                .itertuples(index=False, name=None)
            ),
        )

        entry_columns = [
            "race_id", "horse_id", "horse_number",
            "frame_number", "jockey_name", "carried_weight",
            "win_odds", "popularity", "finish_position",
        ]

        for col in ["horse_number", "frame_number", "jockey_name",
                    "carried_weight"]:
            if col not in df.columns:
                df[col] = None

        con.executemany(
            """
            INSERT INTO entries (
                race_id, horse_id, horse_number,
                frame_number, jockey_name, carried_weight,
                win_odds, popularity, finish_position
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(race_id, horse_id) DO UPDATE SET
                horse_number=excluded.horse_number,
                frame_number=excluded.frame_number,
                jockey_name=excluded.jockey_name,
                carried_weight=excluded.carried_weight,
                win_odds=excluded.win_odds,
                popularity=excluded.popularity,
                finish_position=excluded.finish_position
            """,
            list(df[entry_columns].itertuples(index=False, name=None)),
        )

    print(f"取込完了: {df['race_id'].nunique()}レース")
    print(f"DB: {DB_PATH}")


if __name__ == "__main__":
    main()
