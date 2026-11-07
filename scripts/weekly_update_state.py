from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

if __package__:
    from .database import connect_db, execute, transaction
else:
    from database import connect_db, execute, transaction


JRA_TIMEZONE = ZoneInfo("Asia/Tokyo")


def current_week_key() -> str:
    iso_week = datetime.now(JRA_TIMEZONE).date().isocalendar()
    return f"{iso_week.year}-W{iso_week.week:02d}"


def should_run(connection, week_key: str) -> bool:
    with transaction(connection):
        execute(
            connection,
            """
            CREATE TABLE IF NOT EXISTS weekly_update_runs (
                week_key TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            )
            """,
        )
        result = execute(
            connection,
            "SELECT 1 FROM weekly_update_runs WHERE week_key = ?",
            (week_key,),
        )
        return result.fetchone() is None


def mark_completed(connection, week_key: str) -> None:
    with transaction(connection):
        execute(
            connection,
            """
            CREATE TABLE IF NOT EXISTS weekly_update_runs (
                week_key TEXT PRIMARY KEY,
                completed_at TEXT NOT NULL
            )
            """,
        )
        execute(
            connection,
            """
            INSERT INTO weekly_update_runs (week_key, completed_at)
            VALUES (?, ?)
            ON CONFLICT (week_key) DO NOTHING
            """,
            (
                week_key,
                datetime.now(JRA_TIMEZONE).isoformat(timespec="seconds"),
            ),
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("check", "complete"))
    args = parser.parse_args()

    connection = connect_db()
    try:
        week_key = current_week_key()
        if args.action == "check":
            run = should_run(connection, week_key)
            output_path = os.environ.get("GITHUB_OUTPUT")
            if output_path:
                with Path(output_path).open("a", encoding="utf-8") as output:
                    output.write(f"run={'true' if run else 'false'}\n")
            print(
                f"[WEEKLY UPDATE] {week_key}: "
                f"{'start' if run else 'already completed; skip'}"
            )
        else:
            mark_completed(connection, week_key)
            print(f"[WEEKLY UPDATE] {week_key}: marked complete")
    finally:
        connection.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
