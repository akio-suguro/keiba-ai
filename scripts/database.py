from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
import os
import re
import sqlite3
from typing import Any, Iterator, Mapping, Sequence

import pandas as pd
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import Connection, Engine, URL


ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(os.environ.get("SQLITE_DB_PATH", ROOT / "data" / "keiba.db"))


def database_url() -> str | URL:
    """Return the configured PostgreSQL URL or local SQLite fallback."""
    configured = os.environ.get("DATABASE_URL", "").strip()
    if not configured:
        return URL.create("sqlite", database=str(DB_PATH.resolve()))

    if configured.startswith("postgres://"):
        configured = "postgresql://" + configured[len("postgres://"):]
    if configured.startswith("postgresql://"):
        configured = "postgresql+psycopg://" + configured[len("postgresql://"):]
    if configured.startswith("postgresql+psycopg://") and "sslmode=" not in configured:
        separator = "&" if "?" in configured else "?"
        configured = f"{configured}{separator}sslmode=require"
    return configured


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    """Create the shared database engine."""
    url = database_url()
    options: dict[str, Any] = {"pool_pre_ping": True}
    if isinstance(url, URL) and url.get_backend_name() == "sqlite":
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        options["connect_args"] = {"timeout": 30}
    return create_engine(url, **options)


def is_postgres(connection: Connection | sqlite3.Connection) -> bool:
    return not isinstance(connection, sqlite3.Connection) and (
        connection.dialect.name == "postgresql"
    )


def connect_db() -> Connection:
    """Open a connection to PostgreSQL or the local SQLite database."""
    return get_engine().connect()


def table_columns(
    connection: Connection | sqlite3.Connection,
    table: str,
) -> set[str]:
    if isinstance(connection, sqlite3.Connection):
        rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
        return {row[1] for row in rows}
    return {
        column["name"]
        for column in inspect(connection).get_columns(table)
    }


def _named_query(
    query: str,
    parameters: Sequence[Any],
) -> tuple[str, dict[str, Any]]:
    index = 0

    def replace(_match: re.Match[str]) -> str:
        nonlocal index
        placeholder = f":p{index}"
        index += 1
        return placeholder

    converted = re.sub(r"\?", replace, query)
    if index != len(parameters):
        raise ValueError(
            f"SQL parameter count mismatch: {index} placeholders, "
            f"{len(parameters)} values"
        )
    return converted, {
        f"p{position}": value
        for position, value in enumerate(parameters)
    }


def execute(
    connection: Connection | sqlite3.Connection,
    query: str,
    parameters: Sequence[Any] | Mapping[str, Any] = (),
):
    """Execute SQL using portable named parameters on PostgreSQL."""
    if isinstance(connection, sqlite3.Connection):
        return connection.execute(query, parameters)
    if isinstance(parameters, Mapping):
        return connection.execute(text(query), dict(parameters))
    converted, bound = _named_query(query, parameters)
    return connection.execute(text(converted), bound)


def read_sql_query(
    query: str,
    connection: Connection | sqlite3.Connection,
    params: Sequence[Any] | Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    if isinstance(connection, sqlite3.Connection):
        return pd.read_sql_query(query, connection, params=params)
    if isinstance(params, Mapping):
        return pd.read_sql_query(text(query), connection, params=dict(params))
    if params is None:
        return pd.read_sql_query(text(query), connection)
    converted, bound = _named_query(query, params)
    return pd.read_sql_query(text(converted), connection, params=bound)


def row_value(row, key: str):
    mapping = getattr(row, "_mapping", row)
    return mapping[key]


@contextmanager
def transaction(
    connection: Connection | sqlite3.Connection,
) -> Iterator[Connection | sqlite3.Connection]:
    if isinstance(connection, sqlite3.Connection):
        with connection:
            yield connection
        return

    if connection.in_transaction():
        connection.commit()
    with connection.begin():
        yield connection


def initialize_schema(
    connection: Connection | sqlite3.Connection,
) -> None:
    """Create or update the application's core tables and indexes."""
    identity = (
        "INTEGER PRIMARY KEY AUTOINCREMENT"
        if isinstance(connection, sqlite3.Connection)
        else "SERIAL PRIMARY KEY"
    )
    statements = [
        f"""
        CREATE TABLE IF NOT EXISTS races (
            id {identity},
            url TEXT UNIQUE,
            race_date TEXT,
            venue TEXT,
            race_name TEXT,
            distance_m INTEGER,
            surface TEXT,
            direction TEXT,
            crawled_at TEXT,
            race_number INTEGER,
            start_time TEXT,
            weather TEXT,
            turf_condition TEXT,
            dirt_condition TEXT,
            race_grade TEXT,
            race_conditions TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS horses (
            id {identity},
            race_id INTEGER NOT NULL,
            finish INTEGER,
            frame INTEGER,
            horse_number INTEGER,
            horse_name TEXT,
            sex_age TEXT,
            weight REAL,
            jockey TEXT,
            time TEXT,
            margin TEXT,
            last_3f REAL,
            body_weight INTEGER,
            weight_change INTEGER,
            trainer TEXT,
            popularity INTEGER,
            odds REAL,
            horse_mark TEXT,
            corner_order TEXT,
            horse_url TEXT,
            jockey_url TEXT,
            trainer_url TEXT,
            source_fields TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS payouts (
            id {identity},
            race_id INTEGER NOT NULL,
            bet_type TEXT,
            combination TEXT,
            amount INTEGER,
            popularity INTEGER
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS crawl_errors (
            id {identity},
            url TEXT,
            error TEXT,
            occurred_at TEXT
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS crawl_retry_queue (
            url TEXT PRIMARY KEY,
            retry_count INTEGER NOT NULL DEFAULT 0,
            next_retry_at TEXT NOT NULL,
            last_error TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            resolved_at TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id {identity},
            race_id INTEGER NOT NULL,
            predicted_at TEXT NOT NULL,
            cutoff_at TEXT,
            chaos_probability REAL NOT NULL DEFAULT 0,
            uncertainty_score REAL,
            decision TEXT NOT NULL DEFAULT 'unavailable',
            model_version TEXT NOT NULL,
            top3_predictions TEXT,
            upset_factors TEXT,
            evaluated_at TEXT
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS model_evaluations (
            evaluation_id {identity},
            model_version TEXT NOT NULL,
            train_until TEXT NOT NULL,
            test_from TEXT NOT NULL,
            test_until TEXT NOT NULL,
            skip_rate REAL,
            chaos_recall REAL,
            chaos_precision REAL,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """,
        f"""
        CREATE TABLE IF NOT EXISTS prediction_evaluations (
            evaluation_id {identity},
            model_version TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            test_from TEXT NOT NULL,
            test_until TEXT NOT NULL,
            evaluated_races INTEGER NOT NULL,
            mean_correct_picks REAL NOT NULL,
            precision_at_3 REAL NOT NULL,
            win_return_rate REAL,
            quinella_return_rate REAL,
            trio_return_rate REAL,
            upset_races INTEGER NOT NULL DEFAULT 0,
            upset_brier_score REAL,
            upset_top_10pct_rate REAL
        )
        """,
    ]
    for statement in statements:
        execute(connection, statement)

    optional_columns = {
        "races": {
            "race_number": "INTEGER",
            "start_time": "TEXT",
            "weather": "TEXT",
            "turf_condition": "TEXT",
            "dirt_condition": "TEXT",
            "race_grade": "TEXT",
            "race_conditions": "TEXT",
        },
        "horses": {
            "horse_mark": "TEXT",
            "corner_order": "TEXT",
            "horse_url": "TEXT",
            "jockey_url": "TEXT",
            "trainer_url": "TEXT",
            "source_fields": "TEXT",
        },
        "payouts": {"popularity": "INTEGER"},
        "predictions": {
            "cutoff_at": "TEXT",
            "chaos_probability": "REAL",
            "uncertainty_score": "REAL",
            "decision": "TEXT",
            "model_version": "TEXT",
            "top3_predictions": "TEXT",
            "upset_factors": "TEXT",
            "evaluated_at": "TEXT",
        },
        "prediction_evaluations": {
            "upset_races": "INTEGER NOT NULL DEFAULT 0",
            "upset_brier_score": "REAL",
            "upset_top_10pct_rate": "REAL",
        },
    }
    for table, columns in optional_columns.items():
        existing = table_columns(connection, table)
        for column, data_type in columns.items():
            if column not in existing:
                execute(
                    connection,
                    f'ALTER TABLE "{table}" '
                    f'ADD COLUMN "{column}" {data_type}',
                )

    for index_name, table, column in (
        ("idx_races_date", "races", "race_date"),
        ("idx_horses_race", "horses", "race_id"),
        ("idx_payouts_race", "payouts", "race_id"),
        ("idx_crawl_retry_due", "crawl_retry_queue", "next_retry_at"),
        ("idx_predictions_race", "predictions", "race_id"),
        ("idx_predictions_time", "predictions", "predicted_at"),
        ("idx_prediction_evaluations_model", "prediction_evaluations", "model_version"),
    ):
        execute(
            connection,
            f'CREATE INDEX IF NOT EXISTS "{index_name}" '
            f'ON "{table}"("{column}")',
        )
    if isinstance(connection, sqlite3.Connection):
        connection.commit()
