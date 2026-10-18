import json
import sqlite3
from pathlib import Path

import pandas as pd


DB_PATH = Path("/app/data/keiba.db")
REPORT_DIR = Path("/app/data/quality_report")


def check_database():
    """SQLiteの存在・テーブル・カラムを検査する。"""
    if not DB_PATH.exists():
        raise FileNotFoundError(f"DBが見つかりません: {DB_PATH}")

    with sqlite3.connect(DB_PATH) as conn:
        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            ORDER BY name
            """,
            conn,
        )["name"].tolist()

        required_tables = {
            "races", "horses", "payouts", "crawl_errors"
        }
        missing_tables = required_tables - set(tables)

        if missing_tables:
            raise ValueError(
                f"必要なテーブルがありません: {sorted(missing_tables)}"
            )

        schemas = {}
        for table in sorted(required_tables):
            schemas[table] = pd.read_sql_query(
                f"PRAGMA table_info({table})", conn
            )["name"].tolist()

        required_columns = {
            "races": {
                "id", "url", "race_date", "venue", "race_name"
            },
            "horses": {
                "id", "race_id", "finish", "horse_number", "horse_name"
            },
            "payouts": {"id", "race_id", "bet_type", "combination", "amount"},
            "crawl_errors": {"id", "url", "error", "occurred_at"},
        }

        for table, columns in required_columns.items():
            missing = columns - set(schemas[table])
            if missing:
                raise ValueError(
                    f"{table} に必要なカラムがありません: {sorted(missing)}"
                )

        return tables, schemas


def load_data():
    """検査対象を読み込む。元DBは変更しない。"""
    with sqlite3.connect(DB_PATH) as conn:
        races = pd.read_sql_query("SELECT * FROM races", conn)
        horses = pd.read_sql_query("SELECT * FROM horses", conn)
        payouts = pd.read_sql_query("SELECT * FROM payouts", conn)
        errors = pd.read_sql_query("SELECT * FROM crawl_errors", conn)

    races["race_date_parsed"] = pd.to_datetime(
        races["race_date"], errors="coerce"
    )

    horses["finish_numeric"] = pd.to_numeric(
        horses["finish"], errors="coerce"
    )

    for col in ["horse_number", "frame", "popularity"]:
        if col in horses.columns:
            horses[col + "_numeric"] = pd.to_numeric(
                horses[col], errors="coerce"
            )

    return races, horses, payouts, errors


def build_reports(races, horses, payouts, errors):
    """品質検査結果を集計し、問題レコードを抽出する。"""
    reports = {}

    # 基本件数
    reports["row_counts"] = {
        "races": int(len(races)),
        "horses": int(len(horses)),
        "payouts": int(len(payouts)),
        "crawl_errors": int(len(errors)),
    }

    # 年度別レース件数
    valid_dates = races.dropna(subset=["race_date_parsed"]).copy()
    valid_dates["year"] = valid_dates["race_date_parsed"].dt.year

    reports["races_by_year"] = (
        valid_dates.groupby("year")
        .size()
        .rename("race_count")
        .reset_index()
        .sort_values("year")
    )

    # レースの欠損値
    race_missing_columns = [
        col for col in
        ["url", "race_date", "venue", "race_name"]
        if col in races.columns
    ]

    race_missing = races[
        races[race_missing_columns].isna().any(axis=1)
    ].copy()

    for col in ["race_date", "venue", "race_name", "url"]:
        if col in races.columns:
            race_missing = pd.concat(
                [
                    race_missing,
                    races[races[col].astype(str).str.strip().eq("")],
                ]
            )

    if "race_name" in races.columns:
        race_missing = pd.concat(
            [
                race_missing,
                races[
                    races["race_name"].astype(str).str.contains(
                        "不明|不詳", na=False
                    )
                ],
            ]
        )

    race_missing = race_missing.drop_duplicates(subset=["id"])

    # URLの重複
    duplicate_urls = races[
        races["url"].notna()
        & races["url"].duplicated(keep=False)
    ].copy()

    # レースごとの出走馬数
    horse_counts = (
        horses.groupby("race_id")
        .size()
        .rename("horse_count")
        .reset_index()
    )

    race_horse_counts = races.merge(
        horse_counts,
        left_on="id",
        right_on="race_id",
        how="left",
    )
    race_horse_counts["horse_count"] = (
        race_horse_counts["horse_count"].fillna(0).astype(int)
    )

    unusual_horse_counts = race_horse_counts[
        race_horse_counts["horse_count"] < 5
    ].copy()

    # 着順が欠損・非数値・1未満のレコード
    invalid_finish = horses[
        horses["finish_numeric"].isna()
        | (horses["finish_numeric"] < 1)
    ].copy()

    # 同一レース内の同一馬番の重複
    if "horse_number_numeric" in horses.columns:
        valid_numbers = horses[
            horses["horse_number_numeric"].notna()
        ]
        duplicate_horse_numbers = valid_numbers[
            valid_numbers.duplicated(
                subset=["race_id", "horse_number_numeric"],
                keep=False,
            )
        ].copy()
    else:
        duplicate_horse_numbers = pd.DataFrame()

    # 必須情報が欠けた馬データ
    required_horse_fields = [
        col for col in ["race_id", "horse_name", "finish"]
        if col in horses.columns
    ]
    missing_horses = horses[
        horses[required_horse_fields].isna().any(axis=1)
    ].copy()

    if "horse_name" in horses.columns:
        missing_horses = pd.concat(
            [
                missing_horses,
                horses[
                    horses["horse_name"].astype(str).str.strip().eq("")
                ],
            ]
        ).drop_duplicates(subset=["id"])

    # 払戻が0件のレース
    payout_counts = (
        payouts.groupby("race_id")
        .size()
        .rename("payout_count")
        .reset_index()
    )

    race_payout_counts = races.merge(
        payout_counts,
        left_on="id",
        right_on="race_id",
        how="left",
    )
    race_payout_counts["payout_count"] = (
        race_payout_counts["payout_count"].fillna(0).astype(int)
    )

    no_payouts = race_payout_counts[
        race_payout_counts["payout_count"] == 0
    ].copy()

    # 馬データのカラム別欠損率
    horse_missing_rate = (
        horses.isna().mean()
        .rename("missing_rate")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values("missing_rate", ascending=False)
    )

    # クロールエラーが多いURL
    if errors.empty:
        error_urls = pd.DataFrame(
            columns=["url", "error_count"]
        )
    else:
        error_urls = (
            errors.groupby("url")
            .size()
            .rename("error_count")
            .reset_index()
            .sort_values("error_count", ascending=False)
        )

    reports["summary"] = {
        "races_with_invalid_or_missing_metadata": int(len(race_missing)),
        "duplicate_race_urls": int(len(duplicate_urls)),
        "races_with_fewer_than_5_horses": int(len(unusual_horse_counts)),
        "invalid_finish_records": int(len(invalid_finish)),
        "duplicate_horse_numbers": int(len(duplicate_horse_numbers)),
        "horses_with_missing_required_fields": int(len(missing_horses)),
        "races_without_payout_records": int(len(no_payouts)),
        "invalid_race_dates": int(races["race_date_parsed"].isna().sum()),
        "crawl_error_rows": int(len(errors)),
        "unique_crawl_error_urls": int(errors["url"].nunique())
        if not errors.empty else 0,
    }

    reports["details"] = {
        "race_missing": race_missing,
        "duplicate_urls": duplicate_urls,
        "unusual_horse_counts": unusual_horse_counts,
        "invalid_finish": invalid_finish,
        "duplicate_horse_numbers": duplicate_horse_numbers,
        "missing_horses": missing_horses,
        "no_payouts": no_payouts,
        "horse_missing_rate": horse_missing_rate,
        "error_urls": error_urls,
    }

    return reports


def save_reports(reports):
    """CSVとJSONのレポートを保存する。"""
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    reports["races_by_year"].to_csv(
        REPORT_DIR / "races_by_year.csv",
        index=False,
        encoding="utf-8-sig",
    )

    for name, df in reports["details"].items():
        df.to_csv(
            REPORT_DIR / f"{name}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    with (REPORT_DIR / "summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            reports["summary"],
            f,
            ensure_ascii=False,
            indent=2,
        )


def main():
    print("===== DB品質検査開始 =====")

    tables, schemas = check_database()
    races, horses, payouts, errors = load_data()

    print(f"DB: {DB_PATH}")
    print(f"テーブル: {', '.join(tables)}")

    for table, columns in schemas.items():
        print(f"{table}: {len(columns)} columns")

    reports = build_reports(races, horses, payouts, errors)
    save_reports(reports)

    print("\n===== 基本件数 =====")
    for name, count in reports["row_counts"].items():
        print(f"{name}: {count:,}")

    print("\n===== 品質検査結果 =====")
    for name, count in reports["summary"].items():
        print(f"{name}: {count:,}")

    print("\n===== 年度別レース件数 =====")
    print(reports["races_by_year"].to_string(index=False))

    print(f"\nレポート保存先: {REPORT_DIR}")
    print("元のDBデータは変更していません。")
    print("===== 品質検査完了 =====")


if __name__ == "__main__":
    main()
