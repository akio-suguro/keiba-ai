import sqlite3
import tempfile
import unittest
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import ANY, Mock, patch

from bs4 import BeautifulSoup

from scripts import crawl_current_month, crawl_jra


class CurrentMonthCrawlerEntrypointTests(unittest.TestCase):
    def test_runs_shared_crawler_in_current_month_only_mode(self):
        with patch.object(
            crawl_current_month.crawl_jra,
            "main",
            return_value=0,
        ) as crawl_main:
            result = crawl_current_month.main()

        self.assertEqual(result, 0)
        crawl_main.assert_called_once_with(
            current_month_only=True,
            enforce_database_size_limit=False,
        )

    def test_crawler_uses_japan_local_date(self):
        jst_time = datetime(
            2026,
            11,
            1,
            0,
            0,
            tzinfo=crawl_jra.JRA_TIMEZONE,
        )
        with patch.object(crawl_jra, "datetime") as datetime_mock:
            datetime_mock.now.return_value = jst_time

            self.assertEqual(crawl_jra.jra_today(), date(2026, 11, 1))

        datetime_mock.now.assert_called_once_with(crawl_jra.JRA_TIMEZONE)


class RaceIndexUrlTests(unittest.TestCase):
    def test_years_cover_start_year_through_current_year(self):
        self.assertEqual(
            crawl_jra.YEARS,
            list(range(crawl_jra.CURRENT_YEAR, crawl_jra.START_YEAR - 1, -1)),
        )

    def test_current_year_uses_replay_root(self):
        with patch.object(crawl_jra, "CURRENT_YEAR", 2026):
            self.assertEqual(
                crawl_jra.race_index_url(2026),
                "https://www.jra.go.jp/datafile/seiseki/replay/jyusyo.html",
            )

    def test_previous_year_uses_year_directory(self):
        with patch.object(crawl_jra, "CURRENT_YEAR", 2026):
            self.assertEqual(
                crawl_jra.race_index_url(2025),
                "https://www.jra.go.jp/datafile/seiseki/replay/2025/jyusyo.html",
            )

    def test_discover_current_year_result_links(self):
        html = """
        <table>
          <tr>
            <td class="result">
              <a href="/datafile/seiseki/replay/2026/001.html">レース結果</a>
            </td>
          </tr>
          <tr>
            <td class="result">
              <a href="/datafile/seiseki/replay/2026/002.html">レース結果</a>
            </td>
          </tr>
        </table>
        """
        expected_index_url = (
            "https://www.jra.go.jp/datafile/seiseki/replay/jyusyo.html"
        )

        with (
            patch.object(crawl_jra, "CURRENT_YEAR", 2026),
            patch.object(
                crawl_jra,
                "fetch_html",
                return_value=BeautifulSoup(html, "html.parser"),
            ) as fetch_html,
        ):
            urls = crawl_jra.discover_race_urls(2026)

        fetch_html.assert_called_once_with(expected_index_url)
        self.assertEqual(
            urls,
            [
                "https://www.jra.go.jp/datafile/seiseki/replay/2026/001.html",
                "https://www.jra.go.jp/datafile/seiseki/replay/2026/002.html",
            ],
        )


class JraSearchTests(unittest.TestCase):
    def test_rejects_jra_parameter_error_page(self):
        response = Mock()
        response.text = "<html><title>パラメータエラー JRA</title></html>"
        with (
            patch.object(crawl_jra.SESSION, "post", return_value=response),
            patch.object(crawl_jra.time, "sleep"),
        ):
            with self.assertRaisesRegex(
                crawl_jra.JraParameterError,
                "パラメータエラー",
            ):
                crawl_jra.fetch_jra_search_page("pw01skl10202610/C3")

    def test_current_month_parameter_error_falls_back_to_year_index(self):
        today = crawl_jra.jra_today()
        current_day_cname = (
            f"pw01srl1005{today.year}0401{today:%Y%m%d}/8F"
        )
        previous_month = date(today.year, today.month, 1) - timedelta(days=1)
        old_month_cname = (
            f"pw01srl1005{previous_month.year}0401"
            f"{previous_month:%Y%m%d}/FA"
        )
        future_date = today + timedelta(days=1)
        future_day_cnames = (
            [
                f"pw01srl1005{today.year}0402{future_date:%Y%m%d}/AA"
            ]
            if future_date.month == today.month
            else []
        )
        with (
            patch.object(
                crawl_jra,
                "_YEAR_INDEX_RACE_DAY_CNAMES",
                [
                    current_day_cname,
                    old_month_cname,
                    *future_day_cnames,
                ],
            ),
            patch.object(
                crawl_jra,
                "fetch_jra_search_page",
                side_effect=crawl_jra.JraParameterError("not published"),
            ) as fetch_page,
        ):
            race_days = crawl_jra.discover_race_days(
                today.year,
                today.month,
                {f"{today.year % 100:02d}{today.month:02d}": "C3"},
                today=today,
            )

        fetch_page.assert_called_once_with(
            f"pw01skl10{today.year}{today.month:02d}/C3"
        )
        self.assertEqual(race_days, [current_day_cname])

    def test_extracts_cnames_from_onclick_and_deduplicates(self):
        soup = BeautifulSoup(
            """
            <a href="#" onclick="return doAction(
              '/JRADB/accessS.html',
              'pw01srl10062025010120250105/DA'
            );">開催日</a>
            <a href="/JRADB/accessS.html?CNAME=pw01srl10072025010120250105/24">
              同日別会場
            </a>
            <a href="#" onclick="return doAction(
              '/JRADB/accessS.html',
              'pw01srl10062025010120250105/DA'
            );">重複</a>
            """,
            "html.parser",
        )

        self.assertEqual(
            crawl_jra.extract_cnames(soup, "pw01srl"),
            [
                "pw01srl10062025010120250105/DA",
                "pw01srl10072025010120250105/24",
            ],
        )

    def test_discovers_only_race_days_in_the_requested_past_month(self):
        soup = BeautifulSoup(
            """
            <a onclick="doAction('pw01srl10062025010120250105/DA')">5日</a>
            <a onclick="doAction('pw01srl10062025020120250201/AA')">2月</a>
            <a onclick="doAction('pw01srl10062025013120250131/BB')">31日</a>
            """,
            "html.parser",
        )
        with (
            patch.object(
                crawl_jra,
                "fetch_jra_search_page",
                return_value=soup,
            ) as fetch_page,
            patch.object(crawl_jra.time, "sleep"),
        ):
            days = crawl_jra.discover_race_days(
                2025,
                1,
                {"2501": "3F"},
                today=date(2025, 1, 31),
            )

        fetch_page.assert_called_once_with("pw01skl10202501/3F")
        self.assertEqual(
            days,
            [
                "pw01srl10062025010120250105/DA",
                "pw01srl10062025013120250131/BB",
            ],
        )

    def test_current_month_without_completed_races_is_not_an_error(self):
        today = crawl_jra.jra_today()
        with (
            patch.object(
                crawl_jra,
                "fetch_jra_search_page",
                return_value=BeautifulSoup("", "html.parser"),
            ),
            patch.object(crawl_jra.time, "sleep"),
        ):
            days = crawl_jra.discover_race_days(
                today.year,
                today.month,
                {f"{today.year % 100:02d}{today.month:02d}": "3F"},
                today=today,
            )

        self.assertEqual(days, [])

    def test_discovers_upcoming_cards_across_month_boundary(self):
        top_page = BeautifulSoup(
            """
            <a onclick="doAction('pw01drl00052026040920261030/BC')">
              10月30日 東京
            </a>
            <a onclick="doAction('pw01drl00052026041020261031/AA')">
              10月31日 東京
            </a>
            <a onclick="doAction('pw01drl00052026041120261101/AA')">
              11月1日 東京
            </a>
            <a onclick="doAction('pw01drl00052026041220261108/AA')">
              11月8日 東京
            </a>
            """,
            "html.parser",
        )
        today_page = BeautifulSoup(
            """
            <a onclick="doAction('pw01dde0105202604100120261031/D9')">
              1R
            </a>
            """,
            "html.parser",
        )
        tomorrow_page = BeautifulSoup(
            """
            <a onclick="doAction('pw01dde0105202604110120261101/AA')">
              1R
            </a>
            """,
            "html.parser",
        )
        with patch.object(
            crawl_jra,
            "fetch_jra_card_page",
            side_effect=[top_page, today_page, tomorrow_page],
        ) as fetch_page:
            cnames = crawl_jra.discover_upcoming_race_cnames(
                today=date(2026, 10, 31),
            )

        self.assertEqual(
            cnames,
            [
                "pw01dde0105202604100120261031/D9",
                "pw01dde0105202604110120261101/AA",
            ],
        )
        fetch_page.assert_any_call(crawl_jra.JRA_CARD_INDEX_CNAME)
        self.assertEqual(fetch_page.call_count, 3)

    def test_reads_race_date_venue_and_number_from_result_cname(self):
        cname = "pw01sde1006202501010120250105/6B"

        self.assertEqual(
            crawl_jra.race_cname_details(cname),
            {
                "race_date": "2025-01-05",
                "venue": "中山",
                "race_number": 1,
            },
        )

    def test_reads_race_date_venue_and_number_from_entry_cname(self):
        cname = "pw01dde0105202604080120261024/D9"

        self.assertEqual(
            crawl_jra.race_cname_details(cname),
            {
                "race_date": "2026-10-24",
                "venue": "東京",
                "race_number": 1,
            },
        )

    def test_reads_same_day_result_cname(self):
        cname = "pw01sde0105202604080120261024/DE"

        self.assertEqual(
            crawl_jra.race_cname_details(cname),
            {
                "race_date": "2026-10-24",
                "venue": "東京",
                "race_number": 1,
            },
        )

    def test_rejects_invalid_race_cname(self):
        with self.assertRaisesRegex(ValueError, "形式が不正"):
            crawl_jra.race_cname_details("pw01sde-invalid")


class DatabaseMigrationTests(unittest.TestCase):
    def test_migrates_legacy_predictions_table_for_saved_results(self):
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(
                """
                CREATE TABLE races (id INTEGER PRIMARY KEY, race_date TEXT);
                CREATE TABLE horses (id INTEGER PRIMARY KEY, race_id INTEGER);
                CREATE TABLE payouts (id INTEGER PRIMARY KEY, race_id INTEGER);
                CREATE TABLE predictions (
                    prediction_id INTEGER PRIMARY KEY,
                    race_id INTEGER NOT NULL,
                    predicted_at TEXT NOT NULL,
                    cutoff_at TEXT NOT NULL,
                    chaos_probability REAL NOT NULL,
                    uncertainty_score REAL,
                    decision TEXT NOT NULL,
                    model_version TEXT NOT NULL
                );
                """
            )

            crawl_jra.init_db(conn)

            columns = crawl_jra.get_table_columns(conn, "predictions")

        self.assertIn("top3_predictions", columns)
        self.assertIn("upset_factors", columns)
        self.assertIn("evaluated_at", columns)

    def test_adds_optional_result_columns_to_existing_tables(self):
        with sqlite3.connect(":memory:") as conn:
            conn.executescript(
                """
                CREATE TABLE races (id INTEGER PRIMARY KEY, race_date TEXT);
                CREATE TABLE horses (id INTEGER PRIMARY KEY, race_id INTEGER);
                CREATE TABLE payouts (id INTEGER PRIMARY KEY, race_id INTEGER);
                """
            )

            crawl_jra.init_db(conn)

            races = crawl_jra.get_table_columns(conn, "races")
            horses = crawl_jra.get_table_columns(conn, "horses")
            payouts = crawl_jra.get_table_columns(conn, "payouts")

        self.assertTrue(
            {
                "race_number",
                "start_time",
                "weather",
                "turf_condition",
                "dirt_condition",
                "race_grade",
                "race_conditions",
            }.issubset(races)
        )
        self.assertTrue(
            {
                "horse_mark",
                "corner_order",
                "horse_url",
                "jockey_url",
                "trainer_url",
                "source_fields",
            }.issubset(horses)
        )
        self.assertIn("popularity", payouts)
        with sqlite3.connect(":memory:") as conn:
            crawl_jra.init_db(conn)
            prediction_columns = crawl_jra.get_table_columns(
                conn,
                "predictions",
            )
            retry_columns = crawl_jra.get_table_columns(
                conn,
                "crawl_retry_queue",
            )
            evaluation_columns = crawl_jra.get_table_columns(
                conn,
                "prediction_evaluations",
            )

        self.assertIn("top3_predictions", prediction_columns)
        self.assertIn("evaluated_at", prediction_columns)
        self.assertIn("next_retry_at", retry_columns)
        self.assertIn("mean_correct_picks", evaluation_columns)


class CrawlerRetryQueueTests(unittest.TestCase):
    def test_failed_url_is_queued_then_marked_resolved_on_retry(self):
        url = "https://example.test/race.html"
        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            crawl_jra.init_db(conn)
            crawl_jra.save_error(conn, url, RuntimeError("temporary failure"))
            queued = conn.execute(
                """
                SELECT retry_count, resolved_at
                FROM crawl_retry_queue WHERE url = ?
                """,
                (url,),
            ).fetchone()
            self.assertEqual(queued["retry_count"], 1)
            self.assertIsNone(queued["resolved_at"])
            conn.execute(
                "UPDATE crawl_retry_queue SET next_retry_at = '2000-01-01'"
            )

            with patch.object(
                crawl_jra,
                "fetch_html",
                return_value=BeautifulSoup("<html></html>", "html.parser"),
            ) as fetch_html:
                errors = crawl_jra.process_retry_queue(conn)

            self.assertEqual(errors, 0)
            fetch_html.assert_called_once_with(url)
            resolved = conn.execute(
                "SELECT resolved_at FROM crawl_retry_queue WHERE url = ?",
                (url,),
            ).fetchone()

        self.assertIsNotNone(resolved["resolved_at"])

    def test_repeated_failure_increments_backoff_attempt_count(self):
        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            crawl_jra.init_db(conn)
            crawl_jra.save_error(conn, "https://example.test/fail", ValueError("one"))
            crawl_jra.save_error(conn, "https://example.test/fail", ValueError("two"))

            retry_count = conn.execute(
                "SELECT retry_count FROM crawl_retry_queue"
            ).fetchone()[0]

        self.assertEqual(retry_count, 2)


class SaveRaceDeduplicationTests(unittest.TestCase):
    def test_general_race_updates_matching_existing_static_race(self):
        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            crawl_jra.init_db(conn)
            static_url = (
                "https://www.jra.go.jp/datafile/seiseki/replay/2025/001.html"
            )
            conn.execute(
                """
                INSERT INTO races (
                    url, race_date, venue, race_name, distance_m
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (
                    static_url,
                    "2025-01-05",
                    "中山",
                    "中山金杯",
                    2000,
                ),
            )
            race = {
                "url": (
                    "https://www.jra.go.jp/JRADB/accessS.html?"
                    "CNAME=pw01sde1006202501010120250105/6B"
                ),
                "race_date": "2025-01-05",
                "venue": "中山",
                "race_name": "中山金杯",
                "distance_m": 2000,
                "race_number": 1,
            }
            horses = [
                {
                    "finish": number,
                    "horse_number": number,
                    "horse_name": f"Horse {number}",
                }
                for number in range(1, 6)
            ]

            crawl_jra.save_race(conn, race, horses, [])

            races = conn.execute(
                "SELECT id, url, race_number FROM races"
            ).fetchall()
            horse_count = conn.execute(
                "SELECT COUNT(*) FROM horses WHERE race_id = ?",
                (races[0]["id"],),
            ).fetchone()[0]
            already_saved = crawl_jra.race_result_already_saved(
                conn,
                "pw01sde1006202501010120250105/6B",
            )

        self.assertEqual(len(races), 1)
        self.assertEqual(races[0]["url"], static_url)
        self.assertEqual(races[0]["race_number"], 1)
        self.assertEqual(horse_count, 5)
        self.assertTrue(already_saved)

    def test_entry_only_race_can_be_updated_with_result_later(self):
        entry_cname = "pw01dde0105202604080120261024/D9"
        race = {
            "url": f"{crawl_jra.JRA_CARD_URL}?CNAME={entry_cname}",
            "race_date": "2026-10-24",
            "venue": "東京",
            "race_name": "テストレース",
            "race_number": 1,
        }
        horses = [
            {
                "finish": None,
                "horse_number": number,
                "horse_name": f"Horse {number}",
            }
            for number in range(1, 6)
        ]

        with sqlite3.connect(":memory:") as conn:
            conn.row_factory = sqlite3.Row
            crawl_jra.init_db(conn)
            crawl_jra.save_race(conn, race, horses, [])

            self.assertFalse(
                crawl_jra.race_result_already_saved(conn, entry_cname)
            )
            self.assertFalse(crawl_jra.race_has_results(conn, entry_cname))

            result_race = {
                **race,
                "url": (
                    f"{crawl_jra.JRA_DB_URL}?"
                    "CNAME=pw01sde1005202604080120261024/DE"
                ),
            }
            result_horses = [
                {**horse, "finish": number}
                for number, horse in enumerate(horses, start=1)
            ]
            crawl_jra.save_race(conn, result_race, result_horses, [])

            self.assertTrue(
                crawl_jra.race_result_already_saved(
                    conn,
                    "pw01sde1005202604080120261024/DE",
                )
            )
            self.assertTrue(crawl_jra.race_has_results(conn, entry_cname))


class UnconfirmedRaceTests(unittest.TestCase):
    def test_unconfirmed_current_day_result_is_skipped_without_error_record(self):
        today = crawl_jra.jra_today()
        cname = (
            f"pw01sde1006{today.year}010101{today:%Y%m%d}/6B"
        )
        with sqlite3.connect(":memory:") as conn:
            with (
                patch.object(
                    crawl_jra,
                    "fetch_jra_search_page",
                    return_value=BeautifulSoup("", "html.parser"),
                ),
                patch.object(
                    crawl_jra,
                    "extract_race_info",
                    return_value={"race_date": None},
                ),
                patch.object(crawl_jra, "save_error") as save_error,
            ):
                result = crawl_jra.crawl_one_race(
                    conn,
                    crawl_jra.race_result_url(cname),
                    allow_unconfirmed_today=True,
                )

        self.assertEqual(result, (0, 0, None))
        save_error.assert_not_called()


class MainCrawlerTests(unittest.TestCase):
    def test_main_crawls_all_result_links_discovered_for_a_month(self):
        year = crawl_jra.jra_today().year - 1
        day_cname = f"pw01srl1006{year}0101{year}0105/DA"
        race_cname = f"pw01sde1006{year}010101{year}0105/6B"
        checkdigits = {
            f"{year % 100:02d}{month:02d}": "3F"
            for month in range(1, 13)
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", [year]),
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value=checkdigits,
                ),
                patch.object(
                    crawl_jra,
                    "discover_race_days",
                    side_effect=lambda discovered_year, month, *_args, **_kwargs: (
                        [day_cname] if month == 1 else []
                    ),
                ),
                patch.object(
                    crawl_jra,
                    "discover_race_cnames_for_day",
                    return_value=[race_cname],
                ),
                patch.object(
                    crawl_jra,
                    "race_result_already_saved",
                    return_value=False,
                ),
                patch.object(
                    crawl_jra,
                    "crawl_one_race",
                    return_value=(16, 12, True),
                ) as crawl_one_race,
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main()

        self.assertEqual(result, 0)
        crawl_one_race.assert_called_once_with(
            ANY,
            crawl_jra.race_result_url(race_cname),
        )

    def test_main_scans_years_and_months_newest_first(self):
        years = [2026, 2025]
        checkdigits = {
            f"{year % 100:02d}{month:02d}": "3F"
            for year in years
            for month in range(1, 13)
        }
        scanned_months = []
        expected_newest_months = [
            (2026, month)
            for month in range(crawl_jra.jra_today().month, 0, -1)
        ] + [(2025, month) for month in range(12, 0, -1)]

        def record_month(year, month, *_args, **_kwargs):
            scanned_months.append((year, month))
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", years),
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value=checkdigits,
                ),
                patch.object(
                    crawl_jra,
                    "discover_race_days",
                    side_effect=record_month,
                ),
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main()

        self.assertEqual(result, 0)
        self.assertEqual(scanned_months, expected_newest_months)

    def test_main_can_limit_crawl_to_current_month(self):
        today = crawl_jra.jra_today()
        scanned_months = []

        def record_month(year, month, *_args, **_kwargs):
            scanned_months.append((year, month))
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", [today.year, today.year - 1]),
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value={
                        f"{today.year % 100:02d}{today.month:02d}": "3F"
                    },
                ),
                patch.object(
                    crawl_jra,
                    "discover_race_days",
                    side_effect=record_month,
                ),
                patch.object(
                    crawl_jra,
                    "discover_upcoming_race_cnames",
                    return_value=[],
                ) as discover_entries,
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main(current_month_only=True)

        self.assertEqual(result, 0)
        self.assertEqual(scanned_months, [(today.year, today.month)])
        discover_entries.assert_called_once_with(today=today)

    def test_current_month_crawls_published_future_entries(self):
        today = crawl_jra.jra_today()
        future_date = today + timedelta(days=1)
        entry_cname = (
            f"pw01dde0105{today.year}010101"
            f"{future_date:%Y%m%d}/D9"
        )
        month_key = f"{today.year % 100:02d}{today.month:02d}"
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value={month_key: "3F"},
                ),
                patch.object(crawl_jra, "discover_race_days", return_value=[]),
                patch.object(
                    crawl_jra,
                    "discover_upcoming_race_cnames",
                    return_value=[entry_cname],
                ),
                patch.object(
                    crawl_jra,
                    "crawl_one_entry",
                    return_value=(5, True),
                ) as crawl_entry,
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main(
                    current_month_only=True,
                    enforce_database_size_limit=False,
                )

        self.assertEqual(result, 0)
        crawl_entry.assert_called_once()
        self.assertEqual(crawl_entry.call_args.args[1], entry_cname)

    def test_main_stops_without_crawling_when_database_is_at_size_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", [crawl_jra.jra_today().year]),
                patch.object(
                    crawl_jra,
                    "database_size_bytes",
                    return_value=crawl_jra.MAX_DATABASE_SIZE_BYTES,
                ),
                patch.object(crawl_jra, "year_month_checkdigits") as checkdigits,
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main()

        self.assertEqual(result, 0)
        checkdigits.assert_not_called()

    def test_current_month_crawl_ignores_database_size_limit(self):
        today = crawl_jra.jra_today()
        yymm = f"{today.year % 100:02d}{today.month:02d}"
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(
                    crawl_jra,
                    "database_size_bytes",
                    return_value=crawl_jra.MAX_DATABASE_SIZE_BYTES,
                ) as database_size_bytes,
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value={yymm: "3F"},
                ) as checkdigits,
                patch.object(
                    crawl_jra,
                    "discover_race_days",
                    return_value=[],
                ) as discover_race_days,
                patch.object(
                    crawl_jra,
                    "discover_upcoming_race_cnames",
                    return_value=[],
                ),
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main(
                    current_month_only=True,
                    enforce_database_size_limit=False,
                )

        self.assertEqual(result, 0)
        checkdigits.assert_called_once()
        discover_race_days.assert_called_once()
        database_size_bytes.assert_not_called()

    def test_main_treats_current_month_parameter_error_as_waiting(self):
        today = crawl_jra.jra_today()
        checkdigits = {
            f"{today.year % 100:02d}{month:02d}": "3F"
            for month in range(1, today.month + 1)
        }

        def discover_month(year, month, *_args, **_kwargs):
            if year == today.year and month == today.month:
                raise crawl_jra.JraParameterError("JRA parameter error")
            return []

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", [today.year]),
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value=checkdigits,
                ),
                patch.object(
                    crawl_jra,
                    "discover_race_days",
                    side_effect=discover_month,
                ) as discover_race_days,
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main()

            with sqlite3.connect(Path(temp_dir) / "test.sqlite") as conn:
                error_count = conn.execute(
                    "SELECT COUNT(*) FROM crawl_errors"
                ).fetchone()[0]

        self.assertEqual(result, 0)
        self.assertEqual(discover_race_days.call_count, today.month)
        self.assertEqual(error_count, 0)

    def test_main_stops_after_a_race_reaches_database_size_limit(self):
        year = crawl_jra.jra_today().year - 1
        day_cname = f"pw01srl1006{year}0101{year}0105/DA"
        race_cname = f"pw01sde1006{year}010101{year}0105/6B"
        checkdigits = {
            f"{year % 100:02d}{month:02d}": "3F"
            for month in range(1, 13)
        }
        size_values = iter(
            [0, crawl_jra.MAX_DATABASE_SIZE_BYTES]
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", [year]),
                patch.object(
                    crawl_jra,
                    "database_size_bytes",
                    side_effect=lambda: next(size_values),
                ),
                patch.object(
                    crawl_jra,
                    "year_month_checkdigits",
                    return_value=checkdigits,
                ),
                patch.object(
                    crawl_jra,
                    "discover_race_days",
                    side_effect=lambda _year, month, *_args, **_kwargs: (
                        [day_cname] if month == 1 else []
                    ),
                ) as discover_race_days,
                patch.object(
                    crawl_jra,
                    "discover_race_cnames_for_day",
                    return_value=[race_cname],
                ),
                patch.object(
                    crawl_jra,
                    "race_result_already_saved",
                    return_value=False,
                ),
                patch.object(
                    crawl_jra,
                    "crawl_one_race",
                    return_value=(16, 12, True),
                ) as crawl_one_race,
                patch.object(crawl_jra.SESSION, "close"),
            ):
                result = crawl_jra.main()

        self.assertEqual(result, 0)
        crawl_one_race.assert_called_once()
        self.assertEqual(discover_race_days.call_count, 12)


class LegacyRacePageTests(unittest.TestCase):
    def setUp(self):
        self.soup = BeautifulSoup(
            """
            <table>
              <tr><td>2002年3回中山5日( 10月 19日) 11Ｒ</td></tr>
              <tr><td>第5回 富士ステークス(ＧＩＩＩ)</td></tr>
              <tr><td>1600ｍ 芝・右 外</td></tr>
            </table>
            <table>
              <tr>
                <th>着順</th><th>枠</th><th>馬番</th><th>馬名</th>
                <th>性</th><th>齢</th><th>負担 重量</th><th>騎手</th>
                <th>タイム</th><th>着差</th><th>馬体重</th><th>調教師</th>
                <th>単勝 人気</th>
              </tr>
              <tr>
                <td>1</td><td>4</td><td>6</td><td>テストホース</td>
                <td>牡</td><td>4</td><td>56.0kg</td><td>テスト騎手</td>
                <td>1:32.3</td><td></td><td>456Kg +6</td>
                <td>テスト調教師</td><td>1</td>
              </tr>
            </table>
            <table>
              <tr><td>単勝</td><td>06</td><td>490円</td><td>1番人気</td></tr>
              <tr><td>３連複</td><td>03-06-15</td><td>3,840円</td><td>10番人気</td></tr>
            </table>
            """,
            "html.parser",
        )
        self.url = (
            "https://www.jra.go.jp/datafile/seiseki/replay/2002/101.html"
        )

    def test_extracts_metadata_from_legacy_result_page(self):
        race = crawl_jra.extract_race_info(self.soup, self.url)

        self.assertEqual(race["race_date"], "2002-10-19")
        self.assertEqual(race["venue"], "中山")
        self.assertEqual(race["race_name"], "富士ステークス")
        self.assertEqual(race["distance_m"], 1600)
        self.assertEqual(race["surface"], "芝")
        self.assertEqual(race["direction"], "右")

    def test_parses_legacy_horse_and_payout_tables(self):
        horse = crawl_jra.parse_horses(self.soup)[0]
        payouts = crawl_jra.parse_payouts(self.soup)

        self.assertEqual(horse["sex_age"], "牡4")
        self.assertEqual(horse["weight_change"], 6)
        self.assertEqual(
            payouts,
            [
                {
                    "bet_type": "単勝",
                    "combination": "06",
                    "amount": 490,
                    "popularity": 1,
                },
                {
                    "bet_type": "3連複",
                    "combination": "03-06-15",
                    "amount": 3840,
                    "popularity": 10,
                },
            ],
        )


class TransitionalRacePageTests(unittest.TestCase):
    def test_extracts_race_data_from_2006_table_layout(self):
        soup = BeautifulSoup(
            """
            <table>
              <tr>
                <td>2006年10月8日（日） 5回京都2日</td>
              </tr>
              <tr><td>第４１回 京都大賞典</td></tr>
              <tr><td>2400ｍ 芝・右 外</td></tr>
            </table>
            <table>
              <tr>
                <th>着順</th><th>枠</th><th>馬番</th><th>馬名</th>
                <th>性齢</th><th>負担重量</th><th>騎手</th><th>タイム</th>
              </tr>
              <tr>
                <td>1</td><td>7</td><td>9</td><td>テストホース</td>
                <td>牡4</td><td>56.0</td><td>テスト騎手</td><td>2:31.5</td>
              </tr>
            </table>
            """,
            "html.parser",
        )
        url = (
            "https://www.jra.go.jp/datafile/seiseki/replay/2006/100.html"
        )

        race = crawl_jra.extract_race_info(soup, url)

        self.assertEqual(race["race_date"], "2006-10-08")
        self.assertEqual(race["venue"], "京都")
        self.assertEqual(race["race_name"], "京都大賞典")
        self.assertEqual(race["distance_m"], 2400)
        self.assertEqual(race["surface"], "芝")
        self.assertEqual(race["direction"], "右")

    def test_uses_race_heading_instead_of_2018_page_title(self):
        soup = BeautifulSoup(
            """
            <title>2018年 重賞レース JRA</title>
            <h1>2018年 重賞レース一覧</h1>
            <h2>2018年1月6日（土曜） 1回中山1日</h2>
            <h3>11R 第67回 日刊スポーツ賞中山金杯</h3>
            <table>
              <tr><td>2000m （芝・右）</td></tr>
            </table>
            """,
            "html.parser",
        )
        url = (
            "https://www.jra.go.jp/datafile/seiseki/replay/2018/001.html"
        )

        race = crawl_jra.extract_race_info(soup, url)

        self.assertEqual(race["race_name"], "日刊スポーツ賞中山金杯")
        self.assertEqual(race["distance_m"], 2000)
        self.assertEqual(race["surface"], "芝")
        self.assertEqual(race["direction"], "右")


class HorseFieldExtractionTests(unittest.TestCase):
    def test_extracts_pre_race_horse_fields_from_jra_entry_table(self):
        rows = "".join(
            f"""
            <tr>
              <td class="waku"><img alt="{number}枠"></td>
              <td class="num">{number}</td>
              <td class="horse">
                <div class="name_line">
                  <div class="name">
                    <a href="/JRADB/accessU.html?CNAME=horse{number}">
                      Horse {number}
                    </a>
                  </div>
                  <div class="odds">
                    <span class="num">{number}.5</span>
                    <span class="pop_rank">({number} 番人気)</span>
                  </div>
                </div>
                <p class="trainer">
                  <a href="#" onclick="return doAction(
                    '/JRADB/accessC.html', 'trainer{number}/AA');">
                    Trainer {number}
                  </a>
                  <span class="division">(美浦)</span>
                </p>
              </td>
              <td class="jockey">
                <p class="jockey">
                  <a href="#" onclick="return doAction(
                    '/JRADB/accessK.html', 'jockey{number}/BB');">
                  Jockey {number}
                  </a>
                </p>
                <p class="age">{"牝3" if number == 1 else "牡4"}/鹿</p>
                <p class="weight">56.0 kg</p>
              </td>
            </tr>
            """
            for number in range(1, 6)
        )
        soup = BeautifulSoup(
            f"<table>{rows}</table>",
            "html.parser",
        )

        horses = crawl_jra.parse_entry_horses(soup)

        self.assertEqual(len(horses), 5)
        self.assertEqual(horses[0]["horse_name"], "Horse 1")
        self.assertEqual(horses[0]["horse_number"], 1)
        self.assertEqual(horses[0]["frame"], 1)
        self.assertEqual(horses[0]["sex_age"], "牝3")
        self.assertEqual(horses[0]["weight"], 56.0)
        self.assertEqual(horses[0]["jockey"], "Jockey 1")
        self.assertEqual(horses[0]["trainer"], "Trainer 1")
        self.assertEqual(horses[0]["odds"], 1.5)
        self.assertEqual(horses[0]["popularity"], 1)
        self.assertEqual(
            horses[0]["trainer_url"],
            "https://www.jra.go.jp/JRADB/accessC.html?CNAME=trainer1/AA",
        )
        self.assertEqual(
            horses[0]["jockey_url"],
            "https://www.jra.go.jp/JRADB/accessK.html?CNAME=jockey1/BB",
        )
        self.assertIsNone(horses[0]["finish"])
        self.assertEqual(
            horses[0]["horse_url"],
            "https://www.jra.go.jp/JRADB/accessU.html?CNAME=horse1",
        )

    def test_extracts_trainer_without_profile_link(self):
        soup = BeautifulSoup(
            """
            <table>
              <tr>
                <td class="num">1</td>
                <td class="horse">
                  <a href="/horse.html">Horse</a>
                  <p class="trainer">
                    Trainer Name <span class="division">(本会外)</span>
                  </p>
                </td>
              </tr>
            </table>
            """,
            "html.parser",
        )

        horse = crawl_jra.parse_entry_horses(soup)[0]

        self.assertEqual(horse["trainer"], "Trainer Name")
        self.assertIsNone(horse["trainer_url"])

    def test_keeps_entries_before_horse_numbers_are_assigned(self):
        rows = "".join(
            f"""
            <tr>
              <td class="waku"></td>
              <td class="num"></td>
              <td class="horse">
                <div class="name">
                  <a href="/horse{number}.html">Horse {number}</a>
                </div>
                <p class="trainer">Trainer {number} <span class="division">
                  (美浦)
                </span></p>
              </td>
              <td class="jockey">
                <p class="age">牡3/鹿</p>
                <p class="weight">56.0 <span>kg</span></p>
                <p class="jockey"><a href="#">Jockey {number}</a></p>
              </td>
            </tr>
            """
            for number in range(1, 6)
        )
        soup = BeautifulSoup(f"<table>{rows}</table>", "html.parser")

        horses = crawl_jra.parse_entry_horses(soup)

        self.assertEqual(len(horses), 5)
        self.assertEqual(
            [horse["horse_name"] for horse in horses],
            [f"Horse {number}" for number in range(1, 6)],
        )
        self.assertTrue(
            all(horse["horse_number"] is None for horse in horses)
        )
        self.assertTrue(
            all(horse["frame"] is None for horse in horses)
        )
        self.assertEqual(horses[0]["trainer"], "Trainer 1")
        self.assertEqual(horses[0]["jockey"], "Jockey 1")

    def test_ignores_unassigned_candidates_when_card_has_numbered_entries(self):
        rows = "".join(
            f"""
            <tr>
              <td class="num">{number}</td>
              <td class="horse">
                <div class="name"><a href="/horse{number}.html">
                  Horse {number}
                </a></div>
              </td>
            </tr>
            """
            for number in range(1, 6)
        )
        rows += """
            <tr>
              <td class="num"></td>
              <td class="horse">
                <div class="name"><a href="/candidate.html">
                  Unassigned candidate
                </a></div>
              </td>
            </tr>
        """
        soup = BeautifulSoup(f"<table>{rows}</table>", "html.parser")

        horses = crawl_jra.parse_entry_horses(soup)

        self.assertEqual(len(horses), 5)
        self.assertTrue(
            all(horse["horse_number"] is not None for horse in horses)
        )

    def test_treats_unpublished_entry_page_as_waiting_not_failure(self):
        with (
            sqlite3.connect(":memory:") as conn,
            patch.object(
                crawl_jra,
                "fetch_jra_card_page",
                side_effect=crawl_jra.JraParameterError("not published"),
            ),
            patch.object(crawl_jra, "save_error") as save_error,
        ):
            result = crawl_jra.crawl_one_entry(
                conn,
                "pw01dde0105202604090120261025/09",
            )

        self.assertEqual(result, (0, None))
        save_error.assert_not_called()

    def test_extracts_odds_popularity_and_raw_columns(self):
        soup = BeautifulSoup(
            """
            <table>
              <tr>
                <th>着順</th><th>枠</th><th>馬番</th><th>馬名</th>
                <th>性齢</th><th>負担重量</th><th>騎手</th>
                <th>単勝オッズ</th><th>タイム</th><th>着差</th>
                <th>推定上り</th><th>馬体重</th><th>調教師</th>
                <th>単勝人気</th>
              </tr>
              <tr>
                <td>1</td><td>4</td><td>6</td><td>テストホース</td>
                <td>牡4</td><td>56.0</td><td></td><td>12.5</td>
                <td>1:32.3</td><td></td><td>34.5</td>
                <td>480 (+4)</td><td>テスト調教師</td><td>3</td>
              </tr>
            </table>
            """,
            "html.parser",
        )

        horse = crawl_jra.parse_horses(soup)[0]

        self.assertEqual(horse["jockey"], None)
        self.assertEqual(horse["odds"], 12.5)
        self.assertEqual(horse["popularity"], 3)
        self.assertEqual(horse["time"], "1:32.3")
        self.assertEqual(horse["last_3f"], 34.5)
        self.assertEqual(horse["body_weight"], 480)
        self.assertEqual(horse["weight_change"], 4)
        self.assertEqual(horse["trainer"], "テスト調教師")
        self.assertEqual(horse["source_fields"]["単勝オッズ"], "12.5")

    def test_aligns_separate_zero_body_weight_change_column(self):
        soup = BeautifulSoup(
            """
            <table>
              <tr>
                <th>着順</th><th>枠</th><th>馬番</th><th>馬名</th>
                <th>性齢</th><th>負担重量</th><th>騎手</th>
                <th>タイム</th><th>着差</th><th>推定上り</th>
                <th>馬体重</th><th>調教師</th><th>単勝人気</th>
              </tr>
              <tr>
                <td>1</td><td></td><td>3</td><td>テストホース</td>
                <td>牡6</td><td>56.5</td><td></td><td>テスト騎手</td>
                <td>2:00.6</td><td></td><td>35.6</td><td>480</td>
                <td>0</td><td>テスト調教師</td><td>1</td>
              </tr>
            </table>
            """,
            "html.parser",
        )

        horse = crawl_jra.parse_horses(soup)[0]

        self.assertEqual(horse["jockey"], "テスト騎手")
        self.assertEqual(horse["time"], "2:00.6")
        self.assertEqual(horse["last_3f"], 35.6)
        self.assertEqual(horse["body_weight"], 480)
        self.assertEqual(horse["weight_change"], 0)
        self.assertEqual(horse["trainer"], "テスト調教師")
        self.assertEqual(horse["popularity"], 1)


if __name__ == "__main__":
    unittest.main()
