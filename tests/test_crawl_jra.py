import sqlite3
import tempfile
import unittest
from datetime import date
from pathlib import Path
from unittest.mock import ANY, patch

from bs4 import BeautifulSoup

from scripts import crawl_jra


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
        today = date.today()
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

    def test_rejects_invalid_race_cname(self):
        with self.assertRaisesRegex(ValueError, "形式が不正"):
            crawl_jra.race_cname_details("pw01sde-invalid")


class DatabaseMigrationTests(unittest.TestCase):
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


class UnconfirmedRaceTests(unittest.TestCase):
    def test_unconfirmed_current_day_result_is_skipped_without_error_record(self):
        today = date.today()
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
        year = date.today().year - 1
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
            for month in range(date.today().month, 0, -1)
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

    def test_main_stops_without_crawling_when_database_is_at_size_limit(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                patch.object(
                    crawl_jra,
                    "DB_PATH",
                    Path(temp_dir) / "test.sqlite",
                ),
                patch.object(crawl_jra, "YEARS", [date.today().year]),
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

    def test_main_stops_after_a_race_reaches_database_size_limit(self):
        year = date.today().year - 1
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
