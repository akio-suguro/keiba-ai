import unittest
from unittest.mock import patch

from bs4 import BeautifulSoup

from scripts import crawl_jra


class RaceIndexUrlTests(unittest.TestCase):
    def test_years_cover_start_year_through_current_year(self):
        self.assertEqual(
            crawl_jra.YEARS,
            list(range(crawl_jra.START_YEAR, crawl_jra.CURRENT_YEAR + 1)),
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


if __name__ == "__main__":
    unittest.main()
