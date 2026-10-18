# scripts/crawl_jra.py
# JRA重賞レース結果クローラー
# 対象: 2002年〜現在
#
# HTMLはメモリ上で解析し、ファイルには保存しない。
#
# ログ:
#   [OK]    正常取得したレース
#   [ERROR] 取得・解析・保存エラー
#   [DONE]  最終集計
#
# 既存DB:
#   races(id, url, race_date, venue, race_name,
#         distance_m, surface, direction, crawled_at)
#   horses(id, race_id, finish, frame, horse_number,
#          horse_name, sex_age, weight, jockey, time,
#          margin, last_3f, body_weight, weight_change,
#          trainer, popularity, odds)
#   payouts(id, race_id, bet_type, combination, amount)

from __future__ import annotations

import re
import sqlite3
import sys
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ============================================================
# 設定
# ============================================================

BASE_URL = "https://www.jra.go.jp"

START_YEAR = 2002
CURRENT_YEAR = datetime.now().year
YEARS = list(range(START_YEAR, CURRENT_YEAR + 1))

DB_PATH = Path("data/keiba.db")

REQUEST_INTERVAL = 2
REQUEST_TIMEOUT = 30

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/131.0.0.0 Safari/537.36"
    )
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

BET_TYPES = {
    "単勝",
    "複勝",
    "枠連",
    "ワイド",
    "馬連",
    "馬単",
    "3連複",
    "3連単",
}

RACE_URL_PATTERN = re.compile(
    r"/datafile/seiseki/replay/(\d{4})/(\d{3})\.html"
)

VENUES = (
    "札幌", "函館", "福島", "新潟", "東京",
    "中山", "中京", "京都", "阪神", "小倉",
)


# ============================================================
# 共通処理
# ============================================================

def normalize_text(value: str | None) -> str:
    if value is None:
        return ""

    value = value.replace("\xa0", " ")
    value = value.replace("\u3000", " ")
    return re.sub(r"\s+", " ", value).strip()


def normalize_header(value: str) -> str:
    value = normalize_text(value)
    return re.sub(r"[\s　（）()・]", "", value)


def to_int(value: str | None) -> int | None:
    if not value:
        return None

    value = normalize_text(value).replace(",", "")
    match = re.search(r"[+-]?\d+", value)
    if not match:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def to_float(value: str | None) -> float | None:
    if not value:
        return None

    value = normalize_text(value).replace(",", "")
    match = re.search(r"[+-]?\d+(?:\.\d+)?", value)

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def fetch_html(url: str) -> BeautifulSoup:
    """
    HTMLをHTTP取得してメモリ上で解析する。
    ファイル保存・HTMLキャッシュは行わない。
    """
    response = SESSION.get(
        url,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    response.encoding = (
        response.apparent_encoding or response.encoding
    )

    return BeautifulSoup(response.text, "html.parser")


def header_variants(value: str) -> set[str]:
    """
    ヘッダー表記ゆれを吸収するための候補を作る。
    """
    normalized = normalize_header(value)

    aliases = {
        "着順": {"着順", "順位"},
        "枠": {"枠", "枠番"},
        "枠番": {"枠", "枠番"},
        "馬番": {"馬番", "馬番号"},
        "馬名": {"馬名"},
        "性齢": {"性齢"},
        "斤量": {"斤量", "負担重量"},
        "負担重量": {"斤量", "負担重量"},
        "騎手": {"騎手", "騎手名"},
        "騎手名": {"騎手", "騎手名"},
        "タイム": {"タイム"},
        "着差": {"着差"},
        "上り": {"上り", "推定上り", "上がり"},
        "推定上り": {"上り", "推定上り", "上がり"},
        "馬体重": {"馬体重", "馬体重増減"},
        "調教師": {"調教師", "調教師名"},
        "調教師名": {"調教師", "調教師名"},
        "オッズ": {"オッズ", "単勝オッズ"},
        "単勝オッズ": {"オッズ", "単勝オッズ"},
        "人気": {"人気", "単勝人気"},
        "単勝人気": {"人気", "単勝人気"},
    }

    return aliases.get(normalized, {normalized})


def find_column(
    headers: list[str],
    *keywords: str,
) -> int | None:
    normalized = [normalize_header(h) for h in headers]

    for keyword in keywords:
        candidates = header_variants(keyword)

        for index, header in enumerate(normalized):
            if header in candidates:
                return index

    return None


def cell_text(cells, index: int | None) -> str | None:
    if index is None or index < 0 or index >= len(cells):
        return None

    value = normalize_text(
        cells[index].get_text(" ", strip=True)
    )

    return value or None


def extract_year_race_number(url: str) -> tuple[int, int] | None:
    match = RACE_URL_PATTERN.search(urlparse(url).path)

    if not match:
        return None

    return int(match.group(1)), int(match.group(2))


# ============================================================
# DB
# ============================================================

def connect_db() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """
    テーブルが存在しない新規DBの場合に作成する。
    既存テーブルの列構成は変更しない。
    """
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS races (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT UNIQUE,
            race_date TEXT,
            venue TEXT,
            race_name TEXT,
            distance_m INTEGER,
            surface TEXT,
            direction TEXT,
            crawled_at TEXT
        );

        CREATE TABLE IF NOT EXISTS horses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            odds REAL
        );

        CREATE TABLE IF NOT EXISTS payouts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            race_id INTEGER NOT NULL,
            bet_type TEXT,
            combination TEXT,
            amount INTEGER
        );

        CREATE TABLE IF NOT EXISTS crawl_errors (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            error TEXT,
            occurred_at TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_races_date
            ON races(race_date);

        CREATE INDEX IF NOT EXISTS idx_horses_race
            ON horses(race_id);

        CREATE INDEX IF NOT EXISTS idx_payouts_race
            ON payouts(race_id);
        """
    )
    conn.commit()


def get_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
) -> set[str]:
    rows = conn.execute(
        f'PRAGMA table_info("{table_name}")'
    ).fetchall()

    return {row["name"] for row in rows}


def save_error(
    conn: sqlite3.Connection,
    url: str,
    error: Exception,
) -> None:
    try:
        conn.execute(
            """
            INSERT INTO crawl_errors (url, error, occurred_at)
            VALUES (?, ?, ?)
            """,
            (
                url,
                f"{type(error).__name__}: {error}"[:2000],
                datetime.now().isoformat(timespec="seconds"),
            ),
        )
        conn.commit()
    except sqlite3.Error:
        conn.rollback()


# ============================================================
# 重賞レース一覧
# ============================================================

def race_index_url(year: int) -> str:
    """
    現行年の一覧は replay 直下、それ以前は年別ディレクトリにある。
    """
    year_path = "" if year == CURRENT_YEAR else f"{year}/"
    return f"{BASE_URL}/datafile/seiseki/replay/{year_path}jyusyo.html"


def discover_race_urls(year: int) -> list[str]:
    """
    年別の重賞レース一覧から個別レースURLを取得する。
    """
    index_url = race_index_url(year)
    soup = fetch_html(index_url)
    urls: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()
        absolute_url = urljoin(index_url, href)
        parsed = urlparse(absolute_url)

        if parsed.netloc not in {"www.jra.go.jp", "jra.go.jp"}:
            continue

        match = RACE_URL_PATTERN.search(parsed.path)

        if not match:
            continue

        if int(match.group(1)) != year:
            continue

        # URLのクエリ・フラグメントは識別に不要
        canonical_url = (
            f"{parsed.scheme}://{parsed.netloc}{parsed.path}"
        )
        urls.add(canonical_url)

    return sorted(
        urls,
        key=lambda url: int(
            RACE_URL_PATTERN.search(urlparse(url).path).group(2)
        ),
    )


# ============================================================
# レース情報
# ============================================================

def extract_race_info(
    soup: BeautifulSoup,
    url: str,
) -> dict:
    info = {
        "url": url,
        "race_date": None,
        "venue": None,
        "race_name": None,
        "distance_m": None,
        "surface": None,
        "direction": None,
        "crawled_at": datetime.now().isoformat(timespec="seconds"),
    }
    page_text = normalize_text(soup.get_text(" ", strip=True))
    normalized_page_text = unicodedata.normalize("NFKC", page_text)
    legacy_date_pattern = re.compile(
        r"\d{4}年\s*\d+回\s*(?:"
        + "|".join(VENUES)
        + r")\s*\d+日\s*[（(]\s*\d{1,2}月\s*\d{1,2}日"
    )
    legacy_header = min(
        (
            normalize_text(table.get_text(" ", strip=True))
            for table in soup.find_all("table")
            if legacy_date_pattern.search(
                normalize_text(table.get_text(" ", strip=True))
            )
        ),
        key=len,
        default="",
    )
    generic_titles = {
        "JRAホームページ",
        "レース結果",
        "競走成績",
        "検索ウィンドウ",
        "JRAからのお知らせ",
    }

    def race_name_from_text(text: str) -> str | None:
        normalized = unicodedata.normalize("NFKC", normalize_text(text))
        match = re.search(
            r"第\s*\d+\s*回\s*(.+?)(?:\s*[（(]|$)",
            normalized,
        )
        if match:
            return match.group(1).strip()

        return None

    for element in soup.select(".race_name"):
        text = normalize_text(element.get_text(" ", strip=True))
        if text:
            info["race_name"] = text
            break

    if not info["race_name"]:
        # Earlier layouts put the title in a table cell rather than a heading.
        candidates = []
        for element in soup.find_all(
            ["h1", "h2", "h3", "h4", "td", "th"]
        ):
            text = normalize_text(element.get_text(" ", strip=True))
            race_name = race_name_from_text(text)

            if race_name:
                candidates.append((len(text), race_name))

        if candidates:
            info["race_name"] = min(candidates, key=lambda item: item[0])[1]

    if not info["race_name"] and legacy_header:
        info["race_name"] = race_name_from_text(legacy_header)

    if not info["race_name"]:
        for selector in (".race_title", "h3", "h2", "h1", "title"):
            for element in soup.select(selector):
                text = normalize_text(element.get_text(" ", strip=True))
                normalized = unicodedata.normalize("NFKC", text)

                if (
                    text
                    and text not in generic_titles
                    and "JRAホームページ" not in text
                    and "重賞レース" not in text
                    and not re.search(r"\d{4}年.*JRA", normalized)
                ):
                    info["race_name"] = text
                    break

            if info["race_name"]:
                break

    # ページ全体から日付を探す
    date_patterns = (
        r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日",
        r"(\d{4})[./年-]\s*(\d{1,2})[./月-]\s*(\d{1,2})日?",
    )

    for pattern in date_patterns:
        match = re.search(pattern, page_text) or re.search(
            pattern, legacy_header
        )

        if match:
            year, month, day = map(int, match.groups())

            if 2000 <= year <= datetime.now().year + 1:
                info["race_date"] = (
                    f"{year:04d}-{month:02d}-{day:02d}"
                )
                break

    if not info["race_date"] and legacy_header:
        legacy_date_match = re.search(
            r"(\d{4})年\s*\d+回.*?\d+日\s*[（(]\s*"
            r"(\d{1,2})月\s*(\d{1,2})日",
            legacy_header,
        )

        if legacy_date_match:
            year, month, day = map(int, legacy_date_match.groups())
            info["race_date"] = f"{year:04d}-{month:02d}-{day:02d}"

    # 開催情報は「第N回 東京 N日」などの表記を探す
    venue_pattern = (
        r"\d+回\s*("
        + "|".join(VENUES)
        + r")\s*\d+日"
    )

    venue_match = re.search(venue_pattern, legacy_header) or re.search(
        venue_pattern, page_text
    )

    if venue_match:
        info["venue"] = venue_match.group(1)
    else:
        # 開催情報の書式が異なる古いページ向けの補助判定
        for venue in VENUES:
            if venue in page_text:
                info["venue"] = venue
                break

    # 距離・芝ダート・方向
    course_pattern = re.compile(
        r"(?P<distance>[\d,]+)\s*(?:m|メートル)\s*"
        r"[（(]?\s*(?P<surface>ダート|芝|ダ|障害)"
        r"(?:\s*[・ ]\s*(?P<direction>右|左|直線))?",
        flags=re.IGNORECASE,
    )
    course_match = course_pattern.search(normalized_page_text)

    if course_match:
        info["distance_m"] = to_int(course_match.group("distance"))
        surface = course_match.group("surface")
        info["surface"] = "ダート" if surface == "ダ" else surface
        info["direction"] = course_match.group("direction")

    for selector in (".race_title", "h1", "h2", "title"):
        element = soup.select_one(selector)

        if not element:
            continue

        title_text = normalize_text(
            f"{element.get_text(' ', strip=True)} {legacy_header}"
        )

        distance_match = re.search(
            r"(?:芝|ダート|ダ|障害)?\s*([\d,]{3,5})\s*(?:メートル|ｍ|m)",
            unicodedata.normalize("NFKC", title_text),
            flags=re.IGNORECASE,
        )

        if distance_match and not info["distance_m"]:
            info["distance_m"] = to_int(distance_match.group(1))

        if not info["surface"]:
            if "ダート" in title_text or "ダート" in page_text[:500]:
                info["surface"] = "ダート"
            elif "芝" in title_text:
                info["surface"] = "芝"
            elif "障害" in title_text:
                info["surface"] = "障害"

        direction_match = re.search(r"(右|左|直線)", title_text)

        if direction_match and not info["direction"]:
            info["direction"] = direction_match.group(1)

    if legacy_header:
        distance_match = re.search(
            r"(?:芝|ダート|ダ|障害)?\s*([\d,]{3,5})\s*(?:メートル|ｍ|m)",
            unicodedata.normalize("NFKC", legacy_header),
            flags=re.IGNORECASE,
        )

        if distance_match and not info["distance_m"]:
            info["distance_m"] = to_int(distance_match.group(1))

        if not info["surface"]:
            if "ダート" in legacy_header:
                info["surface"] = "ダート"
            elif "障害" in legacy_header:
                info["surface"] = "障害"
            elif "芝" in legacy_header:
                info["surface"] = "芝"

        direction_match = re.search(r"(右|左|直線)", legacy_header)
        if direction_match and not info["direction"]:
            info["direction"] = direction_match.group(1)

    # 日付が見つからない場合、URL年度だけを使って日付を捏造しない
    return info


# ============================================================
# 出走馬データ
# ============================================================

def get_table_headers(table) -> tuple[list[str], int]:
    """
    テーブル内の先頭数行を調べ、見出し行を探す。
    古いHTMLでは見出しが1行目とは限らない。
    """
    rows = table.find_all("tr")

    for row_index, row in enumerate(rows[:5]):
        cells = row.find_all(["th", "td"], recursive=False)

        if not cells:
            cells = row.find_all(["th", "td"])

        headers = [
            normalize_text(cell.get_text(" ", strip=True))
            for cell in cells
        ]

        normalized = [normalize_header(h) for h in headers]

        has_horse = any("馬名" == h for h in normalized)
        has_number = any(h in {"馬番", "馬番号"} for h in normalized)
        has_finish = any(h in {"着順", "順位"} for h in normalized)

        if has_horse and has_number and has_finish:
            return headers, row_index

    return [], -1


def find_horse_table(soup: BeautifulSoup):
    """
    新旧のテーブル構造を想定して出走馬テーブルを探す。
    """
    best_table = None
    best_headers: list[str] = []
    best_header_row = -1
    best_score = 0

    for table in soup.find_all("table"):
        headers, header_row = get_table_headers(table)

        if not headers:
            continue

        normalized = [normalize_header(h) for h in headers]

        score = 0
        if any(h in {"着順", "順位"} for h in normalized):
            score += 2
        if any(h == "馬名" for h in normalized):
            score += 3
        if any(h in {"馬番", "馬番号"} for h in normalized):
            score += 2
        if any(h in {"騎手", "騎手名"} for h in normalized):
            score += 1

        if score > best_score:
            best_table = table
            best_headers = headers
            best_header_row = header_row
            best_score = score

    if best_score < 7:
        return None, [], -1

    return best_table, best_headers, best_header_row


def parse_horses(soup: BeautifulSoup) -> list[dict]:
    table, headers, header_row = find_horse_table(soup)

    if table is None:
        return []

    indexes = {
        "finish": find_column(headers, "着順", "順位"),
        "frame": find_column(headers, "枠", "枠番"),
        "horse_number": find_column(headers, "馬番", "馬番号"),
        "horse_name": find_column(headers, "馬名"),
        "sex_age": find_column(headers, "性齢"),
        "sex": find_column(headers, "性"),
        "age": find_column(headers, "齢", "年齢"),
        "weight": find_column(headers, "負担重量", "斤量"),
        "jockey": find_column(headers, "騎手名", "騎手"),
        "time": find_column(headers, "タイム"),
        "margin": find_column(headers, "着差"),
        "last_3f": find_column(headers, "推定上り", "上り"),
        "body_weight": find_column(headers, "馬体重", "馬体重（増減）"),
        "trainer": find_column(headers, "調教師名", "調教師"),
        "odds": find_column(headers, "単勝オッズ", "オッズ"),
        "popularity": find_column(headers, "人気", "単勝人気"),
    }

    horses: list[dict] = []
    rows = table.find_all("tr")

    for row in rows[header_row + 1:]:
        cells = row.find_all(["td", "th"], recursive=False)

        if not cells:
            cells = row.find_all(["td", "th"])

        if not cells:
            continue

        horse_name = cell_text(cells, indexes["horse_name"])
        horse_number = to_int(cell_text(cells, indexes["horse_number"]))

        # 見出し行や脚注を誤って馬として登録しない
        if (
            not horse_name
            or horse_number is None
            or not 1 <= horse_number <= 20
        ):
            continue

        finish_text = cell_text(cells, indexes["finish"])
        finish = to_int(finish_text)

        # 除外・取消・中止等は finish が数値にならない場合がある。
        # 馬名と馬番が揃っていれば出走馬行として扱う。
        sex_age = cell_text(cells, indexes["sex_age"])
        if not sex_age:
            sex_age_parts = [
                cell_text(cells, indexes[key])
                for key in ("sex", "age")
            ]
            sex_age = "".join(part or "" for part in sex_age_parts) or None

        jockey = cell_text(cells, indexes["jockey"])

        body_weight_text = cell_text(
            cells, indexes["body_weight"]
        )

        # 例: 426 (+12) → 426 / +12
        body_weight = None
        weight_change = None

        if body_weight_text:
            weight_match = re.search(r"^\s*(\d+)", body_weight_text)
            if weight_match:
                body_weight = int(weight_match.group(1))

            change_match = re.search(
                r"\(?\s*([+-]\d+)\s*\)?",
                body_weight_text,
            )
            if change_match:
                weight_change = int(change_match.group(1))

        horses.append(
            {
                "finish": finish,
                "frame": to_int(cell_text(cells, indexes["frame"])),
                "horse_number": horse_number,
                "horse_name": horse_name,
                "sex_age": sex_age,
                "weight": to_float(cell_text(cells, indexes["weight"])),
                "jockey": jockey,
                "time": cell_text(cells, indexes["time"]),
                "margin": cell_text(cells, indexes["margin"]),
                "last_3f": to_float(cell_text(cells, indexes["last_3f"])),
                "body_weight": body_weight,
                "weight_change": weight_change,
                "trainer": cell_text(cells, indexes["trainer"]),
                "odds": to_float(cell_text(cells, indexes["odds"])),
                "popularity": to_int(
                    cell_text(cells, indexes["popularity"])
                ),
            }
        )

    # 同じ馬番が複数行に出てくる異常ケースを除去
    unique: dict[int, dict] = {}

    for horse in horses:
        unique[horse["horse_number"]] = horse

    return list(unique.values())


# ============================================================
# 払戻金データ
# ============================================================

def parse_payouts(soup: BeautifulSoup) -> list[dict]:
    """
    新旧のJRA結果ページから払戻金を取得する。
    """
    payouts: list[dict] = []

    refund_area = soup.select_one(".refund_area")

    if refund_area is not None:
        for item in refund_area.select("li"):
            dt = item.find("dt")

            if dt is None:
                continue

            bet_type = normalize_text(dt.get_text(" ", strip=True))

            if bet_type not in BET_TYPES:
                continue

            for line in item.select(".line"):
                num_element = line.select_one(".num")
                yen_element = line.select_one(".yen")

                if num_element is None or yen_element is None:
                    continue

                combination = normalize_text(
                    num_element.get_text(" ", strip=True)
                )
                amount = to_int(yen_element.get_text(" ", strip=True))

                if not combination or amount is None:
                    continue

                pop_element = line.select_one(".pop")
                popularity = (
                    to_int(pop_element.get_text(" ", strip=True))
                    if pop_element
                    else None
                )

                payouts.append(
                    {
                        "bet_type": bet_type,
                        "combination": combination,
                        "amount": amount,
                        "popularity": popularity,
                    }
                )

    if not payouts:
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"], recursive=False)

            if len(cells) < 3:
                continue

            bet_type = normalize_text(
                cells[0].get_text(" ", strip=True)
            ).translate(str.maketrans("３", "3"))

            if bet_type not in BET_TYPES:
                continue

            combination = normalize_text(
                cells[1].get_text(" ", strip=True)
            )
            amount = to_int(cells[2].get_text(" ", strip=True))

            if not combination or amount is None:
                continue

            payouts.append(
                {
                    "bet_type": bet_type,
                    "combination": combination,
                    "amount": amount,
                    "popularity": (
                        to_int(cells[3].get_text(" ", strip=True))
                        if len(cells) > 3
                        else None
                    ),
                }
            )

    # 同じ券種・組み合わせの重複行を除去
    unique: dict[tuple[str, str], dict] = {}

    for payout in payouts:
        key = (payout["bet_type"], payout["combination"])
        unique[key] = payout

    return list(unique.values())


# ============================================================
# 保存処理
# ============================================================

def insert_row(
    conn: sqlite3.Connection,
    table: str,
    values: dict,
) -> int | None:
    columns_in_db = get_table_columns(conn, table)

    values = {
        key: value
        for key, value in values.items()
        if key in columns_in_db
    }

    if not values:
        raise RuntimeError(f"{table}に保存可能な列がありません")

    columns = list(values.keys())
    column_sql = ", ".join(f'"{column}"' for column in columns)
    placeholders = ", ".join("?" for _ in columns)

    cursor = conn.execute(
        f"INSERT INTO {table} ({column_sql}) VALUES ({placeholders})",
        tuple(values[column] for column in columns),
    )

    return cursor.lastrowid


def save_race(
    conn: sqlite3.Connection,
    race: dict,
    horses: list[dict],
    payouts: list[dict],
) -> None:
    """
    1レースをトランザクションで保存する。
    レース名・日付・出走馬が取得できない場合は保存しない。
    """
    if not race.get("race_name"):
        raise ValueError("レース名を取得できません。保存を中止しました")

    if not race.get("race_date"):
        raise ValueError("レース日を取得できません。保存を中止しました")

    if not race.get("venue"):
        raise ValueError("競馬場を取得できません。保存を中止しました")

    if len(horses) < 5:
        raise ValueError(
            f"出走馬データが少なすぎます（{len(horses)}件）。保存を中止しました"
        )

    race_columns = get_table_columns(conn, "races")

    race_values = {
        "url": race["url"],
        "race_date": race["race_date"],
        "venue": race["venue"],
        "race_name": race["race_name"],
        "distance_m": race.get("distance_m"),
        "surface": race.get("surface"),
        "direction": race.get("direction"),
        "crawled_at": race.get("crawled_at"),
    }

    race_values = {
        key: value
        for key, value in race_values.items()
        if key in race_columns
    }

    with conn:
        existing = conn.execute(
            "SELECT id FROM races WHERE url = ?",
            (race["url"],),
        ).fetchone()

        if existing:
            race_id = existing["id"]

            update_values = {
                key: value
                for key, value in race_values.items()
                if key != "url"
            }

            if update_values:
                assignments = ", ".join(
                    f'"{column}" = ?'
                    for column in update_values
                )

                conn.execute(
                    f"UPDATE races SET {assignments} WHERE id = ?",
                    (*update_values.values(), race_id),
                )
        else:
            race_id = insert_row(conn, "races", race_values)

        if race_id is None:
            raise RuntimeError("レースIDを取得できません")

        horse_columns = get_table_columns(conn, "horses")

        # 馬データは検証済みの場合のみ置換
        conn.execute(
            "DELETE FROM horses WHERE race_id = ?",
            (race_id,),
        )

        for horse in horses:
            values = {
                "race_id": race_id,
                "finish": horse.get("finish"),
                "frame": horse.get("frame"),
                "horse_number": horse.get("horse_number"),
                "horse_name": horse.get("horse_name"),
                "sex_age": horse.get("sex_age"),
                "weight": horse.get("weight"),
                "jockey": horse.get("jockey"),
                "time": horse.get("time"),
                "margin": horse.get("margin"),
                "last_3f": horse.get("last_3f"),
                "body_weight": horse.get("body_weight"),
                "weight_change": horse.get("weight_change"),
                "trainer": horse.get("trainer"),
                "popularity": horse.get("popularity"),
                "odds": horse.get("odds"),
            }

            values = {
                key: value
                for key, value in values.items()
                if key in horse_columns
            }

            insert_row(conn, "horses", values)

        # 払戻が取得できた場合のみ既存払戻を更新する。
        # 0件の場合は既存データを消さない。
        if payouts:
            conn.execute(
                "DELETE FROM payouts WHERE race_id = ?",
                (race_id,),
            )

            payout_columns = get_table_columns(conn, "payouts")

            for payout in payouts:
                values = {
                    "race_id": race_id,
                    "bet_type": payout.get("bet_type"),
                    "combination": payout.get("combination"),
                    "amount": payout.get("amount"),
                }

                values = {
                    key: value
                    for key, value in values.items()
                    if key in payout_columns
                }

                insert_row(conn, "payouts", values)


# ============================================================
# 個別レース
# ============================================================

def crawl_one_race(
    conn: sqlite3.Connection,
    url: str,
) -> tuple[int, int, bool]:
    try:
        soup = fetch_html(url)

        race = extract_race_info(soup, url)
        horses = parse_horses(soup)
        payouts = parse_payouts(soup)

        if not horses:
            raise ValueError(
                "出走馬データが0件です。HTML構造の確認が必要です"
            )

        save_race(conn, race, horses, payouts)

        print(
            f"[OK] {race['race_date']} {race['venue']} "
            f"{race['race_name']} "
            f"| 馬 {len(horses)}件 "
            f"| 払戻 {len(payouts)}件"
        )

        return len(horses), len(payouts), True

    except Exception as exc:
        print(f"[ERROR] {url} | {type(exc).__name__}: {exc}")
        save_error(conn, url, exc)
        return 0, 0, False


# ============================================================
# メイン
# ============================================================

def main() -> int:
    conn = connect_db()

    total_ok = 0
    total_error = 0
    total_horses = 0
    total_payouts = 0

    try:
        init_db(conn)

        for year in YEARS:
            try:
                urls = discover_race_urls(year)

            except Exception as exc:
                list_url = race_index_url(year)

                print(
                    f"[ERROR] {year}年 | "
                    f"{type(exc).__name__}: {exc}"
                )

                save_error(conn, list_url, exc)
                total_error += 1
                continue

            if not urls:
                exc = RuntimeError(
                    "一覧ページから個別レースURLを取得できませんでした"
                )

                print(f"[ERROR] {year}年 | {exc}")

                save_error(
                    conn,
                    race_index_url(year),
                    exc,
                )

                total_error += 1
                continue

            for url in urls:
                horse_count, payout_count, success = crawl_one_race(
                    conn,
                    url,
                )

                if success:
                    total_ok += 1
                    total_horses += horse_count
                    total_payouts += payout_count
                else:
                    total_error += 1

                time.sleep(REQUEST_INTERVAL)

    finally:
        conn.close()
        SESSION.close()

    print(
        f"[DONE] 成功 {total_ok}レース "
        f"| 失敗 {total_error}件 "
        f"| 馬 {total_horses}件 "
        f"| 払戻 {total_payouts}件"
    )

    return 1 if total_ok == 0 or total_error > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
