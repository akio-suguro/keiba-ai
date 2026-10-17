from __future__ import annotations

import hashlib
import re
import sqlite3
import time
import unicodedata
from datetime import datetime
from pathlib import Path
from urllib.parse import urldefrag, urljoin, urlparse

import requests
from bs4 import BeautifulSoup, Tag


# ============================================================
# 設定
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = DATA_DIR / "html_cache"
DB_PATH = DATA_DIR / "keiba.db"

YEARS = [2025, 2026]

BASE_URL = "https://www.jra.go.jp"
INDEX_URL = (
    "https://www.jra.go.jp/datafile/seiseki/replay/{year}/jyusyo.html"
)

REQUEST_TIMEOUT = 30
REQUEST_INTERVAL_SECONDS = 3
MAX_RETRIES = 3

DATA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

session = requests.Session()
session.headers.update(
    {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    }
)


# ============================================================
# 共通関数
# ============================================================

def normalize_text(value) -> str:
    if value is None:
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text(value) -> str | None:
    text = normalize_text(value)
    return text if text else None


def parse_int(value) -> int | None:
    text = normalize_text(value)
    match = re.search(r"-?\d+", text)

    if not match:
        return None

    try:
        return int(match.group())
    except ValueError:
        return None


def parse_float(value) -> float | None:
    text = normalize_text(value)
    match = re.search(r"\d+(?:\.\d+)?", text)

    if not match:
        return None

    try:
        return float(match.group())
    except ValueError:
        return None


def cache_path_for(url: str) -> Path:
    digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.html"


def fetch_html(url: str, use_cache: bool = True) -> str:
    cache_path = cache_path_for(url)

    if use_cache and cache_path.exists():
        return cache_path.read_text(encoding="utf-8")

    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = session.get(url, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            response.encoding = response.apparent_encoding or "utf-8"

            html = response.text
            cache_path.write_text(html, encoding="utf-8")

            return html

        except requests.RequestException as exc:
            last_error = exc
            print(
                f"[WARN] 取得失敗 {attempt}/{MAX_RETRIES}: "
                f"{url} ({exc})"
            )

            if attempt < MAX_RETRIES:
                time.sleep(attempt * 2)

    raise RuntimeError(f"HTML取得失敗: {url}") from last_error


def soup_from_html(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "html.parser")


def page_text(soup: BeautifulSoup) -> str:
    return normalize_text(soup.get_text(" ", strip=True))


# ============================================================
# URLの検証
# ============================================================

def is_individual_race_url(url: str, year: int) -> bool:
    """
    重賞一覧ページから、個別レース結果ページと思われるURLだけを許可する。

    除外例:
      /datafile/seiseki/
      /datafile/seiseki/replay/2025/jyusyo.html
      /datafile/seiseki/replay/2025/jyusyo.html#contents
      外部サイト
      別年度のURL
    """
    url, _fragment = urldefrag(url)
    parsed = urlparse(url)

    if parsed.scheme not in ("http", "https"):
        return False

    if parsed.netloc.lower() not in (
        "www.jra.go.jp",
        "jra.go.jp",
    ):
        return False

    expected_prefix = f"/datafile/seiseki/replay/{year}/"

    if not parsed.path.startswith(expected_prefix):
        return False

    filename = Path(parsed.path).name.lower()

    # 一覧ページを除外
    if filename in {
        "",
        "jyusyo.html",
        "index.html",
        "index.htm",
    }:
        return False

    # HTML以外は対象外
    if not filename.endswith((".html", ".htm")):
        return False

    # パスが年度直下のHTMLであることを要求する
    # 余分なサブディレクトリは許可しない
    relative_path = parsed.path[len(expected_prefix):]

    if "/" in relative_path:
        return False

    return True


def discover_race_urls(year: int) -> list[str]:
    """
    年別重賞一覧から個別レースURLを収集する。
    一覧ページ自身や、JRAサイト内の別ページは登録しない。
    """
    index_url = INDEX_URL.format(year=year)
    print(f"[INFO] 一覧取得: {index_url}")

    html = fetch_html(index_url)
    soup = soup_from_html(html)

    urls: list[str] = []
    seen: set[str] = set()

    for anchor in soup.select("a[href]"):
        href = anchor.get("href", "").strip()

        if not href:
            continue

        absolute_url = urljoin(index_url, href)

        # #contentsなどのフラグメントを除去
        absolute_url, _ = urldefrag(absolute_url)

        if not is_individual_race_url(absolute_url, year):
            continue

        if absolute_url not in seen:
            seen.add(absolute_url)
            urls.append(absolute_url)

    print(f"[INFO] {year}年: 個別候補 {len(urls)}件")

    for url in urls[:5]:
        print(f"       {url}")

    if not urls:
        print(
            f"[WARN] {year}年の個別レースURLを検出できませんでした。"
            "一覧ページのリンク形式を確認してください。"
        )

    return urls


# ============================================================
# レース基本情報
# ============================================================

def extract_date(soup: BeautifulSoup) -> str | None:
    """
    例: 2025年1月5日（日曜） -> 2025-01-05
    """
    candidates = []

    if soup.title:
        candidates.append(soup.title.get_text(" ", strip=True))

    for selector in (
        "h1",
        "h2",
        ".race_title",
        "#race_result",
        "body",
    ):
        element = soup.select_one(selector)
        if element:
            candidates.append(element.get_text(" ", strip=True))

    for candidate in candidates:
        text = normalize_text(candidate)

        match = re.search(
            r"(20\d{2})年\s*(\d{1,2})月\s*(\d{1,2})日",
            text,
        )

        if not match:
            continue

        try:
            date = datetime(
                int(match.group(1)),
                int(match.group(2)),
                int(match.group(3)),
            )
            return date.strftime("%Y-%m-%d")
        except ValueError:
            continue

    return None


def extract_venue(soup: BeautifulSoup) -> str | None:
    venues = (
        "札幌", "函館", "福島", "新潟", "東京",
        "中山", "中京", "京都", "阪神", "小倉",
    )

    venue_pattern = "|".join(venues)
    pattern = re.compile(
        rf"\d+\s*回\s*({venue_pattern})\s*\d+\s*日"
    )

    # 開催情報が含まれそうな箇所を優先
    for selector in (
        "#race_result",
        ".race_title",
        "h1",
        "h2",
        "body",
    ):
        element = soup.select_one(selector)

        if not element:
            continue

        match = pattern.search(
            normalize_text(element.get_text(" ", strip=True))
        )

        if match:
            return match.group(1)

    return None


def extract_race_name(soup: BeautifulSoup) -> str | None:
    selectors = (
        "#race_result .race_title .race_name",
        "#race_result .race_title h2",
        "#race_result h2",
        ".race_title",
        "h1",
        "h2",
    )

    for selector in selectors:
        for element in soup.select(selector):
            text = clean_text(element.get_text(" ", strip=True))

            if not text:
                continue

            text = re.sub(r"20\d{2}年\s*", "", text)
            text = re.sub(r"\s+", " ", text).strip()

            if text in {
                "レース結果",
                "競走成績",
                "レース一覧",
            }:
                continue

            if len(text) >= 3:
                return text

    # 最後の候補としてページタイトルを使用
    if soup.title:
        title = clean_text(soup.title.get_text(" ", strip=True))

        if title:
            title = re.sub(r"20\d{2}年\s*", "", title)
            return title

    return None


def extract_course(soup: BeautifulSoup) -> dict:
    text = normalize_text(page_text(soup))

    distance_m = None
    surface = None
    direction = None

    distance_match = re.search(
        r"コース\s*[:：]?\s*([\d,]+)\s*メートル",
        text,
    )

    if not distance_match:
        distance_match = re.search(
            r"([\d,]+)\s*メートル\s*[（(]\s*(芝|ダート)",
            text,
        )

    if distance_match:
        try:
            distance_m = int(
                distance_match.group(1).replace(",", "")
            )
        except ValueError:
            pass

    if "ダート" in text:
        surface = "ダート"
    elif "芝" in text:
        surface = "芝"

    direction_match = re.search(
        r"[（(][^）)]{0,30}(右|左)[^）)]{0,30}[）)]",
        text,
    )

    if direction_match:
        direction = direction_match.group(1)

    return {
        "distance_m": distance_m,
        "surface": surface,
        "direction": direction,
    }


# ============================================================
# 出走馬テーブル
# ============================================================

def normalize_header(value: str) -> str:
    return re.sub(r"\s+", "", normalize_text(value))


def find_horse_table(soup: BeautifulSoup) -> Tag | None:
    for table in soup.find_all("table"):
        header_rows = table.find_all("tr")

        for row in header_rows[:3]:
            headers = [
                normalize_header(cell.get_text(" ", strip=True))
                for cell in row.find_all(["th", "td"], recursive=False)
            ]

            header_text = "|".join(headers)

            if (
                "着順" in header_text
                and "馬番" in header_text
                and "馬名" in header_text
            ):
                return table

    return None


def header_indices(headers: list[str]) -> dict[str, int]:
    aliases = {
        "finish": ("着順",),
        "frame": ("枠", "枠番"),
        "horse_number": ("馬番",),
        "horse_name": ("馬名",),
        "sex_age": ("性齢",),
        "weight": ("負担重量", "斤量"),
        "jockey": ("騎手名", "騎手"),
        "time": ("タイム",),
        "margin": ("着差",),
        "last_3f": ("推定上り", "上り"),
        "body_weight": ("馬体重",),
        "trainer": ("調教師名", "調教師"),
        "popularity": ("単勝人気", "人気"),
        "odds": ("単勝オッズ", "オッズ"),
    }

    result = {}

    for key, options in aliases.items():
        for index, header in enumerate(headers):
            if any(option in header for option in options):
                result[key] = index
                break

    return result


def parse_horses(soup: BeautifulSoup) -> list[dict]:
    table = find_horse_table(soup)

    if table is None:
        return []

    rows = table.find_all("tr")

    header_row_index = None
    headers = []

    for row_index, row in enumerate(rows[:3]):
        current_headers = [
            normalize_header(cell.get_text(" ", strip=True))
            for cell in row.find_all(["th", "td"], recursive=False)
        ]

        joined = "|".join(current_headers)

        if (
            "着順" in joined
            and "馬番" in joined
            and "馬名" in joined
        ):
            header_row_index = row_index
            headers = current_headers
            break

    if header_row_index is None:
        return []

    indices = header_indices(headers)
    horses = []

    for row in rows[header_row_index + 1:]:
        cells = row.find_all(["td", "th"], recursive=False)

        if not cells:
            continue

        values = [
            clean_text(cell.get_text(" ", strip=True))
            for cell in cells
        ]

        def get_value(key: str) -> str | None:
            index = indices.get(key)

            if index is None or index >= len(values):
                return None

            return values[index]

        horse_name = get_value("horse_name")
        horse_number = parse_int(get_value("horse_number"))

        if not horse_name or horse_number is None:
            continue

        # 馬体重: 452(+2)
        body_weight = None
        weight_change = None
        body_text = normalize_text(get_value("body_weight"))

        if body_text:
            body_match = re.search(
                r"(\d{2,3})(?:\s*\(\s*([+-]?\d+)\s*\))?",
                body_text,
            )

            if body_match:
                body_weight = int(body_match.group(1))

                if body_match.group(2) is not None:
                    weight_change = int(body_match.group(2))

        # 単勝人気とオッズは別項目
        # オッズ列が実際に存在するときだけ値を取得する
        odds = parse_float(get_value("odds"))

        horses.append(
            {
                "finish": parse_int(get_value("finish")),
                "frame": parse_int(get_value("frame")),
                "horse_number": horse_number,
                "horse_name": horse_name,
                "sex_age": get_value("sex_age"),
                "weight": parse_float(get_value("weight")),
                "jockey": get_value("jockey"),
                "time": get_value("time"),
                "margin": get_value("margin"),
                "last_3f": get_value("last_3f"),
                "body_weight": body_weight,
                "weight_change": weight_change,
                "trainer": get_value("trainer"),
                "popularity": parse_int(get_value("popularity")),
                "odds": odds,
            }
        )

    return horses


# ============================================================
# 払戻金
# ============================================================

def parse_payouts(soup: BeautifulSoup) -> list[dict]:
    payouts = []
    seen = set()

    # 払戻金を含むエリアを探す
    areas = []

    for selector in (
        "#race_result .refund_area",
        "#race_result .payback",
        ".refund_area",
        ".payback",
    ):
        areas.extend(soup.select(selector))

    # 「払戻金」見出しがある場合は、その近くも探索
    if not areas:
        for element in soup.find_all(
            ["h2", "h3", "h4", "caption", "dt", "p", "div"]
        ):
            title = clean_text(element.get_text(" ", strip=True))

            if title == "払戻金":
                parent = element.parent

                if isinstance(parent, Tag):
                    areas.append(parent)

                break

    for area in areas:
        # dl形式
        for dl in area.select("dl"):
            dt = dl.find("dt")
            dd = dl.find("dd")

            if not dt or not dd:
                continue

            bet_type = clean_text(dt.get_text(" ", strip=True))
            detail = normalize_text(dd.get_text(" ", strip=True))

            if not bet_type:
                continue

            amount_match = re.search(
                r"([\d,]+)\s*円",
                detail,
            )

            if not amount_match:
                continue

            amount = parse_int(
                amount_match.group(1).replace(",", "")
            )

            combo_match = re.search(
                r"\d+(?:\s*[-=]\s*\d+)+",
                detail,
            )

            combination = None

            if combo_match:
                combination = re.sub(
                    r"\s+",
                    "",
                    combo_match.group(),
                )

            key = (bet_type, combination, amount)

            if key not in seen:
                seen.add(key)
                payouts.append(
                    {
                        "bet_type": bet_type,
                        "combination": combination,
                        "amount": amount,
                    }
                )

        # テーブル形式
        for table in area.select("table"):
            for row in table.select("tr"):
                cells = [
                    clean_text(cell.get_text(" ", strip=True))
                    for cell in row.find_all(["th", "td"])
                ]

                cells = [value for value in cells if value]

                if len(cells) < 2:
                    continue

                bet_type = cells[0]
                detail = normalize_text(" ".join(cells[1:]))

                amount_match = re.search(
                    r"([\d,]+)\s*円",
                    detail,
                )

                if not amount_match:
                    continue

                amount = parse_int(
                    amount_match.group(1).replace(",", "")
                )

                combo_match = re.search(
                    r"\d+(?:\s*[-=]\s*\d+)+",
                    detail,
                )

                combination = None

                if combo_match:
                    combination = re.sub(
                        r"\s+",
                        "",
                        combo_match.group(),
                    )

                key = (bet_type, combination, amount)

                if key not in seen:
                    seen.add(key)
                    payouts.append(
                        {
                            "bet_type": bet_type,
                            "combination": combination,
                            "amount": amount,
                        }
                    )

    return payouts


# ============================================================
# データベース
# ============================================================

def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS races (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
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
                last_3f TEXT,
                body_weight INTEGER,
                weight_change INTEGER,
                trainer TEXT,
                popularity INTEGER,
                odds REAL,
                FOREIGN KEY (race_id) REFERENCES races(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS payouts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id INTEGER NOT NULL,
                bet_type TEXT,
                combination TEXT,
                amount INTEGER,
                FOREIGN KEY (race_id) REFERENCES races(id)
                    ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS crawl_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT,
                error TEXT,
                occurred_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_races_date
                ON races(race_date);

            CREATE INDEX IF NOT EXISTS idx_races_venue
                ON races(venue);

            CREATE INDEX IF NOT EXISTS idx_horses_race_id
                ON horses(race_id);

            CREATE INDEX IF NOT EXISTS idx_payouts_race_id
                ON payouts(race_id);
            """
        )


def record_error(url: str, error: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO crawl_errors (url, error, occurred_at)
            VALUES (?, ?, ?)
            """,
            (
                url,
                error,
                datetime.now().isoformat(timespec="seconds"),
            ),
        )


def save_race(data: dict) -> int:
    now = datetime.now().isoformat(timespec="seconds")

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO races (
                url, race_date, venue, race_name,
                distance_m, surface, direction, crawled_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(url) DO UPDATE SET
                race_date = excluded.race_date,
                venue = excluded.venue,
                race_name = excluded.race_name,
                distance_m = excluded.distance_m,
                surface = excluded.surface,
                direction = excluded.direction,
                crawled_at = excluded.crawled_at
            """,
            (
                data["url"],
                data["race_date"],
                data["venue"],
                data["race_name"],
                data["distance_m"],
                data["surface"],
                data["direction"],
                now,
            ),
        )

        row = conn.execute(
            "SELECT id FROM races WHERE url = ?",
            (data["url"],),
        ).fetchone()

        race_id = int(row["id"])

        conn.execute(
            "DELETE FROM horses WHERE race_id = ?",
            (race_id,),
        )
        conn.execute(
            "DELETE FROM payouts WHERE race_id = ?",
            (race_id,),
        )

        for horse in data["horses"]:
            conn.execute(
                """
                INSERT INTO horses (
                    race_id, finish, frame, horse_number, horse_name,
                    sex_age, weight, jockey, time, margin, last_3f,
                    body_weight, weight_change, trainer, popularity, odds
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    race_id,
                    horse["finish"],
                    horse["frame"],
                    horse["horse_number"],
                    horse["horse_name"],
                    horse["sex_age"],
                    horse["weight"],
                    horse["jockey"],
                    horse["time"],
                    horse["margin"],
                    horse["last_3f"],
                    horse["body_weight"],
                    horse["weight_change"],
                    horse["trainer"],
                    horse["popularity"],
                    horse["odds"],
                ),
            )

        for payout in data["payouts"]:
            conn.execute(
                """
                INSERT INTO payouts (
                    race_id, bet_type, combination, amount
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    race_id,
                    payout["bet_type"],
                    payout["combination"],
                    payout["amount"],
                ),
            )

        return race_id


# ============================================================
# 個別レースの解析・保存
# ============================================================

def parse_race_page(url: str, html: str) -> dict:
    soup = soup_from_html(html)
    course = extract_course(soup)

    return {
        "url": url,
        "race_date": extract_date(soup),
        "venue": extract_venue(soup),
        "race_name": extract_race_name(soup),
        "distance_m": course["distance_m"],
        "surface": course["surface"],
        "direction": course["direction"],
        "horses": parse_horses(soup),
        "payouts": parse_payouts(soup),
    }


def crawl_one_race(url: str, year: int) -> bool:
    # URL検証に失敗したものは、エラー登録せずにスキップ
    if not is_individual_race_url(url, year):
        print(f"[SKIP] 個別レースURLではありません: {url}")
        return False

    try:
        html = fetch_html(url)
        data = parse_race_page(url, html)

        if not data["race_date"]:
            raise ValueError(
                "レース日付を取得できません。"
                "個別レースページではない可能性があります"
            )

        if not data["horses"]:
            raise ValueError(
                "出走馬テーブルを取得できません。"
                "ページのHTML構造を確認してください"
            )

        race_id = save_race(data)

        print(
            f"[OK] id={race_id} "
            f"date={data['race_date']} "
            f"venue={data['venue']} "
            f"name={data['race_name']} "
            f"horses={len(data['horses'])} "
            f"payouts={len(data['payouts'])}"
        )

        return True

    except Exception as exc:
        message = f"{type(exc).__name__}: {exc}"

        print(f"[ERROR] {url}: {message}")
        record_error(url, message)

        return False


# ============================================================
# 集計表示
# ============================================================

def print_summary() -> None:
    with get_connection() as conn:
        races_count = conn.execute(
            "SELECT COUNT(*) FROM races"
        ).fetchone()[0]

        horses_count = conn.execute(
            "SELECT COUNT(*) FROM horses"
        ).fetchone()[0]

        payouts_count = conn.execute(
            "SELECT COUNT(*) FROM payouts"
        ).fetchone()[0]

        errors_count = conn.execute(
            "SELECT COUNT(*) FROM crawl_errors"
        ).fetchone()[0]

        print()
        print("=" * 55)
        print("クロール結果")
        print("=" * 55)
        print(f"レース数:       {races_count}")
        print(f"出走馬レコード: {horses_count}")
        print(f"払戻金レコード: {payouts_count}")
        print(f"エラーレコード: {errors_count}")

        print()
        print("開催場ごとのレース数:")

        for row in conn.execute(
            """
            SELECT COALESCE(venue, '(未取得)') AS venue,
                   COUNT(*) AS count
            FROM races
            GROUP BY venue
            ORDER BY count DESC
            """
        ):
            print(f"  {row['venue']}: {row['count']}")

        print()
        print("直近のエラー:")

        rows = conn.execute(
            """
            SELECT url, error, occurred_at
            FROM crawl_errors
            ORDER BY id DESC
            LIMIT 10
            """
        ).fetchall()

        if not rows:
            print("  なし")
        else:
            for row in rows:
                print(
                    f"  {row['occurred_at']} "
                    f"{row['url']} - {row['error']}"
                )

        print("=" * 55)


# ============================================================
# メイン処理
# ============================================================

def main() -> None:
    print(f"[INFO] DB: {DB_PATH}")
    print(f"[INFO] 対象年: {YEARS}")

    init_db()

    urls_by_year = {}

    for year in YEARS:
        try:
            urls_by_year[year] = discover_race_urls(year)
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            print(f"[ERROR] {year}年の一覧取得失敗: {message}")
            record_error(INDEX_URL.format(year=year), message)
            urls_by_year[year] = []

    total = sum(len(urls) for urls in urls_by_year.values())

    print(f"[INFO] 個別レースURL合計: {total}件")

    if total == 0:
        print(
            "[STOP] 個別レースURLが0件です。"
            "JRA一覧ページのリンク形式を確認してください。"
        )
        print_summary()
        return

    success_count = 0
    failure_count = 0
    processed = 0

    for year, urls in urls_by_year.items():
        for url in urls:
            processed += 1

            print(f"\n[{processed}/{total}] {url}")

            if crawl_one_race(url, year):
                success_count += 1
            else:
                failure_count += 1

            if processed < total:
                time.sleep(REQUEST_INTERVAL_SECONDS)

    print()
    print(f"[INFO] 今回の成功件数: {success_count}")
    print(f"[INFO] 今回の失敗件数: {failure_count}")

    print_summary()


if __name__ == "__main__":
    main()
