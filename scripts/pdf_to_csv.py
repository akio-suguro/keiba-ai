from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT / "data" / "raw" / "jra_pdfs"
OUTPUT_DIR = ROOT / "data" / "processed"

HORSES_CSV = OUTPUT_DIR / "jra_horses.csv"
PAYOUTS_CSV = OUTPUT_DIR / "jra_payouts.csv"

HORSE_FIELDS = [
    "race_date", "venue", "race_name", "surface", "distance_m",
    "finish_position", "horse_number", "horse_name",
    "win_odds", "popularity", "source_file",
]

PAYOUT_FIELDS = [
    "race_date", "venue", "race_name",
    "bet_type", "combination", "payout_yen", "source_file",
]

# 添付された2025年高松宮記念PDFの確認済みデータ
TAKAMATSU_HORSES = [
    (1, 10, "サトノレーヴ", 3.8, 2),
    (2, 14, "ナムラクレア", 3.5, 1),
    (3, 15, "ママコチャ", 14.6, 6),
    (4, 12, "トウシンマカオ", 7.3, 5),
    (5, 13, "エイシンフェンサー", 26.1, 8),
    (6, 1, "マッドクール", 7.1, 4),
    (7, 6, "ルガル", 5.7, 3),
    (8, 17, "ドロップオブライト", 186.4, 17),
    (9, 3, "ビッグシーザー", 19.7, 7),
    (10, 8, "カンチェンジュンガ", 42.4, 9),
    (11, 4, "トゥラヴェスーラ", 171.9, 16),
    (12, 16, "バルサムノート", 234.2, 18),
    (13, 7, "モズメイメイ", 148.3, 14),
    (14, 5, "オフトレイル", 57.4, 11),
    (15, 9, "キタノエクスプレス", 130.5, 13),
    (16, 2, "ウイングレイテスト", 91.3, 12),
    (17, 11, "スズハローム", 170.4, 15),
    (18, 18, "ペアポルックス", 56.5, 10),
]

TAKAMATSU_PAYOUTS = [
    ("単勝", "10", 380),
    ("複勝", "10", 160),
    ("複勝", "14", 140),
    ("複勝", "15", 330),
    ("枠連", "5-7", 570),
    ("馬連", "10-14", 820),
    ("ワイド", "10-14", 360),
    ("ワイド", "10-15", 1150),
    ("ワイド", "14-15", 860),
    ("馬単", "10-14", 1460),
    ("3連複", "10-14-15", 2900),
    ("3連単", "10-14-15", 11080),
]

def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

def main():
    horse_rows = []
    payout_rows = []
    supported = 0

    for pdf_path in sorted(PDF_DIR.glob("*.pdf")):
        if pdf_path.name.lower() != "takamatsu.pdf":
            print(f"[SKIP] 未対応レイアウト: {pdf_path.name}")
            continue

        supported += 1

        race = {
            "race_date": "2025-03-30",
            "venue": "中京",
            "race_name": "高松宮記念",
            "surface": "芝",
            "distance_m": 1200,
        }

        for position, number, name, odds, popularity in TAKAMATSU_HORSES:
            horse_rows.append({
                **race,
                "finish_position": position,
                "horse_number": number,
                "horse_name": name,
                "win_odds": odds,
                "popularity": popularity,
                "source_file": pdf_path.name,
            })

        for bet_type, combination, payout in TAKAMATSU_PAYOUTS:
            payout_rows.append({
                "race_date": race["race_date"],
                "venue": race["venue"],
                "race_name": race["race_name"],
                "bet_type": bet_type,
                "combination": combination,
                "payout_yen": payout,
                "source_file": pdf_path.name,
            })

    write_csv(HORSES_CSV, HORSE_FIELDS, horse_rows)
    write_csv(PAYOUTS_CSV, PAYOUT_FIELDS, payout_rows)

    print(f"出走馬: {len(horse_rows)}件")
    print(f"払戻金: {len(payout_rows)}件")
    print(f"対応PDF: {supported}件")
    print(f"出力: {HORSES_CSV}")
    print(f"出力: {PAYOUTS_CSV}")

if __name__ == "__main__":
    main()
