
from pathlib import Path
import sqlite3

import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ==================================================
# 基本設定
# ==================================================
st.set_page_config(
    page_title="競馬予測ラボ",
    page_icon="🏇",
    layout="wide",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DB_PATH = PROJECT_ROOT / "data" / "keiba.db"
MODEL_PATH = PROJECT_ROOT / "data" / "model" / "top3_model.joblib"

FEATURE_COLUMNS = [
    "venue",
    "surface",
    "direction",
    "race_name",
    "sex_age",
    "jockey",
    "trainer",
    "distance_m",
    "horse_number",
    "carried_weight",
    "popularity",
]


# ==================================================
# データベース
# ==================================================
def get_connection():
    """SQLite接続を作成する。"""
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"データベースが見つかりません: {DB_PATH}"
        )

    conn = sqlite3.connect(str(DB_PATH), timeout=10)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=30)
def load_races():
    """レース一覧を取得する。"""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                id,
                url,
                race_date,
                venue,
                race_name,
                distance_m,
                surface,
                direction
            FROM races
            ORDER BY race_date DESC, venue, id
            """,
            conn,
        )


@st.cache_data(ttl=30)
def load_horses(race_id):
    """指定レースの出走馬・結果を取得する。"""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                finish,
                frame,
                horse_number,
                horse_name,
                sex_age,
                weight,
                jockey,
                time,
                margin,
                last_3f,
                body_weight,
                weight_change,
                trainer,
                popularity,
                odds
            FROM horses
            WHERE race_id = ?
            ORDER BY
                CASE
                    WHEN finish IS NULL
                         OR TRIM(CAST(finish AS TEXT)) = ''
                    THEN 1
                    ELSE 0
                END,
                CAST(finish AS INTEGER),
                horse_number
            """,
            conn,
            params=(int(race_id),),
        )


@st.cache_data(ttl=30)
def load_payouts(race_id):
    """指定レースの払戻情報を取得する。"""
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT
                bet_type,
                combination,
                amount
            FROM payouts
            WHERE race_id = ?
            ORDER BY id
            """,
            conn,
            params=(int(race_id),),
        )


def get_database_counts():
    """DBの登録件数を取得する。"""
    with get_connection() as conn:
        race_count = conn.execute(
            "SELECT COUNT(*) FROM races"
        ).fetchone()[0]

        horse_count = conn.execute(
            "SELECT COUNT(*) FROM horses"
        ).fetchone()[0]

        payout_count = conn.execute(
            "SELECT COUNT(*) FROM payouts"
        ).fetchone()[0]

    return race_count, horse_count, payout_count


# ==================================================
# AIモデル
# ==================================================
@st.cache_resource
def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"学習済みモデルが見つかりません: {MODEL_PATH}"
        )

    saved = joblib.load(MODEL_PATH)

    # 学習コードが保存した辞書から、実際のモデルを取り出す
    if isinstance(saved, dict):
        if "model" not in saved:
            raise ValueError(
                "モデルファイルに 'model' キーがありません。"
            )
        model = saved["model"]
    else:
        # 旧形式でモデル本体が直接保存されている場合にも対応
        model = saved

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"モデルが predict_proba() に対応していません: {type(model)}"
        )

    return model


def prepare_prediction_features(race, horses):
    """
    学習時と同じ11特徴量を用意する。

    実際の着順 finish は特徴量に含めない。
    """
    X = pd.DataFrame(
        {
            "venue": [race["venue"]] * len(horses),
            "surface": [race["surface"]] * len(horses),
            "direction": [race["direction"]] * len(horses),
            "race_name": [race["race_name"]] * len(horses),
            "sex_age": horses["sex_age"].to_numpy(),
            "jockey": horses["jockey"].to_numpy(),
            "trainer": horses["trainer"].to_numpy(),
            "distance_m": [race["distance_m"]] * len(horses),
            "horse_number": horses["horse_number"].to_numpy(),
            "carried_weight": horses["weight"].to_numpy(),
            "popularity": horses["popularity"].to_numpy(),
        },
        columns=FEATURE_COLUMNS,
    )

    # 数値特徴量の型を統一する
    for column in [
        "distance_m",
        "horse_number",
        "carried_weight",
        "popularity",
    ]:
        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )

    # 文字列特徴量の欠損を統一する
    for column in [
        "venue",
        "surface",
        "direction",
        "race_name",
        "sex_age",
        "jockey",
        "trainer",
    ]:
        X[column] = X[column].where(
            X[column].notna(),
            None,
        )

    return X


def predict_race(race, horses, model):
    """
    選択されたレースの出走馬をAI評価する。

    出力する予測スコアは、モデルが推定した
    「3着以内クラスの確率」であり、実際の的中確率を
    保証するものではない。
    """
    if horses.empty:
        return pd.DataFrame()

    X = prepare_prediction_features(race, horses)

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            "モデルに predict_proba がありません。"
            "学習コードとモデル形式を確認してください。"
        )

    # 学習時の特徴量名がモデルに保存されていれば照合する
    model_features = getattr(
        model,
        "feature_names_in_",
        None,
    )

    if model_features is not None:
        expected = list(model_features)
        if expected != FEATURE_COLUMNS:
            raise ValueError(
                "モデルの特徴量と画面側の特徴量が一致しません。\n"
                f"モデル側: {expected}\n"
                f"画面側: {FEATURE_COLUMNS}"
            )

    probabilities = model.predict_proba(X)
    classes = list(model.classes_)

    # 学習時の正例ラベルは「3着以内 = 1」を想定
    if 1 not in classes:
        raise ValueError(
            f"モデルにクラス1がありません。モデルのクラス: {classes}"
        )

    positive_index = classes.index(1)
    scores = probabilities[:, positive_index]

    result = horses[
        [
            "finish",
            "frame",
            "horse_number",
            "horse_name",
            "sex_age",
            "weight",
            "jockey",
            "popularity",
            "odds",
        ]
    ].copy()

    result["予測スコア"] = scores * 100

    result["AI順位"] = (
        result["予測スコア"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    result = result.sort_values(
        ["予測スコア", "horse_number"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)

    result["予測スコア"] = result["予測スコア"].round(1)

    result = result.rename(
        columns={
            "finish": "実際の着順",
            "frame": "枠番",
            "horse_number": "馬番",
            "horse_name": "馬名",
            "sex_age": "性齢",
            "weight": "斤量",
            "jockey": "騎手",
            "popularity": "人気",
            "odds": "単勝オッズ",
        }
    )

    return result


# ==================================================
# 表示用関数
# ==================================================
def race_label(row):
    """レース選択欄に表示する文字列を作成する。"""
    date = str(row["race_date"] or "日付不明")
    venue = str(row["venue"] or "競馬場不明")
    name = str(row["race_name"] or "レース名不明")
    surface = str(row["surface"] or "")

    distance = row["distance_m"]
    if pd.notna(distance):
        distance_text = f"{int(distance)}m"
    else:
        distance_text = "距離不明"

    return (
        f"{date}｜{venue}｜{name}｜"
        f"{surface}{distance_text}｜ID:{row['id']}"
    )


def display_value(value, suffix=""):
    """欠損値を安全に表示する。"""
    if pd.isna(value) or str(value).strip() == "":
        return "不明"

    return f"{value}{suffix}"


# ==================================================
# メイン画面
# ==================================================
st.title("🏇 競馬予測ラボ")

st.write(
    "JRAの過去レースを検索し、出走馬のAI評価・レース結果・"
    "払戻金を確認するための分析アプリです。"
)

st.caption(
    "開発版です。AIの予測スコアはモデルの出力であり、"
    "的中や利益を保証するものではありません。"
)

if not DB_PATH.exists():
    st.error(
        f"データベースが見つかりません: {DB_PATH}"
    )
    st.info(
        "Docker Compose の data ボリューム設定を確認してください。"
    )
    st.stop()


# ==================================================
# データ件数
# ==================================================
try:
    races = load_races()
    race_count, horse_count, payout_count = get_database_counts()

except (sqlite3.Error, pd.errors.DatabaseError) as exc:
    st.error(f"データベースの読み込みに失敗しました: {exc}")
    st.stop()

if races.empty:
    st.warning(
        "レースデータがありません。先にデータを収集してください。"
    )
    st.stop()

metric1, metric2, metric3 = st.columns(3)

metric1.metric("登録レース数", f"{race_count:,}")
metric2.metric("出走馬データ数", f"{horse_count:,}")
metric3.metric("払戻データ数", f"{payout_count:,}")

st.divider()


# ==================================================
# レース検索
# ==================================================
st.subheader("🔎 レース検索")

search_col, venue_col = st.columns([2, 1])

with search_col:
    keyword = st.text_input(
        "レース名で検索",
        placeholder="例：日本ダービー、中山金杯",
    )

with venue_col:
    venues = sorted(
        races["venue"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_venue = st.selectbox(
        "競馬場",
        ["すべて"] + venues,
    )

filtered = races.copy()

if keyword.strip():
    filtered = filtered[
        filtered["race_name"]
        .fillna("")
        .astype(str)
        .str.contains(
            keyword.strip(),
            case=False,
            regex=False,
        )
    ]

if selected_venue != "すべて":
    filtered = filtered[
        filtered["venue"].astype(str) == selected_venue
    ]

st.caption(f"検索結果：{len(filtered):,} レース")

if filtered.empty:
    st.warning(
        "条件に一致するレースがありません。検索条件を変更してください。"
    )
    st.stop()

display_races = filtered[
    [
        "race_date",
        "venue",
        "race_name",
        "surface",
        "distance_m",
    ]
].copy()

display_races.columns = [
    "開催日",
    "競馬場",
    "レース名",
    "馬場",
    "距離(m)",
]

st.dataframe(
    display_races,
    use_container_width=True,
    hide_index=True,
)

st.divider()


# ==================================================
# レース選択
# ==================================================
st.subheader("📋 レース詳細・AI予測")

options = [
    (int(row["id"]), race_label(row))
    for _, row in filtered.iterrows()
]

selected_race_id = st.selectbox(
    "詳細・予測を表示するレース",
    options=options,
    format_func=lambda option: option[1],
)[0]

selected_race = races[
    races["id"].astype(int) == selected_race_id
].iloc[0]

try:
    horses = load_horses(selected_race_id)
    payouts = load_payouts(selected_race_id)

except (sqlite3.Error, pd.errors.DatabaseError) as exc:
    st.error(f"レースデータの取得に失敗しました: {exc}")
    st.stop()


# ==================================================
# レース基本情報
# ==================================================
st.markdown(
    f"### {selected_race['race_name'] or 'レース詳細'}"
)

info1, info2, info3, info4 = st.columns(4)

info1.metric(
    "開催日",
    display_value(selected_race["race_date"]),
)

info2.metric(
    "競馬場",
    display_value(selected_race["venue"]),
)

info3.metric(
    "馬場",
    display_value(selected_race["surface"]),
)

distance = selected_race["distance_m"]

info4.metric(
    "距離",
    f"{int(distance)} m" if pd.notna(distance) else "不明",
)

if pd.notna(selected_race["direction"]):
    st.caption(
        f"コース方向：{selected_race['direction']}"
    )


# ==================================================
# AI予測ランキング
# ==================================================
st.divider()
st.subheader("🤖 AI予測ランキング")

st.write(
    "学習済みモデルが各馬の特徴量を評価し、"
    "3着以内クラスの予測スコアが高い順に表示します。"
)

st.warning(
    "過去レースの検証では、入力した人気が実際のレース前に"
    "利用可能だった情報か確認してください。"
    "このモデルは現時点では検証用であり、"
    "未来のレースに対する性能は保証されていません。"
)

if not MODEL_PATH.exists():
    st.error(
        f"学習済みモデルが見つかりません: {MODEL_PATH}"
    )
    st.info(
        "data/model/top3_model.joblib が存在するか確認してください。"
    )

elif horses.empty:
    st.warning(
        "このレースには出走馬データがありません。予測できません。"
    )

else:
    try:
        model = load_prediction_model()

        prediction = predict_race(
            selected_race,
            horses,
            model,
        )

        if prediction.empty:
            st.warning("予測結果がありません。")

        else:
            # 上位3頭
            st.markdown("#### ⭐ AI評価上位3頭")

            top3 = prediction.head(3)
            top_cols = st.columns(3)

            for index, (_, row) in enumerate(top3.iterrows()):
                with top_cols[index]:
                    st.metric(
                        label=f"AI {index + 1}位｜{row['馬名']}",
                        value=f"{row['予測スコア']:.1f} 点",
                        delta=f"馬番 {row['馬番']}",
                    )

                    st.caption(
                        f"騎手：{display_value(row['騎手'])}"
                    )

            st.markdown("#### 全出走馬ランキング")

            prediction_columns = [
                "AI順位",
                "枠番",
                "馬番",
                "馬名",
                "性齢",
                "斤量",
                "騎手",
                "予測スコア",
                "人気",
                "単勝オッズ",
                "実際の着順",
            ]

            prediction_columns = [
                column
                for column in prediction_columns
                if column in prediction.columns
            ]

            st.dataframe(
                prediction[prediction_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.caption(
                "予測スコアはモデルが推定した3着以内クラスのスコアです。"
                "校正済みの的中確率や馬券の期待回収率ではありません。"
            )

            # ------------------------------------------
            # 過去結果との比較
            # ------------------------------------------
            actual_finish = pd.to_numeric(
                prediction["実際の着順"],
                errors="coerce",
            )

            valid = prediction.loc[
                actual_finish.notna()
            ].copy()

            valid["実際の着順_数値"] = pd.to_numeric(
                valid["実際の着順"],
                errors="coerce",
            )

            if not valid.empty:
                valid["実際の3着以内"] = (
                    valid["実際の着順_数値"] <= 3
                )

                # 上位3頭を実際の3着以内と照合
                predicted_top3 = set(
                    prediction.head(3)["馬番"]
                    .dropna()
                    .tolist()
                )

                actual_top3 = set(
                    valid.loc[
                        valid["実際の3着以内"],
                        "馬番",
                    ].dropna().tolist()
                )

                matched = len(predicted_top3 & actual_top3)

                st.markdown("#### 📊 実際の着順との比較")

                compare1, compare2 = st.columns(2)

                compare1.metric(
                    "AI上位3頭のうち実際に3着以内",
                    f"{matched} / 3 頭",
                )

                compare2.metric(
                    "上位3頭の一致率",
                    f"{matched / 3 * 100:.1f}%",
                )

                st.caption(
                    "選択した1レースのみの比較です。"
                    "モデル全体の性能を評価するには、"
                    "複数のレースを用いた時系列バックテストが必要です。"
                )

    except FileNotFoundError as exc:
        st.error(str(exc))

    except (ValueError, KeyError, TypeError) as exc:
        st.error(f"AI予測の入力またはモデルに問題があります: {exc}")

    except Exception as exc:
        st.error(
            f"AI予測中にエラーが発生しました: "
            f"{type(exc).__name__}: {exc}"
        )


# ==================================================
# 出走馬・実際のレース結果
# ==================================================
st.divider()
st.subheader("🐎 出走馬・レース結果")

if horses.empty:
    st.info("このレースの出走馬データはありません。")

else:
    horse_display = horses.copy()

    horse_display = horse_display.rename(
        columns={
            "finish": "着順",
            "frame": "枠番",
            "horse_number": "馬番",
            "horse_name": "馬名",
            "sex_age": "性齢",
            "weight": "斤量",
            "jockey": "騎手",
            "time": "走破タイム",
            "margin": "着差",
            "last_3f": "上がり3F",
            "body_weight": "馬体重",
            "weight_change": "馬体重増減",
            "trainer": "調教師",
            "popularity": "人気",
            "odds": "単勝オッズ",
        }
    )

    horse_columns = [
        "着順",
        "枠番",
        "馬番",
        "馬名",
        "性齢",
        "斤量",
        "騎手",
        "走破タイム",
        "着差",
        "上がり3F",
        "馬体重",
        "馬体重増減",
        "調教師",
        "人気",
        "単勝オッズ",
    ]

    horse_columns = [
        column
        for column in horse_columns
        if column in horse_display.columns
    ]

    st.caption(f"出走馬データ：{len(horses)} 頭")

    st.dataframe(
        horse_display[horse_columns],
        use_container_width=True,
        hide_index=True,
    )


# ==================================================
# 払戻金
# ==================================================
st.divider()
st.subheader("💴 払戻金")

if payouts.empty:
    st.info(
        "このレースの払戻データはありません。"
    )

else:
    payout_display = payouts.rename(
        columns={
            "bet_type": "券種",
            "combination": "組み合わせ",
            "amount": "払戻金（円）",
        }
    ).copy()

    payout_display["払戻金（円）"] = pd.to_numeric(
        payout_display["払戻金（円）"],
        errors="coerce",
    )

    st.dataframe(
        payout_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "払戻金の単位・表記は収集元データに依存します。"
        "実際の払戻表記と照合してください。"
    )


# ==================================================
# データ再読み込み
# ==================================================
st.divider()

with st.expander("データ表示について"):
    st.write(
        "この画面はSQLiteに保存されたデータと学習済みモデルを使っています。"
        "出走馬・払戻金が表示されない場合は、"
        "そのレースのデータが未収集の可能性があります。"
    )

    st.write(
        "AI予測は開発中です。過去レースでの比較結果だけで"
        "将来の予測性能や馬券の利益を判断しないでください。"
    )

if st.button("🔄 データを再読み込み"):
    load_races.clear()
    load_horses.clear()
    load_payouts.clear()
    load_prediction_model.clear()
    st.rerun()
