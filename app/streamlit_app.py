from pathlib import Path
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_features import make_prediction_feature_rows
from scripts.train_upset_model import add_race_features
from scripts.database import (
    connect_db as open_database,
    execute as db_execute,
    read_sql_query,
    table_columns,
)


st.set_page_config(
    page_title="競馬予測ラボ",
    page_icon="🏇",
    layout="wide",
)

DB_PATH = PROJECT_ROOT / "data" / "keiba.db"
MODEL_PATH = PROJECT_ROOT / "data" / "model" / "top3_model.joblib"
UPSET_MODEL_PATH = (
    PROJECT_ROOT / "data" / "model" / "upset" / "upset_model.joblib"
)
UPSET_METRICS_PATH = (
    PROJECT_ROOT / "data" / "model" / "upset" / "metrics.json"
)


@st.cache_data(ttl=30)
def load_races():
    """レース一覧を取得する。"""
    with get_connection() as conn:
        columns = table_columns(conn, "races")
        optional = (
            "race_number",
            "start_time",
            "weather",
            "turf_condition",
            "dirt_condition",
            "race_grade",
            "race_conditions",
        )
        extra_columns = ", ".join(
            f'"{column}"'
            if column in columns
            else f'NULL AS "{column}"'
            for column in optional
        )
        return read_sql_query(
            f"""
            SELECT
                id, url, race_date, venue, race_name, distance_m,
                surface, direction, {extra_columns}
            FROM races
            ORDER BY race_date DESC, venue, id
            """,
            conn,
        )


def get_connection():
    """PostgreSQLまたはローカルSQLiteへの接続を作成する。"""
    if not os.environ.get("DATABASE_URL", "").strip() and not DB_PATH.exists():
        raise FileNotFoundError(f"データベースが見つかりません: {DB_PATH}")
    return open_database()


@st.cache_data(ttl=30)
def load_horses(race_id):
    """指定レースの出走馬・結果を取得する。"""
    with get_connection() as conn:
        columns = table_columns(conn, "horses")
        optional = ("horse_mark", "source_fields")
        extra_columns = ", ".join(
            f'"{column}"'
            if column in columns
            else f'NULL AS "{column}"'
            for column in optional
        )
        return read_sql_query(
            f"""
            SELECT
                finish, frame, horse_number, horse_name, sex_age, weight,
                jockey, time, margin, last_3f, body_weight, weight_change,
                trainer, popularity, odds, {extra_columns}
            FROM horses
            WHERE race_id = :race_id
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
            params={"race_id": int(race_id)},
        )


@st.cache_data(ttl=300)
def load_historical_horses():
    """日付をまたいだ過去成績の特徴量作成用データを取得する。"""
    with get_connection() as conn:
        return read_sql_query(
            """
            SELECT
                h.race_id, r.race_date, h.finish, h.horse_name,
                h.jockey, h.trainer
            FROM horses h
            JOIN races r ON r.id = h.race_id
            WHERE r.race_date IS NOT NULL
            """,
            conn,
        )


@st.cache_data(ttl=300)
def load_upset_feature_rows():
    """波乱モデルの学習時と同じ方法で、全レースの特徴量を作る。"""
    with get_connection() as conn:
        races = read_sql_query("SELECT * FROM races", conn)
        horses = read_sql_query("SELECT * FROM horses", conn)

    if races.empty or horses.empty:
        return pd.DataFrame()

    races["race_date"] = pd.to_datetime(races["race_date"], errors="coerce")
    races["id"] = races["id"].astype(str)
    horses["race_id"] = horses["race_id"].astype(str)

    race_frame = races.copy()
    race_frame["race_id"] = race_frame["id"]
    field_sizes = horses.groupby("race_id").size()
    race_frame["field_size"] = (
        race_frame["race_id"].map(field_sizes).fillna(0).astype(int)
    )
    features, _, _ = add_race_features(race_frame, races, horses)
    return features


@st.cache_data(ttl=30)
def load_payouts(race_id):
    """指定レースの払戻情報を取得する。"""
    with get_connection() as conn:
        return read_sql_query(
            """
            SELECT bet_type, combination, amount
            FROM payouts
            WHERE race_id = :race_id
            ORDER BY id
            """,
            conn,
            params={"race_id": int(race_id)},
        )


def get_database_counts():
    """DBの登録件数を取得する。"""
    with get_connection() as conn:
        race_count = db_execute(
            conn,
            "SELECT COUNT(*) FROM races",
        ).scalar_one()
        horse_count = db_execute(
            conn,
            "SELECT COUNT(*) FROM horses",
        ).scalar_one()
        payout_count = db_execute(
            conn,
            "SELECT COUNT(*) FROM payouts",
        ).scalar_one()
    return race_count, horse_count, payout_count


@st.cache_resource
def load_prediction_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"学習済みモデルが見つかりません: {MODEL_PATH}")
    saved = joblib.load(MODEL_PATH)

    if isinstance(saved, dict):
        if "model" not in saved:
            raise ValueError("モデルファイルに 'model' キーがありません。")
        model = saved["model"]
        feature_columns = saved.get("feature_columns")
    else:
        model = saved
        feature_columns = None

    if not hasattr(model, "predict_proba"):
        raise ValueError(
            f"モデルが predict_proba() に対応していません: {type(model)}"
        )
    if feature_columns is None:
        model_features = getattr(model, "feature_names_in_", None)
        feature_columns = (
            list(model_features)
            if model_features is not None
            else [
                "venue", "surface", "direction", "race_name", "sex_age",
                "jockey", "trainer", "distance_m", "horse_number",
                "carried_weight", "popularity",
            ]
        )
    return model, list(feature_columns)


@st.cache_resource
def load_upset_prediction_model():
    if not UPSET_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"波乱予測モデルが見つかりません: {UPSET_MODEL_PATH}"
        )
    if not UPSET_METRICS_PATH.exists():
        raise FileNotFoundError(
            f"波乱モデルの評価情報が見つかりません: {UPSET_METRICS_PATH}"
        )
    model = joblib.load(UPSET_MODEL_PATH)
    metrics = json.loads(UPSET_METRICS_PATH.read_text(encoding="utf-8"))
    threshold = float(
        metrics["classification_threshold_selected_on_validation"]
    )
    if not hasattr(model, "predict_proba"):
        raise ValueError("波乱予測モデルが predict_proba() に対応していません。")
    return model, threshold


def prepare_prediction_features(
    race,
    horses,
    historical_horses,
    feature_columns,
):
    """学習時の出走前特徴量を構築し、モデルが必要とする列だけ返す。"""
    features = make_prediction_feature_rows(race, horses, historical_horses)
    for column in feature_columns:
        if column not in features.columns:
            features[column] = np.nan
    return features.reindex(columns=feature_columns)


def predict_race(
    race,
    horses,
    model,
    feature_columns,
    historical_horses,
):
    """出走馬ごとの3着以内スコアを算出し、レース内で順位付けする。"""
    if horses.empty:
        return pd.DataFrame()

    features = prepare_prediction_features(
        race, horses, historical_horses, feature_columns
    )
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError(f"モデルにクラス1がありません。モデルのクラス: {classes}")
    scores = model.predict_proba(features)[:, classes.index(1)]

    result = horses[
        [
            "frame", "horse_number", "horse_name", "sex_age", "weight",
            "jockey", "popularity", "odds",
        ]
    ].copy()
    result["予測スコア"] = scores * 100
    result = result.sort_values(
        ["予測スコア", "horse_number"],
        ascending=[False, True],
        na_position="last",
    ).reset_index(drop=True)
    result.insert(0, "AI順位", np.arange(1, len(result) + 1))
    result["予測スコア"] = result["予測スコア"].round(1)
    return result.rename(
        columns={
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


def predict_upset_race(race_id, feature_rows, model, threshold):
    """波乱確率と、予測を押し上げ／押し下げた特徴量を返す。"""
    matches = feature_rows.loc[
        feature_rows["race_id"].astype(str) == str(race_id)
    ]
    if matches.empty:
        raise ValueError("選択レースの波乱予測特徴量が見つかりません。")

    columns = list(model.feature_names_in_)
    row = matches.iloc[[0]].reindex(columns=columns)
    probability = float(model.predict_proba(row)[0, 1])
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    transformed = preprocessor.transform(row)
    transformed = (
        transformed.toarray().ravel()
        if hasattr(transformed, "toarray")
        else np.asarray(transformed).ravel()
    )
    contributions = transformed * classifier.coef_[0]
    names = preprocessor.get_feature_names_out()

    is_upset = probability >= threshold
    ranked = sorted(
        zip(names, contributions),
        key=lambda item: item[1],
        reverse=is_upset,
    )
    if is_upset:
        ranked = [item for item in ranked if item[1] > 0][:3]
    else:
        ranked = [item for item in ranked if item[1] < 0][:3]

    factors = [
        {
            "特徴量": feature_label(name),
            "寄与度（log-odds）": f"{contribution:+.2f}",
        }
        for name, contribution in ranked
    ]
    return probability, is_upset, factors


def feature_label(feature_name):
    """特徴量名を簡潔な日本語にする。"""
    name = str(feature_name)
    for prefix in ("numeric__", "categorical__"):
        if name.startswith(prefix):
            name = name[len(prefix):]
            break

    category_features = (
        "venue", "surface", "direction", "race_name", "distance_band",
        "weather", "turf_condition", "dirt_condition", "race_grade",
        "race_conditions",
    )
    for feature in category_features:
        if name.startswith(f"{feature}_"):
            return f"{feature_label(feature)} = {name[len(feature) + 1:]}"

    labels = {
        "venue": "競馬場",
        "surface": "馬場",
        "direction": "コース方向",
        "race_name": "レース名",
        "distance_band": "距離区分",
        "distance_m": "距離",
        "race_number": "レース番号",
        "start_minutes": "発走時刻",
        "field_size": "出走頭数",
        "frame": "枠番",
        "horse_number": "馬番",
        "weight": "斤量",
        "body_weight": "馬体重",
        "weight_change": "馬体重増減",
        "age": "年齢",
        "horse_prior_starts": "馬の過去出走数",
        "horse_prior_win_rate": "馬の過去勝率",
        "horse_prior_top3_rate": "馬の過去3着内率",
        "jockey_prior_starts": "騎手の過去出走数",
        "jockey_prior_win_rate": "騎手の過去勝率",
        "jockey_prior_top3_rate": "騎手の過去3着内率",
        "trainer_prior_starts": "調教師の過去出走数",
        "trainer_prior_win_rate": "調教師の過去勝率",
        "trainer_prior_top3_rate": "調教師の過去3着内率",
        "weather": "天候",
        "turf_condition": "芝状態",
        "dirt_condition": "ダート状態",
        "race_grade": "レース格",
        "race_conditions": "レース条件",
    }

    stat_labels = {
        "mean": "平均",
        "std": "ばらつき",
        "min": "最小",
        "max": "最大",
    }
    parts = name.split("_")
    stat = parts.pop() if parts[-1] in stat_labels else None
    name = "_".join(parts)
    if name.startswith("field_"):
        name = name[len("field_"):]
        parts = name.split("_")
        if parts[-1] in stat_labels:
            stat = parts.pop()
            name = "_".join(parts)
        label = f"出走馬の{labels.get(name, name.replace('_', ' '))}"
    elif name.startswith("horse_mark_"):
        label = f"出走馬の馬印「{name[len('horse_mark_'):]}」"
    else:
        label = labels.get(name, name.replace("_", " "))
    if stat:
        label = f"{label}（{stat_labels[stat]}）"
    return label


def race_label(row):
    """レース選択欄に表示する文字列を作成する。"""
    date = str(row["race_date"] or "日付不明")
    venue = str(row["venue"] or "競馬場不明")
    name = str(row["race_name"] or "レース名不明")
    race_number = row["race_number"]
    race_number_text = (
        f"{int(race_number)}R" if pd.notna(race_number) else ""
    )
    surface = str(row["surface"] or "")
    distance = row["distance_m"]
    distance_text = f"{int(distance)}m" if pd.notna(distance) else "距離不明"
    return (
        f"{date}｜{venue}｜"
        f"{race_number_text + '｜' if race_number_text else ''}"
        f"{name}｜{surface}{distance_text}"
    )


def filter_races(
    races,
    *,
    race_date="すべて",
    venue="すべて",
    race_number="すべて",
    race_name="",
):
    """開催日・競馬場・レース番号・レース名で一覧を絞り込む。"""
    filtered = races
    if race_date != "すべて":
        filtered = filtered[filtered["race_date"].astype(str) == race_date]
    if venue != "すべて":
        filtered = filtered[filtered["venue"].astype(str) == venue]
    if race_number != "すべて":
        numbers = pd.to_numeric(filtered["race_number"], errors="coerce")
        filtered = filtered[numbers == race_number]
    if race_name.strip():
        filtered = filtered[
            filtered["race_name"]
            .fillna("")
            .astype(str)
            .str.contains(race_name.strip(), case=False, regex=False)
        ]
    return filtered


def race_filter_controls(races, key_prefix):
    """レース一覧用の開催日・競馬場・レース番号・レース名を表示する。"""
    dates = sorted(
        races["race_date"].dropna().astype(str).unique().tolist(),
        reverse=True,
    )
    venues = sorted(
        races["venue"].dropna().astype(str).unique().tolist()
    )
    race_numbers = sorted(
        pd.to_numeric(races["race_number"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
        .tolist()
    )
    date_col, venue_col, number_col, name_col = st.columns([1, 1, 1, 2])
    with date_col:
        selected_date = st.selectbox(
            "開催日",
            ["すべて"] + dates,
            key=f"{key_prefix}_date",
        )
    with venue_col:
        selected_venue = st.selectbox(
            "競馬場",
            ["すべて"] + venues,
            key=f"{key_prefix}_venue",
        )
    with number_col:
        selected_race_number = st.selectbox(
            "レース番号",
            ["すべて"] + race_numbers,
            format_func=lambda number: (
                f"{number}R" if number != "すべて" else number
            ),
            key=f"{key_prefix}_number",
        )
    with name_col:
        race_name = st.text_input(
            "レース名",
            placeholder="レース名で検索",
            key=f"{key_prefix}_keyword",
        )

    filtered = filter_races(
        races,
        race_date=selected_date,
        venue=selected_venue,
        race_number=selected_race_number,
        race_name=race_name,
    )
    st.caption(f"検索結果：{len(filtered):,} レース")
    return filtered


def display_value(value, suffix=""):
    """欠損値を安全に表示する。"""
    if pd.isna(value) or str(value).strip() == "":
        return "不明"
    return f"{value}{suffix}"


def select_race_from_search():
    st.session_state["selected_race_id"] = st.session_state["search_race_id"]
    st.session_state["race_select_date"] = "すべて"
    st.session_state["race_select_venue"] = "すべて"
    st.session_state["race_select_keyword"] = ""


st.markdown(
    """
    <style>
    [data-testid="stMainBlockContainer"] {
      max-width: 1440px;
      padding-top: 1.25rem;
      padding-bottom: 1.5rem;
    }
    h1 {
      font-size: clamp(1.5rem, 2.2vw, 2rem) !important;
      line-height: 1.2 !important;
      margin-bottom: 0.25rem !important;
    }
    h2 {
      font-size: 1.3rem !important;
      line-height: 1.25 !important;
      margin-top: 1rem !important;
      margin-bottom: 0.35rem !important;
    }
    h3 {
      font-size: 1.05rem !important;
      line-height: 1.3 !important;
    }
    [data-testid="stVerticalBlock"] {
      gap: 0.75rem;
    }
    [data-testid="stMetric"] {
      padding: 0.45rem 0.6rem;
    }
    [data-testid="stMetricLabel"] {
      font-size: 0.82rem;
    }
    [data-testid="stMetricValue"] {
      font-size: 1.25rem;
    }
    [data-testid="stExpander"] summary p {
      font-size: 0.95rem;
    }
    @media (max-width: 700px) {
      [data-testid="stMainBlockContainer"] {
        padding: 0.65rem 0.7rem 1rem;
      }
      [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        gap: 0.4rem;
      }
      [data-testid="stColumn"] {
        box-sizing: border-box;
        width: calc(50% - 0.2rem) !important;
        flex: 1 1 calc(50% - 0.2rem) !important;
        min-width: 0 !important;
      }
      h1 {
        font-size: 1.35rem !important;
      }
      h2 {
        font-size: 1.1rem !important;
        margin-top: 0.75rem !important;
      }
      h3 {
        font-size: 0.98rem !important;
      }
      [data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem;
        line-height: 1.45;
      }
      [data-testid="stCaptionContainer"] {
        font-size: 0.75rem;
        line-height: 1.4;
      }
      [data-testid="stMetric"] {
        padding: 0.35rem 0.45rem;
      }
      [data-testid="stMetricValue"] {
        font-size: 1.05rem;
      }
      [data-testid="stSelectbox"] label,
      [data-testid="stTextInput"] label {
        font-size: 0.82rem;
      }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🏇 競馬予測ラボ")
st.caption(
    "データベースに登録されたレースを選び、着順と波乱の予測を確認できます。"
    "予測は参考情報であり、的中や利益を保証するものではありません。"
)

if not os.environ.get("DATABASE_URL", "").strip() and not DB_PATH.exists():
    st.error(f"データベースが見つかりません: {DB_PATH}")
    st.info("Docker Compose の data ボリューム設定を確認してください。")
    st.stop()

try:
    races = load_races()
except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
    st.error(f"データベースの読み込みに失敗しました: {exc}")
    st.stop()

if races.empty:
    st.warning("レースデータがありません。先にデータを収集してください。")
    st.stop()


# ① レース選択・上位3頭予想
st.header("① レースを選んで上位3頭を予想")
race_labels = {
    int(row["id"]): race_label(row)
    for _, row in races.iterrows()
}
selection_races = race_filter_controls(races, "race_select")
if selection_races.empty:
    st.info(
        "条件に一致するレースがありません。"
        "対象レースは全件から選択できます。"
    )
    selection_races = races
race_ids = selection_races["id"].astype(int).tolist()
if (
    "selected_race_id" not in st.session_state
    or st.session_state["selected_race_id"] not in race_ids
):
    st.session_state["selected_race_id"] = race_ids[0]

selected_race_id = st.selectbox(
    "対象レース",
    options=race_ids,
    format_func=lambda race_id: race_labels[race_id],
    key="selected_race_id",
)
selected_race = races.loc[
    races["id"].astype(int) == int(selected_race_id)
].iloc[0]

basic1, basic2, basic3 = st.columns(3)
basic1.metric("開催日", display_value(selected_race["race_date"]))
basic2.metric("競馬場", display_value(selected_race["venue"]))
basic3.metric(
    "コース",
    f"{display_value(selected_race['surface'])} "
    f"{display_value(selected_race['distance_m'], 'm')}",
)

try:
    horses = load_horses(selected_race_id)
except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
    st.error(f"出走馬データの取得に失敗しました: {exc}")
    st.stop()

st.subheader("出走馬・出走情報")
if horses.empty:
    st.info("このレースの出走馬データはありません。")
else:
    entry_display = horses.sort_values(
        "horse_number",
        key=lambda values: pd.to_numeric(values, errors="coerce"),
        na_position="last",
    ).rename(
        columns={
            "horse_number": "馬番",
            "horse_name": "馬名",
            "sex_age": "性齢",
            "weight": "斤量",
            "jockey": "騎手",
            "trainer": "調教師",
            "body_weight": "馬体重",
            "weight_change": "馬体重増減",
            "popularity": "人気",
            "odds": "単勝オッズ",
        }
    )
    entry_columns = [
        "馬番",
        "馬名",
        "性齢",
        "斤量",
        "騎手",
        "調教師",
        "馬体重",
        "馬体重増減",
        "人気",
        "単勝オッズ",
    ]
    st.dataframe(
        entry_display[entry_columns],
        width="stretch",
        hide_index=True,
    )

if not MODEL_PATH.exists():
    st.error(f"上位3頭予測モデルが見つかりません: {MODEL_PATH}")
elif horses.empty:
    st.warning("このレースには出走馬データがありません。予測できません。")
else:
    try:
        model, feature_columns = load_prediction_model()
        prediction = predict_race(
            selected_race,
            horses,
            model,
            feature_columns,
            load_historical_horses(),
        )
        if prediction.empty:
            st.info("予測対象の出走馬がありません。")
        else:
            st.caption(
                "3着以内クラスの予測スコア順です。"
                "1着・2着・3着を個別に予測するモデルではありません。"
            )
            top3 = prediction.head(3)
            top_columns = st.columns(len(top3))
            for rank, (_, horse) in enumerate(top3.iterrows(), start=1):
                with top_columns[rank - 1]:
                    st.metric(
                        f"AI評価{rank}番手｜{horse['馬名']}",
                        f"{horse['予測スコア']:.1f} 点",
                        f"馬番 {horse['馬番']}",
                    )
                    st.caption(f"騎手：{display_value(horse['騎手'])}")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"上位3頭の予測に失敗しました: {exc}")
    except Exception as exc:
        st.error(f"上位3頭の予測中にエラーが発生しました: {type(exc).__name__}: {exc}")


# ② 波乱予想
st.header("② このレースは荒れる？")
st.caption(
    "波乱の定義：7番人気以下の馬が3着以内に入るレース。"
    "現在のレースの人気・オッズは、ラベルとの情報漏洩を避けるため予測に使いません。"
)
if not UPSET_MODEL_PATH.exists():
    st.error(f"波乱予測モデルが見つかりません: {UPSET_MODEL_PATH}")
elif horses.empty:
    st.info("出走馬データがないため、波乱予測は表示できません。")
else:
    try:
        upset_model, upset_threshold = load_upset_prediction_model()
        upset_features = load_upset_feature_rows()
        probability, is_upset, factors = predict_upset_race(
            selected_race_id,
            upset_features,
            upset_model,
            upset_threshold,
        )
        result1, result2 = st.columns(2)
        result1.metric("波乱確率", f"{probability:.1%}")
        result2.metric(
            "モデル判定",
            "荒れる予想" if is_upset else "荒れにくい予想",
        )
        st.caption(
            f"判定基準：{upset_threshold:.1%}。"
            "確率はモデルの推定値で、校正済みの的中確率ではありません。"
        )
        if factors:
            st.markdown("**予測に影響した特徴量**")
            st.dataframe(
                pd.DataFrame(factors),
                width="stretch",
                hide_index=True,
            )
            st.caption(
                "寄与度はモデル内部の log-odds への影響であり、"
                "因果関係や確実な理由を示すものではありません。"
            )
        else:
            st.info("この判定を押し上げる／押し下げる特徴量は特定できませんでした。")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"波乱予測に失敗しました: {exc}")
    except Exception as exc:
        st.error(f"波乱予測中にエラーが発生しました: {type(exc).__name__}: {exc}")


# ③ レース検索・保存データ
with st.expander("③ レース検索・データベースの内容", expanded=False):
    st.write("レースを検索し、選択中のレースの出走馬・結果・払戻を確認できます。")
    filtered = race_filter_controls(races, "race_search")
    display_races = filtered[
        ["race_date", "venue", "race_name", "surface", "distance_m"]
    ].rename(
        columns={
            "race_date": "開催日",
            "venue": "競馬場",
            "race_name": "レース名",
            "surface": "馬場",
            "distance_m": "距離(m)",
        }
    )
    st.dataframe(
        display_races.head(100),
        width="stretch",
        hide_index=True,
    )
    if len(display_races) > 100:
        st.caption("一覧は先頭100件まで表示しています。検索条件を追加して絞り込めます。")

    if not filtered.empty:
        search_ids = [int(value) for value in filtered["id"]]
        search_labels = {
            race_id: race_labels[race_id]
            for race_id in search_ids
        }
        if int(selected_race_id) not in search_ids:
            search_ids.insert(0, int(selected_race_id))
            search_labels[int(selected_race_id)] = (
                f"{race_labels[int(selected_race_id)]}（現在の選択）"
            )
            st.caption(
                "現在の予測対象が検索条件外のため、"
                "選択肢に追加しています。"
            )
        if (
            "search_race_id" not in st.session_state
            or st.session_state["search_race_id"] not in search_ids
            or st.session_state["search_race_id"] != int(selected_race_id)
        ):
            st.session_state["search_race_id"] = (
                int(selected_race_id)
                if int(selected_race_id) in search_ids
                else search_ids[0]
            )
        st.selectbox(
            "検索結果から予測対象を変更",
            options=search_ids,
            index=search_ids.index(int(selected_race_id)),
            format_func=lambda race_id: search_labels[race_id],
            key="search_race_id",
            on_change=select_race_from_search,
        )

    try:
        race_count, horse_count, payout_count = get_database_counts()
        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("登録レース数", f"{race_count:,}")
        metric2.metric("出走馬データ数", f"{horse_count:,}")
        metric3.metric("払戻データ数", f"{payout_count:,}")
    except SQLAlchemyError as exc:
        st.error(f"データ件数の取得に失敗しました: {exc}")

    st.subheader(f"選択中：{race_labels[int(selected_race_id)]}")
    horse_display = horses.rename(
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
        "着順", "枠番", "馬番", "馬名", "性齢", "斤量", "騎手",
        "走破タイム", "着差", "上がり3F", "馬体重", "馬体重増減",
        "調教師", "人気", "単勝オッズ",
    ]
    horse_columns = [
        column for column in horse_columns if column in horse_display.columns
    ]
    st.dataframe(
        horse_display[horse_columns],
        width="stretch",
        hide_index=True,
    )

    try:
        payouts = load_payouts(selected_race_id)
        if payouts.empty:
            st.info("このレースの払戻データはありません。")
        else:
            st.markdown("**払戻金**")
            st.dataframe(
                payouts.rename(
                    columns={
                        "bet_type": "券種",
                        "combination": "組み合わせ",
                        "amount": "払戻金（円）",
                    }
                ),
                width="stretch",
                hide_index=True,
            )
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        st.error(f"払戻データの取得に失敗しました: {exc}")


# ④ モデル特徴量一覧
with st.expander("④ 学習に使った特徴量一覧", expanded=False):
    try:
        _, top3_features = load_prediction_model()
        upset_model, _ = load_upset_prediction_model()
        catalog = pd.DataFrame(
            [
                {"モデル": model_name, "特徴量": feature_label(feature)}
                for model_name, features in (
                    ("上位3頭予想", top3_features),
                    ("波乱予想", list(upset_model.feature_names_in_)),
                )
                for feature in features
            ]
        )
        st.dataframe(catalog, width="stretch", hide_index=True)
        st.caption(
            "波乱予想は現在レースの人気・オッズを使用しません。"
            "両モデルの過去成績特徴量は、同日結果を含めず以前の開催日のみから計算します。"
        )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"特徴量一覧を読み込めませんでした: {exc}")


if st.button("🔄 データを再読み込み"):
    load_races.clear()
    load_horses.clear()
    load_historical_horses.clear()
    load_upset_feature_rows.clear()
    load_payouts.clear()
    load_prediction_model.clear()
    load_upset_prediction_model.clear()
    st.rerun()
