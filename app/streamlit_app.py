from html import escape
import json
import os
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from sqlalchemy.exc import SQLAlchemyError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.model_features import make_prediction_feature_rows
from scripts.database import (
    connect_db as open_database,
    execute as db_execute,
    read_sql_query,
    table_columns,
)
from scripts.model_artifacts import (
    ModelReleaseUnavailableError,
    download_model_artifacts,
)


st.set_page_config(
    page_title="競馬予測ラボ",
    page_icon="🏇",
    layout="wide",
)


def configure_database_url() -> None:
    """Load the database URL from Streamlit Cloud secrets when available."""
    if os.environ.get("DATABASE_URL", "").strip():
        return

    try:
        database_url = (
            st.secrets.get("DATABASE_URL")
            or st.secrets.get("NEON_DATABASE_URL")
        )
    except StreamlitSecretNotFoundError:
        return

    if isinstance(database_url, str) and database_url.strip():
        os.environ["DATABASE_URL"] = database_url.strip()


configure_database_url()

DB_PATH = PROJECT_ROOT / "data" / "keiba.db"
MODEL_CACHE_DIR = Path(tempfile.gettempdir()) / "keiba-ai" / "models"
MODEL_PATH = MODEL_CACHE_DIR / "top3_model.joblib"
UPSET_MODEL_PATH = MODEL_CACHE_DIR / "upset_model.joblib"
UPSET_METRICS_PATH = MODEL_CACHE_DIR / "upset_metrics.json"


@st.cache_data(ttl=300)
def load_races(
    race_date="すべて",
    venue="すべて",
    race_number="すべて",
    race_name="",
    limit=5000,
):
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
        conditions = []
        params = {"limit": int(limit)}
        if race_date != "すべて":
            conditions.append("race_date = :race_date")
            params["race_date"] = race_date
        if venue != "すべて":
            conditions.append("venue = :venue")
            params["venue"] = venue
        if race_number != "すべて":
            conditions.append("race_number = :race_number")
            params["race_number"] = int(race_number)
        if race_name.strip():
            conditions.append(
                        "LOWER(COALESCE(race_name, '')) LIKE :race_name ESCAPE '\\'"
            )
            params["race_name"] = (
                "%"
                + race_name.strip().lower().replace("%", "\\%").replace("_", "\\_")
                + "%"
            )
        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        return read_sql_query(
            f"""
            SELECT
                id, url, race_date, venue, race_name, distance_m,
                surface, direction, {extra_columns}
            FROM races
            {where}
            ORDER BY race_date DESC, venue, id
            LIMIT :limit
            """,
            conn,
            params=params,
        )


@st.cache_data(ttl=300)
def load_race_filter_options():
    with get_connection() as conn:
        dates = read_sql_query(
            """
            SELECT DISTINCT race_date
            FROM races
            WHERE race_date IS NOT NULL
            ORDER BY race_date DESC
            """,
            conn,
        )["race_date"].astype(str).tolist()
        venues = read_sql_query(
            """
            SELECT DISTINCT venue
            FROM races
            WHERE venue IS NOT NULL
            ORDER BY venue
            """,
            conn,
        )["venue"].astype(str).tolist()
        numbers = read_sql_query(
            """
            SELECT DISTINCT race_number
            FROM races
            WHERE race_number IS NOT NULL
            ORDER BY race_number
            """,
            conn,
        )["race_number"].tolist()
    return dates, venues, numbers


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


@st.cache_data(ttl=30)
def load_saved_prediction(race_id):
    with get_connection() as conn:
        columns = table_columns(conn, "predictions")
        required = {"race_id", "top3_predictions"}
        missing = required - columns
        if missing:
            raise ValueError(
                f"predictions テーブルに列がありません: {sorted(missing)}"
            )
        return read_sql_query(
            """
            SELECT
                top3_predictions, chaos_probability, decision, upset_factors
            FROM predictions
            WHERE race_id = :race_id
            ORDER BY predicted_at DESC
            LIMIT 1
            """,
            conn,
            params={"race_id": int(race_id)},
        )


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


@st.cache_data(ttl=3600)
def refresh_model_artifacts():
    return download_model_artifacts(MODEL_CACHE_DIR)


@st.cache_resource(ttl=3600)
def load_prediction_model():
    if not refresh_model_artifacts():
        raise ModelReleaseUnavailableError("No published model release is available.")
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

    if not hasattr(model, "predict_proba") and not hasattr(model, "predict"):
        raise ValueError(
            f"モデルが predict() に対応していません: {type(model)}"
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


@st.cache_resource(ttl=3600)
def load_upset_prediction_model():
    if not refresh_model_artifacts():
        raise ModelReleaseUnavailableError("No published model release is available.")
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
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        if 1 not in classes:
            raise ValueError(
                f"モデルにクラス1がありません。モデルのクラス: {classes}"
            )
        scores = model.predict_proba(features)[:, classes.index(1)]
    else:
        scores = np.asarray(model.predict(features), dtype=float)
        scores = pd.Series(scores).rank(method="average", pct=True).to_numpy()

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
        "horse_recent_win_rate_5": "馬の直近5走勝率（新しい走りを重視）",
        "horse_recent_top3_rate_5": "馬の直近5走3着内率（新しい走りを重視）",
        "horse_prior_days_since_last_start": "前走からの日数",
        "horse_prior_time_seconds": "前走タイム（秒）",
        "horse_prior_last_3f": "前走の上がり3F",
        "horse_prior_last_body_weight": "前走の馬体重",
        "horse_prior_body_weight_change": "馬体重の前走比",
        "horse_surface_prior_top3_rate": "馬の同馬場3着内率",
        "horse_course_distance_prior_top3_rate": "馬の同コース・距離3着内率",
        "horse_course_distance_prior_starts": "馬の同コース・距離出走数",
        "jockey_horse_prior_top3_rate": "騎手と馬の組合せ3着内率",
        "jockey_horse_prior_starts": "騎手と馬の組合せ出走数",
        "race_name_prior_top3_rate": "レース名の過去3着内率",
        "race_name_prior_starts": "レース名の過去出走頭数",
        "race_name_prior_upset_rate": "レース名の過去波乱率",
        "race_name_prior_races": "レース名の過去開催数",
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
    dates, venues, race_numbers = load_race_filter_options()
    race_numbers = [
        int(value)
        for value in pd.to_numeric(
            pd.Series(race_numbers),
            errors="coerce",
        ).dropna().unique()
    ]
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

    filtered = load_races(
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
      padding-top: 2.75rem;
      padding-bottom: 0.4rem;
    }
    [data-testid="stHeading"] {
      overflow: visible;
    }
    h1 {
      font-size: clamp(1.5rem, 2.2vw, 2rem) !important;
      line-height: 1.3 !important;
      padding-top: 0 !important;
      margin-top: 0 !important;
      margin-bottom: 0 !important;
    }
    h2 {
      font-size: 1.3rem !important;
      line-height: 1.25 !important;
      margin-top: 0.45rem !important;
      margin-bottom: 0.1rem !important;
    }
    h3 {
      font-size: 1.05rem !important;
      line-height: 1.3 !important;
      margin-top: 0.35rem !important;
      margin-bottom: 0.1rem !important;
    }
    [data-testid="stVerticalBlock"] {
      gap: 0.3rem;
    }
    [data-testid="stMetric"] {
      padding: 0.2rem 0.35rem;
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
    .jockey-name {
      color: #111 !important;
      font-weight: 600;
      line-height: 1.25;
      margin: 0.15rem 0 0 !important;
    }
    @media (max-width: 700px) {
      [data-testid="stMainBlockContainer"] {
        padding: 3.25rem 0.55rem 0.4rem;
      }
      [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap;
        gap: 0.25rem;
      }
      [data-testid="stHorizontalBlock"]:has(
        > [data-testid="stColumn"]:nth-child(3):last-child
      ) > [data-testid="stColumn"] {
        width: calc(33.33% - 0.17rem) !important;
        flex: 1 1 calc(33.33% - 0.17rem) !important;
      }
      [data-testid="stColumn"] {
        box-sizing: border-box;
        width: calc(50% - 0.2rem) !important;
        flex: 1 1 calc(50% - 0.2rem) !important;
        min-width: 0 !important;
      }
      h1 {
        font-size: 1.4rem !important;
        line-height: 1.3 !important;
      }
      h2 {
        font-size: 1.1rem !important;
        margin-top: 0.35rem !important;
      }
      h3 {
        font-size: 0.98rem !important;
      }
      [data-testid="stVerticalBlock"] {
        gap: 0.2rem;
      }
      [data-testid="stMarkdownContainer"] p {
        font-size: 0.84rem;
        line-height: 1.2;
      }
      [data-testid="stCaptionContainer"] {
        font-size: 0.7rem;
        line-height: 1.25;
      }
      [data-testid="stMetric"] {
        padding: 0.2rem 0.3rem;
      }
      [data-testid="stMetricValue"] {
        font-size: 0.98rem;
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

if not os.environ.get("DATABASE_URL", "").strip() and not DB_PATH.exists():
    st.error(f"データベースが見つかりません: {DB_PATH}")
    st.info(
        "Streamlit Cloudではアプリ設定のSecretsにDATABASE_URLを設定してください。"
    )
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
race_labels.update(
    {
        int(row["id"]): race_label(row)
        for _, row in selection_races.iterrows()
    }
)
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
selected_race = selection_races.loc[
    selection_races["id"].astype(int) == int(selected_race_id)
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
        height=280,
        hide_index=True,
    )

if horses.empty:
    st.warning("このレースには出走馬データがありません。予測できません。")
else:
    try:
        saved_predictions = load_saved_prediction(selected_race_id)
        if saved_predictions.empty:
            st.info(
                "週次ジョブの予測結果がまだありません。"
                "出走表の更新後に週次ジョブを実行してください。"
            )
        else:
            saved = saved_predictions.iloc[0]
            ranked = sorted(
                json.loads(saved["top3_predictions"]),
                key=lambda item: item["rank"],
            )
            top3 = ranked[:3]
            top_columns = st.columns(len(top3))
            for rank, horse in enumerate(top3, start=1):
                with top_columns[rank - 1]:
                    st.metric(
                        f"AI評価{rank}番手｜{horse['horse_name'] or '不明'}",
                        f"{horse['score']:.1f} 点",
                        f"馬番 {horse['horse_number']}",
                    )
                    jockey = escape(str(display_value(horse["jockey"])))
                    st.markdown(
                        f'<p class="jockey-name">騎手：{jockey}</p>',
                        unsafe_allow_html=True,
                    )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"保存済み上位3頭予測の読み込みに失敗しました: {exc}")
    except (SQLAlchemyError, pd.errors.DatabaseError) as exc:
        st.error(f"予測結果のDB読込に失敗しました: {exc}")


# ② 波乱予想
st.header("② このレースは荒れる？")
if horses.empty:
    st.info("出走馬データがないため、波乱予測は表示できません。")
else:
    try:
        saved_predictions = load_saved_prediction(selected_race_id)
        if saved_predictions.empty:
            st.info("保存済みの波乱予測はありません。")
        else:
            saved = saved_predictions.iloc[0]
            probability = float(saved["chaos_probability"])
            result1, result2 = st.columns(2)
            result1.metric("波乱確率", f"{probability:.1%}")
            result2.metric(
                "モデル判定",
                "荒れる予想" if saved["decision"] == "upset" else "荒れにくい予想",
            )
            factors = [
                {
                    "特徴量": feature_label(item["feature"]),
                    "寄与度（log-odds）": f"{item['contribution']:+.2f}",
                }
                for item in json.loads(saved["upset_factors"] or "[]")
            ]
            if factors:
                st.markdown("**予測に影響した特徴量**")
                st.dataframe(
                    pd.DataFrame(factors),
                    width="stretch",
                    hide_index=True,
                )
            else:
                st.info(
                    "この判定を押し上げる／押し下げる特徴量は特定できませんでした。"
                )
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"波乱予測に失敗しました: {exc}")
    except Exception as exc:
        st.error(f"波乱予測中にエラーが発生しました: {type(exc).__name__}: {exc}")


# ③ レース検索・保存データ
with st.expander("③ レース検索・データベースの内容", expanded=False):
    st.write("レースを検索し、選択中のレースの出走馬・結果・払戻を確認できます。")
    filtered = race_filter_controls(races, "race_search")
    race_labels.update(
        {
            int(row["id"]): race_label(row)
            for _, row in filtered.iterrows()
        }
    )
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
        height=280,
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
        height=280,
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
    except ModelReleaseUnavailableError:
        st.info(
            "モデルの公開 Release がまだありません。"
            "Actions の「Weekly Keiba Update」で Run workflow を実行し、"
            "初回のモデル公開を行ってください。"
        )
    except requests.RequestException as exc:
        st.warning(f"モデルファイルを取得できないため特徴量を表示できません: {exc}")
    except (FileNotFoundError, ValueError, KeyError, TypeError) as exc:
        st.error(f"特徴量一覧を読み込めませんでした: {exc}")


if st.button("🔄 データを再読み込み"):
    load_races.clear()
    load_race_filter_options.clear()
    load_horses.clear()
    load_saved_prediction.clear()
    load_payouts.clear()
    refresh_model_artifacts.clear()
    load_prediction_model.clear()
    load_upset_prediction_model.clear()
    st.rerun()
