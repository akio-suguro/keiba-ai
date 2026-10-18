from __future__ import annotations

import numpy as np
import pandas as pd


HISTORICAL_FEATURE_COLUMNS = [
    f"{entity}_prior_{metric}"
    for entity in ("horse", "jockey", "trainer")
    for metric in ("starts", "win_rate", "top3_rate")
]


def add_historical_performance_features(
    horses: pd.DataFrame,
) -> pd.DataFrame:
    """Add prior-date performance only; same-day outcomes are never visible."""
    required = {"race_date", "finish"}
    missing = required - set(horses.columns)
    if missing:
        raise ValueError(
            "Historical features require columns: "
            f"{sorted(missing)}"
        )

    result = horses.copy()
    result["race_date"] = pd.to_datetime(
        result["race_date"],
        errors="coerce",
    ).dt.normalize()
    result["finish"] = pd.to_numeric(
        result["finish"],
        errors="coerce",
    )

    entity_columns = {
        "horse": "horse_name",
        "jockey": "jockey",
        "trainer": "trainer",
    }
    row_order = "__model_feature_row_order"
    result[row_order] = np.arange(len(result))

    for entity, column in entity_columns.items():
        feature_names = {
            "starts": f"{entity}_prior_starts",
            "win_rate": f"{entity}_prior_win_rate",
            "top3_rate": f"{entity}_prior_top3_rate",
        }
        if column not in result.columns:
            for feature in feature_names.values():
                result[feature] = 0 if feature.endswith("_starts") else np.nan
            continue

        history = result.loc[
            result[column].notna()
            & result["race_date"].notna()
            & result["finish"].notna()
            & result["finish"].ge(1),
            [column, "race_date", "finish"],
        ].copy()
        history["__win"] = history["finish"].eq(1).astype(int)
        history["__top3"] = history["finish"].le(3).astype(int)

        daily_counts = (
            history.groupby(
                [column, "race_date"],
                as_index=False,
                sort=True,
                observed=True,
            )
            .agg(
                __starts=("finish", "size"),
                __wins=("__win", "sum"),
                __top3s=("__top3", "sum"),
            )
        )
        entity_dates = result.loc[
            result[column].notna() & result["race_date"].notna(),
            [column, "race_date"],
        ].drop_duplicates()
        daily = entity_dates.merge(
            daily_counts,
            on=[column, "race_date"],
            how="left",
            validate="one_to_one",
        )
        daily[["__starts", "__wins", "__top3s"]] = daily[
            ["__starts", "__wins", "__top3s"]
        ].fillna(0)
        daily = daily.sort_values(
            [column, "race_date"],
            kind="stable",
        )

        if daily.empty:
            for feature in feature_names.values():
                result[feature] = 0 if feature.endswith("_starts") else np.nan
            continue

        grouped = daily.groupby(column, sort=False, observed=True)
        daily["__starts_before"] = (
            grouped["__starts"].cumsum() - daily["__starts"]
        )
        daily["__wins_before"] = (
            grouped["__wins"].cumsum() - daily["__wins"]
        )
        daily["__top3s_before"] = (
            grouped["__top3s"].cumsum() - daily["__top3s"]
        )
        daily[feature_names["starts"]] = daily["__starts_before"]
        daily[feature_names["win_rate"]] = (
            daily["__wins_before"]
            .div(daily["__starts_before"].replace(0, np.nan))
        )
        daily[feature_names["top3_rate"]] = (
            daily["__top3s_before"]
            .div(daily["__starts_before"].replace(0, np.nan))
        )

        lookup_columns = [
            column,
            "race_date",
            *feature_names.values(),
        ]
        result = result.merge(
            daily[lookup_columns],
            on=[column, "race_date"],
            how="left",
            validate="many_to_one",
            sort=False,
        )
        result = result.sort_values(row_order, kind="stable")
        result[feature_names["starts"]] = (
            result[feature_names["starts"]].fillna(0)
        )

    return result.drop(columns=[row_order]).reset_index(drop=True)


def add_sex_age_features(horses: pd.DataFrame) -> pd.DataFrame:
    """Parse the JRA combined sex/age value without requiring a fixed format."""
    result = horses.copy()
    if "sex_age" not in result.columns:
        result["sex_age"] = np.nan

    sex_age = result["sex_age"].fillna("").astype(str).str.strip()
    result["sex"] = sex_age.str.extract(
        r"^(セン|騸|牡|牝|セ)",
        expand=False,
    )
    result["age"] = pd.to_numeric(
        sex_age.str.extract(r"(\d+)", expand=False),
        errors="coerce",
    )
    return result


def make_prediction_feature_rows(
    race: dict,
    horses: pd.DataFrame,
    historical_horses: pd.DataFrame,
) -> pd.DataFrame:
    """Build the pre-race feature shape used by the top-three model."""
    if horses.empty:
        return pd.DataFrame()

    history = historical_horses.copy()
    history = history.drop(
        columns=HISTORICAL_FEATURE_COLUMNS,
        errors="ignore",
    )
    for column in (
        "race_id",
        "race_date",
        "finish",
        "horse_name",
        "jockey",
        "trainer",
    ):
        if column not in history.columns:
            history[column] = np.nan

    current = horses.copy()
    current = current.drop(
        columns=HISTORICAL_FEATURE_COLUMNS,
        errors="ignore",
    )
    current["race_id"] = race.get("id", race.get("race_id"))
    current["race_date"] = race.get("race_date")
    current["finish"] = np.nan
    current["__prediction_row"] = True
    history["__prediction_row"] = False
    combined = pd.concat(
        [history, current],
        ignore_index=True,
        sort=False,
    )
    combined = add_historical_performance_features(combined)
    result = combined.loc[combined["__prediction_row"]].copy()
    result = add_sex_age_features(result)

    for column in (
        "venue",
        "surface",
        "direction",
        "race_name",
        "race_number",
        "start_time",
        "weather",
        "turf_condition",
        "dirt_condition",
        "race_grade",
        "race_conditions",
    ):
        result[column] = race.get(column)
    result["distance_m"] = race.get("distance_m")
    result["weight"] = result.get("weight", pd.Series(np.nan, index=result.index))

    start_parts = result["start_time"].astype("string").str.extract(
        r"(\d{1,2})[:：](\d{2})"
    )
    result["start_minutes"] = (
        pd.to_numeric(start_parts[0], errors="coerce") * 60
        + pd.to_numeric(start_parts[1], errors="coerce")
    )

    for column in (
        "distance_m",
        "race_number",
        "start_minutes",
        "frame",
        "horse_number",
        "age",
        "weight",
        "body_weight",
        "weight_change",
        "popularity",
        "odds",
        *HISTORICAL_FEATURE_COLUMNS,
    ):
        if column in result.columns:
            result[column] = pd.to_numeric(
                result[column],
                errors="coerce",
            )

    for column in (
        "venue",
        "surface",
        "direction",
        "race_name",
        "weather",
        "turf_condition",
        "dirt_condition",
        "race_grade",
        "race_conditions",
        "sex",
        "jockey",
        "trainer",
        "horse_mark",
    ):
        if column in result.columns:
            result[column] = (
                result[column]
                .fillna("不明")
                .astype(str)
                .replace("", "不明")
            )

    return result.reset_index(drop=True)
