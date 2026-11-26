from __future__ import annotations

import numpy as np
import pandas as pd


HISTORICAL_FEATURE_COLUMNS = [
    f"{entity}_prior_{metric}"
    for entity in ("horse", "jockey", "trainer")
    for metric in ("starts", "win_rate", "top3_rate")
] + [
    f"{entity}_recent_{metric}_5"
    for entity in ("horse", "jockey", "trainer")
    for metric in ("win_rate", "top3_rate")
] + [
    "horse_prior_days_since_last_start",
    "horse_prior_time_seconds",
    "horse_prior_last_3f",
    "horse_prior_last_body_weight",
    "horse_prior_body_weight_change",
    "horse_surface_prior_top3_rate",
    "horse_course_distance_prior_top3_rate",
    "horse_course_distance_prior_starts",
    "jockey_horse_prior_top3_rate",
    "jockey_horse_prior_starts",
    "race_name_prior_top3_rate",
    "race_name_prior_starts",
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

        history_columns = [column, "race_date", "finish"]
        if entity == "horse":
            history_columns.extend(
                name
                for name in ("time", "last_3f", "body_weight")
                if name in result.columns
            )
        history = result.loc[
            result[column].notna()
            & result["race_date"].notna()
            & result["finish"].notna()
            & result["finish"].ge(1),
            history_columns,
        ].copy()
        history["__win"] = history["finish"].eq(1).astype(int)
        history["__top3"] = history["finish"].le(3).astype(int)

        aggregations = {
            "__starts": ("finish", "size"),
            "__wins": ("__win", "sum"),
            "__top3s": ("__top3", "sum"),
        }
        if entity == "horse":
            for name in ("last_3f", "body_weight"):
                if name in history.columns:
                    history[name] = pd.to_numeric(
                        history[name],
                        errors="coerce",
                    )
                    aggregations[f"__last_{name}"] = (name, "last")
            if "time" in history.columns:
                time_parts = (
                    history["time"]
                    .astype("string")
                    .str.extract(r"(?:(\d+):)?(\d{1,2}(?:\.\d+)?)$")
                )
                history["__time_seconds"] = (
                    pd.to_numeric(time_parts[0], errors="coerce").fillna(0) * 60
                    + pd.to_numeric(time_parts[1], errors="coerce")
                )
                aggregations["__last_time_seconds"] = (
                    "__time_seconds",
                    "last",
                )
        daily_counts = history.groupby(
            [column, "race_date"],
            as_index=False,
            sort=True,
            observed=True,
        ).agg(**aggregations)
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
        grouped = daily.groupby(column, sort=False, observed=True)
        for source, feature in (
            ("__wins", f"{entity}_recent_win_rate_5"),
            ("__top3s", f"{entity}_recent_top3_rate_5"),
        ):
            rate_column = f"{source}_rate"
            daily[rate_column] = daily[source].div(
                daily["__starts"].replace(0, np.nan)
            )
            weighted_sum = pd.Series(0.0, index=daily.index)
            weight_total = pd.Series(0.0, index=daily.index)
            for lag in range(1, 6):
                weight = 6 - lag
                previous_rate = grouped[rate_column].shift(lag)
                weighted_sum += previous_rate.fillna(0) * weight
                weight_total += previous_rate.notna().astype(float) * weight
            daily[feature] = weighted_sum.div(
                weight_total.replace(0, np.nan)
            )

        lookup_columns = [
            column,
            "race_date",
            *feature_names.values(),
            f"{entity}_recent_win_rate_5",
            f"{entity}_recent_top3_rate_5",
        ]
        if entity == "horse":
            daily["horse_prior_days_since_last_start"] = (
                grouped["race_date"].diff().dt.days
            )
            for name in ("last_3f", "body_weight"):
                source = f"__last_{name}"
                if source in daily.columns:
                    feature_name = (
                        "horse_prior_last_body_weight"
                        if name == "body_weight"
                        else "horse_prior_last_3f"
                    )
                    daily[feature_name] = grouped[source].shift()
                    lookup_columns.append(feature_name)
            if "__last_time_seconds" in daily.columns:
                daily["horse_prior_time_seconds"] = grouped[
                    "__last_time_seconds"
                ].shift()
                lookup_columns.append("horse_prior_time_seconds")
            lookup_columns.append("horse_prior_days_since_last_start")
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

    if "body_weight" in result.columns and "horse_prior_last_body_weight" in result:
        current_weight = pd.to_numeric(result["body_weight"], errors="coerce")
        result["horse_prior_body_weight_change"] = (
            current_weight - result["horse_prior_last_body_weight"]
        )

    return add_contextual_history_features(
        result.drop(columns=[row_order]).reset_index(drop=True)
    )


def _add_prior_group_rates(
    frame: pd.DataFrame,
    keys: list[str],
    prefix: str,
) -> pd.DataFrame:
    """Join cumulative results strictly from dates before each race."""
    if not set(keys).issubset(frame.columns):
        return frame

    result = frame.copy()
    row_order = "__contextual_feature_row_order"
    result[row_order] = np.arange(len(result))
    eligible = result.loc[
        result[keys].notna().all(axis=1)
        & result["race_date"].notna()
        & result["finish"].notna()
        & result["finish"].ge(1),
        [*keys, "race_date", "finish"],
    ].copy()
    if eligible.empty:
        result[f"{prefix}_prior_starts"] = 0
        result[f"{prefix}_prior_top3_rate"] = np.nan
        return result.drop(columns=row_order)

    eligible["__wins"] = eligible["finish"].eq(1).astype(int)
    eligible["__top3"] = eligible["finish"].le(3).astype(int)
    group_keys = [*keys, "race_date"]
    daily = eligible.groupby(
        group_keys,
        as_index=False,
        sort=True,
        observed=True,
    ).agg(
        __starts=("finish", "size"),
        __wins=("__wins", "sum"),
        __top3=("__top3", "sum"),
    )
    entity_dates = result.loc[
        result[keys].notna().all(axis=1) & result["race_date"].notna(),
        group_keys,
    ].drop_duplicates()
    daily = entity_dates.merge(
        daily,
        on=group_keys,
        how="left",
        validate="one_to_one",
    )
    daily[["__starts", "__wins", "__top3"]] = daily[
        ["__starts", "__wins", "__top3"]
    ].fillna(0)
    daily = daily.sort_values(group_keys, kind="stable")
    grouped = daily.groupby(keys, sort=False, observed=True)
    daily[f"{prefix}_prior_starts"] = (
        grouped["__starts"].cumsum() - daily["__starts"]
    )
    prior_top3 = grouped["__top3"].cumsum() - daily["__top3"]
    daily[f"{prefix}_prior_top3_rate"] = prior_top3.div(
        daily[f"{prefix}_prior_starts"].replace(0, np.nan)
    )
    result = result.merge(
        daily[
            [
                *keys,
                "race_date",
                f"{prefix}_prior_starts",
                f"{prefix}_prior_top3_rate",
            ]
        ],
        on=[*keys, "race_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.sort_values(row_order, kind="stable")
    result[f"{prefix}_prior_starts"] = (
        result[f"{prefix}_prior_starts"].fillna(0)
    )
    return result.drop(columns=row_order).reset_index(drop=True)


def add_contextual_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add course-, distance-, and jockey/horse-pair history."""
    result = frame.copy()
    if "distance_m" in result.columns:
        result["__distance_band"] = pd.cut(
            pd.to_numeric(result["distance_m"], errors="coerce"),
            bins=[0, 1400, 1800, 2200, 10000],
            labels=["short", "mile_middle", "middle_long", "long"],
            include_lowest=True,
        ).astype("object")

    contexts = (
        (
            ["horse_name", "surface"],
            "horse_surface",
        ),
        (
            ["horse_name", "venue", "surface", "__distance_band"],
            "horse_course_distance",
        ),
        (
            ["horse_name", "jockey"],
            "jockey_horse",
        ),
    )
    for keys, prefix in contexts:
        if all(key in result.columns for key in keys):
            result = _add_prior_group_rates(result, keys, prefix)

    return result.drop(columns=["__distance_band"], errors="ignore")


def add_race_name_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Encode race names with expanding historical outcomes, without leakage."""
    required = {"race_id", "race_date", "race_name", "finish"}
    if not required.issubset(frame.columns):
        return frame

    result = frame.copy()
    row_order = "__race_name_feature_row_order"
    result[row_order] = np.arange(len(result))
    eligible = result.loc[
        result["race_id"].notna()
        & result["race_date"].notna()
        & result["race_name"].notna()
        & result["finish"].notna()
        & result["finish"].ge(1),
        ["race_id", "race_date", "race_name", "finish"],
    ].copy()
    if eligible.empty:
        result["race_name_prior_starts"] = 0
        result["race_name_prior_top3_rate"] = np.nan
        return result.drop(columns=row_order)

    eligible["__top3"] = eligible["finish"].le(3).astype(int)
    by_race = eligible.groupby(
        ["race_id", "race_date", "race_name"],
        as_index=False,
        sort=False,
        observed=True,
    ).agg(
        __starts=("finish", "size"),
        __top3=("__top3", "sum"),
    )
    daily = by_race.groupby(
        ["race_name", "race_date"],
        as_index=False,
        sort=True,
        observed=True,
    ).agg(
        __starts=("__starts", "sum"),
        __top3=("__top3", "sum"),
    )
    grouped = daily.groupby("race_name", sort=False, observed=True)
    daily["race_name_prior_starts"] = (
        grouped["__starts"].cumsum() - daily["__starts"]
    )
    prior_top3 = grouped["__top3"].cumsum() - daily["__top3"]
    daily["race_name_prior_top3_rate"] = prior_top3.div(
        daily["race_name_prior_starts"].replace(0, np.nan)
    )
    result = result.merge(
        daily[
            [
                "race_name",
                "race_date",
                "race_name_prior_starts",
                "race_name_prior_top3_rate",
            ]
        ],
        on=["race_name", "race_date"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result = result.sort_values(row_order, kind="stable")
    result["race_name_prior_starts"] = (
        result["race_name_prior_starts"].fillna(0)
    )
    return result.drop(columns=row_order).reset_index(drop=True)


def add_race_relative_features(
    frame: pd.DataFrame,
    columns: list[str],
) -> pd.DataFrame:
    """Add within-race ordinal percentiles for numeric entry features."""
    result = frame.copy()
    if "race_id" not in result.columns:
        return result
    for column in columns:
        if column not in result.columns:
            continue
        values = pd.to_numeric(result[column], errors="coerce")
        result[f"{column}_race_percentile"] = values.groupby(
            result["race_id"]
        ).rank(method="average", pct=True, ascending=True)
    return result


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
        "race_name",
        "venue",
        "surface",
        "distance_m",
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
    current["race_name"] = race.get("race_name")
    current["venue"] = race.get("venue")
    current["surface"] = race.get("surface")
    current["distance_m"] = race.get("distance_m")
    current["__prediction_row"] = True
    history["__prediction_row"] = False
    combined = pd.concat(
        [history, current],
        ignore_index=True,
        sort=False,
    )
    combined = add_historical_performance_features(combined)
    combined = add_race_name_history_features(combined)
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

    result = add_race_relative_features(
        result,
        [
            "popularity",
            "odds",
            "horse_prior_starts",
            "horse_prior_win_rate",
            "horse_prior_top3_rate",
            "horse_recent_win_rate_5",
            "horse_recent_top3_rate_5",
            "horse_surface_prior_top3_rate",
            "horse_course_distance_prior_top3_rate",
            "jockey_horse_prior_top3_rate",
        ],
    )
    return result.reset_index(drop=True)
