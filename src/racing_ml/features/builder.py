from __future__ import annotations

import numpy as np
import pandas as pd


def _group_shifted_rolling_mean(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    shifted = frame.groupby(group_col, sort=False)[value_col].shift(1)
    rolled = (
        shifted.groupby(frame[group_col], sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return rolled.astype(float)


def _race_level_shifted_rolling_mean(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    required = {"race_id", group_col, value_col}
    if not required.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index, dtype=float)

    race_frame = (
        frame[["race_id", group_col, value_col]]
        .dropna(subset=[group_col])
        .groupby("race_id", sort=False)
        .agg({group_col: "first", value_col: "mean"})
        .reset_index()
    )
    if race_frame.empty:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    shifted = race_frame.groupby(group_col, sort=False)[value_col].shift(1)
    rolled = (
        shifted.groupby(race_frame[group_col], sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    mapping = pd.Series(rolled.to_numpy(dtype=float), index=race_frame["race_id"])
    return frame["race_id"].map(mapping).astype(float)


def _compose_key(
    frame: pd.DataFrame,
    columns: list[str],
    out_col: str,
    required_columns: list[str] | None = None,
) -> None:
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return

    key = frame[available_columns[0]].astype("string").fillna("unknown")
    for column in available_columns[1:]:
        key = key + "|" + frame[column].astype("string").fillna("unknown")

    if required_columns:
        required_mask = pd.Series(True, index=frame.index, dtype=bool)
        for column in required_columns:
            if column not in frame.columns:
                required_mask &= False
                continue
            required_mask &= _clean_group_series(frame, column).notna()
        key = key.where(required_mask)

    frame[out_col] = key.astype("string")


def _clean_group_series(frame: pd.DataFrame, column: str) -> pd.Series:
    return frame[column].astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})


def _fill_history_defaults(
    frame: pd.DataFrame,
    group_col: str,
    values: pd.Series,
    fill_value: float | int | None,
) -> pd.Series:
    if pd.isna(fill_value) or group_col not in frame.columns:
        return values

    valid_group_mask = _clean_group_series(frame, group_col).notna()
    if not valid_group_mask.any():
        return values

    filled = values.copy()
    filled.loc[valid_group_mask] = filled.loc[valid_group_mask].fillna(float(fill_value))
    return filled


def _build_horse_name_history_key(frame: pd.DataFrame) -> pd.Series | None:
    if "horse_name" not in frame.columns:
        return None

    horse_name = _clean_group_series(frame, "horse_name")
    if "sex" not in frame.columns:
        return horse_name

    sex = _clean_group_series(frame, "sex").fillna("unknown")
    return (horse_name + "|sex=" + sex).where(horse_name.notna())


def _entity_race_shifted_rolling_mean(
    frame: pd.DataFrame,
    group_col: str,
    value_col: str,
    window: int,
    min_periods: int = 1,
) -> pd.Series:
    required = {"race_id", group_col, value_col}
    if not required.issubset(frame.columns):
        return pd.Series(np.nan, index=frame.index, dtype=float)

    work = frame[["race_id", group_col, value_col]].copy()
    work[group_col] = _clean_group_series(work, group_col)
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work["_row_order"] = np.arange(len(work), dtype=int)
    work = work.dropna(subset=[group_col])
    if work.empty:
        return pd.Series(np.nan, index=frame.index, dtype=float)

    race_group = (
        work.groupby(["race_id", group_col], sort=False)
        .agg({value_col: "mean", "_row_order": "min"})
        .reset_index()
        .sort_values("_row_order")
        .reset_index(drop=True)
    )
    shifted = race_group.groupby(group_col, sort=False)[value_col].shift(1)
    rolled = (
        shifted.groupby(race_group[group_col], sort=False)
        .rolling(window=window, min_periods=min_periods)
        .mean()
        .reset_index(level=0, drop=True)
    )
    race_group["_feature_value"] = rolled.to_numpy(dtype=float)

    lookup = race_group[["race_id", group_col, "_feature_value"]]
    joined = frame[["race_id", group_col]].copy()
    joined[group_col] = _clean_group_series(joined, group_col)
    joined = joined.merge(lookup, on=["race_id", group_col], how="left", sort=False)
    joined.index = frame.index
    return joined["_feature_value"].astype(float)


def _select_horse_fallback_key(frame: pd.DataFrame) -> str | None:
    has_horse_id = "horse_id" in frame.columns
    has_horse_name = "horse_name" in frame.columns
    if not has_horse_id and not has_horse_name:
        return None

    if has_horse_id and not has_horse_name:
        return "horse_id"
    if has_horse_name and not has_horse_id:
        return "horse_name"

    horse_id = _clean_group_series(frame, "horse_id")
    horse_name = _clean_group_series(frame, "horse_name")

    valid_id = horse_id.dropna()
    id_repeat_ratio = (1.0 - (valid_id.nunique() / len(valid_id))) if len(valid_id) > 0 else 0.0

    valid_name = horse_name.dropna()
    name_repeat_ratio = (1.0 - (valid_name.nunique() / len(valid_name))) if len(valid_name) > 0 else 0.0

    if id_repeat_ratio >= 0.05 and id_repeat_ratio >= name_repeat_ratio * 0.8:
        return "horse_id"
    return "horse_name"


def _build_horse_history_key(frame: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    history_key = pd.Series(pd.NA, index=frame.index, dtype="string")
    key_source = pd.Series(pd.NA, index=frame.index, dtype="string")

    if "horse_key" in frame.columns:
        horse_key = _clean_group_series(frame, "horse_key")
        horse_key_mask = horse_key.notna()
        if horse_key_mask.any():
            history_key.loc[horse_key_mask] = "horse_key|" + horse_key.loc[horse_key_mask]
            key_source.loc[horse_key_mask] = "horse_key"

    fallback_col = _select_horse_fallback_key(frame)
    if fallback_col is None and not history_key.notna().any():
        return None, None

    if fallback_col == "horse_name":
        fallback_key = _build_horse_name_history_key(frame)
    elif fallback_col is not None:
        fallback_key = _clean_group_series(frame, fallback_col)
    else:
        fallback_key = None

    if fallback_key is not None:
        fallback_mask = history_key.isna() & fallback_key.notna()
        if fallback_mask.any():
            history_key.loc[fallback_mask] = f"{fallback_col}|" + fallback_key.loc[fallback_mask]
            key_source.loc[fallback_mask] = fallback_col

    if not history_key.notna().any():
        return None, None

    return history_key, key_source


def _build_early_corner_position(frame: pd.DataFrame) -> pd.Series | None:
    corner_columns = [
        column
        for column in ["corner_2_position", "corner_1_position", "corner_3_position"]
        if column in frame.columns
    ]
    if not corner_columns:
        return None
    return frame[corner_columns].bfill(axis=1).iloc[:, 0].astype(float)


def _group_shift(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    required = {group_col, value_col}
    if not required.issubset(frame.columns):
        return pd.Series(pd.NA, index=frame.index, dtype="object")
    return frame.groupby(group_col, sort=False)[value_col].shift(1)


def _build_surface_series(frame: pd.DataFrame) -> pd.Series | None:
    surface_columns = [column for column in ["芝・ダート区分", "芝・ダート区分2"] if column in frame.columns]
    if not surface_columns:
        return None

    surface_frame = pd.DataFrame(
        {column: _clean_group_series(frame, column) for column in surface_columns},
        index=frame.index,
    )
    return surface_frame.bfill(axis=1).iloc[:, 0].astype("string")


def _build_race_class_score(frame: pd.DataFrame) -> pd.Series | None:
    if "競争条件" not in frame.columns and "リステッド・重賞競走" not in frame.columns:
        return None

    score = pd.Series(np.nan, index=frame.index, dtype=float)

    if "リステッド・重賞競走" in frame.columns:
        stakes = _clean_group_series(frame, "リステッド・重賞競走").str.upper()
        stakes_specs = [
            (r"J\.G1|G1", 10.0),
            (r"J\.G2|G2", 9.0),
            (r"J\.G3|G3", 8.0),
            (r"^L$", 7.5),
            (r"^G$", 8.5),
        ]
        for pattern, value in stakes_specs:
            mask = stakes.str.contains(pattern, regex=True, na=False)
            score = score.mask(mask, value)

    if "競争条件" in frame.columns:
        condition = _clean_group_series(frame, "競争条件")
        condition_specs = [
            (r"未出走|新馬", 0.5),
            (r"未勝利", 1.0),
            (r"300万下", 2.0),
            (r"400万下|500万下|1勝", 3.0),
            (r"600万下|700万下", 4.0),
            (r"800万下|900万下|1000万下|2勝", 5.0),
            (r"1400万下|1500万下|1600万下|3勝", 6.0),
            (r"オープン|OPEN|OP", 7.0),
        ]
        for pattern, value in condition_specs:
            mask = score.isna() & condition.str.contains(pattern, regex=True, na=False)
            score = score.mask(mask, value)

    return score


def _bucketize_gate_ratio(gate_ratio: pd.Series) -> pd.Series:
    numeric_ratio = pd.to_numeric(gate_ratio, errors="coerce")
    bucket = pd.Series(pd.NA, index=gate_ratio.index, dtype="string")
    bucket.loc[numeric_ratio.notna() & (numeric_ratio <= 0.34)] = "inner"
    bucket.loc[numeric_ratio.notna() & (numeric_ratio > 0.34) & (numeric_ratio <= 0.67)] = "middle"
    bucket.loc[numeric_ratio.notna() & (numeric_ratio > 0.67)] = "outer"
    return bucket


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    date_series = None
    if "date" in data.columns:
        date_series = pd.to_datetime(data["date"], errors="coerce")
        data["race_year"] = date_series.dt.year.astype("Int64")
        data["race_month"] = date_series.dt.month.astype("Int64")
        data["race_dayofweek"] = date_series.dt.dayofweek.astype("Int64")

    race_surface = _build_surface_series(data)
    if race_surface is not None:
        data["_race_surface_tmp"] = race_surface

    race_class_score = _build_race_class_score(data)
    if race_class_score is not None:
        data["race_class_score"] = race_class_score

    if "rank" in data.columns:
        data["rank"] = pd.to_numeric(data["rank"], errors="coerce")

    if "distance" in data.columns:
        data["distance"] = pd.to_numeric(data["distance"], errors="coerce")

    if "is_win" in data.columns:
        data["is_win"] = pd.to_numeric(data["is_win"], errors="coerce").fillna(0).astype(int)

    if "race_id" in data.columns:
        data["field_size"] = data.groupby("race_id", sort=False)["race_id"].transform("size").astype(float)

    for column in ["frame_no", "gate_no"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if {"gate_no", "field_size"}.issubset(data.columns):
        data["gate_ratio"] = np.where(data["field_size"] > 0, data["gate_no"] / data["field_size"], np.nan)

    if {"frame_no", "field_size"}.issubset(data.columns):
        approx_frame_count = np.ceil(data["field_size"] / 2.0)
        data["frame_ratio"] = np.where(approx_frame_count > 0, data["frame_no"] / approx_frame_count, np.nan)

    if "finish_time_sec" in data.columns:
        data["finish_time_sec"] = pd.to_numeric(data["finish_time_sec"], errors="coerce")

    if "closing_time_3f" in data.columns:
        data["closing_time_3f"] = pd.to_numeric(data["closing_time_3f"], errors="coerce")

    for column in ["corner_1_position", "corner_2_position", "corner_3_position", "corner_4_position", "race_pace_front3f", "race_pace_back3f"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    early_corner_position = _build_early_corner_position(data)
    if early_corner_position is not None:
        data["corner_2_position"] = early_corner_position

    if {"corner_2_position", "field_size"}.issubset(data.columns):
        data["corner_2_ratio"] = np.where(data["field_size"] > 0, data["corner_2_position"] / data["field_size"], np.nan)

    if {"corner_4_position", "field_size"}.issubset(data.columns):
        data["corner_4_ratio"] = np.where(data["field_size"] > 0, data["corner_4_position"] / data["field_size"], np.nan)

    if {"corner_2_position", "corner_4_position"}.issubset(data.columns):
        data["corner_gain_2_to_4"] = data["corner_2_position"] - data["corner_4_position"]

    if {"race_pace_front3f", "race_pace_back3f"}.issubset(data.columns):
        data["race_pace_balance_3f"] = data["race_pace_back3f"] - data["race_pace_front3f"]

    if {"finish_time_sec", "distance"}.issubset(data.columns):
        distance_km = data["distance"] / 1000.0
        data["time_per_1000m"] = np.where(distance_km > 0, data["finish_time_sec"] / distance_km, np.nan)

    if {"race_id", "finish_time_sec"}.issubset(data.columns):
        race_best_time = data.groupby("race_id", sort=False)["finish_time_sec"].transform("min")
        data["time_margin_sec"] = data["finish_time_sec"] - race_best_time

    _compose_key(data, ["track", "distance", "ground_condition"], "course_history_key")
    if {"track", "distance", "_race_surface_tmp"}.issubset(data.columns):
        course_surface_columns = ["track", "distance"]
        if "ground_condition" in data.columns:
            course_surface_columns.append("ground_condition")
        course_surface_columns.append("_race_surface_tmp")
        _compose_key(data, course_surface_columns, "course_surface_history_key")

    if "gate_ratio" in data.columns:
        data["gate_bucket"] = _bucketize_gate_ratio(data["gate_ratio"])

    if {"course_surface_history_key", "gate_bucket"}.issubset(data.columns):
        _compose_key(
            data,
            ["course_surface_history_key", "gate_bucket"],
            "course_gate_bucket_key",
            required_columns=["course_surface_history_key", "gate_bucket"],
        )

    if {"course_history_key", "time_per_1000m"}.issubset(data.columns):
        data["course_baseline_time_per_1000m"] = _race_level_shifted_rolling_mean(
            data,
            "course_history_key",
            "time_per_1000m",
            window=120,
            min_periods=3,
        )
        data["time_deviation"] = data["time_per_1000m"] - data["course_baseline_time_per_1000m"]

    course_pace_specs = [
        ("race_pace_front3f", "course_baseline_race_pace_front3f"),
        ("race_pace_back3f", "course_baseline_race_pace_back3f"),
        ("race_pace_balance_3f", "course_baseline_race_pace_balance_3f"),
    ]
    if "course_history_key" in data.columns:
        for source_col, output_col in course_pace_specs:
            if source_col in data.columns:
                data[output_col] = _race_level_shifted_rolling_mean(
                    data,
                    "course_history_key",
                    source_col,
                    window=120,
                    min_periods=3,
                )

    horse_history_key, horse_key_source = _build_horse_history_key(data)
    if horse_history_key is not None:
        data["horse_history_key"] = horse_history_key

    if horse_history_key is not None and date_series is not None:
        previous_race_date = pd.to_datetime(_group_shift(data, "horse_history_key", "date"), errors="coerce")
        days_since_last_race = (date_series - previous_race_date).dt.days.astype(float)
        data["horse_days_since_last_race"] = days_since_last_race
        data["horse_days_since_last_race_log1p"] = np.log1p(days_since_last_race.clip(lower=0))

        short_turnaround = (days_since_last_race <= 14).astype(float)
        data["horse_is_short_turnaround"] = short_turnaround.where(days_since_last_race.notna(), np.nan)

        long_layoff = (days_since_last_race >= 90).astype(float)
        data["horse_is_long_layoff"] = long_layoff.where(days_since_last_race.notna(), np.nan)

    if horse_history_key is not None and "weight" in data.columns:
        current_weight = pd.to_numeric(data["weight"], errors="coerce")
        previous_weight = pd.to_numeric(_group_shift(data, "horse_history_key", "weight"), errors="coerce")
        data["horse_weight_change"] = current_weight - previous_weight
        data["horse_weight_change_abs"] = data["horse_weight_change"].abs()

    if horse_history_key is not None and "斤量" in data.columns:
        current_carried_weight = pd.to_numeric(data["斤量"], errors="coerce")
        previous_carried_weight = pd.to_numeric(_group_shift(data, "horse_history_key", "斤量"), errors="coerce")
        data["horse_carried_weight_change"] = current_carried_weight - previous_carried_weight
        data["horse_carried_weight_change_abs"] = data["horse_carried_weight_change"].abs()

    if horse_history_key is not None and "distance" in data.columns:
        previous_distance = pd.to_numeric(_group_shift(data, "horse_history_key", "distance"), errors="coerce")
        data["horse_distance_change"] = data["distance"] - previous_distance
        data["horse_distance_change_abs"] = data["horse_distance_change"].abs()

    if horse_history_key is not None and "_race_surface_tmp" in data.columns:
        previous_surface = _group_shift(data, "horse_history_key", "_race_surface_tmp").astype("string")
        surface_switch = (data["_race_surface_tmp"] != previous_surface).astype(float)
        data["horse_surface_switch"] = surface_switch.where(previous_surface.notna(), np.nan)

    if horse_history_key is not None and "race_class_score" in data.columns:
        data["horse_last_class_score"] = pd.to_numeric(
            _group_shift(data, "horse_history_key", "race_class_score"),
            errors="coerce",
        )
        data["horse_class_change"] = data["race_class_score"] - data["horse_last_class_score"]
        class_up = (data["horse_class_change"] > 0).astype(float)
        data["horse_is_class_up"] = class_up.where(data["horse_class_change"].notna(), np.nan)
        class_down = (data["horse_class_change"] < 0).astype(float)
        data["horse_is_class_down"] = class_down.where(data["horse_class_change"].notna(), np.nan)

    if horse_history_key is not None and {"rank"}.issubset(data.columns):
        data["horse_last_3_avg_rank"] = _group_shifted_rolling_mean(data, "horse_history_key", "rank", window=3).fillna(data["rank"].median())

    if horse_history_key is not None and {"is_win"}.issubset(data.columns):
        data["horse_last_5_win_rate"] = _group_shifted_rolling_mean(data, "horse_history_key", "is_win", window=5).fillna(0.0)
    if horse_history_key is not None and {"track", "distance"}.issubset(data.columns):
        _compose_key(
            data,
            ["horse_history_key", "track", "distance"],
            "horse_track_distance_key",
            required_columns=["horse_history_key"],
        )

    history_specs = [
        ("time_per_1000m", "horse_last_3_avg_time_per_1000m", 3),
        ("time_deviation", "horse_last_5_avg_time_deviation", 5),
        ("time_margin_sec", "horse_last_3_avg_time_margin_sec", 3),
        ("closing_time_3f", "horse_last_3_avg_closing_time_3f", 3),
        ("corner_2_position", "horse_last_3_avg_corner_2_position", 3),
        ("corner_2_ratio", "horse_last_3_avg_corner_2_ratio", 3),
        ("corner_4_position", "horse_last_3_avg_corner_4_position", 3),
        ("corner_4_ratio", "horse_last_3_avg_corner_4_ratio", 3),
        ("corner_gain_2_to_4", "horse_last_3_avg_corner_gain_2_to_4", 3),
        ("race_pace_front3f", "horse_last_3_avg_race_pace_front3f", 3),
        ("race_pace_back3f", "horse_last_3_avg_race_pace_back3f", 3),
        ("race_pace_balance_3f", "horse_last_3_avg_race_pace_balance_3f", 3),
        ("field_size", "horse_last_3_avg_field_size", 3),
    ]
    if horse_history_key is not None:
        for source_col, output_col, window in history_specs:
            if source_col in data.columns:
                data[output_col] = _group_shifted_rolling_mean(
                    data,
                    "horse_history_key",
                    source_col,
                    window=window,
                )

    if horse_key_source is not None:
        data["horse_history_key_source"] = horse_key_source

    if {"jockey_id", "is_win"}.issubset(data.columns):
        data["jockey_last_30_win_rate"] = _entity_race_shifted_rolling_mean(
            data,
            "jockey_id",
            "is_win",
            window=30,
        ).fillna(0.0)

    if {"trainer_id", "is_win"}.issubset(data.columns):
        data["trainer_last_30_win_rate"] = _entity_race_shifted_rolling_mean(
            data,
            "trainer_id",
            "is_win",
            window=30,
        ).fillna(0.0)

    if {"jockey_id", "trainer_id"}.issubset(data.columns):
        _compose_key(
            data,
            ["jockey_id", "trainer_id"],
            "jockey_trainer_combo_key",
            required_columns=["jockey_id", "trainer_id"],
        )

    combo_history_specs = [
        (
            "jockey_trainer_combo_key",
            "is_win",
            "jockey_trainer_combo_last_50_win_rate",
            50,
            3,
            0.0,
        ),
        (
            "jockey_trainer_combo_key",
            "rank",
            "jockey_trainer_combo_last_50_avg_rank",
            50,
            3,
            float(data["rank"].median()) if "rank" in data.columns and data["rank"].notna().any() else np.nan,
        ),
    ]
    for group_col, value_col, output_col, window, min_periods, fill_value in combo_history_specs:
        if {group_col, value_col}.issubset(data.columns):
            history_values = _entity_race_shifted_rolling_mean(
                data,
                group_col,
                value_col,
                window=window,
                min_periods=min_periods,
            )
            history_values = _fill_history_defaults(data, group_col, history_values, fill_value)
            data[output_col] = history_values

    jockey_trainer_style_specs = [
        ("jockey_id", "corner_gain_2_to_4", "jockey_last_30_avg_corner_gain_2_to_4", 30),
        ("trainer_id", "corner_gain_2_to_4", "trainer_last_30_avg_corner_gain_2_to_4", 30),
        ("jockey_id", "closing_time_3f", "jockey_last_30_avg_closing_time_3f", 30),
        ("trainer_id", "closing_time_3f", "trainer_last_30_avg_closing_time_3f", 30),
    ]
    for group_col, value_col, output_col, window in jockey_trainer_style_specs:
        if {group_col, value_col}.issubset(data.columns):
            data[output_col] = _entity_race_shifted_rolling_mean(
                data,
                group_col,
                value_col,
                window=window,
            )

    track_distance_history_specs = [
        ("horse_track_distance_key", "rank", "horse_track_distance_last_3_avg_rank", 3, 1, float(data["rank"].median()) if "rank" in data.columns and data["rank"].notna().any() else np.nan),
        ("horse_track_distance_key", "is_win", "horse_track_distance_last_5_win_rate", 5, 1, 0.0),
        ("jockey_track_distance_key", "is_win", "jockey_track_distance_last_50_win_rate", 50, 3, 0.0),
        ("jockey_track_distance_key", "rank", "jockey_track_distance_last_50_avg_rank", 50, 3, float(data["rank"].median()) if "rank" in data.columns and data["rank"].notna().any() else np.nan),
        ("trainer_track_distance_key", "is_win", "trainer_track_distance_last_50_win_rate", 50, 3, 0.0),
        ("trainer_track_distance_key", "rank", "trainer_track_distance_last_50_avg_rank", 50, 3, float(data["rank"].median()) if "rank" in data.columns and data["rank"].notna().any() else np.nan),
    ]
    if {"jockey_id", "track", "distance"}.issubset(data.columns):
        _compose_key(
            data,
            ["jockey_id", "track", "distance"],
            "jockey_track_distance_key",
            required_columns=["jockey_id"],
        )
    if {"trainer_id", "track", "distance"}.issubset(data.columns):
        _compose_key(
            data,
            ["trainer_id", "track", "distance"],
            "trainer_track_distance_key",
            required_columns=["trainer_id"],
        )
    for group_col, value_col, output_col, window, min_periods, fill_value in track_distance_history_specs:
        if {group_col, value_col}.issubset(data.columns):
            history_values = _entity_race_shifted_rolling_mean(
                data,
                group_col,
                value_col,
                window=window,
                min_periods=min_periods,
            )
            history_values = _fill_history_defaults(data, group_col, history_values, fill_value)
            data[output_col] = history_values

    gate_bias_history_specs = [
        ("course_gate_bucket_key", "is_win", "course_gate_bucket_last_100_win_rate", 100, 5, 0.0),
        (
            "course_gate_bucket_key",
            "rank",
            "course_gate_bucket_last_100_avg_rank",
            100,
            5,
            float(data["rank"].median()) if "rank" in data.columns and data["rank"].notna().any() else np.nan,
        ),
    ]
    for group_col, value_col, output_col, window, min_periods, fill_value in gate_bias_history_specs:
        if {group_col, value_col}.issubset(data.columns):
            history_values = _entity_race_shifted_rolling_mean(
                data,
                group_col,
                value_col,
                window=window,
                min_periods=min_periods,
            )
            history_values = _fill_history_defaults(data, group_col, history_values, fill_value)
            data[output_col] = history_values

    pedigree_history_specs = [
        ("owner_name", "is_win", "owner_last_50_win_rate", 50, 3, 0.0),
        ("breeder_name", "is_win", "breeder_last_50_win_rate", 50, 3, 0.0),
        ("sire_name", "is_win", "sire_last_100_win_rate", 100, 5, 0.0),
        ("sire_name", "rank", "sire_last_100_avg_rank", 100, 5, float(data["rank"].median()) if "rank" in data.columns and data["rank"].notna().any() else np.nan),
        ("damsire_name", "is_win", "damsire_last_100_win_rate", 100, 5, 0.0),
    ]
    for group_col, value_col, output_col, window, min_periods, fill_value in pedigree_history_specs:
        if {group_col, value_col}.issubset(data.columns):
            history_values = _entity_race_shifted_rolling_mean(
                data,
                group_col,
                value_col,
                window=window,
                min_periods=min_periods,
            )
            history_values = _fill_history_defaults(data, group_col, history_values, fill_value)
            data[output_col] = history_values

    if {"sire_name", "track", "distance"}.issubset(data.columns):
        _compose_key(
            data,
            ["sire_name", "track", "distance"],
            "sire_track_distance_key",
            required_columns=["sire_name"],
        )
    if {"sire_track_distance_key", "is_win"}.issubset(data.columns):
        history_values = _entity_race_shifted_rolling_mean(
            data,
            "sire_track_distance_key",
            "is_win",
            window=80,
            min_periods=3,
        )
        data["sire_track_distance_last_80_win_rate"] = _fill_history_defaults(
            data,
            "sire_track_distance_key",
            history_values,
            0.0,
        )

    if {"horse_last_3_avg_corner_gain_2_to_4", "course_baseline_race_pace_balance_3f"}.issubset(data.columns):
        data["horse_closing_pace_fit"] = data["horse_last_3_avg_corner_gain_2_to_4"] * data["course_baseline_race_pace_balance_3f"]

    if {"horse_last_3_avg_corner_2_ratio", "course_baseline_race_pace_balance_3f"}.issubset(data.columns):
        data["horse_front_pace_fit"] = (1.0 - data["horse_last_3_avg_corner_2_ratio"]) * (-data["course_baseline_race_pace_balance_3f"])

    if {"horse_last_3_avg_closing_time_3f", "course_baseline_race_pace_back3f"}.issubset(data.columns):
        data["horse_closing_vs_course"] = data["course_baseline_race_pace_back3f"] - data["horse_last_3_avg_closing_time_3f"]

    if "_race_surface_tmp" in data.columns:
        data = data.drop(columns=["_race_surface_tmp"])

    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].fillna("unknown")

    return data
