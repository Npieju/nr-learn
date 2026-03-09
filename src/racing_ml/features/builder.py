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


def _compose_key(frame: pd.DataFrame, columns: list[str], out_col: str) -> None:
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return

    key = frame[available_columns[0]].astype("string").fillna("unknown")
    for column in available_columns[1:]:
        key = key + "|" + frame[column].astype("string").fillna("unknown")
    frame[out_col] = key.astype(str)


def _select_horse_key(frame: pd.DataFrame) -> str | None:
    has_horse_id = "horse_id" in frame.columns
    has_horse_name = "horse_name" in frame.columns
    if not has_horse_id and not has_horse_name:
        return None

    if has_horse_id and not has_horse_name:
        return "horse_id"
    if has_horse_name and not has_horse_id:
        return "horse_name"

    horse_id = frame["horse_id"].astype(str).str.strip().replace({"": "unknown", "nan": "unknown", "None": "unknown"})
    horse_name = frame["horse_name"].astype(str).str.strip().replace({"": "unknown", "nan": "unknown", "None": "unknown"})

    valid_id = horse_id[horse_id != "unknown"]
    id_repeat_ratio = (1.0 - (valid_id.nunique() / len(valid_id))) if len(valid_id) > 0 else 0.0

    valid_name = horse_name[horse_name != "unknown"]
    name_repeat_ratio = (1.0 - (valid_name.nunique() / len(valid_name))) if len(valid_name) > 0 else 0.0

    if id_repeat_ratio >= 0.05 and id_repeat_ratio >= name_repeat_ratio * 0.8:
        return "horse_id"
    return "horse_name"


def _build_horse_history_key(frame: pd.DataFrame) -> tuple[pd.Series | None, str | None]:
    key_col = _select_horse_key(frame)
    if key_col is None:
        return None, None

    key = frame[key_col].astype(str).str.strip().replace({"": "unknown", "nan": "unknown", "None": "unknown"})

    if key_col == "horse_name" and "sex" in frame.columns:
        sex = frame["sex"].astype(str).str.strip().replace({"": "unknown", "nan": "unknown", "None": "unknown"})
        key = key + "|sex=" + sex

    return key, key_col


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    if "date" in data.columns:
        date_series = pd.to_datetime(data["date"], errors="coerce")
        data["race_year"] = date_series.dt.year.astype("Int64")
        data["race_month"] = date_series.dt.month.astype("Int64")
        data["race_dayofweek"] = date_series.dt.dayofweek.astype("Int64")

    if "rank" in data.columns:
        data["rank"] = pd.to_numeric(data["rank"], errors="coerce")

    if "distance" in data.columns:
        data["distance"] = pd.to_numeric(data["distance"], errors="coerce")

    if "is_win" in data.columns:
        data["is_win"] = pd.to_numeric(data["is_win"], errors="coerce").fillna(0).astype(int)

    if "race_id" in data.columns:
        data["field_size"] = data.groupby("race_id", sort=False)["race_id"].transform("size").astype(float)

    if "finish_time_sec" in data.columns:
        data["finish_time_sec"] = pd.to_numeric(data["finish_time_sec"], errors="coerce")

    if "closing_time_3f" in data.columns:
        data["closing_time_3f"] = pd.to_numeric(data["closing_time_3f"], errors="coerce")

    for column in ["corner_1_position", "corner_2_position", "corner_3_position", "corner_4_position", "race_pace_front3f", "race_pace_back3f"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    if {"finish_time_sec", "distance"}.issubset(data.columns):
        distance_km = data["distance"] / 1000.0
        data["time_per_1000m"] = np.where(distance_km > 0, data["finish_time_sec"] / distance_km, np.nan)

    if {"race_id", "finish_time_sec"}.issubset(data.columns):
        race_best_time = data.groupby("race_id", sort=False)["finish_time_sec"].transform("min")
        data["time_margin_sec"] = data["finish_time_sec"] - race_best_time

    _compose_key(data, ["track", "distance", "ground_condition"], "course_history_key")
    if {"course_history_key", "time_per_1000m"}.issubset(data.columns):
        data["course_baseline_time_per_1000m"] = _race_level_shifted_rolling_mean(
            data,
            "course_history_key",
            "time_per_1000m",
            window=120,
            min_periods=3,
        )
        data["time_deviation"] = data["time_per_1000m"] - data["course_baseline_time_per_1000m"]

    horse_history_key, horse_key_source = _build_horse_history_key(data)
    if horse_history_key is not None:
        data["horse_history_key"] = horse_history_key

    if horse_history_key is not None and {"rank"}.issubset(data.columns):
        data["horse_last_3_avg_rank"] = _group_shifted_rolling_mean(data, "horse_history_key", "rank", window=3).fillna(data["rank"].median())

    if horse_history_key is not None and {"is_win"}.issubset(data.columns):
        data["horse_last_5_win_rate"] = _group_shifted_rolling_mean(data, "horse_history_key", "is_win", window=5).fillna(0.0)

    history_specs = [
        ("time_per_1000m", "horse_last_3_avg_time_per_1000m", 3),
        ("time_deviation", "horse_last_5_avg_time_deviation", 5),
        ("time_margin_sec", "horse_last_3_avg_time_margin_sec", 3),
        ("closing_time_3f", "horse_last_3_avg_closing_time_3f", 3),
        ("corner_2_position", "horse_last_3_avg_corner_2_position", 3),
        ("corner_4_position", "horse_last_3_avg_corner_4_position", 3),
        ("race_pace_front3f", "horse_last_3_avg_race_pace_front3f", 3),
        ("race_pace_back3f", "horse_last_3_avg_race_pace_back3f", 3),
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
        data["jockey_last_30_win_rate"] = _group_shifted_rolling_mean(data, "jockey_id", "is_win", window=30).fillna(0.0)

    if {"trainer_id", "is_win"}.issubset(data.columns):
        data["trainer_last_30_win_rate"] = _group_shifted_rolling_mean(data, "trainer_id", "is_win", window=30).fillna(0.0)

    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].fillna("unknown")

    return data
