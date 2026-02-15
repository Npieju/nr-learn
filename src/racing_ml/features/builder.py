from __future__ import annotations

import pandas as pd


def _group_cumulative_mean(frame: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    grouped = frame.groupby(group_col, sort=False)[value_col]
    prev_sum = grouped.cumsum() - frame[value_col]
    prev_count = grouped.cumcount()
    denominator = prev_count.replace(0, float("nan")).astype(float)
    result = prev_sum.astype(float) / denominator
    return result.astype(float)


def _select_horse_key(frame: pd.DataFrame) -> str | None:
    if "horse_name" in frame.columns:
        return "horse_name"
    if "horse_id" in frame.columns:
        return "horse_id"
    return None


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    if "rank" in data.columns:
        data["rank"] = pd.to_numeric(data["rank"], errors="coerce")

    if "is_win" in data.columns:
        data["is_win"] = pd.to_numeric(data["is_win"], errors="coerce").fillna(0).astype(int)

    horse_key = _select_horse_key(data)

    if horse_key and {"rank"}.issubset(data.columns):
        data["horse_last_3_avg_rank"] = _group_cumulative_mean(data, horse_key, "rank").fillna(data["rank"].median())

    if horse_key and {"is_win"}.issubset(data.columns):
        data["horse_last_5_win_rate"] = _group_cumulative_mean(data, horse_key, "is_win").fillna(0.0)

    if {"jockey_id", "is_win"}.issubset(data.columns):
        data["jockey_last_30_win_rate"] = _group_cumulative_mean(data, "jockey_id", "is_win").fillna(0.0)

    if {"trainer_id", "is_win"}.issubset(data.columns):
        data["trainer_last_30_win_rate"] = _group_cumulative_mean(data, "trainer_id", "is_win").fillna(0.0)

    for column in data.columns:
        if data[column].dtype == "object":
            data[column] = data[column].fillna("unknown")

    return data
