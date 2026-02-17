from __future__ import annotations

import pandas as pd


def _group_shifted_rolling_mean(frame: pd.DataFrame, group_col: str, value_col: str, window: int) -> pd.Series:
    shifted = frame.groupby(group_col, sort=False)[value_col].shift(1)
    rolled = (
        shifted.groupby(frame[group_col], sort=False)
        .rolling(window=window, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)
    )
    return rolled.astype(float)


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
    return None


def build_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()

    if "rank" in data.columns:
        data["rank"] = pd.to_numeric(data["rank"], errors="coerce")

    if "is_win" in data.columns:
        data["is_win"] = pd.to_numeric(data["is_win"], errors="coerce").fillna(0).astype(int)

    horse_history_key, horse_key_source = _build_horse_history_key(data)
    if horse_history_key is not None:
        data["horse_history_key"] = horse_history_key

    if horse_history_key is not None and {"rank"}.issubset(data.columns):
        data["horse_last_3_avg_rank"] = _group_shifted_rolling_mean(data, "horse_history_key", "rank", window=3).fillna(data["rank"].median())

    if horse_history_key is not None and {"is_win"}.issubset(data.columns):
        data["horse_last_5_win_rate"] = _group_shifted_rolling_mean(data, "horse_history_key", "is_win", window=5).fillna(0.0)

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
