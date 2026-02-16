from __future__ import annotations

import numpy as np
import pandas as pd


def normalize_position_probabilities(
    frame: pd.DataFrame,
    raw_columns: list[str],
    race_id_col: str = "race_id",
    output_prefix: str = "",
) -> pd.DataFrame:
    normalized = frame.copy()

    if race_id_col not in normalized.columns:
        raise ValueError(f"Missing race id column: {race_id_col}")

    for column in raw_columns:
        if column not in normalized.columns:
            raise ValueError(f"Missing probability column: {column}")

        series = pd.to_numeric(normalized[column], errors="coerce").fillna(0.0)
        series = series.clip(lower=0.0)
        sums = series.groupby(normalized[race_id_col]).transform("sum")
        race_sizes = series.groupby(normalized[race_id_col]).transform("size").astype(float)

        fallback = np.where(race_sizes > 0, 1.0 / race_sizes, 0.0)
        norm_values = np.where(sums > 0, series / sums, fallback)
        out_col = f"{output_prefix}{column}"
        normalized[out_col] = norm_values.astype(float)

    return normalized
