from __future__ import annotations

import pandas as pd


def simple_hit_rate(predicted: pd.Series, actual: pd.Series, threshold: float = 0.5) -> float:
    picked = predicted >= threshold
    if picked.sum() == 0:
        return 0.0
    return float(actual[picked].mean())
