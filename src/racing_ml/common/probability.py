from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RaceProbabilityDiagnostics:
    row_count: int
    race_count: int
    valid_race_count: int
    invalid_value_count: int
    race_sum_min: float | None
    race_sum_mean: float | None
    race_sum_max: float | None
    race_sum_abs_error_mean: float | None
    race_sum_violation_count: int
    probability_contract_ok: bool
    tolerance: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def diagnose_race_probabilities(
    frame: pd.DataFrame,
    probability_col: str,
    *,
    race_id_col: str = "race_id",
    tolerance: float = 1e-6,
) -> RaceProbabilityDiagnostics:
    if race_id_col not in frame.columns:
        raise ValueError(f"Missing race id column: {race_id_col}")
    if probability_col not in frame.columns:
        raise ValueError(f"Missing probability column: {probability_col}")

    probabilities = pd.to_numeric(frame[probability_col], errors="coerce")
    finite_mask = np.isfinite(probabilities.to_numpy(dtype=float, na_value=np.nan))
    range_mask = probabilities.between(0.0, 1.0, inclusive="both").fillna(False).to_numpy(dtype=bool)
    valid_value_mask = finite_mask & range_mask
    invalid_value_count = int((~valid_value_mask).sum())

    work = pd.DataFrame(
        {
            race_id_col: frame[race_id_col],
            "probability": probabilities.where(valid_value_mask),
            "valid": valid_value_mask,
        }
    )
    race_valid = work.groupby(race_id_col, sort=False)["valid"].all()
    race_sums = work.groupby(race_id_col, sort=False)["probability"].sum(min_count=1)
    valid_sums = race_sums[race_valid & race_sums.notna()]
    absolute_error = (valid_sums - 1.0).abs()
    violation_count = int((absolute_error > float(tolerance)).sum())
    race_count = int(work[race_id_col].nunique(dropna=True))
    valid_race_count = int(len(valid_sums))

    return RaceProbabilityDiagnostics(
        row_count=int(len(frame)),
        race_count=race_count,
        valid_race_count=valid_race_count,
        invalid_value_count=invalid_value_count,
        race_sum_min=float(valid_sums.min()) if not valid_sums.empty else None,
        race_sum_mean=float(valid_sums.mean()) if not valid_sums.empty else None,
        race_sum_max=float(valid_sums.max()) if not valid_sums.empty else None,
        race_sum_abs_error_mean=float(absolute_error.mean()) if not absolute_error.empty else None,
        race_sum_violation_count=violation_count,
        probability_contract_ok=bool(
            race_count > 0
            and valid_race_count == race_count
            and invalid_value_count == 0
            and violation_count == 0
        ),
        tolerance=float(tolerance),
    )


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
