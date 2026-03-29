from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from racing_ml.data.dataset_loader import _normalize_columns, _read_csv_tail

TailReader = Callable[[Path, int], tuple[pd.DataFrame, int]]


def _series_value_diff_mask(left: pd.Series, right: pd.Series) -> pd.Series:
    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    numeric_mask = left_numeric.notna() | right_numeric.notna()

    equal_mask = pd.Series(False, index=left.index, dtype=bool)
    if numeric_mask.any():
        numeric_equal = (left_numeric == right_numeric) | (left_numeric.isna() & right_numeric.isna())
        equal_mask.loc[numeric_mask] = numeric_equal.loc[numeric_mask]

    non_numeric_mask = ~numeric_mask
    if non_numeric_mask.any():
        left_text = left.astype("string")
        right_text = right.astype("string")
        text_equal = (left_text == right_text) | (left_text.isna() & right_text.isna())
        equal_mask.loc[non_numeric_mask] = text_equal.loc[non_numeric_mask]

    return ~equal_mask


def _classify_dtype_difference(left: pd.Series, right: pd.Series) -> str:
    if not left.notna().any() and not right.notna().any():
        return "all_null"

    left_numeric = pd.to_numeric(left, errors="coerce")
    right_numeric = pd.to_numeric(right, errors="coerce")
    numeric_coverage = (left_numeric.notna() | right_numeric.notna()).all()
    if numeric_coverage:
        left_integral = bool(((left_numeric.dropna() % 1) == 0).all())
        right_integral = bool(((right_numeric.dropna() % 1) == 0).all())
        if left_integral and right_integral:
            return "numeric_integral_equivalent"
        return "numeric_dtype_only"

    return "other"


def read_csv_tail_deque_trim_candidate(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
    if tail_rows <= 0:
        raise ValueError("tail_rows must be greater than 0")

    chunk_size = max(min(int(tail_rows) * 4, 200000), 50000)
    total_rows = 0
    kept_rows = 0
    max_kept_rows = int(tail_rows)
    chunks: deque[pd.DataFrame] = deque()

    for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunk_size):
        total_rows += int(len(chunk))
        if len(chunk) > max_kept_rows:
            chunk = chunk.tail(max_kept_rows)
        chunks.append(chunk)
        kept_rows += int(len(chunk))
        if kept_rows > max_kept_rows:
            overflow = kept_rows - max_kept_rows
            while chunks and overflow >= int(len(chunks[0])):
                left_chunk = chunks.popleft()
                overflow -= int(len(left_chunk))
                kept_rows -= int(len(left_chunk))
            if overflow > 0 and chunks:
                chunks[0] = chunks[0].iloc[int(overflow):]
                kept_rows -= int(overflow)

    if not chunks:
        return pd.DataFrame(), 0

    tail_frame = pd.concat(list(chunks), ignore_index=True)
    if len(tail_frame) <= int(tail_rows):
        return tail_frame.reset_index(drop=True), total_rows
    return tail_frame.tail(int(tail_rows)).reset_index(drop=True), total_rows


TAIL_READER_CANDIDATES: dict[str, TailReader] = {
    "current": _read_csv_tail,
    "deque_trim": read_csv_tail_deque_trim_candidate,
}


def _build_frame_comparison(
    *,
    left_frame: pd.DataFrame,
    right_frame: pd.DataFrame,
    sample_limit: int,
) -> dict[str, Any]:
    left_columns = [str(column) for column in left_frame.columns]
    right_columns = [str(column) for column in right_frame.columns]
    shared_columns = [column for column in left_columns if column in right_frame.columns]
    column_order_equal = left_columns == right_columns
    shape_equal = left_frame.shape == right_frame.shape
    exact_equal = bool(column_order_equal and shape_equal and left_frame.equals(right_frame))
    value_equal = bool(column_order_equal and shape_equal)

    first_diff_column: str | None = None
    first_diff_indices: list[int] = []
    first_diff_samples: list[dict[str, Any]] = []
    dtype_differences: list[dict[str, str]] = []
    dtype_difference_categories: list[dict[str, str]] = []
    value_difference_count = 0

    for column in shared_columns:
        left_dtype = str(left_frame[column].dtype)
        right_dtype = str(right_frame[column].dtype)
        if left_dtype != right_dtype:
            classification = _classify_dtype_difference(left_frame[column], right_frame[column])
            dtype_differences.append(
                {
                    "column": column,
                    "left_dtype": left_dtype,
                    "right_dtype": right_dtype,
                }
            )
            dtype_difference_categories.append(
                {
                    "column": column,
                    "left_dtype": left_dtype,
                    "right_dtype": right_dtype,
                    "classification": classification,
                }
            )

    if shared_columns and shape_equal:
        for column in shared_columns:
            left_series = left_frame[column]
            right_series = right_frame[column]
            diff_mask = _series_value_diff_mask(left_series, right_series)
            diff_count = int(diff_mask.sum())
            value_difference_count += diff_count
            if diff_count == 0:
                continue
            first_diff_column = column
            first_diff_indices = [int(index) for index in diff_mask[diff_mask].index[:sample_limit]]
            for index in first_diff_indices:
                first_diff_samples.append(
                    {
                        "index": int(index),
                        "column": column,
                        "left_value": repr(left_series.iloc[index]),
                        "right_value": repr(right_series.iloc[index]),
                    }
                )
            break
        value_equal = value_difference_count == 0

    dtype_only_difference = bool((not exact_equal) and value_equal)
    canonical_dtype_only_difference = bool(
        dtype_only_difference
        and dtype_difference_categories
        and all(
            item["classification"] in {"all_null", "numeric_integral_equivalent", "numeric_dtype_only"}
            for item in dtype_difference_categories
        )
    )
    return {
        "shape_equal": bool(shape_equal),
        "column_order_equal": bool(column_order_equal),
        "shared_column_count": int(len(shared_columns)),
        "exact_equal": bool(exact_equal),
        "value_equal": bool(value_equal),
        "dtype_only_difference": dtype_only_difference,
        "canonical_dtype_only_difference": canonical_dtype_only_difference,
        "value_difference_count": int(value_difference_count),
        "dtype_differences": dtype_differences,
        "dtype_difference_categories": dtype_difference_categories,
        "first_diff_column": first_diff_column,
        "first_diff_indices": first_diff_indices,
        "first_diff_samples": first_diff_samples,
    }


def compare_tail_readers(
    *,
    left_name: str,
    left_reader: TailReader,
    right_name: str,
    right_reader: TailReader,
    csv_path: Path,
    tail_rows: int,
    sample_limit: int = 5,
) -> dict[str, Any]:
    left_frame, left_total_rows = left_reader(csv_path, tail_rows)
    right_frame, right_total_rows = right_reader(csv_path, tail_rows)

    left_frame = left_frame.reset_index(drop=True)
    right_frame = right_frame.reset_index(drop=True)
    totals_equal = int(left_total_rows) == int(right_total_rows)
    raw_comparison = _build_frame_comparison(
        left_frame=left_frame,
        right_frame=right_frame,
        sample_limit=sample_limit,
    )
    normalized_left_frame = _normalize_columns(left_frame).reset_index(drop=True)
    normalized_right_frame = _normalize_columns(right_frame).reset_index(drop=True)
    normalized_comparison = _build_frame_comparison(
        left_frame=normalized_left_frame,
        right_frame=normalized_right_frame,
        sample_limit=sample_limit,
    )
    left_columns = [str(column) for column in left_frame.columns]
    right_columns = [str(column) for column in right_frame.columns]

    return {
        "csv_path": str(csv_path),
        "tail_rows": int(tail_rows),
        "left": {
            "name": left_name,
            "total_rows": int(left_total_rows),
            "shape": [int(left_frame.shape[0]), int(left_frame.shape[1])],
            "columns": left_columns,
        },
        "right": {
            "name": right_name,
            "total_rows": int(right_total_rows),
            "shape": [int(right_frame.shape[0]), int(right_frame.shape[1])],
            "columns": right_columns,
        },
        "comparison": {
            "totals_equal": bool(totals_equal),
            "raw": raw_comparison,
            "normalized": normalized_comparison,
        },
    }
