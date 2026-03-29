from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from racing_ml.data.dataset_loader import _read_csv_tail

TailReader = Callable[[Path, int], tuple[pd.DataFrame, int]]


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

    left_columns = [str(column) for column in left_frame.columns]
    right_columns = [str(column) for column in right_frame.columns]
    shared_columns = [column for column in left_columns if column in right_frame.columns]
    column_order_equal = left_columns == right_columns
    shape_equal = left_frame.shape == right_frame.shape
    totals_equal = int(left_total_rows) == int(right_total_rows)
    exact_equal = bool(column_order_equal and shape_equal and left_frame.equals(right_frame))

    first_diff_column: str | None = None
    first_diff_indices: list[int] = []
    first_diff_samples: list[dict[str, Any]] = []
    dtype_differences: list[dict[str, str]] = []

    for column in shared_columns:
        left_dtype = str(left_frame[column].dtype)
        right_dtype = str(right_frame[column].dtype)
        if left_dtype != right_dtype:
            dtype_differences.append(
                {
                    "column": column,
                    "left_dtype": left_dtype,
                    "right_dtype": right_dtype,
                }
            )

    if not exact_equal and shared_columns and shape_equal:
        for column in shared_columns:
            left_series = left_frame[column]
            right_series = right_frame[column]
            if left_series.equals(right_series):
                continue
            first_diff_column = column
            diff_mask = left_series.astype("string") != right_series.astype("string")
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
            "shape_equal": bool(shape_equal),
            "column_order_equal": bool(column_order_equal),
            "shared_column_count": int(len(shared_columns)),
            "exact_equal": bool(exact_equal),
            "dtype_differences": dtype_differences,
            "first_diff_column": first_diff_column,
            "first_diff_indices": first_diff_indices,
            "first_diff_samples": first_diff_samples,
        },
    }
