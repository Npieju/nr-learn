from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import pandas as pd

from racing_ml.common.artifacts import display_path, ensure_output_file_path, utc_now_iso, write_json


def _parse_label_glob(values: list[str]) -> list[tuple[str, str]]:
    parsed: list[tuple[str, str]] = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected LABEL=GLOB, got: {value}")
        label, pattern = value.split("=", 1)
        label = label.strip()
        pattern = pattern.strip()
        if not label or not pattern:
            raise ValueError(f"Expected non-empty LABEL=GLOB, got: {value}")
        parsed.append((label, pattern))
    if not parsed:
        raise ValueError("At least one --prediction-glob LABEL=GLOB is required")
    return parsed


def _resolve_files(pattern: str) -> list[Path]:
    files = sorted(ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files matched: {pattern}")
    return files


def _to_float_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _bucket_popularity(popularity: pd.Series) -> pd.Series:
    bucket = pd.Series("unknown", index=popularity.index, dtype="object")
    bucket.loc[popularity == 1] = "favorite"
    bucket.loc[popularity == 2] = "second_favorite"
    bucket.loc[popularity == 3] = "third_favorite"
    bucket.loc[(popularity >= 4) & (popularity <= 6)] = "pop_4_6"
    bucket.loc[(popularity >= 7) & (popularity <= 10)] = "pop_7_10"
    bucket.loc[popularity >= 11] = "pop_11_plus"
    return bucket


def _mean_or_none(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.mean())


def _sum_or_none(series: pd.Series) -> float | None:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return None
    return float(clean.sum())


def _roi_or_none(frame: pd.DataFrame) -> float | None:
    if "rank" not in frame.columns or "odds" not in frame.columns:
        return None
    rank = _to_float_series(frame, "rank")
    odds = _to_float_series(frame, "odds")
    if frame.empty:
        return None
    returns = odds.where(rank == 1, 0.0)
    return float(returns.mean())


def _summarize_label(label: str, files: list[Path]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        frame["__source_file"] = display_path(path, workspace_root=ROOT)
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if "race_id" not in data.columns:
        raise ValueError(f"{label}: prediction files must contain race_id")

    score = _to_float_series(data, "score")
    policy_prob = _to_float_series(data, "policy_prob")
    policy_selected = (
        data["policy_selected"].fillna(False).astype(bool)
        if "policy_selected" in data.columns
        else pd.Series(False, index=data.index)
    )
    race_count = int(data["race_id"].nunique())
    row_count = int(len(data))

    race_sum = pd.DataFrame({"race_id": data["race_id"], "score": score, "policy_prob": policy_prob})
    score_sum = race_sum.groupby("race_id")["score"].sum(min_count=1)
    policy_prob_sum = race_sum.groupby("race_id")["policy_prob"].sum(min_count=1)

    summary: dict[str, Any] = {
        "label": label,
        "files": [display_path(path, workspace_root=ROOT) for path in files],
        "file_count": int(len(files)),
        "race_count": race_count,
        "row_count": row_count,
        "policy_selected_rows": int(policy_selected.sum()),
        "policy_selected_races": int(data.loc[policy_selected, "race_id"].nunique()) if policy_selected.any() else 0,
        "score_sum_mean": _mean_or_none(score_sum),
        "score_sum_min": _mean_or_none(pd.Series([score_sum.min()])),
        "score_sum_max": _mean_or_none(pd.Series([score_sum.max()])),
        "policy_prob_sum_mean": _mean_or_none(policy_prob_sum),
        "policy_prob_sum_min": _mean_or_none(pd.Series([policy_prob_sum.min()])),
        "policy_prob_sum_max": _mean_or_none(pd.Series([policy_prob_sum.max()])),
        "score_sum_abs_error_mean": _mean_or_none((score_sum - 1.0).abs()),
        "policy_prob_sum_abs_error_mean": _mean_or_none((policy_prob_sum - 1.0).abs()),
        "selected_roi": _roi_or_none(data.loc[policy_selected].copy()),
    }

    popularity = _to_float_series(data, "popularity")
    data = data.copy()
    data["__popularity_bucket"] = _bucket_popularity(popularity)
    data["__score"] = score
    data["__policy_prob"] = policy_prob
    data["__policy_expected_value"] = _to_float_series(data, "policy_expected_value")
    data["__expected_value"] = _to_float_series(data, "expected_value")
    data["__odds"] = _to_float_series(data, "odds")
    data["__is_win"] = (_to_float_series(data, "rank") == 1).astype(float) if "rank" in data.columns else pd.NA

    bucket_rows: list[dict[str, Any]] = []
    for bucket, group in data.groupby("__popularity_bucket", sort=False):
        if str(bucket) == "unknown":
            continue
        bucket_rows.append(
            {
                "label": label,
                "bucket": str(bucket),
                "rows": int(len(group)),
                "win_rate": _mean_or_none(group["__is_win"]),
                "mean_odds": _mean_or_none(group["__odds"]),
                "mean_score": _mean_or_none(group["__score"]),
                "mean_policy_prob": _mean_or_none(group["__policy_prob"]),
                "mean_expected_value": _mean_or_none(group["__expected_value"]),
                "mean_policy_expected_value": _mean_or_none(group["__policy_expected_value"]),
                "mean_return": _roi_or_none(group),
                "policy_selected_rows": int(policy_selected.loc[group.index].sum()),
            }
        )

    return summary, bucket_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize race-closure and popularity diagnostics for prediction CSVs.")
    parser.add_argument(
        "--prediction-glob",
        action="append",
        default=[],
        metavar="LABEL=GLOB",
        help="Prediction CSV glob relative to workspace root. Can be passed multiple times.",
    )
    parser.add_argument("--output-json", default="artifacts/reports/probability_path_diagnostics.json")
    parser.add_argument("--output-csv", default="artifacts/reports/probability_path_diagnostics_by_bucket.csv")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    for label, pattern in _parse_label_glob(args.prediction_glob):
        summary, rows = _summarize_label(label, _resolve_files(pattern))
        summaries.append(summary)
        bucket_rows.extend(rows)

    output_json = ensure_output_file_path(args.output_json, label="output-json", workspace_root=ROOT)
    output_csv = ensure_output_file_path(args.output_csv, label="output-csv", workspace_root=ROOT)
    payload = {
        "created_at": utc_now_iso(),
        "summaries": summaries,
        "bucket_csv": display_path(output_csv, workspace_root=ROOT),
    }
    write_json(output_json, payload)
    pd.DataFrame(bucket_rows).to_csv(output_csv, index=False)
    print(f"wrote {display_path(output_json, workspace_root=ROOT)}")
    print(f"wrote {display_path(output_csv, workspace_root=ROOT)}")


if __name__ == "__main__":
    main()
