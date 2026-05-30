from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import numpy as np
import pandas as pd

from racing_ml.common.artifacts import display_path, ensure_output_file_path, utc_now_iso, write_json


def _resolve_files(pattern: str) -> list[Path]:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return paths


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _read_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = display_path(path, workspace_root=ROOT)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _bucket_series(frame: pd.DataFrame) -> pd.Series:
    popularity = _numeric(frame, "popularity")
    bucket = pd.Series("unknown", index=frame.index, dtype="object")
    bucket.loc[popularity == 1] = "favorite"
    bucket.loc[popularity == 2] = "second_favorite"
    bucket.loc[popularity == 3] = "third_favorite"
    bucket.loc[(popularity >= 4) & (popularity <= 6)] = "pop_4_6"
    bucket.loc[(popularity >= 7) & (popularity <= 10)] = "pop_7_10"
    bucket.loc[popularity >= 11] = "pop_11_plus"
    return bucket


def _margin_band_series(margin: pd.Series) -> pd.Series:
    bins = [-np.inf, -0.20, -0.10, -0.05, -0.02, 0.0, 0.02, 0.05, 0.10, 0.20, np.inf]
    labels = [
        "lt_-0.20",
        "-0.20_-0.10",
        "-0.10_-0.05",
        "-0.05_-0.02",
        "-0.02_0.00",
        "0.00_0.02",
        "0.02_0.05",
        "0.05_0.10",
        "0.10_0.20",
        "gte_0.20",
    ]
    return pd.cut(margin, bins=bins, labels=labels, include_lowest=True, right=False).astype("object").fillna("unknown")


def _roi(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    if not bool(mask.any()):
        return None
    odds = _numeric(frame.loc[mask], "odds")
    rank = _numeric(frame.loc[mask], "rank")
    return float(odds.where(rank == 1, 0.0).mean())


def _rate(frame: pd.DataFrame, mask: pd.Series) -> float | None:
    if not bool(mask.any()):
        return None
    return float((_numeric(frame.loc[mask], "rank") == 1).mean())


def _quantiles(series: pd.Series) -> dict[str, float | None]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return {"p10": None, "p25": None, "p50": None, "p75": None, "p90": None}
    return {f"p{int(q * 100):02d}": float(values.quantile(q)) for q in [0.10, 0.25, 0.50, 0.75, 0.90]}


def _summarize_mask(frame: pd.DataFrame, mask: pd.Series, *, label: str, group: str) -> dict[str, Any]:
    selected = frame["policy_selected"].fillna(False).astype(bool)
    margin = _numeric(frame, "ev_margin")
    return {
        "label": label,
        "group": group,
        "rows": int(mask.sum()),
        "races": int(frame.loc[mask, "race_id"].nunique()) if bool(mask.any()) else 0,
        "selected_rows": int((selected & mask).sum()),
        "win_rate": _rate(frame, mask),
        "roi": _roi(frame, mask),
        "mean_margin": None if not bool(mask.any()) else float(margin.loc[mask].mean()),
        "margin_quantiles": _quantiles(margin.loc[mask]),
    }


def _band_rows(frame: pd.DataFrame, *, label: str) -> list[dict[str, Any]]:
    selected = frame["policy_selected"].fillna(False).astype(bool)
    band = frame["ev_margin_band"].astype(str)
    bucket = frame["popularity_bucket"].astype(str)
    rows: list[dict[str, Any]] = []
    for band_name, mask_index in band.groupby(band).groups.items():
        mask = pd.Series(False, index=frame.index)
        mask.loc[mask_index] = True
        rows.append(
            {
                "label": label,
                "kind": "margin_band",
                "bucket": "",
                "band": str(band_name),
                "rows": int(mask.sum()),
                "selected_rows": int((selected & mask).sum()),
                "win_rate": _rate(frame, mask),
                "roi": _roi(frame, mask),
            }
        )
    for (bucket_name, band_name), group_index in frame.groupby([bucket, band], sort=False).groups.items():
        mask = pd.Series(False, index=frame.index)
        mask.loc[group_index] = True
        rows.append(
            {
                "label": label,
                "kind": "bucket_margin_band",
                "bucket": str(bucket_name),
                "band": str(band_name),
                "rows": int(mask.sum()),
                "selected_rows": int((selected & mask).sum()),
                "win_rate": _rate(frame, mask),
                "roi": _roi(frame, mask),
            }
        )
    return rows


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    required = ["policy_expected_value", "policy_min_expected_value", "policy_selected", "rank", "odds", "race_id"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    work = frame.copy()
    work["ev_margin"] = _numeric(work, "policy_expected_value") - _numeric(work, "policy_min_expected_value")
    work["ev_margin_band"] = _margin_band_series(work["ev_margin"])
    work["popularity_bucket"] = _bucket_series(work)
    return work


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze policy EV threshold margin distributions.")
    parser.add_argument("--prediction-glob", action="append", required=True, help="label=glob")
    parser.add_argument("--near-miss-lower", type=float, default=-0.05)
    parser.add_argument("--near-miss-upper", type=float, default=0.0)
    parser.add_argument("--output-json", default="artifacts/reports/policy_ev_margin_analysis.json")
    parser.add_argument("--output-csv", default="artifacts/reports/policy_ev_margin_analysis_by_band.csv")
    args = parser.parse_args()

    summaries: list[dict[str, Any]] = []
    band_rows: list[dict[str, Any]] = []
    inputs: dict[str, list[str]] = {}
    for spec in args.prediction_glob:
        if "=" not in spec:
            raise ValueError(f"--prediction-glob must be label=glob, got: {spec}")
        label, pattern = spec.split("=", 1)
        label = label.strip()
        paths = _resolve_files(pattern.strip())
        inputs[label] = [display_path(path, workspace_root=ROOT) for path in paths]
        frame = _prepare(_read_files(paths))
        selected = frame["policy_selected"].fillna(False).astype(bool)
        winner = _numeric(frame, "rank") == 1
        margin = _numeric(frame, "ev_margin")
        near_miss = (~selected) & (margin >= args.near_miss_lower) & (margin < args.near_miss_upper)

        summaries.extend(
            [
                _summarize_mask(frame, pd.Series(True, index=frame.index), label=label, group="all"),
                _summarize_mask(frame, selected, label=label, group="selected"),
                _summarize_mask(frame, selected & winner, label=label, group="selected_winners"),
                _summarize_mask(frame, selected & ~winner, label=label, group="selected_losers"),
                _summarize_mask(frame, near_miss, label=label, group="near_miss_rejected"),
                _summarize_mask(frame, near_miss & winner, label=label, group="near_miss_rejected_winners"),
            ]
        )
        band_rows.extend(_band_rows(frame, label=label))

    output_json = ensure_output_file_path(args.output_json, label="output-json", workspace_root=ROOT)
    output_csv = ensure_output_file_path(args.output_csv, label="output-csv", workspace_root=ROOT)
    report = {
        "created_at": utc_now_iso(),
        "near_miss_lower": args.near_miss_lower,
        "near_miss_upper": args.near_miss_upper,
        "inputs": inputs,
        "summaries": summaries,
        "band_csv": display_path(output_csv, workspace_root=ROOT),
    }
    write_json(output_json, report)
    pd.DataFrame(band_rows).to_csv(output_csv, index=False)
    print(f"wrote {display_path(output_json, workspace_root=ROOT)}")
    print(f"wrote {display_path(output_csv, workspace_root=ROOT)}")


if __name__ == "__main__":
    main()
