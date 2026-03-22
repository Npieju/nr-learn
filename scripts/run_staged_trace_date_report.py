from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[staged-trace-date-report {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _trace_steps(trace: Any) -> list[str]:
    text = _string_or_none(trace)
    if text is None:
        return []
    return [part.strip() for part in text.split(">") if part.strip()]


def _trace_depth(trace: Any) -> int:
    return len(_trace_steps(trace))


def _selected_race_rows(raw_rows: pd.DataFrame) -> pd.DataFrame:
    if raw_rows.empty:
        return raw_rows.copy()
    stage1 = raw_rows[pd.to_numeric(raw_rows.get("stage_index"), errors="coerce") == 1].copy()
    if not stage1.empty:
        return stage1
    return raw_rows.sort_values(["race_id", "stage_index"], ascending=[True, True]).drop_duplicates(["race_id"], keep="first")


def _aggregate_date_rows(raw_rows: pd.DataFrame) -> dict[str, Any]:
    race_rows = _selected_race_rows(raw_rows)
    highest_stage_index = int(pd.to_numeric(raw_rows.get("stage_index"), errors="coerce").max()) if not raw_rows.empty else 0
    final_stage_counts: Counter[str] = Counter()
    final_trace_counts: Counter[str] = Counter()
    final_reason_counts: Counter[str] = Counter()
    trace_depth_counts: Counter[int] = Counter()
    deepest_selected_stage_counts: Counter[str] = Counter()
    max_depth_races: list[str] = []
    final_selected_net_units = 0.0
    final_selected_hit_count = 0
    stage2_plus_trace_race_count = 0
    stage3_plus_trace_race_count = 0
    stage2_plus_selected_race_count = 0
    stage3_plus_selected_race_count = 0
    deepest_stage_selected_race_count = 0

    per_race_depths: list[tuple[str, int]] = []
    per_race_selected_depths: list[tuple[str, int]] = []
    for _, row in race_rows.iterrows():
        race_id = str(row.get("race_id") or "")
        final_stage_name = _string_or_none(row.get("final_stage_name"))
        final_stage_trace = _string_or_none(row.get("final_stage_trace"))
        final_stage_fallback_reasons = _string_or_none(row.get("final_stage_fallback_reasons"))
        depth = _trace_depth(final_stage_trace)

        if final_stage_name is not None:
            final_stage_counts[final_stage_name] += 1
        if final_stage_trace is not None:
            final_trace_counts[final_stage_trace] += 1
        if final_stage_fallback_reasons is not None:
            final_reason_counts[final_stage_fallback_reasons] += 1
        trace_depth_counts[depth] += 1
        per_race_depths.append((race_id, depth))

        if depth >= 2:
            stage2_plus_trace_race_count += 1
        if depth >= 3:
            stage3_plus_trace_race_count += 1

    for race_id, race_group in raw_rows.groupby("race_id", sort=False):
        selected_rows = race_group[pd.to_numeric(race_group.get("selected_count"), errors="coerce").fillna(0) > 0].copy() if "selected_count" in race_group.columns else pd.DataFrame()
        if selected_rows.empty:
            continue
        deepest_row = selected_rows.sort_values(["stage_index"], ascending=[False]).iloc[0]
        deepest_index = int(pd.to_numeric(deepest_row.get("stage_index"), errors="coerce"))
        deepest_name = _string_or_none(deepest_row.get("stage_name")) or f"stage_{deepest_index}"
        per_race_selected_depths.append((str(race_id), deepest_index))
        deepest_selected_stage_counts[deepest_name] += 1
        if deepest_index >= 2:
            stage2_plus_selected_race_count += 1
        if deepest_index >= 3:
            stage3_plus_selected_race_count += 1
        if highest_stage_index > 0 and deepest_index >= highest_stage_index:
            deepest_stage_selected_race_count += 1

    if not race_rows.empty:
        max_depth = max(depth for _, depth in per_race_depths)
        max_depth_races = sorted(race_id for race_id, depth in per_race_depths if depth == max_depth)
    else:
        max_depth = 0

    max_selected_stage_index = max((depth for _, depth in per_race_selected_depths), default=0)

    final_stage_rows = raw_rows[raw_rows["stage_is_final"].fillna(False).astype(bool)].copy() if "stage_is_final" in raw_rows.columns else pd.DataFrame()
    if not final_stage_rows.empty:
        final_selected_net_units = float(pd.to_numeric(final_stage_rows.get("selected_net_units"), errors="coerce").fillna(0.0).sum())
        final_selected_hit_count = int(final_stage_rows["selected_hit"].fillna(False).astype(bool).sum()) if "selected_hit" in final_stage_rows.columns else 0

    final_stage_names = sorted(final_stage_counts)
    return {
        "num_races": int(len(race_rows)),
        "races_with_final_selection": int(sum(final_stage_counts.values())),
        "no_final_selection_race_count": int(len(race_rows) - sum(final_stage_counts.values())),
        "final_selected_hit_count": int(final_selected_hit_count),
        "final_selected_net_units": float(final_selected_net_units),
        "highest_stage_index": int(highest_stage_index),
        "max_trace_depth": int(max_depth),
        "mean_trace_depth": float(pd.Series([depth for _, depth in per_race_depths], dtype=float).mean()) if per_race_depths else 0.0,
        "stage2_plus_trace_race_count": int(stage2_plus_trace_race_count),
        "stage3_plus_trace_race_count": int(stage3_plus_trace_race_count),
        "max_selected_stage_index": int(max_selected_stage_index),
        "stage2_plus_selected_race_count": int(stage2_plus_selected_race_count),
        "stage3_plus_selected_race_count": int(stage3_plus_selected_race_count),
        "deepest_stage_selected_race_count": int(deepest_stage_selected_race_count),
        "max_trace_depth_races": max_depth_races,
        "final_stage_counts": dict(final_stage_counts),
        "final_trace_counts": dict(final_trace_counts),
        "final_fallback_reason_counts": dict(final_reason_counts),
        "deepest_selected_stage_counts": dict(deepest_selected_stage_counts),
        "trace_depth_counts": {str(depth): int(count) for depth, count in sorted(trace_depth_counts.items())},
        "final_stage_names": final_stage_names,
        "reaches_stage2": bool(stage2_plus_selected_race_count > 0),
        "reaches_stage3": bool(stage3_plus_selected_race_count > 0),
        "reaches_deepest_stage": bool(deepest_stage_selected_race_count > 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--window-label", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[staged-trace-date-report]", logger=log_progress, min_interval_sec=0.0)
    try:
        input_csv = _resolve_path(args.input_csv)
        if not input_csv.exists():
            raise FileNotFoundError(f"input CSV not found: {input_csv}")
        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"staged_trace_date_report_{args.window_label}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"staged_trace_date_report_{args.window_label}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        progress.start(message="loading staged diagnostic csv")
        raw_rows = pd.read_csv(input_csv)
        if "date" not in raw_rows.columns or "race_id" not in raw_rows.columns:
            raise ValueError("input CSV must contain date and race_id columns")
        progress.update(message=f"loaded rows={len(raw_rows)}")

        report_rows: list[dict[str, Any]] = []
        date_progress = ProgressBar(total=max(int(raw_rows["date"].nunique()), 1), prefix="[staged-trace-date-report dates]", logger=log_progress, min_interval_sec=0.0)
        date_progress.start(message="aggregating per-date trace signals")
        for date_value, date_rows in raw_rows.groupby("date", sort=True):
            with Heartbeat("[staged-trace-date-report]", f"processing {date_value}", logger=log_progress):
                row = {
                    "date": str(date_value),
                    "input_csv": artifact_display_path(input_csv, workspace_root=ROOT),
                    "window_label": args.window_label,
                }
                row.update(_aggregate_date_rows(date_rows.copy()))
                report_rows.append(row)
            date_progress.update(message=f"processed {date_value}")

        progress.update(message=f"report rows complete count={len(report_rows)}")

        payload = {
            "input_csv": artifact_display_path(input_csv, workspace_root=ROOT),
            "window_label": args.window_label,
            "dates": sorted(str(value) for value in raw_rows["date"].dropna().astype(str).unique().tolist()),
            "rows": report_rows,
        }
        rows_df = pd.DataFrame(report_rows).sort_values(["date"], ascending=[True])

        with Heartbeat("[staged-trace-date-report]", "writing outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, rows_df, index=False)

        print(f"saved staged trace date report json to {output_json.relative_to(ROOT)}")
        print(f"saved staged trace date report csv to {output_csv.relative_to(ROOT)}")
        progress.complete(message="staged trace date report completed")
        return 0
    except KeyboardInterrupt:
        print("[staged-trace-date-report] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[staged-trace-date-report] failed: {error}")
        return 1
    except Exception as error:
        print(f"[staged-trace-date-report] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())