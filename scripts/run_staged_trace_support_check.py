from __future__ import annotations

import argparse
import ast
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
    print(f"[staged-trace-support {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return None
    return int(numeric_value)


def _parse_labelled_input(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"report must be label=path: {raw}")
    label, path_value = raw.split("=", 1)
    normalized_label = str(label).strip()
    if not normalized_label:
        raise ValueError(f"report label is empty: {raw}")
    return normalized_label, _resolve_path(path_value.strip())


def _parse_stage_counts(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {str(key): int(count) for key, count in value.items()}
    text = str(value or "").strip()
    if not text:
        return {}
    parsed = ast.literal_eval(text)
    if not isinstance(parsed, dict):
        return {}
    return {str(key): int(count) for key, count in parsed.items()}


def _net_sign(value: Any) -> str:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "unknown"
    if numeric_value > 0.0:
        return "positive"
    if numeric_value < 0.0:
        return "negative"
    return "zero"


def _load_report_rows(label: str, path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"report not found: {path}")
    frame = pd.read_csv(path)
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        final_selected_net_units = _safe_float(row.get("final_selected_net_units"))
        stage3_selected_count = _safe_int(pd.to_numeric(row.get("stage3_plus_selected_race_count"), errors="coerce")) or 0
        stage2_selected_count = _safe_int(pd.to_numeric(row.get("stage2_plus_selected_race_count"), errors="coerce")) or 0
        highest_stage_index = _safe_int(pd.to_numeric(row.get("highest_stage_index"), errors="coerce")) or 0
        max_selected_stage_index = _safe_int(pd.to_numeric(row.get("max_selected_stage_index"), errors="coerce")) or 0
        deepest_stage_selected_count = _safe_int(pd.to_numeric(row.get("deepest_stage_selected_race_count"), errors="coerce"))
        deepest_selected_stage_counts = _parse_stage_counts(row.get("deepest_selected_stage_counts"))
        deepest_stage3_selected_count = int(deepest_selected_stage_counts.get("kelly_fallback_2", 0))
        if highest_stage_index == 0 and deepest_stage3_selected_count > 0:
            highest_stage_index = max_selected_stage_index
        if deepest_stage_selected_count is None:
            if deepest_stage3_selected_count > 0:
                deepest_stage_selected_count = deepest_stage3_selected_count
            elif highest_stage_index > 0 and max_selected_stage_index >= highest_stage_index:
                deepest_stage_selected_count = stage3_selected_count
            else:
                deepest_stage_selected_count = 0
        rows.append(
            {
                "report_label": label,
                "report_file": artifact_display_path(path, workspace_root=ROOT),
                "window_label": str(row.get("window_label") or label),
                "date": str(row.get("date") or ""),
                "races_with_final_selection": _safe_int(pd.to_numeric(row.get("races_with_final_selection"), errors="coerce")) or 0,
                "final_selected_net_units": final_selected_net_units,
                "net_sign": _net_sign(final_selected_net_units),
                "max_selected_stage_index": max_selected_stage_index,
                "highest_stage_index": highest_stage_index,
                "stage2_plus_selected_race_count": stage2_selected_count,
                "stage3_plus_selected_race_count": stage3_selected_count,
                "deepest_stage_selected_race_count": deepest_stage_selected_count,
                "deepest_stage_selected_present": bool(deepest_stage_selected_count > 0),
                "intermediate_stage_selected_present": bool(stage2_selected_count > 0 and deepest_stage_selected_count == 0),
                "deepest_selected_stage_counts": deepest_selected_stage_counts,
                "deepest_stage3_selected_count": deepest_stage3_selected_count,
            }
        )
    return rows


def _selection_depth_bucket(row: dict[str, Any]) -> str:
    if bool(row.get("deepest_stage_selected_present")):
        return "deepest_stage_selected"
    if bool(row.get("intermediate_stage_selected_present")):
        return "intermediate_stage_selected"
    if int(row.get("races_with_final_selection") or 0) > 0:
        return "stage1_only_selected"
    return "no_final_selection"


def _build_support_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {
            "window_count": 0,
            "date_count": 0,
            "stage3_candidate_date_count": 0,
            "stage3_candidate_positive_net_date_count": 0,
            "stage3_candidate_non_positive_net_date_count": 0,
            "stage3_candidate_windows": [],
            "stage3_candidate_dates": [],
        }

    stage3 = frame[frame["deepest_stage_selected_present"]].copy()
    intermediate = frame[frame["intermediate_stage_selected_present"]].copy()
    frame = frame.copy()
    frame["selection_depth_bucket"] = frame.apply(lambda row: _selection_depth_bucket(row.to_dict()), axis=1)
    return {
        "window_count": int(frame["report_label"].nunique()),
        "date_count": int(len(frame)),
        "deepest_stage_selected_date_count": int(len(stage3)),
        "deepest_stage_selected_positive_net_date_count": int((stage3["net_sign"] == "positive").sum()),
        "deepest_stage_selected_non_positive_net_date_count": int((stage3["net_sign"].isin(["negative", "zero"])).sum()),
        "deepest_stage_selected_windows": sorted(stage3["report_label"].dropna().astype(str).unique().tolist()),
        "deepest_stage_selected_dates": sorted(stage3["date"].dropna().astype(str).unique().tolist()),
        "intermediate_stage_selected_date_count": int(len(intermediate)),
        "intermediate_stage_selected_positive_net_date_count": int((intermediate["net_sign"] == "positive").sum()),
        "intermediate_stage_selected_non_positive_net_date_count": int((intermediate["net_sign"].isin(["negative", "zero"])).sum()),
        "intermediate_stage_selected_dates": sorted(intermediate["date"].dropna().astype(str).unique().tolist()),
        "net_sign_counts": frame["net_sign"].value_counts().to_dict(),
        "deepest_stage_selected_net_sign_counts": stage3["net_sign"].value_counts().to_dict(),
        "intermediate_stage_selected_net_sign_counts": intermediate["net_sign"].value_counts().to_dict(),
        "selection_depth_bucket_counts": frame["selection_depth_bucket"].value_counts().to_dict(),
        "rows_by_report": {
            label: {
                "date_count": int(len(group)),
                "deepest_stage_selected_date_count": int(group["deepest_stage_selected_present"].sum()),
                "deepest_stage_selected_dates": sorted(group.loc[group["deepest_stage_selected_present"], "date"].astype(str).tolist()),
                "deepest_stage_selected_net_sign_counts": group.loc[group["deepest_stage_selected_present"], "net_sign"].value_counts().to_dict(),
                "intermediate_stage_selected_date_count": int(group["intermediate_stage_selected_present"].sum()),
                "intermediate_stage_selected_dates": sorted(group.loc[group["intermediate_stage_selected_present"], "date"].astype(str).tolist()),
                "intermediate_stage_selected_net_sign_counts": group.loc[group["intermediate_stage_selected_present"], "net_sign"].value_counts().to_dict(),
                "selection_depth_bucket_counts": group["selection_depth_bucket"].value_counts().to_dict(),
            }
            for label, group in frame.groupby("report_label", sort=True)
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", required=True, help="label=path to staged_trace_date_report CSV; repeatable")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--window-label", default="staged_trace_support_check")
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[staged-trace-support]", logger=log_progress, min_interval_sec=0.0)
    try:
        report_specs = [_parse_labelled_input(raw) for raw in args.report]
        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"staged_trace_support_check_{args.window_label}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"staged_trace_support_check_{args.window_label}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        progress.start(message=f"loading reports count={len(report_specs)}")
        rows: list[dict[str, Any]] = []
        for label, path in report_specs:
            with Heartbeat("[staged-trace-support]", f"loading {label}", logger=log_progress):
                rows.extend(_load_report_rows(label, path))
        progress.update(message=f"rows loaded count={len(rows)}")

        frame = pd.DataFrame(rows).sort_values(["report_label", "date"], ascending=[True, True]).reset_index(drop=True)
        if not frame.empty:
            frame["selection_depth_bucket"] = frame.apply(lambda row: _selection_depth_bucket(row.to_dict()), axis=1)
        summary = _build_support_summary(rows)
        payload = {
            "window_label": args.window_label,
            "reports": [
                {
                    "label": label,
                    "report_file": artifact_display_path(path, workspace_root=ROOT),
                }
                for label, path in report_specs
            ],
            "summary": summary,
            "rows": frame.to_dict(orient="records"),
        }
        progress.update(message=f"support summary assembled deepest_stage_dates={summary['deepest_stage_selected_date_count']}")

        with Heartbeat("[staged-trace-support]", "writing outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, frame, index=False)

        print(f"saved staged trace support json to {output_json.relative_to(ROOT)}")
        print(f"saved staged trace support csv to {output_csv.relative_to(ROOT)}")
        progress.complete(message="staged trace support check completed")
        return 0
    except KeyboardInterrupt:
        print("[staged-trace-support] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError, SyntaxError) as error:
        print(f"[staged-trace-support] failed: {error}")
        return 1
    except Exception as error:
        print(f"[staged-trace-support] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())