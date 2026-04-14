from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.local_nankan_provenance import annotate_market_timing_bucket, build_provenance_summary, filter_pre_race_only


REQUIRED_MARKET_PROVENANCE_COLUMNS = [
    "scheduled_post_at",
    "card_snapshot_at",
    "card_snapshot_relation",
    "odds_snapshot_at",
    "odds_snapshot_relation",
]


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-provenance {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _display_path_value(path_text: object) -> str | None:
    if not isinstance(path_text, str) or not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        return path_text
    return _display_path(path)


def _normalize_display_paths(value: object) -> object:
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if isinstance(child, str) and (
                key.endswith("_input")
                or key.endswith("_output")
                or key.endswith("_manifest")
                or key.endswith("_file")
                or key.endswith("_dir")
                or key.endswith("_path")
            ):
                normalized[key] = _display_path_value(child)
            else:
                normalized[key] = _normalize_display_paths(child)
        return normalized
    if isinstance(value, list):
        return [_normalize_display_paths(child) for child in value]
    if isinstance(value, str):
        return _display_path_value(value) or value
    return value


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _build_readiness(
    *,
    frame: pd.DataFrame,
    summary: dict[str, object],
    fail_on_missing_columns: bool,
    max_unknown_ratio: float | None,
    max_post_race_ratio: float | None,
    min_pre_race_rows: int | None,
) -> tuple[dict[str, object], str, str | None, str | None]:
    row_count = int(summary.get("row_count") or len(frame) or 0)
    pre_race_rows = int(summary.get("pre_race_only_rows") or 0)
    unknown_rows = int(summary.get("unknown_rows") or 0)
    post_race_rows = int(summary.get("post_race_rows") or 0)
    missing_columns = [column for column in REQUIRED_MARKET_PROVENANCE_COLUMNS if column not in frame.columns]

    unknown_ratio = _ratio(unknown_rows, row_count)
    post_race_ratio = _ratio(post_race_rows, row_count)
    blocking_reasons: list[str] = []
    recommended_action = "inspect_local_nankan_provenance_summary"

    if fail_on_missing_columns and missing_columns:
        blocking_reasons.append(
            "required market provenance columns are missing: " + ", ".join(missing_columns)
        )
        recommended_action = "rebuild_local_primary_with_provenance_columns"

    if max_unknown_ratio is not None and unknown_ratio > float(max_unknown_ratio):
        blocking_reasons.append(
            f"unknown market timing ratio {unknown_ratio:.6f} exceeds threshold {float(max_unknown_ratio):.6f}"
        )
        if recommended_action == "inspect_local_nankan_provenance_summary":
            recommended_action = "rerun_local_backfill_with_pre_race_market_capture"

    if max_post_race_ratio is not None and post_race_ratio > float(max_post_race_ratio):
        blocking_reasons.append(
            f"post-race market timing ratio {post_race_ratio:.6f} exceeds threshold {float(max_post_race_ratio):.6f}"
        )
        if recommended_action == "inspect_local_nankan_provenance_summary":
            recommended_action = "exclude_post_race_market_rows_and_recrawl"

    if min_pre_race_rows is not None and pre_race_rows < int(min_pre_race_rows):
        blocking_reasons.append(
            f"pre-race rows {pre_race_rows} are below required minimum {int(min_pre_race_rows)}"
        )
        if recommended_action == "inspect_local_nankan_provenance_summary":
            recommended_action = "capture_pre_race_market_rows_before_benchmark"

    strict_ready = len(blocking_reasons) == 0
    readiness = {
        "strict_trust_ready": bool(strict_ready),
        "recommended_action": recommended_action,
        "blocking_reasons": blocking_reasons,
        "required_market_provenance_columns": REQUIRED_MARKET_PROVENANCE_COLUMNS,
        "missing_market_provenance_columns": missing_columns,
        "unknown_ratio": unknown_ratio,
        "post_race_ratio": post_race_ratio,
        "pre_race_rows": pre_race_rows,
        "row_count": row_count,
        "thresholds": {
            "fail_on_missing_columns": bool(fail_on_missing_columns),
            "max_unknown_ratio": max_unknown_ratio,
            "max_post_race_ratio": max_post_race_ratio,
            "min_pre_race_rows": min_pre_race_rows,
        },
    }
    if strict_ready:
        return readiness, "completed", None, None
    return readiness, "not_ready", "market_provenance_not_ready", "; ".join(blocking_reasons)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/local_nankan/raw/local_nankan_primary.csv")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_provenance_summary.json")
    parser.add_argument("--manifest-output", default="artifacts/reports/local_nankan_provenance_audit.json")
    parser.add_argument("--annotated-output", default=None)
    parser.add_argument("--pre-race-output", default=None)
    parser.add_argument("--fail-on-missing-columns", action="store_true")
    parser.add_argument("--max-unknown-ratio", type=float, default=None)
    parser.add_argument("--max-post-race-ratio", type=float, default=None)
    parser.add_argument("--min-pre-race-rows", type=int, default=None)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[local-nankan-provenance]", logger=log_progress, min_interval_sec=0.0)
        input_path = _resolve_path(args.input_file)
        summary_path = _resolve_path(args.summary_output)
        manifest_path = _resolve_path(args.manifest_output)
        annotated_path = _resolve_path(args.annotated_output) if args.annotated_output else None
        pre_race_path = _resolve_path(args.pre_race_output) if args.pre_race_output else None

        progress.start(message=f"loading input={input_path}")
        frame = pd.read_csv(input_path, low_memory=False)
        progress.update(current=1, message=f"loaded rows={len(frame)}")

        with Heartbeat("[local-nankan-provenance]", "building provenance summary", logger=log_progress):
            annotated = annotate_market_timing_bucket(frame)
            summary = build_provenance_summary(annotated)
            readiness, status, error_code, error_message = _build_readiness(
                frame=annotated,
                summary=summary,
                fail_on_missing_columns=bool(args.fail_on_missing_columns),
                max_unknown_ratio=args.max_unknown_ratio,
                max_post_race_ratio=args.max_post_race_ratio,
                min_pre_race_rows=args.min_pre_race_rows,
            )

        progress.update(current=2, message=f"bucketed pre_race={summary['pre_race_only_rows']} unknown={summary['unknown_rows']} post_race={summary['post_race_rows']}")
        summary = _normalize_display_paths(summary)
        write_json(summary_path, summary)

        if annotated_path is not None:
            write_csv_file(annotated_path, annotated, index=False, label="local_nankan annotated provenance output")
        progress.update(current=3, message="annotated output ready")

        if pre_race_path is not None:
            pre_race_only = filter_pre_race_only(annotated)
            write_csv_file(pre_race_path, pre_race_only, index=False, label="local_nankan pre-race only output")
        manifest = _normalize_display_paths({
            "started_at": None,
            "finished_at": None,
            "status": status,
            "current_phase": "strict_pre_race_trust_ready" if status == "completed" else "strict_pre_race_trust_blocked",
            "error_code": error_code,
            "error_message": error_message,
            "recommended_action": readiness.get("recommended_action"),
            "artifacts": {
                "input_file": str(input_path),
                "summary_output": str(summary_path),
                "manifest_output": str(manifest_path),
                "annotated_output": str(annotated_path) if annotated_path is not None else None,
                "pre_race_output": str(pre_race_path) if pre_race_path is not None else None,
            },
            "readiness": readiness,
            "provenance_summary": summary,
        })
        write_json(manifest_path, manifest)
        progress.complete(message=f"summary ready output={summary_path}")
        if status != "completed":
            return 2
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-provenance] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-provenance] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
