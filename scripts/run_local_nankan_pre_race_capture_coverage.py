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

from racing_ml.common.artifacts import read_json, write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.local_nankan_provenance import (
    build_pre_race_capture_coverage_summary,
    build_pre_race_capture_date_coverage,
)


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-pre-race-capture {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json")
    parser.add_argument("--date-coverage-output", default="artifacts/reports/local_nankan_pre_race_capture_date_coverage.csv")
    parser.add_argument("--baseline-summary-input", default=None)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=5, prefix="[local-nankan-pre-race-capture]", logger=log_progress, min_interval_sec=0.0)
        race_card_path = _resolve_path(args.race_card_input)
        race_result_path = _resolve_path(args.race_result_input)
        summary_path = _resolve_path(args.summary_output)
        date_coverage_path = _resolve_path(args.date_coverage_output)
        baseline_summary_path = _resolve_path(args.baseline_summary_input) if args.baseline_summary_input else None

        progress.start(message=f"loading race_card={race_card_path}")
        race_card_frame = pd.read_csv(race_card_path, low_memory=False)
        progress.update(current=1, message=f"race_card loaded rows={len(race_card_frame)}")

        race_result_frame = pd.read_csv(race_result_path, usecols=["race_id"], low_memory=False) if race_result_path.exists() else None
        progress.update(
            current=2,
            message=(
                f"race_result {'loaded' if race_result_frame is not None else 'missing'}"
                + (f" races={race_result_frame['race_id'].nunique()}" if race_result_frame is not None else "")
            ),
        )

        previous_summary = None
        if baseline_summary_path is not None and baseline_summary_path.exists():
            payload = read_json(baseline_summary_path)
            previous_summary = payload if isinstance(payload, dict) else None
        progress.update(current=3, message=f"baseline {'loaded' if previous_summary is not None else 'missing'}")

        with Heartbeat("[local-nankan-pre-race-capture]", "building capture coverage summary", logger=log_progress):
            summary = build_pre_race_capture_coverage_summary(
                race_card_frame,
                result_frame=race_result_frame,
                previous_summary=previous_summary,
            )
            date_coverage = build_pre_race_capture_date_coverage(race_card_frame, result_frame=race_result_frame)
        progress.update(
            current=4,
            message=(
                f"phase={summary['current_phase']} "
                f"rows={summary['pre_race_only_rows']} races={summary['pre_race_only_races']}"
            ),
        )

        summary["race_card_input"] = str(race_card_path)
        summary["race_result_input"] = str(race_result_path) if race_result_path.exists() else None
        summary["baseline_summary_input"] = str(baseline_summary_path) if baseline_summary_path is not None else None
        summary["date_coverage_output"] = str(date_coverage_path)
        write_json(summary_path, summary)
        write_csv_file(date_coverage_path, date_coverage, index=False, label="local_nankan pre-race capture date coverage")
        progress.complete(message=f"summary ready output={summary_path}")
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-pre-race-capture] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-pre-race-capture] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
