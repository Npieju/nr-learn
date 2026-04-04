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


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-provenance {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/local_nankan/raw/local_nankan_primary.csv")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_provenance_summary.json")
    parser.add_argument("--annotated-output", default=None)
    parser.add_argument("--pre-race-output", default=None)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[local-nankan-provenance]", logger=log_progress, min_interval_sec=0.0)
        input_path = _resolve_path(args.input_file)
        summary_path = _resolve_path(args.summary_output)
        annotated_path = _resolve_path(args.annotated_output) if args.annotated_output else None
        pre_race_path = _resolve_path(args.pre_race_output) if args.pre_race_output else None

        progress.start(message=f"loading input={input_path}")
        frame = pd.read_csv(input_path, low_memory=False)
        progress.update(current=1, message=f"loaded rows={len(frame)}")

        with Heartbeat("[local-nankan-provenance]", "building provenance summary", logger=log_progress):
            annotated = annotate_market_timing_bucket(frame)
            summary = build_provenance_summary(annotated)

        progress.update(current=2, message=f"bucketed pre_race={summary['pre_race_only_rows']} unknown={summary['unknown_rows']} post_race={summary['post_race_rows']}")
        write_json(summary_path, summary)

        if annotated_path is not None:
            write_csv_file(annotated_path, annotated, index=False, label="local_nankan annotated provenance output")
        progress.update(current=3, message="annotated output ready")

        if pre_race_path is not None:
            pre_race_only = filter_pre_race_only(annotated)
            write_csv_file(pre_race_path, pre_race_only, index=False, label="local_nankan pre-race only output")
        progress.complete(message=f"summary ready output={summary_path}")
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
