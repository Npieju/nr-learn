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
from racing_ml.data.local_nankan_provenance import (
    build_pre_race_only_materialization_summary,
    build_provenance_summary,
    filter_pre_race_only,
)


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-pre-race {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--output-file", default="data/local_nankan/raw/local_nankan_race_card_pre_race_only.csv")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_pre_race_only_materialize_summary.json")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=5, prefix="[local-nankan-pre-race]", logger=log_progress, min_interval_sec=0.0)
        race_card_path = _resolve_path(args.race_card_input)
        race_result_path = _resolve_path(args.race_result_input)
        output_path = _resolve_path(args.output_file)
        summary_path = _resolve_path(args.summary_output)

        progress.start(message=f"loading race_card={race_card_path}")
        frame = pd.read_csv(race_card_path, low_memory=False)
        progress.update(current=1, message=f"race_card loaded rows={len(frame)}")

        result_frame = None
        if race_result_path.exists():
            result_frame = pd.read_csv(race_result_path, usecols=["race_id"], low_memory=False)
        progress.update(
            current=2,
            message=(
                f"race_result {'loaded' if result_frame is not None else 'missing'}"
                + (f" races={result_frame['race_id'].nunique()}" if result_frame is not None else "")
            ),
        )

        with Heartbeat("[local-nankan-pre-race]", "filtering strict pre-race subset", logger=log_progress):
            pre_race_only = filter_pre_race_only(frame)
            provenance_summary = build_provenance_summary(frame)
            materialization_summary = build_pre_race_only_materialization_summary(frame, result_frame=result_frame)
        progress.update(
            current=3,
            message=(
                f"filtered rows={materialization_summary['pre_race_only_rows']} "
                f"races={materialization_summary['pre_race_only_races']}"
            ),
        )

        write_csv_file(output_path, pre_race_only, index=False, label="local_nankan pre-race only race_card")
        progress.update(current=4, message=f"subset output ready rows={len(pre_race_only)}")

        summary = {
            "status": "completed",
            "current_phase": "pre_race_only_materialized",
            "recommended_action": "run_pre_race_primary_materialization",
            "read_order": [
                "status",
                "current_phase",
                "recommended_action",
                "materialization_summary.pre_race_only_rows",
                "materialization_summary.result_ready_races",
                "provenance_summary.pre_race_rows",
            ],
            "race_card_input": _display_path(race_card_path),
            "race_result_input": _display_path(race_result_path),
            "output_file": _display_path(output_path),
            "provenance_summary": provenance_summary,
            "materialization_summary": materialization_summary,
        }
        write_json(summary_path, summary)
        progress.complete(message=f"summary ready output={summary_path}")
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-pre-race] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-pre-race] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
