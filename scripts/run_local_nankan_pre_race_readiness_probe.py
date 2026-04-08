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

from racing_ml.common.artifacts import write_json
from racing_ml.common.progress import ProgressBar
from racing_ml.data.local_nankan_provenance import apply_source_timing_context_to_readiness_summary, build_pre_race_readiness_probe_summary


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-readiness-probe {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path | None) -> dict[str, object]:
    if path is None or not path.exists():
        return {}
    try:
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_pre_race_readiness_probe_summary.json")
    parser.add_argument("--source-timing-summary-input", default="artifacts/reports/local_nankan_source_timing_audit_issue121.json")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[local-nankan-readiness-probe]", logger=log_progress, min_interval_sec=0.0)
        race_card_path = _resolve_path(args.race_card_input)
        race_result_path = _resolve_path(args.race_result_input)
        summary_output = _resolve_path(args.summary_output)
        source_timing_summary_path = _resolve_path(args.source_timing_summary_input) if args.source_timing_summary_input else None

        progress.start(message=f"loading race_card={race_card_path}")
        race_card_frame = pd.read_csv(race_card_path, low_memory=False)
        progress.update(current=1, message=f"race_card loaded rows={len(race_card_frame)}")

        race_result_frame = pd.read_csv(race_result_path, low_memory=False) if race_result_path.exists() else None
        progress.update(
            current=2,
            message=(
                f"race_result {'loaded' if race_result_frame is not None else 'missing'}"
                + (f" rows={len(race_result_frame)}" if race_result_frame is not None else "")
            ),
        )

        summary = build_pre_race_readiness_probe_summary(race_card_frame, result_frame=race_result_frame)
        source_timing_summary = _read_json_dict(source_timing_summary_path)
        summary = apply_source_timing_context_to_readiness_summary(
            summary,
            source_timing_summary=source_timing_summary,
        )
        summary["race_card_input"] = str(race_card_path)
        summary["race_result_input"] = str(race_result_path)
        summary["source_timing_summary_input"] = str(source_timing_summary_path) if source_timing_summary_path is not None else None
        write_json(summary_output, summary)
        progress.complete(message=f"status={summary['status']} output={summary_output}")
        return 0 if summary["status"] == "ready" else 2
    except KeyboardInterrupt:
        print("[local-nankan-readiness-probe] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-readiness-probe] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
