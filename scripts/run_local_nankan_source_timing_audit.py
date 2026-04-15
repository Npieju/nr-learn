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
from racing_ml.data.local_nankan_provenance import build_source_timing_audit_summary


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-source-timing {now}] {message}", flush=True)


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_source_timing_audit.json")
    parser.add_argument("--date-output", default="artifacts/reports/local_nankan_source_timing_audit_by_date.csv")
    parser.add_argument("--year-output", default="artifacts/reports/local_nankan_source_timing_audit_by_year.csv")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[local-nankan-source-timing]", logger=log_progress, min_interval_sec=0.0)
        race_card_path = _resolve_path(args.race_card_input)
        race_result_path = _resolve_path(args.race_result_input)
        summary_output = _resolve_path(args.summary_output)
        date_output = _resolve_path(args.date_output)
        year_output = _resolve_path(args.year_output)

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

        with Heartbeat("[local-nankan-source-timing]", "building source timing audit", logger=log_progress):
            summary, by_date, by_year = build_source_timing_audit_summary(
                race_card_frame,
                result_frame=race_result_frame,
            )
        progress.update(
            current=3,
            message=(
                f"phase={summary['current_phase']} result_ready_pre_race="
                f"{summary['historical_pre_race_recoverability']['result_ready_pre_race_rows']}"
            ),
        )

        summary = _normalize_display_paths(summary)
        summary["read_order"] = [
            "status",
            "current_phase",
            "recommended_action",
            "historical_pre_race_recoverability.result_ready_pre_race_rows",
            "historical_pre_race_recoverability.future_only_pre_race_rows",
            "historical_pre_race_recoverability.status",
        ]
        summary["race_card_input"] = _display_path(race_card_path)
        summary["race_result_input"] = _display_path(race_result_path) if race_result_path.exists() else None
        summary["date_output"] = _display_path(date_output)
        summary["year_output"] = _display_path(year_output)
        write_json(summary_output, summary)
        write_csv_file(date_output, by_date, index=False, label="local_nankan source timing audit by date")
        write_csv_file(year_output, by_year, index=False, label="local_nankan source timing audit by year")
        progress.complete(message=f"summary ready output={summary_output}")
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-source-timing] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-source-timing] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())