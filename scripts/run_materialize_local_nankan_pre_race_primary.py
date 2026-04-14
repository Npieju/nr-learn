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
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.local_nankan_primary import materialize_local_nankan_primary_from_config
from racing_ml.data.local_nankan_provenance import (
    build_pre_race_only_materialization_summary,
    filter_result_ready_pre_race_only,
)


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-pre-race-primary {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _display_path_value(value: object) -> object:
    if value is None:
        return None
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, str):
        return _display_path(_resolve_path(value)) if Path(value).is_absolute() else value
    return value


def _normalize_display_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_display_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_display_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_display_paths(item) for item in value]
    return _display_path_value(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_local_nankan_pre_race_ready.yaml")
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--pedigree-input", default="data/external/local_nankan/pedigree/local_pedigree.csv")
    parser.add_argument("--filtered-race-card-output", default="data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv")
    parser.add_argument("--filtered-race-result-output", default="data/local_nankan_pre_race_ready/raw/local_nankan_race_result_pre_race_ready.csv")
    parser.add_argument("--primary-output-file", default="data/local_nankan_pre_race_ready/raw/local_nankan_primary_pre_race_ready.csv")
    parser.add_argument("--manifest-file", default="artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json")
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_pre_race_ready_summary.json")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=6, prefix="[local-nankan-pre-race-primary]", logger=log_progress, min_interval_sec=0.0)
        data_config = load_yaml(ROOT / args.data_config)
        race_card_path = _resolve_path(args.race_card_input)
        race_result_path = _resolve_path(args.race_result_input)
        pedigree_path = _resolve_path(args.pedigree_input)
        filtered_card_path = _resolve_path(args.filtered_race_card_output)
        filtered_result_path = _resolve_path(args.filtered_race_result_output)
        primary_output_path = _resolve_path(args.primary_output_file)
        manifest_path = _resolve_path(args.manifest_file)
        summary_path = _resolve_path(args.summary_output)

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

        with Heartbeat("[local-nankan-pre-race-primary]", "filtering result-ready pre-race subset", logger=log_progress):
            ready_card_frame = filter_result_ready_pre_race_only(race_card_frame, race_result_frame)
            materialization_summary = build_pre_race_only_materialization_summary(race_card_frame, result_frame=race_result_frame)
        progress.update(
            current=3,
            message=(
                f"ready_races={materialization_summary['result_ready_races']} "
                f"pending_races={materialization_summary['pending_result_races']}"
            ),
        )

        if race_result_frame is None or "race_id" not in ready_card_frame.columns:
            ready_result_frame = pd.DataFrame()
        else:
            ready_race_ids = {str(value) for value in ready_card_frame["race_id"].dropna().tolist()}
            ready_result_ids = race_result_frame["race_id"].astype(str)
            ready_result_frame = race_result_frame.loc[ready_result_ids.isin(ready_race_ids)].reset_index(drop=True)

        write_csv_file(filtered_card_path, ready_card_frame, index=False, label="local_nankan result-ready pre-race race_card")
        write_csv_file(filtered_result_path, ready_result_frame, index=False, label="local_nankan result-ready pre-race race_result")
        progress.update(current=4, message=f"filtered outputs ready card_rows={len(ready_card_frame)} result_rows={len(ready_result_frame)}")

        if materialization_summary["result_ready_races"] == 0:
            summary = _normalize_display_paths({
                "status": "not_ready",
                "current_phase": "await_result_arrival",
                "recommended_action": "wait_for_result_ready_pre_race_races",
                "filtered_race_card_output": filtered_card_path,
                "filtered_race_result_output": filtered_result_path,
                "primary_output_file": primary_output_path,
                "manifest_file": manifest_path,
                "materialization_summary": materialization_summary,
            })
            write_json(summary_path, summary)
            progress.complete(message=f"not ready output={summary_path}")
            return 2

        with Heartbeat("[local-nankan-pre-race-primary]", "materializing result-ready primary raw", logger=log_progress):
            primary_summary = materialize_local_nankan_primary_from_config(
                data_config,
                base_dir=ROOT,
                race_result_path=filtered_result_path,
                race_card_path=filtered_card_path,
                pedigree_path=pedigree_path,
                output_file=primary_output_path,
                manifest_file=manifest_path,
                dry_run=False,
            )
        progress.update(current=5, message=f"primary status={primary_summary.get('status')} rows={primary_summary.get('row_count')}")

        summary = _normalize_display_paths({
            "status": "completed" if primary_summary.get("status") == "completed" else "failed",
            "current_phase": primary_summary.get("current_phase"),
            "recommended_action": primary_summary.get("recommended_action"),
            "filtered_race_card_output": filtered_card_path,
            "filtered_race_result_output": filtered_result_path,
            "primary_output_file": primary_output_path,
            "manifest_file": manifest_path,
            "materialization_summary": materialization_summary,
            "primary_materialize_summary": primary_summary,
        })
        write_json(summary_path, summary)
        progress.complete(message=f"summary ready output={summary_path}")
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-pre-race-primary] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-pre-race-primary] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
