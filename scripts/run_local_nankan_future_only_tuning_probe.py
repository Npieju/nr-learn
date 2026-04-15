from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-future-probe {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _display_path_value(value: object) -> object:
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    try:
        path = Path(value)
    except (TypeError, ValueError):
        return value
    if not path.is_absolute():
        return value
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return value


def _normalize_display_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_display_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_display_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_display_paths(item) for item in value]
    if isinstance(value, Path):
        try:
            return str(value.relative_to(ROOT))
        except ValueError:
            return str(value)
    return _display_path_value(value)


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[local-nankan-future-probe] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[local-nankan-future-probe]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _parse_scenario(raw_value: str) -> dict[str, Any]:
    text = str(raw_value).strip()
    if not text:
        raise ValueError("empty scenario")
    parts = [part.strip() for part in text.split(":") if part.strip()]
    if len(parts) != 3:
        raise ValueError(f"scenario must be label:horizon_days:max_passes, got {text!r}")
    label, horizon_text, passes_text = parts
    return {
        "label": label,
        "horizon_days": int(horizon_text),
        "max_passes": int(passes_text),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--wrapper-script", default="scripts/run_local_nankan_future_only_readiness_cycle.py")
    parser.add_argument("--output", default="artifacts/reports/local_nankan_future_only_tuning_probe_issue122.json")
    parser.add_argument(
        "--scenario",
        action="append",
        default=None,
        help="label:horizon_days:max_passes; repeatable",
    )
    parser.add_argument("--include-completed", action="store_true")
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scenario_inputs = args.scenario or ["h1_p1:1:1", "h7_p1:7:1", "h7_p2:7:2"]
    scenarios = [_parse_scenario(value) for value in scenario_inputs]
    progress = ProgressBar(total=max(1, len(scenarios)), prefix="[local-nankan-future-probe]", logger=log_progress, min_interval_sec=0.0)

    try:
        python_executable = str(_resolve_path(args.python_executable)) if Path(args.python_executable).is_absolute() else args.python_executable
        wrapper_script = str(_resolve_path(args.wrapper_script))
        progress.start(message=f"starting tuning probe scenarios={len(scenarios)}")

        records: list[dict[str, Any]] = []
        for index, scenario in enumerate(scenarios, start=1):
            label = str(scenario["label"])
            horizon_days = int(scenario["horizon_days"])
            max_passes = int(scenario["max_passes"])
            capture_manifest = f"artifacts/reports/{label}_local_nankan_pre_race_capture_loop_issue122_probe.json"
            watcher_manifest = f"artifacts/reports/{label}_local_nankan_readiness_watcher_issue122_probe.json"
            bootstrap_manifest = f"artifacts/reports/{label}_local_nankan_result_ready_bootstrap_handoff_issue122_probe.json"
            status_board_manifest = f"artifacts/reports/{label}_local_nankan_data_status_board_issue122_probe.json"
            wrapper_manifest = f"artifacts/reports/{label}_local_nankan_future_only_readiness_cycle_issue122_probe.json"

            command = [
                python_executable,
                wrapper_script,
                "--default-horizon-days",
                str(horizon_days),
                "--max-passes",
                str(max_passes),
                "--capture-loop-manifest-output",
                capture_manifest,
                "--watcher-manifest-output",
                watcher_manifest,
                "--bootstrap-manifest-output",
                bootstrap_manifest,
                "--status-board-output",
                status_board_manifest,
                "--wrapper-manifest-output",
                wrapper_manifest,
            ]
            if args.include_completed:
                command.append("--include-completed")

            exit_code = _run_command(label=f"scenario={label}", command=command)
            wrapper_payload = _read_json_dict(_resolve_path(wrapper_manifest))
            capture_payload = _read_json_dict(_resolve_path(capture_manifest))
            status_board_payload = _read_json_dict(_resolve_path(status_board_manifest))
            latest_summary = capture_payload.get("latest_summary") if isinstance(capture_payload.get("latest_summary"), dict) else {}
            readiness_payload = status_board_payload.get("readiness") if isinstance(status_board_payload.get("readiness"), dict) else {}
            record = {
                "label": label,
                "exit_code": exit_code,
                "horizon_days": horizon_days,
                "max_passes": max_passes,
                "wrapper_status": wrapper_payload.get("status"),
                "wrapper_current_phase": wrapper_payload.get("current_phase"),
                "wrapper_recommended_action": wrapper_payload.get("recommended_action"),
                "run_context": wrapper_payload.get("run_context"),
                "capture_loop_status": capture_payload.get("status"),
                "capture_loop_current_phase": capture_payload.get("current_phase"),
                "pre_race_only_rows": latest_summary.get("pre_race_only_rows"),
                "pre_race_only_races": latest_summary.get("pre_race_only_races"),
                "result_ready_races": latest_summary.get("result_ready_races"),
                "pending_result_races": latest_summary.get("pending_result_races"),
                "delta_pre_race_only_rows": latest_summary.get("baseline_comparison", {}).get("delta_pre_race_only_rows") if isinstance(latest_summary.get("baseline_comparison"), dict) else None,
                "status_board_phase": status_board_payload.get("current_phase"),
                "status_board_action": status_board_payload.get("recommended_action"),
                "benchmark_rerun_ready": readiness_payload.get("benchmark_rerun_ready"),
                "readiness_reasons": readiness_payload.get("reasons"),
                "artifacts": {
                    "wrapper_manifest": wrapper_manifest,
                    "capture_loop_manifest": capture_manifest,
                    "status_board_manifest": status_board_manifest,
                },
            }
            records.append(record)
            progress.update(current=index, message=f"scenario={label} rows={record['pre_race_only_rows']} pending={record['pending_result_races']}")

        output_payload = {
            "status": "completed",
            "current_phase": "scenario_matrix_completed",
            "recommended_action": "review_probe_scenarios",
            "read_order": [
                "status",
                "current_phase",
                "recommended_action",
                "scenario_count",
                "scenarios[0].wrapper_status",
                "scenarios[0].pending_result_races",
            ],
            "scenario_count": len(records),
            "scenarios": records,
        }
        write_json(output_path, _normalize_display_paths(output_payload))
        progress.complete(message=f"probe summary output={output_path}")
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-future-probe] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-future-probe] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())