from __future__ import annotations

import argparse
from datetime import date, timedelta
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

from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-future-readiness {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[local-nankan-future-readiness] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[local-nankan-future-readiness]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _resolve_capture_window(*, start_date_text: str | None, end_date_text: str | None, horizon_days: int) -> tuple[str, str]:
    if start_date_text and end_date_text:
        return start_date_text, end_date_text
    today = date.today()
    start_date_value = start_date_text or today.isoformat()
    end_date_value = end_date_text or (today + timedelta(days=max(0, int(horizon_days)))).isoformat()
    return start_date_value, end_date_value


def _default_bootstrap_log_prefix(bootstrap_manifest_output: str) -> str:
    return Path(bootstrap_manifest_output).stem


def _default_bootstrap_revision(bootstrap_manifest_output: str) -> str:
    return Path(bootstrap_manifest_output).stem


def _capture_snapshot_dir(capture_loop_manifest_output: str) -> str:
    manifest_path = Path(capture_loop_manifest_output)
    base_name = manifest_path.stem
    if base_name.endswith("_pre_race_capture_loop"):
        base_name = base_name[: -len("_pre_race_capture_loop")]
    elif base_name.endswith("_capture_loop"):
        base_name = base_name[: -len("_capture_loop")]
    elif base_name.endswith("_manifest"):
        base_name = base_name[: -len("_manifest")]
    return str(manifest_path.parent / f"{base_name}_pre_race_capture_snapshots")


def _probe_summary_output(watcher_manifest_output: str) -> str:
    manifest_path = Path(watcher_manifest_output)
    base_name = manifest_path.stem
    if base_name.endswith("_readiness_watcher"):
        base_name = base_name[: -len("_readiness_watcher")]
    elif base_name.endswith("_watcher"):
        base_name = base_name[: -len("_watcher")]
    elif base_name.endswith("_manifest"):
        base_name = base_name[: -len("_manifest")]
    return str(manifest_path.parent / f"{base_name}_readiness_probe_summary.json")


def _bootstrap_cycle_artifacts(bootstrap_manifest_output: str) -> dict[str, str]:
    manifest_path = Path(bootstrap_manifest_output)
    base_name = manifest_path.stem
    if base_name.endswith("_bootstrap_handoff"):
        base_name = base_name[: -len("_bootstrap_handoff")]
    elif base_name.endswith("_bootstrap_manifest"):
        base_name = base_name[: -len("_bootstrap_manifest")]
    elif base_name.endswith("_manifest"):
        base_name = base_name[: -len("_manifest")]
    parent = manifest_path.parent
    raw_parent = ROOT / "data" / "local_nankan_pre_race_ready" / "raw"
    return {
        "handoff_manifest_output": str(parent / f"{base_name}_pre_race_benchmark_handoff.json"),
        "pre_race_summary_output": str(parent / f"{base_name}_pre_race_ready_summary.json"),
        "primary_manifest_file": str(parent / f"{base_name}_pre_race_ready_primary_materialize.json"),
        "benchmark_manifest_output": str(parent / f"{base_name}_pre_race_ready_benchmark_gate.json"),
        "filtered_race_card_output": str(raw_parent / f"{base_name}_race_card.csv"),
        "filtered_race_result_output": str(raw_parent / f"{base_name}_race_result.csv"),
        "primary_output_file": str(raw_parent / f"{base_name}_primary.csv"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one local Nankan future-only readiness evaluation cycle. This wrapper "
            "updates readiness surfaces and does not ingest external data by itself."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--capture-loop-script", default="scripts/run_local_nankan_pre_race_capture_loop.py")
    parser.add_argument("--watcher-script", default="scripts/run_local_nankan_readiness_watcher.py")
    parser.add_argument("--bootstrap-handoff-script", default="scripts/run_local_nankan_result_ready_bootstrap_handoff.py")
    parser.add_argument("--status-board-script", default="scripts/run_local_nankan_status_board.py")
    parser.add_argument("--source-timing-summary-input", default="artifacts/reports/local_nankan_source_timing_audit_issue121.json")
    parser.add_argument("--capture-loop-manifest-output", default="artifacts/reports/local_nankan_pre_race_capture_loop_issue122_cycle.json")
    parser.add_argument("--watcher-manifest-output", default="artifacts/reports/local_nankan_readiness_watcher_issue122_cycle.json")
    parser.add_argument("--bootstrap-manifest-output", default="artifacts/reports/local_nankan_result_ready_bootstrap_handoff_issue122.json")
    parser.add_argument("--status-board-output", default="artifacts/reports/local_nankan_data_status_board_issue122_cycle.json")
    parser.add_argument("--wrapper-manifest-output", default="artifacts/reports/local_nankan_future_only_readiness_cycle_issue122.json")
    parser.add_argument("--max-passes", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=int, default=1)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--default-horizon-days", type=int, default=7)
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--bootstrap-log-prefix", default=None)
    parser.add_argument("--bootstrap-revision", default=None)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)
    wrapper_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=4, prefix="[local-nankan-future-readiness]", logger=log_progress, min_interval_sec=0.0)

    run_started_at = utc_now_iso()

    try:
        python_executable = str(_resolve_path(args.python_executable)) if Path(args.python_executable).is_absolute() else args.python_executable
        start_date_text, end_date_text = _resolve_capture_window(
            start_date_text=args.start_date,
            end_date_text=args.end_date,
            horizon_days=args.default_horizon_days,
        )
        progress.start(message="starting future-only readiness cycle")

        capture_command = [
            python_executable,
            str(_resolve_path(args.capture_loop_script)),
            "--wrapper-manifest-output",
            args.capture_loop_manifest_output,
            "--snapshot-dir",
            _capture_snapshot_dir(args.capture_loop_manifest_output),
            "--max-passes",
            str(args.max_passes),
            "--poll-interval-seconds",
            str(args.poll_interval_seconds),
        ]
        capture_command.extend(["--start-date", start_date_text, "--end-date", end_date_text])
        if args.include_completed:
            capture_command.append("--include-completed")
        capture_command.append("--overwrite" if args.overwrite else "--no-overwrite")

        capture_exit = _run_command(label="capture_loop", command=capture_command)
        capture_manifest = _read_json_dict(_resolve_path(args.capture_loop_manifest_output))
        progress.update(current=1, message=f"capture_loop exit_code={capture_exit} status={capture_manifest.get('status')}")

        watcher_command = [
            python_executable,
            str(_resolve_path(args.watcher_script)),
            "--probe-summary-output",
            _probe_summary_output(args.watcher_manifest_output),
            "--source-timing-summary-input",
            args.source_timing_summary_input,
            "--watcher-manifest-output",
            args.watcher_manifest_output,
        ]
        watcher_exit = _run_command(label="readiness_watcher", command=watcher_command)
        watcher_manifest = _read_json_dict(_resolve_path(args.watcher_manifest_output))
        progress.update(current=2, message=f"watcher exit_code={watcher_exit} phase={watcher_manifest.get('current_phase')}")

        bootstrap_artifacts = _bootstrap_cycle_artifacts(args.bootstrap_manifest_output)
        bootstrap_command = [
            python_executable,
            str(_resolve_path(args.bootstrap_handoff_script)),
            "--source-timing-summary-input",
            args.source_timing_summary_input,
            "--wrapper-manifest-output",
            args.bootstrap_manifest_output,
            "--handoff-manifest-output",
            bootstrap_artifacts["handoff_manifest_output"],
            "--filtered-race-card-output",
            bootstrap_artifacts["filtered_race_card_output"],
            "--filtered-race-result-output",
            bootstrap_artifacts["filtered_race_result_output"],
            "--primary-output-file",
            bootstrap_artifacts["primary_output_file"],
            "--pre-race-summary-output",
            bootstrap_artifacts["pre_race_summary_output"],
            "--primary-manifest-file",
            bootstrap_artifacts["primary_manifest_file"],
            "--benchmark-manifest-output",
            bootstrap_artifacts["benchmark_manifest_output"],
            "--log-prefix",
            args.bootstrap_log_prefix or _default_bootstrap_log_prefix(args.bootstrap_manifest_output),
            "--bootstrap-revision",
            args.bootstrap_revision or _default_bootstrap_revision(args.bootstrap_manifest_output),
        ]
        bootstrap_exit = _run_command(label="bootstrap_handoff", command=bootstrap_command)
        bootstrap_manifest = _read_json_dict(_resolve_path(args.bootstrap_manifest_output))
        progress.update(current=3, message=f"bootstrap exit_code={bootstrap_exit} phase={bootstrap_manifest.get('current_phase')}")

        board_command = [
            python_executable,
            str(_resolve_path(args.status_board_script)),
            "--capture-loop-manifest",
            args.capture_loop_manifest_output,
            "--readiness-probe-summary",
            _probe_summary_output(args.watcher_manifest_output),
            "--pre-race-handoff-manifest",
            bootstrap_artifacts["handoff_manifest_output"],
            "--readiness-watcher-manifest",
            args.watcher_manifest_output,
            "--bootstrap-handoff-manifest",
            args.bootstrap_manifest_output,
            "--output",
            args.status_board_output,
        ]
        board_exit = _run_command(label="status_board", command=board_command)
        status_board = _read_json_dict(_resolve_path(args.status_board_output))
        progress.update(current=4, message=f"status_board exit_code={board_exit} phase={status_board.get('current_phase')}")

        wrapper_manifest = {
            "started_at": run_started_at,
            "finished_at": utc_now_iso(),
            "status": str(status_board.get("status") or ("failed" if board_exit != 0 else "completed")),
            "current_phase": str(status_board.get("current_phase") or watcher_manifest.get("current_phase") or capture_manifest.get("current_phase") or "future_only_readiness_track"),
            "recommended_action": str(status_board.get("recommended_action") or watcher_manifest.get("recommended_action") or capture_manifest.get("recommended_action") or "capture_future_pre_race_rows_and_wait_for_results"),
            "execution_role": "readiness_cycle_wrapper",
            "data_update_mode": "capture_refresh_with_readiness",
            "execution_mode": "single_cycle",
            "trigger_contract": "direct_refresh_plus_readiness",
            "artifacts": {
                "capture_loop_manifest": args.capture_loop_manifest_output,
                "capture_snapshot_dir": _capture_snapshot_dir(args.capture_loop_manifest_output),
                "readiness_probe_summary": _probe_summary_output(args.watcher_manifest_output),
                "watcher_manifest": args.watcher_manifest_output,
                "bootstrap_manifest": args.bootstrap_manifest_output,
                "bootstrap_handoff_manifest": bootstrap_artifacts["handoff_manifest_output"],
                "bootstrap_filtered_race_card_output": bootstrap_artifacts["filtered_race_card_output"],
                "bootstrap_filtered_race_result_output": bootstrap_artifacts["filtered_race_result_output"],
                "bootstrap_primary_output_file": bootstrap_artifacts["primary_output_file"],
                "bootstrap_pre_race_summary": bootstrap_artifacts["pre_race_summary_output"],
                "bootstrap_primary_manifest": bootstrap_artifacts["primary_manifest_file"],
                "bootstrap_benchmark_manifest": bootstrap_artifacts["benchmark_manifest_output"],
                "status_board": args.status_board_output,
            },
            "run_context": {
                "start_date": start_date_text,
                "end_date": end_date_text,
                "default_horizon_days": int(args.default_horizon_days),
                "max_passes": int(args.max_passes),
            },
            "steps": {
                "capture_loop": {"exit_code": capture_exit, "manifest": capture_manifest},
                "readiness_watcher": {"exit_code": watcher_exit, "manifest": watcher_manifest},
                "bootstrap_handoff": {"exit_code": bootstrap_exit, "manifest": bootstrap_manifest},
                "status_board": {"exit_code": board_exit, "manifest": status_board},
            },
        }
        write_json(wrapper_manifest_path, wrapper_manifest)
        progress.complete(message=f"future-only readiness cycle output={wrapper_manifest_path}")

        if board_exit != 0:
            return int(board_exit)
        if capture_exit != 0:
            return int(capture_exit)
        if watcher_exit not in {0, 2}:
            return int(watcher_exit)
        if bootstrap_exit not in {0, 2}:
            return int(bootstrap_exit)
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-future-readiness] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-future-readiness] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())