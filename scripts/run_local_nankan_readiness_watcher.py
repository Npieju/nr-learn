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
from racing_ml.data.local_nankan_bootstrap import resolve_python_executable
from racing_ml.data.local_nankan_watch import build_readiness_watcher_manifest, should_trigger_handoff


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-watcher {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[local-nankan-watcher] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[local-nankan-watcher]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


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
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--probe-script", default="scripts/run_local_nankan_pre_race_readiness_probe.py")
    parser.add_argument("--handoff-script", default="scripts/run_local_nankan_result_ready_bootstrap_handoff.py")
    parser.add_argument("--probe-summary-output", default="artifacts/reports/local_nankan_pre_race_readiness_probe_summary.json")
    parser.add_argument("--capture-loop-manifest", default="artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json")
    parser.add_argument("--handoff-manifest-output", default="artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json")
    parser.add_argument("--watcher-manifest-output", default="artifacts/reports/local_nankan_readiness_watcher_manifest.json")
    parser.add_argument("--source-timing-summary-input", default="artifacts/reports/local_nankan_source_timing_audit.json")
    parser.add_argument("--wait-for-ready", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--run-bootstrap", action="store_true")
    args = parser.parse_args()

    watcher_manifest_path = _resolve_path(args.watcher_manifest_output)
    watcher_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=3, prefix="[local-nankan-watcher]", logger=log_progress, min_interval_sec=0.0)

    try:
        python_executable = resolve_python_executable(workspace_root=ROOT, fallback=args.python_executable or sys.executable)
        probe_command = [
            python_executable,
            str(_resolve_path(args.probe_script)),
            "--summary-output",
            args.probe_summary_output,
            "--source-timing-summary-input",
            str(_resolve_path(args.source_timing_summary_input)),
        ]
        handoff_command = [
            python_executable,
            str(_resolve_path(args.handoff_script)),
            "--source-timing-summary-input",
            str(_resolve_path(args.source_timing_summary_input)),
        ]
        if args.run_bootstrap:
            handoff_command.append("--run-bootstrap")

        wait_started = time.monotonic()
        attempts = 0
        progress.start(message="starting readiness watcher")

        while True:
            attempts += 1
            probe_exit = _run_command(label=f"probe attempt={attempts}", command=probe_command)
            probe_summary = _read_json_dict(_resolve_path(args.probe_summary_output))
            capture_loop_manifest = _read_json_dict(_resolve_path(args.capture_loop_manifest))
            progress.update(
                current=1,
                message=f"probe attempt={attempts} exit_code={probe_exit} status={probe_summary.get('status')}",
            )

            if should_trigger_handoff(probe_summary):
                handoff_exit = _run_command(label="result_ready_handoff", command=handoff_command)
                handoff_manifest = _read_json_dict(_resolve_path(args.handoff_manifest_output))
                progress.update(current=2, message=f"handoff exit_code={handoff_exit} status={handoff_manifest.get('status')}")
                manifest = build_readiness_watcher_manifest(
                    status="completed" if handoff_exit == 0 else "failed",
                    current_phase="handoff_completed" if handoff_exit == 0 else "handoff_failed",
                    recommended_action="review_handoff_outputs" if handoff_exit == 0 else "inspect_handoff_manifest",
                    attempts=attempts,
                    waited_seconds=int(max(0, time.monotonic() - wait_started)),
                    timed_out=False,
                    probe_summary_output=_display_path(_resolve_path(args.probe_summary_output)),
                    probe_summary=_normalize_display_paths(probe_summary),
                    capture_loop_manifest_output=_display_path(_resolve_path(args.capture_loop_manifest)),
                    capture_loop_manifest=_normalize_display_paths(capture_loop_manifest),
                    handoff_manifest=_normalize_display_paths(handoff_manifest),
                )
                write_json(watcher_manifest_path, manifest)
                progress.complete(message=f"watcher completed output={watcher_manifest_path}")
                return 0 if handoff_exit == 0 else 1

            waited_seconds = int(max(0, time.monotonic() - wait_started))
            timed_out = (not args.wait_for_ready) or (args.max_wait_seconds >= 0 and waited_seconds >= max(0, args.max_wait_seconds))
            if timed_out:
                manifest = build_readiness_watcher_manifest(
                    status="not_ready",
                    current_phase=str(probe_summary.get("current_phase") or "await_result_arrival"),
                    recommended_action=str(
                        probe_summary.get("recommended_action")
                        or "wait_for_result_ready_pre_race_races"
                    ),
                    attempts=attempts,
                    waited_seconds=waited_seconds,
                    timed_out=True,
                    probe_summary_output=_display_path(_resolve_path(args.probe_summary_output)),
                    probe_summary=_normalize_display_paths(probe_summary),
                    capture_loop_manifest_output=_display_path(_resolve_path(args.capture_loop_manifest)),
                    capture_loop_manifest=_normalize_display_paths(capture_loop_manifest),
                )
                write_json(watcher_manifest_path, manifest)
                progress.complete(message=f"not ready output={watcher_manifest_path}")
                return 2

            sleep_seconds = max(1, int(args.poll_interval_seconds))
            print(f"[local-nankan-watcher] waiting for readiness sleep_seconds={sleep_seconds}", flush=True)
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("[local-nankan-watcher] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-watcher] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
