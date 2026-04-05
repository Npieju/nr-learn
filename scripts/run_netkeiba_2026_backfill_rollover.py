from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_BACKFILL_SCRIPT = "scripts/run_netkeiba_2026_ytd_backfill.py"
DEFAULT_BACKFILL_MANIFEST = "artifacts/reports/netkeiba_backfill_manifest_2026_ytd.json"
DEFAULT_RACE_RESULT_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_race_result.json"
DEFAULT_RACE_CARD_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_race_card.json"
DEFAULT_PEDIGREE_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_pedigree.json"
DEFAULT_CRAWL_LOCK = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd.json.lock"
DEFAULT_OUTPUT = "artifacts/reports/netkeiba_2026_backfill_rollover_manifest.json"
DEFAULT_LOG_DIR = "artifacts/logs"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-2026-rollover {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _pid_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _target_summary(path: Path) -> dict[str, Any]:
    payload = _read_json_dict(path)
    return {
        "manifest": str(path.relative_to(ROOT)) if path.is_absolute() else str(path),
        "status": payload.get("status"),
        "requested_ids": payload.get("requested_ids"),
        "processed_ids": payload.get("processed_ids"),
        "parsed_ids": payload.get("parsed_ids"),
        "rows_written": payload.get("rows_written"),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
    }


def _read_lock_pid(lock_path: Path) -> tuple[int | None, dict[str, Any]]:
    payload = _read_json_dict(lock_path)
    raw_pid = payload.get("pid")
    try:
        pid = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        pid = None
    return pid, payload


def _any_running(targets: dict[str, dict[str, Any]]) -> bool:
    return any(str(payload.get("status") or "") == "running" for payload in targets.values())


def _build_manifest(
    *,
    status: str,
    current_phase: str,
    recommended_action: str,
    waited_seconds: int,
    active_pid: int | None,
    active_pid_running: bool,
    lock_payload: dict[str, Any],
    backfill_manifest_path: Path,
    targets: dict[str, dict[str, Any]],
    safe_to_restart: bool,
    restart_command: list[str],
    restart_log_file: str | None = None,
    restarted_pid: int | None = None,
    terminated_at: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    highlights = [
        f"status={status}",
        f"current_phase={current_phase}",
        f"recommended_action={recommended_action}",
        f"active_pid={active_pid}",
        f"safe_to_restart={safe_to_restart}",
    ]
    for name, payload in targets.items():
        highlights.append(f"{name}={payload.get('status')}:{payload.get('processed_ids')}/{payload.get('requested_ids')}")
    if restart_log_file:
        highlights.append(f"restart_log={restart_log_file}")
    if error:
        highlights.append(error)
    return {
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "observed_at": utc_now_iso(),
        "waited_seconds": waited_seconds,
        "active_pid": active_pid,
        "active_pid_running": active_pid_running,
        "lock": lock_payload,
        "backfill_manifest": str(backfill_manifest_path.relative_to(ROOT)),
        "targets": targets,
        "safe_to_restart": safe_to_restart,
        "restart_command": restart_command,
        "restart_log_file": restart_log_file,
        "restarted_pid": restarted_pid,
        "terminated_at": terminated_at,
        "error": error,
        "highlights": highlights,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--backfill-script", default=DEFAULT_BACKFILL_SCRIPT)
    parser.add_argument("--backfill-manifest", default=DEFAULT_BACKFILL_MANIFEST)
    parser.add_argument("--race-result-manifest", default=DEFAULT_RACE_RESULT_MANIFEST)
    parser.add_argument("--race-card-manifest", default=DEFAULT_RACE_CARD_MANIFEST)
    parser.add_argument("--pedigree-manifest", default=DEFAULT_PEDIGREE_MANIFEST)
    parser.add_argument("--crawl-lock-path", default=DEFAULT_CRAWL_LOCK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--poll-interval-seconds", type=int, default=15)
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--terminate-timeout-seconds", type=int, default=60)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
    backfill_manifest_path = _resolve_path(args.backfill_manifest)
    race_result_manifest_path = _resolve_path(args.race_result_manifest)
    race_card_manifest_path = _resolve_path(args.race_card_manifest)
    pedigree_manifest_path = _resolve_path(args.pedigree_manifest)
    lock_path = _resolve_path(args.crawl_lock_path)
    log_dir = _resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    restart_command = [str(args.python_executable), str(_resolve_path(args.backfill_script))]
    progress = ProgressBar(total=3, prefix="[netkeiba-2026-rollover]", logger=log_progress, min_interval_sec=0.0)
    wait_started = time.monotonic()

    try:
        progress.start(message="waiting for safe cycle boundary")
        while True:
            pid, lock_payload = _read_lock_pid(lock_path)
            active_pid_running = _pid_running(pid)
            targets = {
                "race_result": _target_summary(race_result_manifest_path),
                "race_card": _target_summary(race_card_manifest_path),
                "pedigree": _target_summary(pedigree_manifest_path),
            }
            safe_to_restart = active_pid_running and not _any_running(targets)
            waited_seconds = int(max(0, time.monotonic() - wait_started))
            manifest = _build_manifest(
                status="waiting",
                current_phase="wait_cycle_boundary",
                recommended_action="continue_waiting",
                waited_seconds=waited_seconds,
                active_pid=pid,
                active_pid_running=active_pid_running,
                lock_payload=lock_payload,
                backfill_manifest_path=backfill_manifest_path,
                targets=targets,
                safe_to_restart=safe_to_restart,
                restart_command=restart_command,
            )
            write_json(output_path, manifest)
            progress.update(current=1, message=f"safe_to_restart={safe_to_restart} pid={pid}")

            if safe_to_restart:
                if args.dry_run:
                    write_json(
                        output_path,
                        _build_manifest(
                            status="dry_run_ready",
                            current_phase="ready_to_restart",
                            recommended_action="launch_rollover_without_dry_run",
                            waited_seconds=waited_seconds,
                            active_pid=pid,
                            active_pid_running=active_pid_running,
                            lock_payload=lock_payload,
                            backfill_manifest_path=backfill_manifest_path,
                            targets=targets,
                            safe_to_restart=True,
                            restart_command=restart_command,
                        ),
                    )
                    progress.complete(message="dry-run reached safe cycle boundary")
                    return 0

                os.kill(pid, signal.SIGINT)
                terminated_at = utc_now_iso()
                progress.update(current=2, message=f"sent SIGINT pid={pid}")

                terminate_started = time.monotonic()
                while _pid_running(pid):
                    if time.monotonic() - terminate_started > max(int(args.terminate_timeout_seconds), 1):
                        write_json(
                            output_path,
                            _build_manifest(
                                status="terminate_timeout",
                                current_phase="wait_old_backfill_exit",
                                recommended_action="inspect_backfill_process",
                                waited_seconds=int(max(0, time.monotonic() - wait_started)),
                                active_pid=pid,
                                active_pid_running=True,
                                lock_payload=lock_payload,
                                backfill_manifest_path=backfill_manifest_path,
                                targets=targets,
                                safe_to_restart=True,
                                restart_command=restart_command,
                                terminated_at=terminated_at,
                                error="backfill did not exit before terminate timeout",
                            ),
                        )
                        return 1
                    time.sleep(1)

                restart_log_file = log_dir / f"netkeiba_2026_ytd_backfill_rollover_{time.strftime('%Y%m%d_%H%M%S')}.log"
                with restart_log_file.open("ab") as handle:
                    process = subprocess.Popen(
                        restart_command,
                        cwd=ROOT,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )

                write_json(
                    output_path,
                    _build_manifest(
                        status="restarted",
                        current_phase="backfill_restarted",
                        recommended_action="monitor_new_backfill_log",
                        waited_seconds=int(max(0, time.monotonic() - wait_started)),
                        active_pid=pid,
                        active_pid_running=False,
                        lock_payload=lock_payload,
                        backfill_manifest_path=backfill_manifest_path,
                        targets=targets,
                        safe_to_restart=True,
                        restart_command=restart_command,
                        restart_log_file=str(restart_log_file.relative_to(ROOT)),
                        restarted_pid=int(process.pid),
                        terminated_at=terminated_at,
                    ),
                )
                progress.complete(message=f"restarted backfill pid={process.pid}")
                return 0

            if args.max_wait_seconds and waited_seconds >= int(args.max_wait_seconds):
                write_json(
                    output_path,
                    _build_manifest(
                        status="timeout",
                        current_phase="wait_cycle_boundary",
                        recommended_action="re-run_rollover_or_restart_manually",
                        waited_seconds=waited_seconds,
                        active_pid=pid,
                        active_pid_running=active_pid_running,
                        lock_payload=lock_payload,
                        backfill_manifest_path=backfill_manifest_path,
                        targets=targets,
                        safe_to_restart=False,
                        restart_command=restart_command,
                        error="max wait exceeded before safe cycle boundary",
                    ),
                )
                return 1

            time.sleep(max(int(args.poll_interval_seconds), 1))
    except KeyboardInterrupt:
        print("[netkeiba-2026-rollover] interrupted by user")
        return 130
    except Exception as error:
        print(f"[netkeiba-2026-rollover] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())