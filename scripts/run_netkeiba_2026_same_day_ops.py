from __future__ import annotations

import argparse
import json
from pathlib import Path
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


DEFAULT_STATUS_BOARD_SCRIPT = "scripts/run_netkeiba_2026_status_board.py"
DEFAULT_BACKFILL_SCRIPT = "scripts/run_netkeiba_2026_ytd_backfill.py"
DEFAULT_HANDOFF_SCRIPT = "scripts/run_netkeiba_2026_live_handoff.py"
DEFAULT_ROLLOVER_SCRIPT = "scripts/run_netkeiba_2026_backfill_rollover.py"
DEFAULT_STATUS_BOARD = "artifacts/reports/netkeiba_2026_status_board.json"
DEFAULT_HANDOFF_MANIFEST = "artifacts/reports/netkeiba_2026_live_handoff_manifest.json"
DEFAULT_OUTPUT = "artifacts/reports/netkeiba_2026_same_day_ops_manifest.json"
DEFAULT_LOG_DIR = "artifacts/logs"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-2026-same-day-ops {now}] {message}", flush=True)


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


def _refresh_status_board(*, python_executable: str, status_board_script: str) -> int:
    command = [python_executable, str(_resolve_path(status_board_script))]
    print(f"[netkeiba-2026-same-day-ops] refreshing board: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _find_running_processes(needle: str) -> list[dict[str, Any]]:
    result = subprocess.run(["ps", "-eo", "pid=,args="], cwd=ROOT, capture_output=True, text=True, check=False)
    matches: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if not line or needle not in line:
            continue
        parts = line.split(maxsplit=1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        args = parts[1] if len(parts) > 1 else ""
        matches.append({"pid": pid, "args": args})
    return matches


def _launch_background(command: list[str], *, log_file: Path) -> dict[str, Any]:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handle = log_file.open("ab")
    try:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    finally:
        handle.close()
    return {
        "status": "started",
        "pid": int(process.pid),
        "command": command,
        "log_file": str(log_file.relative_to(ROOT)),
        "started_at": utc_now_iso(),
    }


def _timestamped_log(log_dir: Path, prefix: str) -> Path:
    return log_dir / f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}.log"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--race-date", required=True)
    parser.add_argument("--headline-contains", default=None)
    parser.add_argument("--poll-interval-seconds", type=int, default=300)
    parser.add_argument("--status-board-script", default=DEFAULT_STATUS_BOARD_SCRIPT)
    parser.add_argument("--backfill-script", default=DEFAULT_BACKFILL_SCRIPT)
    parser.add_argument("--handoff-script", default=DEFAULT_HANDOFF_SCRIPT)
    parser.add_argument("--rollover-script", default=DEFAULT_ROLLOVER_SCRIPT)
    parser.add_argument("--status-board", default=DEFAULT_STATUS_BOARD)
    parser.add_argument("--handoff-manifest", default=DEFAULT_HANDOFF_MANIFEST)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
    log_dir = _resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    progress = ProgressBar(total=4, prefix="[netkeiba-2026-same-day-ops]", logger=log_progress, min_interval_sec=0.0)

    try:
        progress.start(message=f"preparing same-day ops race_date={args.race_date}")
        board_exit = _refresh_status_board(
            python_executable=str(args.python_executable),
            status_board_script=args.status_board_script,
        )
        board_payload = _read_json_dict(_resolve_path(args.status_board))
        handoff_payload = _read_json_dict(_resolve_path(args.handoff_manifest))
        progress.update(current=1, message=f"board_refreshed exit_code={board_exit} board_status={board_payload.get('status')}")

        actions: dict[str, Any] = {}
        completed = (
            str(board_payload.get("status") or "") in {"completed", "handed_off"}
            and str(_read_json_dict(_resolve_path(args.handoff_manifest)).get("status") or "") == "completed"
        )
        if completed:
            payload = {
                "status": "completed",
                "current_phase": "already_completed",
                "recommended_action": "review_live_prediction_outputs",
                "observed_at": utc_now_iso(),
                "race_date": args.race_date,
                "headline_contains": args.headline_contains,
                "dry_run": bool(args.dry_run),
                "actions": actions,
                "board": board_payload,
                "handoff": handoff_payload,
                "highlights": [
                    "status=completed",
                    "same-day ops already completed",
                    f"prediction_file={_dict_or_empty(board_payload.get('live_outputs')).get('prediction_file')}",
                ],
            }
            write_json(output_path, payload)
            progress.complete(message="same-day ops already completed")
            return 0

        backfill_processes = _find_running_processes("scripts/run_netkeiba_2026_ytd_backfill.py")
        handoff_processes = _find_running_processes("scripts/run_netkeiba_2026_live_handoff.py")
        rollover_processes = _find_running_processes("scripts/run_netkeiba_2026_backfill_rollover.py")

        if backfill_processes:
            actions["backfill"] = {"status": "already_running", "processes": backfill_processes}
        elif args.dry_run:
            actions["backfill"] = {"status": "would_start"}
        else:
            actions["backfill"] = _launch_background(
                [str(args.python_executable), str(_resolve_path(args.backfill_script))],
                log_file=_timestamped_log(log_dir, "netkeiba_backfill_2026_ytd"),
            )

        handoff_completed = (
            str(handoff_payload.get("status") or "") == "completed"
            and str(handoff_payload.get("race_date") or "") == args.race_date
        )
        if handoff_completed:
            actions["handoff"] = {"status": "already_completed", "manifest": args.handoff_manifest}
        elif handoff_processes:
            actions["handoff"] = {"status": "already_running", "processes": handoff_processes}
        elif args.dry_run:
            actions["handoff"] = {"status": "would_start"}
        else:
            handoff_command = [
                str(args.python_executable),
                str(_resolve_path(args.handoff_script)),
                "--race-date",
                args.race_date,
                "--wait-for-ready",
                "--poll-interval-seconds",
                str(args.poll_interval_seconds),
            ]
            if args.headline_contains:
                handoff_command.extend(["--headline-contains", args.headline_contains])
            actions["handoff"] = _launch_background(
                handoff_command,
                log_file=_timestamped_log(log_dir, "netkeiba_2026_live_handoff"),
            )

        backfill_needs_rollover = bool(backfill_processes)
        if not backfill_needs_rollover:
            actions["rollover"] = {"status": "not_needed"}
        elif rollover_processes:
            actions["rollover"] = {"status": "already_running", "processes": rollover_processes}
        elif args.dry_run:
            actions["rollover"] = {"status": "would_start"}
        else:
            actions["rollover"] = _launch_background(
                [str(args.python_executable), str(_resolve_path(args.rollover_script))],
                log_file=_timestamped_log(log_dir, "netkeiba_2026_backfill_rollover"),
            )

        progress.update(current=3, message="background action decisions recorded")

        board_exit = _refresh_status_board(
            python_executable=str(args.python_executable),
            status_board_script=args.status_board_script,
        )
        board_payload = _read_json_dict(_resolve_path(args.status_board))
        payload = {
            "status": "running" if not args.dry_run else "dry_run",
            "current_phase": "ops_armed" if not args.dry_run else "ops_plan_ready",
            "recommended_action": "monitor_status_board",
            "observed_at": utc_now_iso(),
            "race_date": args.race_date,
            "headline_contains": args.headline_contains,
            "dry_run": bool(args.dry_run),
            "status_board_refresh_exit_code": board_exit,
            "actions": actions,
            "board": board_payload,
            "highlights": [
                f"backfill={actions.get('backfill', {}).get('status')}",
                f"handoff={actions.get('handoff', {}).get('status')}",
                f"rollover={actions.get('rollover', {}).get('status')}",
                f"board_status={board_payload.get('status')}",
            ],
        }
        write_json(output_path, payload)
        progress.complete(message=f"ops manifest ready path={args.output}")
        return 0
    except KeyboardInterrupt:
        print("[netkeiba-2026-same-day-ops] interrupted by user")
        return 130
    except Exception as error:
        print(f"[netkeiba-2026-same-day-ops] failed: {error}")
        traceback.print_exc()
        return 1


def _dict_or_empty(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


if __name__ == "__main__":
    raise SystemExit(main())