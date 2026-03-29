from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _pid_is_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def _refresh_status_board(status_command: list[str]) -> None:
    _run(status_command)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-file", default="artifacts/reports/local_nankan_backfill_race_card_20y.json")
    parser.add_argument("--status-board-output", default="artifacts/reports/local_nankan_data_status_board.json")
    parser.add_argument("--poll-sec", type=float, default=30.0)
    parser.add_argument("--max-windows", type=int, default=None)
    parser.add_argument("--crawl-config", default="configs/crawl_local_nankan_template.yaml")
    parser.add_argument("--data-config", default="configs/data_local_nankan.yaml")
    parser.add_argument("--target", default="race_card")
    parser.add_argument("--race-id-source", default="race_list")
    parser.add_argument("--start-date", default="2006-03-29")
    parser.add_argument("--end-date", default="2026-03-28")
    parser.add_argument("--date-order", default="asc")
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--coverage-output", default="artifacts/reports/coverage_snapshot_local_nankan_current.json")
    args = parser.parse_args()

    manifest_path = ROOT / args.manifest_file if not Path(args.manifest_file).is_absolute() else Path(args.manifest_file)
    status_command = [
        sys.executable,
        str(ROOT / "scripts/run_local_nankan_status_board.py"),
        "--coverage-snapshot",
        args.coverage_output,
        "--backfill-aggregate",
        args.manifest_file,
        "--output",
        args.status_board_output,
    ]
    next_window_command = [
        sys.executable,
        str(ROOT / "scripts/run_local_nankan_next_window.py"),
        "--crawl-config",
        args.crawl_config,
        "--data-config",
        args.data_config,
        "--target",
        args.target,
        "--race-id-source",
        args.race_id_source,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--date-order",
        args.date_order,
        "--chunk-months",
        str(args.chunk_months),
        "--max-cycles",
        str(args.max_cycles),
        "--manifest-file",
        args.manifest_file,
        "--coverage-output",
        args.coverage_output,
        "--status-board-output",
        args.status_board_output,
        "--backfill-aggregate",
        args.manifest_file,
    ]

    launched_windows = 0
    print(f"[local-nankan-window-queue] started manifest={args.manifest_file} poll_sec={args.poll_sec}", flush=True)
    while True:
        payload = _read_json(manifest_path)
        status = str(payload.get("status") or "")
        requested_window_count = _optional_int(payload.get("requested_window_count")) or 0
        completed_window_count = _optional_int(payload.get("completed_window_count")) or 0
        active_window = _optional_int(payload.get("active_window"))
        pid = _optional_int(payload.get("pid"))
        pid_running = _pid_is_running(pid)
        remaining_window_count = _optional_int(payload.get("remaining_window_count"))

        _refresh_status_board(status_command)
        print(
            f"[local-nankan-window-queue] status={status or 'missing'} completed={completed_window_count}/{requested_window_count} remaining={remaining_window_count} active_window={active_window} pid_running={pid_running}",
            flush=True,
        )

        if requested_window_count > 0 and completed_window_count >= requested_window_count:
            print("[local-nankan-window-queue] all windows completed", flush=True)
            return 0

        if args.max_windows is not None and launched_windows >= args.max_windows:
            print(f"[local-nankan-window-queue] reached max_windows={args.max_windows}", flush=True)
            return 0

        if status == "running" and pid_running:
            time.sleep(args.poll_sec)
            continue

        if status == "failed":
            print("[local-nankan-window-queue] stop on failed status; inspect manifests", flush=True)
            return 2

        if remaining_window_count is not None and remaining_window_count <= 0:
            print("[local-nankan-window-queue] no remaining windows", flush=True)
            return 0

        print("[local-nankan-window-queue] launching next window", flush=True)
        exit_code = _run(next_window_command)
        launched_windows += 1
        print(f"[local-nankan-window-queue] next_window_exit_code={exit_code}", flush=True)
        if exit_code != 0:
            _refresh_status_board(status_command)
            return exit_code

        time.sleep(max(args.poll_sec, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())