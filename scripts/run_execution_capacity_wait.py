from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import write_json
from racing_ml.common.execution_capacity import build_execution_capacity_status
from racing_ml.common.progress import Heartbeat


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[execution-capacity-wait {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-pattern", default="scripts/run_train.py")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--output",
        default="artifacts/reports/execution_capacity_wait.json",
    )
    args = parser.parse_args()

    started_at = time.monotonic()
    timeout_seconds = max(int(args.timeout_seconds), 0)
    poll_interval_seconds = max(int(args.poll_interval_seconds), 1)

    with Heartbeat("[execution-capacity-wait]", "waiting for quiet heavy-job lane", logger=log_progress, interval_sec=poll_interval_seconds):
        while True:
            payload = build_execution_capacity_status(current_script_pattern=args.script_pattern)
            payload["timeout_seconds"] = timeout_seconds
            payload["poll_interval_seconds"] = poll_interval_seconds
            payload["elapsed_seconds"] = int(time.monotonic() - started_at)
            write_json(args.output, payload)
            if payload["status"] == "ready":
                log_progress("status=ready conflicts=0")
                print(f"[execution-capacity-wait] status saved: {ROOT / args.output}")
                return 0
            if payload["elapsed_seconds"] >= timeout_seconds:
                log_progress(f"status=blocked timeout_reached conflicts={payload['conflict_count']}")
                print(f"[execution-capacity-wait] status saved: {ROOT / args.output}")
                return 2
            time.sleep(poll_interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
