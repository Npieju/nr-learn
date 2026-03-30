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


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[execution-capacity {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--script-pattern", default="scripts/run_train.py")
    parser.add_argument(
        "--output",
        default="artifacts/reports/execution_capacity_status.json",
    )
    args = parser.parse_args()

    log_progress(f"checking heavy-job lane for script_pattern={args.script_pattern}")
    payload = build_execution_capacity_status(current_script_pattern=args.script_pattern)
    write_json(args.output, payload)
    log_progress(f"status={payload['status']} conflicts={payload['conflict_count']}")
    print(f"[execution-capacity] status saved: {ROOT / args.output}")
    return 0 if payload["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
