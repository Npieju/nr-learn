from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[train-when-ready {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--poll-interval-seconds", type=int, default=30)
    parser.add_argument(
        "--wait-output",
        default="artifacts/reports/execution_capacity_wait_train.json",
    )
    parser.add_argument("train_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    train_args = list(args.train_args)
    if train_args and train_args[0] == "--":
        train_args = train_args[1:]
    if not train_args:
        raise ValueError("train_args must include the arguments to pass to scripts/run_train.py")

    wait_command = [
        sys.executable,
        str(ROOT / "scripts/run_execution_capacity_wait.py"),
        "--script-pattern",
        "scripts/run_train.py",
        "--timeout-seconds",
        str(args.timeout_seconds),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
        "--output",
        args.wait_output,
    ]
    log_progress(f"waiting for quiet lane: {' '.join(wait_command)}")
    wait_result = subprocess.run(wait_command, cwd=ROOT, check=False)
    if wait_result.returncode != 0:
        log_progress(f"wait step returned non-zero exit code={wait_result.returncode}")
        return int(wait_result.returncode)

    train_command = [
        sys.executable,
        str(ROOT / "scripts/run_train.py"),
        *train_args,
    ]
    log_progress(f"starting train command: {' '.join(train_command)}")
    train_result = subprocess.run(train_command, cwd=ROOT, check=False)
    log_progress(f"train command finished exit_code={train_result.returncode}")
    return int(train_result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
