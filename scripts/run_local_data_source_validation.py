from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.progress import Heartbeat, ProgressBar

DEFAULT_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_OUTPUT = "artifacts/reports/data_source_validation_local_nankan.json"


def log_progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    progress = ProgressBar(total=2, prefix="[local-data-validate]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message=f"starting config={args.config} output={args.output}")

    command = [
        sys.executable,
        str(ROOT / "scripts/run_validate_data_sources.py"),
        "--config",
        args.config,
        "--output",
        args.output,
    ]
    progress.update(message="launching delegated data source validation")
    print(f"[local-data-validate] running: {' '.join(command)}", flush=True)
    with Heartbeat("[local-data-validate]", "data validation child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    progress.complete(message=f"child command finished exit_code={int(result.returncode)}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
