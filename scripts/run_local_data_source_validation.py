from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_OUTPUT = "artifacts/reports/data_source_validation_local_nankan.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "scripts/run_validate_data_sources.py"),
        "--config",
        args.config,
        "--output",
        args.output,
    ]
    print(f"[local-data-validate] running: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())