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
DEFAULT_FEATURE_CONFIG = "configs/features_local_baseline.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_local_baseline.yaml"
DEFAULT_TEMPLATE_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_SUMMARY_OUTPUT = "artifacts/reports/feature_gap_summary_local_nankan.json"
DEFAULT_FEATURE_OUTPUT = "artifacts/reports/feature_gap_feature_coverage_local_nankan.csv"
DEFAULT_RAW_OUTPUT = "artifacts/reports/feature_gap_raw_column_coverage_local_nankan.csv"


def log_progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--template-config", default=DEFAULT_TEMPLATE_CONFIG)
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--summary-output", default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--feature-output", default=DEFAULT_FEATURE_OUTPUT)
    parser.add_argument("--raw-output", default=DEFAULT_RAW_OUTPUT)
    args = parser.parse_args()
    progress = ProgressBar(total=2, prefix="[local-feature-gap]", logger=log_progress, min_interval_sec=0.0)
    progress.start(
        message=(
            f"starting config={args.config} feature_config={args.feature_config} "
            f"max_rows={args.max_rows}"
        )
    )

    command = [
        sys.executable,
        str(ROOT / "scripts/run_feature_gap_report.py"),
        "--config",
        args.config,
        "--feature-config",
        args.feature_config,
        "--model-config",
        args.model_config,
        "--template-config",
        args.template_config,
        "--max-rows",
        str(args.max_rows),
        "--coverage-threshold",
        str(args.coverage_threshold),
        "--summary-output",
        args.summary_output,
        "--feature-output",
        args.feature_output,
        "--raw-output",
        args.raw_output,
    ]
    progress.update(message="launching delegated feature gap report")
    print(f"[local-feature-gap] running: {' '.join(command)}", flush=True)
    with Heartbeat("[local-feature-gap]", "feature gap child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    progress.complete(message=f"child command finished exit_code={int(result.returncode)}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
