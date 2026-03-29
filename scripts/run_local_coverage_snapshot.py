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

DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_OUTPUT = "artifacts/reports/coverage_snapshot_local_nankan.json"
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_SCHEMA_VERSION = "local.coverage_snapshot.v1"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_PATH = "data/external/local_nankan/results/local_race_result.csv"
DEFAULT_RACE_CARD_PATH = "data/external/local_nankan/racecard/local_racecard.csv"
DEFAULT_PEDIGREE_PATH = "data/external/local_nankan/pedigree/local_pedigree.csv"
DEFAULT_RACE_RESULT_MANIFEST = "artifacts/reports/local_nankan_crawl_manifest_race_result.json"
DEFAULT_RACE_CARD_MANIFEST = "artifacts/reports/local_nankan_crawl_manifest_race_card.json"
DEFAULT_PEDIGREE_MANIFEST = "artifacts/reports/local_nankan_crawl_manifest_pedigree.json"
DEFAULT_CRAWL_LOCK_PATH = "artifacts/reports/crawl_manifest_local_nankan.json.lock"


def log_progress(message: str) -> None:
    print(message, flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--race-result-manifest", default=DEFAULT_RACE_RESULT_MANIFEST)
    parser.add_argument("--race-card-manifest", default=DEFAULT_RACE_CARD_MANIFEST)
    parser.add_argument("--pedigree-manifest", default=DEFAULT_PEDIGREE_MANIFEST)
    parser.add_argument("--crawl-lock-path", default=DEFAULT_CRAWL_LOCK_PATH)
    parser.add_argument("--columns", nargs="*", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=2, prefix="[local-coverage-snapshot]", logger=log_progress, min_interval_sec=0.0)
    progress.start(
        message=(
            f"starting data_config={args.data_config} tail_rows={args.tail_rows} "
            f"output={args.output}"
        )
    )

    command = [
        sys.executable,
        str(ROOT / "scripts/run_netkeiba_coverage_snapshot.py"),
        "--config",
        args.data_config,
        "--tail-rows",
        str(args.tail_rows),
        "--output",
        args.output,
        "--universe",
        args.universe,
        "--source-scope",
        args.source_scope,
        "--schema-version",
        args.schema_version,
        "--baseline-reference",
        args.baseline_reference,
        "--race-result-path",
        args.race_result_path,
        "--race-card-path",
        args.race_card_path,
        "--pedigree-path",
        args.pedigree_path,
        "--race-result-manifest",
        args.race_result_manifest,
        "--race-card-manifest",
        args.race_card_manifest,
        "--pedigree-manifest",
        args.pedigree_manifest,
        "--crawl-lock-path",
        args.crawl_lock_path,
    ]
    if args.columns:
        command.extend(["--columns", *args.columns])

    progress.update(message="launching delegated netkeiba coverage snapshot")
    print(f"[local-coverage-snapshot] running: {' '.join(command)}", flush=True)
    with Heartbeat("[local-coverage-snapshot]", "coverage snapshot child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    progress.complete(message=f"child command finished exit_code={int(result.returncode)}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
