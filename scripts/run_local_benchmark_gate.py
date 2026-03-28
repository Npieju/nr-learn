from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_local_baseline.yaml"
DEFAULT_FEATURE_CONFIG = "configs/features_local_baseline.yaml"
DEFAULT_SNAPSHOT_OUTPUT = "artifacts/reports/coverage_snapshot_local_nankan.json"
DEFAULT_MANIFEST_OUTPUT = "artifacts/reports/benchmark_gate_local_nankan.json"
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_SCHEMA_VERSION = "local.benchmark_gate.v1"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_PATH = "data/external/local_nankan/results/local_race_result.csv"
DEFAULT_RACE_CARD_PATH = "data/external/local_nankan/racecard/local_racecard.csv"
DEFAULT_PEDIGREE_PATH = "data/external/local_nankan/pedigree/local_pedigree.csv"
DEFAULT_PREFLIGHT_OUTPUT = "artifacts/reports/data_preflight_local_nankan.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--snapshot-output", default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--preflight-output", default=DEFAULT_PREFLIGHT_OUTPUT)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "scripts/run_netkeiba_benchmark_gate.py"),
        "--data-config",
        args.data_config,
        "--model-config",
        args.model_config,
        "--feature-config",
        args.feature_config,
        "--tail-rows",
        str(args.tail_rows),
        "--snapshot-output",
        args.snapshot_output,
        "--manifest-output",
        args.manifest_output,
        "--max-rows",
        str(args.max_rows),
        "--wf-mode",
        args.wf_mode,
        "--wf-scheme",
        args.wf_scheme,
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
        "--preflight-output",
        args.preflight_output,
    ]
    if args.pre_feature_max_rows is not None:
        command.extend(["--pre-feature-max-rows", str(args.pre_feature_max_rows)])
    if args.skip_train:
        command.append("--skip-train")
    if args.skip_evaluate:
        command.append("--skip-evaluate")

    print(f"[local-benchmark-gate] running: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())