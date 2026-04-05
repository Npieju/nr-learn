from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.progress import Heartbeat, ProgressBar

DEFAULT_DATA_CONFIG = "configs/data_2025_latest.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_catboost_fundamental_enriched.yaml"
DEFAULT_FEATURE_CONFIG = "configs/features_catboost_fundamental_enriched.yaml"
DEFAULT_SNAPSHOT_OUTPUT = "artifacts/reports/netkeiba_coverage_snapshot_2026_benchmark_gate.json"
DEFAULT_MANIFEST_OUTPUT = "artifacts/reports/netkeiba_2026_benchmark_gate.json"
DEFAULT_PREFLIGHT_OUTPUT = "artifacts/reports/netkeiba_2026_benchmark_preflight.json"
DEFAULT_UNIVERSE = "jra"
DEFAULT_SOURCE_SCOPE = "netkeiba"
DEFAULT_SCHEMA_VERSION = "netkeiba.2026.benchmark_gate.v1"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_PATH = "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
DEFAULT_RACE_CARD_PATH = "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"
DEFAULT_PEDIGREE_PATH = "data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv"
DEFAULT_STATUS_BOARD_COMMAND = f"{sys.executable} {ROOT / 'scripts' / 'run_netkeiba_2026_status_board.py'}"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-2026-benchmark {now}] {message}", flush=True)


def _run_command(*, label: str, command: list[str]) -> subprocess.CompletedProcess[bytes]:
    print(f"[netkeiba-2026-benchmark] running {label}: {' '.join(command)}", flush=True)
    with Heartbeat("[netkeiba-2026-benchmark]", f"{label} child command", logger=log_progress):
        return subprocess.run(command, cwd=ROOT, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--snapshot-output", default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--preflight-output", default=DEFAULT_PREFLIGHT_OUTPUT)
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
    parser.add_argument("--status-board-command", default=DEFAULT_STATUS_BOARD_COMMAND)
    parser.add_argument("--skip-status-board", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    progress = ProgressBar(total=2, prefix="[netkeiba-2026-benchmark]", logger=log_progress, min_interval_sec=0.0)
    progress.start(
        message=(
            f"starting data_config={args.data_config} model_config={args.model_config} "
            f"feature_config={args.feature_config}"
        )
    )

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
        "--preflight-output",
        args.preflight_output,
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
    ]
    if args.pre_feature_max_rows is not None:
        command.extend(["--pre-feature-max-rows", str(args.pre_feature_max_rows)])
    if args.skip_train:
        command.append("--skip-train")
    if args.skip_evaluate:
        command.append("--skip-evaluate")

    try:
        result = _run_command(label="benchmark_gate", command=command)
        progress.update(message=f"benchmark gate finished exit_code={int(result.returncode)}")
        exit_code = int(result.returncode)
        if not args.skip_status_board:
            board_result = subprocess.run(args.status_board_command, cwd=ROOT, shell=True, check=False)
            if board_result.returncode != 0:
                print(
                    f"[netkeiba-2026-benchmark] status board update failed: exit_code={board_result.returncode}",
                    flush=True,
                )
                if exit_code == 0:
                    return int(board_result.returncode)
        progress.complete(message=f"benchmark gate finished manifest={args.manifest_output}")
        return exit_code
    except KeyboardInterrupt:
        print("[netkeiba-2026-benchmark] interrupted by user")
        return 130
    except Exception as error:
        print(f"[netkeiba-2026-benchmark] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())