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


DEFAULT_DATA_CONFIG = "configs/data_2025_latest.yaml"
DEFAULT_OUTPUT = "artifacts/reports/netkeiba_coverage_snapshot_2026_ytd.json"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_race_result.json"
DEFAULT_RACE_CARD_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_race_card.json"
DEFAULT_PEDIGREE_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_pedigree.json"
DEFAULT_CRAWL_LOCK = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd.json.lock"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-2026-snapshot {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--race-result-path", default="data/external/netkeiba/results/netkeiba_race_result_crawled.csv")
    parser.add_argument("--race-card-path", default="data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv")
    parser.add_argument("--pedigree-path", default="data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv")
    parser.add_argument("--race-result-manifest", default=DEFAULT_RACE_RESULT_MANIFEST)
    parser.add_argument("--race-card-manifest", default=DEFAULT_RACE_CARD_MANIFEST)
    parser.add_argument("--pedigree-manifest", default=DEFAULT_PEDIGREE_MANIFEST)
    parser.add_argument("--crawl-lock-path", default=DEFAULT_CRAWL_LOCK)
    args = parser.parse_args()

    command = [
        sys.executable,
        str(ROOT / "scripts" / "run_netkeiba_coverage_snapshot.py"),
        "--config",
        args.config,
        "--tail-rows",
        str(args.tail_rows),
        "--output",
        args.output,
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

    try:
        log_progress(
            f"starting snapshot config={args.config} output={args.output} baseline_reference={args.baseline_reference}"
        )
        result = subprocess.run(command, cwd=ROOT, check=False)
        if result.returncode != 0:
            print(f"[netkeiba-2026-snapshot] failed: exit_code={result.returncode}")
            return int(result.returncode)
        print(f"[netkeiba-2026-snapshot] output: {ROOT / args.output}")
        return 0
    except KeyboardInterrupt:
        print("[netkeiba-2026-snapshot] interrupted by user")
        return 130
    except Exception as error:
        print(f"[netkeiba-2026-snapshot] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())