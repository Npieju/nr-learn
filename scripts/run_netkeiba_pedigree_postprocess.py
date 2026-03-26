from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _run(command: list[str], *, label: str) -> int:
    print(f"[netkeiba-postprocess] running {label}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--crawl-lock-path", default="artifacts/reports/netkeiba_crawl_manifest_2025_pedigree.json.lock")
    parser.add_argument("--missing-script", default="scripts/run_netkeiba_refresh_pedigree_missing.py")
    parser.add_argument("--snapshot-script", default="scripts/run_netkeiba_coverage_snapshot.py")
    args = parser.parse_args()

    lock_path = _resolve_path(args.crawl_lock_path)
    missing_script = _resolve_path(args.missing_script)
    snapshot_script = _resolve_path(args.snapshot_script)
    deadline = time.monotonic() + args.wait_timeout_seconds if args.wait_timeout_seconds > 0 else None

    while lock_path.exists():
        if deadline is not None and time.monotonic() >= deadline:
            print("[netkeiba-postprocess] timeout waiting for crawl lock to clear", flush=True)
            return 124
        print("[netkeiba-postprocess] waiting for pedigree crawl to finish", flush=True)
        time.sleep(max(int(args.poll_seconds), 1))

    refresh_code = _run([sys.executable, str(missing_script)], label="refresh_missing")
    if refresh_code != 0:
        return refresh_code

    snapshot_code = _run(
        [
            sys.executable,
            str(snapshot_script),
            "--config",
            "configs/data_2025_latest.yaml",
            "--tail-rows",
            "5000",
            "--output",
            "artifacts/reports/netkeiba_coverage_snapshot_2025_latest.json",
            "--race-result-manifest",
            "artifacts/reports/netkeiba_crawl_manifest_2025_missing_result_race_result.json",
            "--race-card-manifest",
            "artifacts/reports/netkeiba_crawl_manifest_2025_backfill_race_card.json",
            "--pedigree-manifest",
            "artifacts/reports/netkeiba_crawl_manifest_2025_pedigree_pedigree.json",
            "--crawl-lock-path",
            "artifacts/reports/netkeiba_crawl_manifest_2025_pedigree.json.lock",
        ],
        label="coverage_snapshot",
    )
    return snapshot_code


if __name__ == "__main__":
    raise SystemExit(main())