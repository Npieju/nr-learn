from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import ProgressBar
from racing_ml.data.dataset_loader import inspect_primary_tail_cache_status


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[primary-tail-cache-status {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_2025_latest_primary_tail_cache.yaml")
    parser.add_argument("--tail-rows", type=int, default=10000)
    args = parser.parse_args()

    progress = ProgressBar(total=3, prefix="[primary-tail-cache-status]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message=f"loading config tail_rows={args.tail_rows}")
    data_config = load_yaml(ROOT / args.data_config)
    dataset_cfg = data_config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    progress.update(message="config ready")
    status = inspect_primary_tail_cache_status(
        raw_dir,
        tail_rows=int(args.tail_rows),
        dataset_config=dataset_cfg,
        base_dir=ROOT,
    )
    progress.update(message=f"status={status.get('status')} recommended_action={status.get('recommended_action')}")
    print(status, flush=True)
    progress.complete(message="primary tail cache status completed")
    state = str(status.get("status") or "")
    if state == "fresh":
        return 0
    if state in {"stale", "missing", "tail_mismatch", "manifest_invalid", "cache_invalid", "cache_short"}:
        return 2
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
