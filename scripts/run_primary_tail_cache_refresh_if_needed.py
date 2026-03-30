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
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import refresh_primary_tail_cache_if_needed


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[primary-tail-cache-refresh {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_2025_latest_primary_tail_cache.yaml")
    parser.add_argument("--tail-rows", type=int, default=10000)
    args = parser.parse_args()

    progress = ProgressBar(total=3, prefix="[primary-tail-cache-refresh]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message=f"loading config tail_rows={args.tail_rows}")
    data_config = load_yaml(ROOT / args.data_config)
    dataset_cfg = data_config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    progress.update(message="config ready")
    with Heartbeat(
        "[primary-tail-cache-refresh]",
        f"checking and refreshing primary tail cache rows={args.tail_rows}",
        logger=log_progress,
    ):
        result = refresh_primary_tail_cache_if_needed(
            raw_dir,
            tail_rows=int(args.tail_rows),
            dataset_config=dataset_cfg,
            base_dir=ROOT,
        )
    progress.update(message=f"status={result.get('status')} action={result.get('action')}")
    print(result, flush=True)
    progress.complete(message="primary tail cache refresh completed")
    state = str(result.get("status") or "")
    return 0 if state == "fresh" else 1


if __name__ == "__main__":
    raise SystemExit(main())
