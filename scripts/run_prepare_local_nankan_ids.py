from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.local_nankan_id_prep import prepare_local_nankan_ids_from_config


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[prepare-local-nankan-ids {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-config", default="configs/crawl_local_nankan_template.yaml")
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--race-id-source", choices=["seed_file", "race_list"], default="seed_file")
    parser.add_argument("--target", choices=["all", "race_result", "race_card", "pedigree"], default="all")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-completed", action="store_true")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[prepare-local-nankan-ids]", logger=log_progress, min_interval_sec=0.0)
        crawl_config = load_yaml(ROOT / args.crawl_config)
        progress.start(message=f"config loaded target={args.target}")
        with Heartbeat("[prepare-local-nankan-ids]", "preparing ids from seed", logger=log_progress):
            summary = prepare_local_nankan_ids_from_config(
                crawl_config,
                base_dir=ROOT,
                seed_file=args.seed_file,
                target_filter=args.target,
                start_date=args.start_date,
                end_date=args.end_date,
                date_order=args.date_order,
                limit=args.limit,
                include_completed=args.include_completed,
                race_id_source=args.race_id_source,
            )
        progress.update(message=f"reports ready count={len(summary.get('reports', []))}")

        for report in summary.get("reports", []):
            output_files = ", ".join(report.get("output_files", []))
            targets = ",".join(report.get("targets", []))
            print(
                "[prepare-local-nankan-ids] "
                f"kind={report.get('kind')} targets={targets} rows={report.get('row_count')} source={report.get('source')}"
            )
            print(f"[prepare-local-nankan-ids] outputs: {output_files}")
        progress.complete(message="id preparation completed")
        return 0
    except KeyboardInterrupt:
        print("[prepare-local-nankan-ids] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[prepare-local-nankan-ids] failed: {error}")
        return 1
    except Exception as error:
        print(f"[prepare-local-nankan-ids] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())