from __future__ import annotations

import argparse
from datetime import date
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.netkeiba_backfill import run_netkeiba_backfill_from_config


DEFAULT_START_DATE = "2026-01-01"
DEFAULT_END_DATE = date.today().isoformat()
DEFAULT_DATA_CONFIG = "configs/data_2025_latest.yaml"
DEFAULT_CRAWL_CONFIG = "configs/crawl_netkeiba_backfill_2026_ytd.yaml"
DEFAULT_MANIFEST_FILE = "artifacts/reports/netkeiba_backfill_manifest_2026_ytd.json"
DEFAULT_POST_CYCLE_SNAPSHOT_COMMAND = f"{sys.executable} {ROOT / 'scripts' / 'run_netkeiba_2026_ytd_snapshot.py'}"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[backfill-netkeiba-2026 {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--crawl-config", default=DEFAULT_CRAWL_CONFIG)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE)
    parser.add_argument("--end-date", default=DEFAULT_END_DATE)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--race-id-source", choices=["training_table", "race_list"], default="race_list")
    parser.add_argument("--race-batch-size", type=int, default=100)
    parser.add_argument("--pedigree-batch-size", type=int, default=500)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--post-cycle-command", default=None)
    parser.add_argument("--skip-post-cycle-snapshot", action="store_true")
    parser.add_argument("--stop-on-post-cycle-failure", action="store_true")
    parser.add_argument("--skip-race-card", action="store_true")
    parser.add_argument("--skip-pedigree", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--manifest-file", default=DEFAULT_MANIFEST_FILE)
    args = parser.parse_args()

    try:
        if args.start_date > args.end_date:
            raise ValueError("--start-date must be less than or equal to --end-date")
        if args.post_cycle_command and not args.skip_post_cycle_snapshot:
            raise ValueError(
                "--post-cycle-command cannot be combined with automatic snapshot; "
                "use --skip-post-cycle-snapshot if you want to provide a custom post-cycle command"
            )

        progress = ProgressBar(total=3, prefix="[backfill-netkeiba-2026]", logger=log_progress, min_interval_sec=0.0)
        manifest_path = Path(args.manifest_file)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        artifact_ensure_output_file_path(manifest_path, label="manifest output", workspace_root=ROOT)
        data_config = load_yaml(ROOT / args.data_config)
        crawl_config = load_yaml(ROOT / args.crawl_config)
        post_cycle_command = args.post_cycle_command
        if not args.skip_post_cycle_snapshot:
            post_cycle_command = DEFAULT_POST_CYCLE_SNAPSHOT_COMMAND
        progress.start(
            message=(
                f"configs loaded data={args.data_config} crawl={args.crawl_config} "
                f"window={args.start_date}..{args.end_date}"
            )
        )
        with Heartbeat("[backfill-netkeiba-2026]", "running 2026 YTD backfill cycles", logger=log_progress):
            summary = run_netkeiba_backfill_from_config(
                data_config,
                crawl_config,
                base_dir=ROOT,
                start_date=args.start_date,
                end_date=args.end_date,
                date_order=args.date_order,
                race_id_source=args.race_id_source,
                race_batch_size=args.race_batch_size,
                pedigree_batch_size=args.pedigree_batch_size,
                include_race_card=not args.skip_race_card,
                include_pedigree=not args.skip_pedigree,
                max_cycles=(args.max_cycles if args.max_cycles > 0 else None),
                post_cycle_command=post_cycle_command,
                stop_on_post_cycle_failure=args.stop_on_post_cycle_failure,
                refresh=args.refresh,
                parse_only=args.parse_only,
                manifest_file=args.manifest_file,
            )
        progress.update(message=f"backfill completed cycles={summary.get('completed_cycles')}")
        print(
            "[backfill-netkeiba-2026] "
            f"cycles={summary.get('completed_cycles')} stopped_reason={summary.get('stopped_reason')} "
            f"date_order={summary.get('date_order')} race_id_source={summary.get('race_id_source')}"
        )
        print(f"[backfill-netkeiba-2026] manifest: {manifest_path}")
        progress.complete(message="manifest ready")
        return 0
    except KeyboardInterrupt:
        print("[backfill-netkeiba-2026] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[backfill-netkeiba-2026] failed: {error}")
        return 1
    except Exception as error:
        print(f"[backfill-netkeiba-2026] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())