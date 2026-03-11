import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.data.netkeiba_backfill import run_netkeiba_backfill_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--crawl-config", default="configs/crawl_netkeiba_template.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="desc")
    parser.add_argument("--race-batch-size", type=int, default=100)
    parser.add_argument("--pedigree-batch-size", type=int, default=500)
    parser.add_argument("--max-cycles", type=int, default=0)
    parser.add_argument("--post-cycle-command", default=None)
    parser.add_argument("--stop-on-post-cycle-failure", action="store_true")
    parser.add_argument("--skip-race-card", action="store_true")
    parser.add_argument("--skip-pedigree", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    parser.add_argument("--manifest-file", default="artifacts/reports/netkeiba_backfill_manifest.json")
    args = parser.parse_args()

    try:
        data_config = load_yaml(ROOT / args.data_config)
        crawl_config = load_yaml(ROOT / args.crawl_config)
        summary = run_netkeiba_backfill_from_config(
            data_config,
            crawl_config,
            base_dir=ROOT,
            start_date=args.start_date,
            end_date=args.end_date,
            date_order=args.date_order,
            race_batch_size=args.race_batch_size,
            pedigree_batch_size=args.pedigree_batch_size,
            include_race_card=not args.skip_race_card,
            include_pedigree=not args.skip_pedigree,
            max_cycles=(args.max_cycles if args.max_cycles > 0 else None),
            post_cycle_command=args.post_cycle_command,
            stop_on_post_cycle_failure=args.stop_on_post_cycle_failure,
            refresh=args.refresh,
            parse_only=args.parse_only,
            manifest_file=args.manifest_file,
        )
        print(
            "[backfill-netkeiba] "
            f"cycles={summary.get('completed_cycles')} stopped_reason={summary.get('stopped_reason')} date_order={summary.get('date_order')}"
        )
        print(f"[backfill-netkeiba] manifest: {ROOT / args.manifest_file}")
        return 0
    except KeyboardInterrupt:
        print("[backfill-netkeiba] interrupted by user")
        return 130
    except Exception as error:
        print(f"[backfill-netkeiba] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())