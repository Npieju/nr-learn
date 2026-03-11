import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.data.netkeiba_id_prep import prepare_netkeiba_ids_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--crawl-config", default="configs/crawl_netkeiba_template.yaml")
    parser.add_argument("--target", choices=["all", "race_result", "race_card", "pedigree"], default="all")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--include-completed", action="store_true")
    args = parser.parse_args()

    try:
        data_config = load_yaml(ROOT / args.data_config)
        crawl_config = load_yaml(ROOT / args.crawl_config)
        summary = prepare_netkeiba_ids_from_config(
            data_config,
            crawl_config,
            base_dir=ROOT,
            target_filter=args.target,
            start_date=args.start_date,
            end_date=args.end_date,
            date_order=args.date_order,
            limit=args.limit,
            include_completed=args.include_completed,
        )
        for report in summary.get("reports", []):
            output_files = ", ".join(report.get("output_files", []))
            targets = ",".join(report.get("targets", []))
            print(
                "[prepare-netkeiba-ids] "
                f"kind={report.get('kind')} targets={targets} rows={report.get('row_count')}"
            )
            print(f"[prepare-netkeiba-ids] outputs: {output_files}")
        return 0
    except KeyboardInterrupt:
        print("[prepare-netkeiba-ids] interrupted by user")
        return 130
    except Exception as error:
        print(f"[prepare-netkeiba-ids] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())