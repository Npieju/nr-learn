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
from racing_ml.data.netkeiba_crawler import crawl_netkeiba_from_config


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[collect-netkeiba {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crawl_netkeiba_template.yaml")
    parser.add_argument("--target", choices=["race_result", "race_card", "pedigree"], default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--parse-only", action="store_true")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[collect-netkeiba]", logger=log_progress, min_interval_sec=0.0)
        config = load_yaml(ROOT / args.config)
        progress.start(message=f"config loaded target={args.target or 'all'}")
        with Heartbeat("[collect-netkeiba]", "crawling targets", logger=log_progress):
            summary = crawl_netkeiba_from_config(
                config,
                base_dir=ROOT,
                target_filter=args.target,
                override_limit=args.limit,
                refresh=args.refresh,
                parse_only=args.parse_only,
            )
        progress.update(message=f"crawl completed targets={len(summary.get('targets', []))}")
        for target in summary.get("targets", []):
            print(
                "[collect-netkeiba] "
                f"target={target.get('target')} parsed={target.get('parsed_ids')}/{target.get('requested_ids')} "
                f"rows={target.get('rows_written')} failures={target.get('failure_count')}"
            )
            print(f"[collect-netkeiba] output: {target.get('output_file')}")
            print(f"[collect-netkeiba] manifest: {target.get('manifest_file')}")
        print(f"[collect-netkeiba] summary: {ROOT / 'artifacts/reports/netkeiba_crawl_manifest.json'}")
        progress.complete(message="summary written")
        return 0
    except KeyboardInterrupt:
        print("[collect-netkeiba] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[collect-netkeiba] failed: {error}")
        return 1
    except Exception as error:
        print(f"[collect-netkeiba] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())