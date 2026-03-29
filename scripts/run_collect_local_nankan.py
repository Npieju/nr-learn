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
from racing_ml.data.local_nankan_collect import collect_local_nankan_from_config


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[collect-local-nankan {now}] {message}", flush=True)


def _apply_crawl_overrides(
    config: dict[str, object],
    *,
    delay_sec: float | None,
    timeout_sec: float | None,
    retry_count: int | None,
    retry_backoff_sec: float | None,
    overwrite: bool | None,
) -> dict[str, object]:
    output = dict(config)
    crawl_cfg = output.get("crawl")
    crawl_section = dict(crawl_cfg) if isinstance(crawl_cfg, dict) else output
    if delay_sec is not None:
        crawl_section["delay_sec"] = float(delay_sec)
    if timeout_sec is not None:
        crawl_section["timeout_sec"] = float(timeout_sec)
    if retry_count is not None:
        crawl_section["retry_count"] = int(retry_count)
    if retry_backoff_sec is not None:
        crawl_section["retry_backoff_sec"] = float(retry_backoff_sec)
    if overwrite is not None:
        crawl_section["overwrite"] = bool(overwrite)
    if isinstance(crawl_cfg, dict):
        output["crawl"] = crawl_section
    else:
        output = crawl_section
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/crawl_local_nankan_template.yaml")
    parser.add_argument("--target", choices=["all", "race_result", "race_card", "pedigree"], default="all")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--delay-sec", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--retry-count", type=int, default=None)
    parser.add_argument("--retry-backoff-sec", type=float, default=None)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[collect-local-nankan]", logger=log_progress, min_interval_sec=0.0)
        config = _apply_crawl_overrides(
            load_yaml(ROOT / args.config),
            delay_sec=args.delay_sec,
            timeout_sec=args.timeout_sec,
            retry_count=args.retry_count,
            retry_backoff_sec=args.retry_backoff_sec,
            overwrite=args.overwrite,
        )
        progress.start(message=f"config loaded target={args.target}")
        with Heartbeat("[collect-local-nankan]", "running local_nankan crawl", logger=log_progress):
            summary = collect_local_nankan_from_config(
                config,
                base_dir=ROOT,
                target_filter=args.target,
                override_limit=args.limit,
                dry_run=args.dry_run,
            )
        progress.update(message=f"collect summary ready targets={len(summary.get('targets', []))}")
        for target in summary.get("targets", []):
            print(
                "[collect-local-nankan] "
                f"target={target.get('target')} status={target.get('status')} requested_ids={target.get('requested_ids')} output={target.get('output_file')}"
            )
            print(f"[collect-local-nankan] manifest: {target.get('manifest_file')}")
        print(f"[collect-local-nankan] summary: {summary.get('manifest_file')}")
        progress.complete(message="collect completed")
        if str(summary.get("status")) in {"planned", "completed", "partial"}:
            return 0
        return 2
    except KeyboardInterrupt:
        print("[collect-local-nankan] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[collect-local-nankan] failed: {error}")
        return 1
    except Exception as error:
        print(f"[collect-local-nankan] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())