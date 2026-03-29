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
from racing_ml.data.local_nankan_primary import materialize_local_nankan_primary_from_config


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-primary {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_local_nankan.yaml")
    parser.add_argument("--race-result-path", default=None)
    parser.add_argument("--race-card-path", default=None)
    parser.add_argument("--pedigree-path", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--manifest-file", default="artifacts/reports/local_nankan_primary_materialize_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[local-nankan-primary]", logger=log_progress, min_interval_sec=0.0)
        data_config = load_yaml(ROOT / args.data_config)
        progress.start(message="config loaded")
        with Heartbeat("[local-nankan-primary]", "materializing primary raw", logger=log_progress):
            summary = materialize_local_nankan_primary_from_config(
                data_config,
                base_dir=ROOT,
                race_result_path=args.race_result_path,
                race_card_path=args.race_card_path,
                pedigree_path=args.pedigree_path,
                output_file=args.output_file,
                manifest_file=args.manifest_file,
                dry_run=args.dry_run,
            )
        progress.update(message=f"materialize status={summary.get('status')} rows={summary.get('row_count')}")
        print(
            "[local-nankan-primary] "
            f"status={summary.get('status')} phase={summary.get('current_phase')} output={summary.get('output_file')}"
        )
        progress.complete(message="primary materialize completed")
        if summary.get("status") == "completed":
            return 0
        if summary.get("status") == "planned":
            return 0
        if summary.get("status") == "not_ready":
            return 2
        return 1
    except KeyboardInterrupt:
        print("[local-nankan-primary] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[local-nankan-primary] failed: {error}")
        return 1
    except Exception as error:
        print(f"[local-nankan-primary] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())