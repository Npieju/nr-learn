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

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import materialize_primary_tail_cache


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[primary-tail-cache {now}] {message}", flush=True)


def _resolve_output_path(path: str | None) -> Path | None:
    if path is None:
        return None
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else (ROOT / path_obj)


def _default_output_path(tail_rows: int) -> Path:
    return ROOT / "data/processed/primary" / f"race_result_tail{int(tail_rows)}_exact.pkl"


def _default_manifest_path(tail_rows: int) -> Path:
    return ROOT / "artifacts/reports" / f"primary_tail_cache_tail{int(tail_rows)}.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_2025_latest.yaml")
    parser.add_argument("--tail-rows", type=int, default=10000)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--manifest-file", default=None)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[primary-tail-cache]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"loading config tail_rows={args.tail_rows}")
        data_config = load_yaml(ROOT / args.data_config)
        dataset_cfg = data_config.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        output_path = _resolve_output_path(args.output_file) if args.output_file is not None else _default_output_path(args.tail_rows)
        manifest_path = _resolve_output_path(args.manifest_file) if args.manifest_file is not None else _default_manifest_path(args.tail_rows)
        artifact_ensure_output_file_path(output_path, label="primary tail cache output", workspace_root=ROOT)
        artifact_ensure_output_file_path(manifest_path, label="primary tail cache manifest", workspace_root=ROOT)
        progress.update(message="config ready")
        with Heartbeat(
            "[primary-tail-cache]",
            f"materializing primary tail cache rows={args.tail_rows}",
            logger=log_progress,
        ):
            summary = materialize_primary_tail_cache(
                raw_dir,
                tail_rows=int(args.tail_rows),
                dataset_config=dataset_cfg,
                base_dir=ROOT,
                output_file=output_path,
                manifest_file=manifest_path,
            )
        progress.update(message=f"materialize status={summary.get('status')} rows={summary.get('row_count')}")
        progress.update(message=f"manifest written path={artifact_display_path(manifest_path, workspace_root=ROOT)}")
        print(
            "[primary-tail-cache] "
            f"status={summary.get('status')} output={summary.get('output_file')} rows={summary.get('row_count')}",
            flush=True,
        )
        progress.complete(message="primary tail cache completed")
        return 0 if summary.get("status") == "completed" else 2
    except KeyboardInterrupt:
        print("[primary-tail-cache] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[primary-tail-cache] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[primary-tail-cache] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
