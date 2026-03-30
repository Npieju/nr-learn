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
from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import materialize_config_table


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[supplemental-materialize {now}] {message}", flush=True)


def _resolve_output_path(path: str | None) -> Path | None:
    if path is None:
        return None
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else (ROOT / path_obj)


def _default_output_path(table_name: str) -> Path:
    slug = str(table_name).strip().lower().replace("-", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return ROOT / "data/processed/supplemental" / f"{slug}.csv"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_2025_latest.yaml")
    parser.add_argument("--table-name", default="corner_passing_order")
    parser.add_argument("--table-kind", choices=["auto", "append", "supplemental"], default="supplemental")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--manifest-file", default="artifacts/reports/supplemental_materialize_manifest.json")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[supplemental-materialize]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"loading config table={args.table_name} kind={args.table_kind}")
        data_config = load_yaml(ROOT / args.data_config)
        dataset_cfg = data_config.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        output_path = _resolve_output_path(args.output_file) if args.output_file is not None else _default_output_path(args.table_name)
        manifest_path = _resolve_output_path(args.manifest_file)
        assert manifest_path is not None
        artifact_ensure_output_file_path(manifest_path, label="materialize manifest", workspace_root=ROOT)
        if output_path is not None:
            artifact_ensure_output_file_path(output_path, label="materialized supplemental output", workspace_root=ROOT)
        progress.update(message="config ready")
        with Heartbeat(
            "[supplemental-materialize]",
            f"materializing table={args.table_name} kind={args.table_kind}",
            logger=log_progress,
        ):
            summary = materialize_config_table(
                raw_dir,
                table_name=args.table_name,
                dataset_config=dataset_cfg,
                base_dir=ROOT,
                output_file=output_path,
                table_kind=args.table_kind,
            )
        progress.update(message=f"materialize status={summary.get('status')} rows={summary.get('row_count')}")
        summary["data_config"] = args.data_config
        summary["table_name"] = args.table_name
        summary["table_kind"] = args.table_kind
        summary["manifest_file"] = artifact_display_path(manifest_path, workspace_root=ROOT)
        write_json(manifest_path, summary)
        progress.update(message=f"manifest written path={artifact_display_path(manifest_path, workspace_root=ROOT)}")
        print(
            "[supplemental-materialize] "
            f"status={summary.get('status')} output={summary.get('output_file')} rows={summary.get('row_count')}",
            flush=True,
        )
        progress.complete(message="supplemental materialize completed")
        return 0 if summary.get("status") == "completed" else 2
    except KeyboardInterrupt:
        print("[supplemental-materialize] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[supplemental-materialize] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[supplemental-materialize] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
