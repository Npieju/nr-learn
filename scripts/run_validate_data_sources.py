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
from racing_ml.data.dataset_loader import inspect_dataset_sources


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[data-validate {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="artifacts/reports/data_source_validation.json")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[data-validate]", logger=log_progress, min_interval_sec=0.0)
        progress.start("starting validation")
        output_path = ROOT / args.output
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        data_cfg = load_yaml(ROOT / args.config)
        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        progress.update(message="config loaded")

        with Heartbeat("[data-validate]", "inspecting dataset sources", logger=log_progress):
            report = inspect_dataset_sources(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
        progress.update(message="dataset sources inspected")

        with Heartbeat("[data-validate]", "writing validation report", logger=log_progress):
            write_json(output_path, report)
        progress.complete(message="validation report written")

        healthy_statuses = {"ok", "optional_missing"}
        append_ok = sum(1 for row in report.get("append_tables", []) if row.get("status") in healthy_statuses)
        append_total = len(report.get("append_tables", []))
        supplemental_ok = sum(1 for row in report.get("supplemental_tables", []) if row.get("status") in healthy_statuses)
        supplemental_total = len(report.get("supplemental_tables", []))
        append_optional = sum(1 for row in report.get("append_tables", []) if row.get("status") == "optional_missing")
        supplemental_optional = sum(1 for row in report.get("supplemental_tables", []) if row.get("status") == "optional_missing")
        print(f"[data-validate] primary={report.get('primary_dataset', {}).get('status')}")
        print(f"[data-validate] append_tables_ok={append_ok}/{append_total} optional_missing={append_optional}")
        print(f"[data-validate] supplemental_tables_ok={supplemental_ok}/{supplemental_total} optional_missing={supplemental_optional}")
        print(f"[data-validate] report saved: {output_path}")
        return 0
    except KeyboardInterrupt:
        print("[data-validate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[data-validate] failed: {error}")
        return 1
    except Exception as error:
        print(f"[data-validate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())