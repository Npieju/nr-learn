import argparse
import json
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
        data_cfg = load_yaml(ROOT / args.config)
        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        progress.update(message="config loaded")

        with Heartbeat("[data-validate]", "inspecting dataset sources", logger=log_progress):
            report = inspect_dataset_sources(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
        progress.update(message="dataset sources inspected")

        output_path = ROOT / args.output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with Heartbeat("[data-validate]", "writing validation report", logger=log_progress):
            with output_path.open("w", encoding="utf-8") as file:
                json.dump(report, file, ensure_ascii=False, indent=2)
        progress.complete(message="validation report written")

        append_ok = sum(1 for row in report.get("append_tables", []) if row.get("status") == "ok")
        append_total = len(report.get("append_tables", []))
        supplemental_ok = sum(1 for row in report.get("supplemental_tables", []) if row.get("status") == "ok")
        supplemental_total = len(report.get("supplemental_tables", []))
        print(f"[data-validate] primary={report.get('primary_dataset', {}).get('status')}")
        print(f"[data-validate] append_tables_ok={append_ok}/{append_total}")
        print(f"[data-validate] supplemental_tables_ok={supplemental_ok}/{supplemental_total}")
        print(f"[data-validate] report saved: {output_path}")
        return 0
    except KeyboardInterrupt:
        print("[data-validate] interrupted by user")
        return 130
    except Exception as error:
        print(f"[data-validate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())