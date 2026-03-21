from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.serving.replay_backtest import write_prediction_backtest_artifacts


DATE_PATTERN = re.compile(r"predictions_(\d{8})")


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-replay {now}] {message}", flush=True)


def _normalize_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _date_from_prediction_path(path: Path) -> str:
    match = DATE_PATTERN.search(path.name)
    if match is None:
        raise ValueError(f"could not extract date from prediction file: {path}")
    text = match.group(1)
    return f"{text[:4]}-{text[4:6]}-{text[6:8]}"


def _score_source(frame: pd.DataFrame) -> str:
    if "score_source" not in frame.columns:
        return "default"
    values = frame["score_source"].dropna().astype(str)
    return values.iloc[0] if not values.empty else "default"


def _relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction-files", nargs="+", required=True)
    parser.add_argument("--artifact-suffix", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    try:
        config_path = _normalize_path(args.config)
        prediction_paths = [_normalize_path(path) for path in args.prediction_files]
        output_file = _normalize_path(args.output_file) if args.output_file else ROOT / "artifacts" / "reports" / f"serving_smoke_{args.artifact_suffix}.json"
        artifact_ensure_output_file_path(output_file, label="output file", workspace_root=ROOT)
        report_dir = ROOT / "artifacts" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        progress = ProgressBar(total=max(len(prediction_paths) + 2, 3), prefix="[serving-replay]", logger=log_progress, min_interval_sec=0.0)
        model_config = load_yaml(config_path)
        progress.start(message=f"config loaded files={len(prediction_paths)}")

        cases: list[dict[str, Any]] = []
        for prediction_path in prediction_paths:
            frame = pd.read_csv(prediction_path)
            stem = prediction_path.stem.replace("predictions_", "")
            archived_backtest_json = report_dir / f"backtest_{stem}_{args.artifact_suffix}.json"
            archived_backtest_png = report_dir / f"backtest_{stem}_{args.artifact_suffix}.png"
            with Heartbeat("[serving-replay]", f"replaying {prediction_path.name}", logger=log_progress):
                metrics = write_prediction_backtest_artifacts(
                    frame,
                    model_config=model_config,
                    config_path=config_path,
                    prediction_path=prediction_path,
                    workspace_root=ROOT,
                    output_json_path=archived_backtest_json,
                    output_png_path=archived_backtest_png,
                )

            case = {
                "date": _date_from_prediction_path(prediction_path),
                "status": "ok",
                "prediction_file": _relative(prediction_path),
                "score_source": _score_source(frame),
                "policy_name": metrics.get("policy_name"),
                "policy_selected_rows": int(metrics.get("policy_selected_rows") or 0),
                "policy_bets": int(metrics.get("policy_bets") or 0),
                "policy_roi": metrics.get("policy_roi"),
                "archived_artifacts": {
                    "prediction_csv": _relative(prediction_path),
                    "backtest_json": _relative(archived_backtest_json),
                    "backtest_png": _relative(archived_backtest_png),
                },
            }
            cases.append(case)
            progress.update(message=f"processed {prediction_path.name}")

        payload = {
            "profile": args.profile or args.artifact_suffix,
            "config_file": _relative(config_path),
            "artifact_suffix": args.artifact_suffix,
            "cases": cases,
        }
        with Heartbeat("[serving-replay]", "writing replay summary", logger=log_progress):
            write_json(output_file, payload)

        print(f"saved replay summary to {_relative(output_file)}")
        print(f"processed_cases={len(cases)}")
        progress.complete(message="replay summary written")
        return 0
    except KeyboardInterrupt:
        print("[serving-replay] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[serving-replay] failed: {error}")
        return 1
    except Exception as error:
        print(f"[serving-replay] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())