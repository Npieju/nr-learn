from __future__ import annotations

import argparse
from pathlib import Path
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
from racing_ml.common.progress import Heartbeat
from racing_ml.serving.prediction_market_refresh import refresh_prediction_market_data
from racing_ml.serving.replay_backtest import write_prediction_backtest_artifacts


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[market-refresh {now}] {message}", flush=True)


def _normalize_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _default_output_path(prediction_path: Path, artifact_suffix: str | None) -> Path:
    suffix = f"_{artifact_suffix}" if artifact_suffix else "_market_refresh"
    return prediction_path.with_name(f"{prediction_path.stem}{suffix}{prediction_path.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh market columns in an existing prediction CSV without rerunning the model.")
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--market-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--backtest-json", default=None)
    parser.add_argument("--backtest-png", default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--join-keys", default=None, help="Comma-separated join keys. Default: auto-detect.")
    parser.add_argument("--market-columns", default=None, help="Comma-separated market columns to refresh. Default: odds,popularity,timestamps.")
    args = parser.parse_args()

    try:
        prediction_path = _normalize_path(args.predictions_file)
        market_path = _normalize_path(args.market_file)
        output_path = _normalize_path(args.output_file) if args.output_file else _default_output_path(prediction_path, args.artifact_suffix)
        summary_path = _normalize_path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
        artifact_ensure_output_file_path(output_path, label="output file", workspace_root=ROOT)
        artifact_ensure_output_file_path(summary_path, label="summary output", workspace_root=ROOT)

        join_keys = [token.strip() for token in str(args.join_keys or "").split(",") if token.strip()] or None
        market_columns = [token.strip() for token in str(args.market_columns or "").split(",") if token.strip()] or None

        with Heartbeat("[market-refresh]", "loading inputs", logger=log_progress):
            predictions = pd.read_csv(prediction_path, low_memory=False)
            market = pd.read_csv(market_path, low_memory=False)

        with Heartbeat("[market-refresh]", "refreshing market columns", logger=log_progress):
            refreshed, summary = refresh_prediction_market_data(
                predictions,
                market,
                join_keys=join_keys,
                market_columns=market_columns,
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            refreshed.to_csv(output_path, index=False)

        payload: dict[str, Any] = {
            "prediction_file": artifact_display_path(prediction_path, workspace_root=ROOT),
            "market_file": artifact_display_path(market_path, workspace_root=ROOT),
            "output_file": artifact_display_path(output_path, workspace_root=ROOT),
            "refresh_summary": summary,
        }

        if args.config:
            config_path = _normalize_path(args.config)
            backtest_json = _normalize_path(args.backtest_json) if args.backtest_json else output_path.with_suffix(".backtest.json")
            backtest_png = _normalize_path(args.backtest_png) if args.backtest_png else output_path.with_suffix(".backtest.png")
            artifact_ensure_output_file_path(backtest_json, label="backtest json", workspace_root=ROOT)
            artifact_ensure_output_file_path(backtest_png, label="backtest png", workspace_root=ROOT)
            model_config = load_yaml(config_path)
            with Heartbeat("[market-refresh]", "replaying refreshed predictions", logger=log_progress):
                backtest_metrics = write_prediction_backtest_artifacts(
                    refreshed,
                    model_config=model_config,
                    config_path=config_path,
                    prediction_path=output_path,
                    workspace_root=ROOT,
                    output_json_path=backtest_json,
                    output_png_path=backtest_png,
                )
            payload["config_file"] = artifact_display_path(config_path, workspace_root=ROOT)
            payload["backtest_json"] = artifact_display_path(backtest_json, workspace_root=ROOT)
            payload["backtest_png"] = artifact_display_path(backtest_png, workspace_root=ROOT)
            payload["backtest_metrics"] = backtest_metrics

        write_json(summary_path, payload)
        print(f"[market-refresh] refreshed predictions saved: {artifact_display_path(output_path, workspace_root=ROOT)}")
        print(f"[market-refresh] summary saved: {artifact_display_path(summary_path, workspace_root=ROOT)}")
        if args.config:
            print(f"[market-refresh] backtest replay complete: {payload['backtest_json']}")
        return 0
    except KeyboardInterrupt:
        print("[market-refresh] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[market-refresh] failed: {error}")
        return 1
    except Exception as error:
        print(f"[market-refresh] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())