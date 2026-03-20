import argparse
from pathlib import Path
import json
import sys
import time
import traceback

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.progress import Heartbeat, ProgressBar


def latest_file(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {path}/{pattern}")
    return files[-1]


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[dashboard {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--backtest-file", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=4, prefix="[dashboard]", logger=log_progress, min_interval_sec=0.0)

    try:
        progress.start("resolving dashboard inputs")
        artifacts = ROOT / "artifacts"
        pred_dir = artifacts / "predictions"
        report_dir = artifacts / "reports"

        pred_path = Path(args.predictions_file) if args.predictions_file else latest_file(pred_dir, "predictions_*.csv")
        backtest_path = Path(args.backtest_file) if args.backtest_file else latest_file(report_dir, "backtest_*.json")
        train_metrics_path = report_dir / "train_metrics.json"
        progress.update(
            message=(
                f"inputs resolved predictions={pred_path.name} backtest={backtest_path.name} "
                f"train_metrics={train_metrics_path.name}"
            )
        )

        with Heartbeat("[dashboard]", "loading dashboard inputs", logger=log_progress):
            pred_df = pd.read_csv(pred_path)
            with backtest_path.open("r", encoding="utf-8") as file:
                backtest = json.load(file)
            with train_metrics_path.open("r", encoding="utf-8") as file:
                train_metrics = json.load(file)
        progress.update(message=f"inputs loaded rows={len(pred_df):,}")

        summary = {
            "prediction_file": pred_path.name,
            "rows": int(len(pred_df)),
            "races": int(pred_df["race_id"].nunique()) if "race_id" in pred_df.columns else None,
            "top1_hit_rate": backtest.get("top1_hit_rate"),
            "top3_hit_rate": backtest.get("top3_hit_rate"),
            "top5_hit_rate": backtest.get("top5_hit_rate"),
            "simple_top1_win_roi": backtest.get("simple_top1_win_roi"),
            "train_auc": train_metrics.get("auc"),
            "train_logloss": train_metrics.get("logloss"),
        }

        out_dir = report_dir / "dashboard"
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = pred_path.stem.replace("predictions_", "")
        summary_path = out_dir / f"dashboard_summary_{stem}.json"
        chart_path = out_dir / f"dashboard_{stem}.png"
        top20_path = out_dir / f"dashboard_top20_{stem}.csv"

        with Heartbeat("[dashboard]", "writing dashboard outputs", logger=log_progress):
            with summary_path.open("w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)

            fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
            axes[0].hist(pred_df["score"], bins=25, color="#3b82f6", alpha=0.85)
            axes[0].set_title("Prediction Score Distribution")
            axes[0].set_xlabel("score")

            if "race_id" in pred_df.columns:
                race_top = pred_df.groupby("race_id", as_index=False)["score"].max()
                axes[1].boxplot(race_top["score"], vert=True)
                axes[1].set_title("Per-race Top Score Boxplot")
                axes[1].set_ylabel("top score")
            else:
                axes[1].text(0.5, 0.5, "race_id not found", ha="center", va="center")

            plt.tight_layout()
            fig.savefig(chart_path, dpi=140)
            plt.close(fig)

            top_cols = [c for c in ["race_id", "horse_id", "horse_name", "score", "pred_rank", "rank"] if c in pred_df.columns]
            pred_df[top_cols].sort_values("score", ascending=False).head(20).to_csv(top20_path, index=False)
        progress.update(message=f"dashboard outputs saved stem={stem}")
        progress.complete(message="dashboard flow finished")

        print(f"[dashboard] summary saved: {summary_path}")
        print(f"[dashboard] chart saved: {chart_path}")
        print(f"[dashboard] top20 saved: {top20_path}")
        print(f"[dashboard] metrics: {summary}")
        return 0
    except KeyboardInterrupt:
        print("[dashboard] interrupted by user")
        return 130
    except Exception as error:
        print(f"[dashboard] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
