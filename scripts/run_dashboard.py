import argparse
from pathlib import Path
import json
import sys
import traceback

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def latest_file(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {path}/{pattern}")
    return files[-1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--backtest-file", default=None)
    args = parser.parse_args()

    try:
        artifacts = ROOT / "artifacts"
        pred_dir = artifacts / "predictions"
        report_dir = artifacts / "reports"

        pred_path = Path(args.predictions_file) if args.predictions_file else latest_file(pred_dir, "predictions_*.csv")
        backtest_path = Path(args.backtest_file) if args.backtest_file else latest_file(report_dir, "backtest_*.json")
        train_metrics_path = report_dir / "train_metrics.json"

        pred_df = pd.read_csv(pred_path)
        with backtest_path.open("r", encoding="utf-8") as file:
            backtest = json.load(file)
        with train_metrics_path.open("r", encoding="utf-8") as file:
            train_metrics = json.load(file)

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
        top20_path = out_dir / f"dashboard_top20_{stem}.csv"
        pred_df[top_cols].sort_values("score", ascending=False).head(20).to_csv(top20_path, index=False)

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
