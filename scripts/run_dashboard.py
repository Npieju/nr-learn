import argparse
from pathlib import Path
import json
import sys
import time
import traceback
from typing import Any

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


def display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else (Path.cwd() / path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        return None
    return payload


def resolve_profile(*payloads: dict[str, Any] | None) -> str | None:
    for payload in payloads:
        if not isinstance(payload, dict):
            continue
        profile = payload.get("profile")
        if profile:
            return str(profile)
        run_context = payload.get("run_context")
        if isinstance(run_context, dict):
            run_context_profile = run_context.get("profile")
            if run_context_profile:
                return str(run_context_profile)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--backtest-file", default=None)
    parser.add_argument("--train-metrics-file", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=4, prefix="[dashboard]", logger=log_progress, min_interval_sec=0.0)

    try:
        progress.start("resolving dashboard inputs")
        artifacts = ROOT / "artifacts"
        pred_dir = artifacts / "predictions"
        report_dir = artifacts / "reports"

        pred_path = Path(args.predictions_file) if args.predictions_file else latest_file(pred_dir, "predictions_*.csv")
        backtest_path = Path(args.backtest_file) if args.backtest_file else latest_file(report_dir, "backtest_*.json")
        train_metrics_path = Path(args.train_metrics_file) if args.train_metrics_file else report_dir / "train_metrics.json"
        pred_summary_path = pred_path.with_suffix(".summary.json")
        progress.update(
            message=(
                f"inputs resolved predictions={pred_path.name} backtest={backtest_path.name} "
                f"train_metrics={train_metrics_path.name}"
            )
        )

        with Heartbeat("[dashboard]", "loading dashboard inputs", logger=log_progress):
            pred_df = pd.read_csv(pred_path)
            backtest = load_optional_json(backtest_path)
            train_metrics = load_optional_json(train_metrics_path)
            prediction_summary = load_optional_json(pred_summary_path)

            if backtest is None:
                raise RuntimeError(f"Failed to load dashboard backtest JSON: {backtest_path}")
            if train_metrics is None:
                raise RuntimeError(f"Failed to load dashboard train metrics JSON: {train_metrics_path}")
        progress.update(message=f"inputs loaded rows={len(pred_df):,}")

        train_run_context = train_metrics.get("run_context") if isinstance(train_metrics.get("run_context"), dict) else {}
        profile_name = resolve_profile(prediction_summary, backtest, train_metrics)
        score_sources = backtest.get("score_sources") if isinstance(backtest.get("score_sources"), dict) else None
        summary_score_source = prediction_summary.get("score_source") if prediction_summary else None
        if summary_score_source is None and score_sources and len(score_sources) == 1:
            summary_score_source = next(iter(score_sources))
        policy_name = prediction_summary.get("policy_name") if prediction_summary else backtest.get("policy_name")
        policy_strategy_kind = (
            prediction_summary.get("policy_strategy_kind") if prediction_summary else backtest.get("policy_strategy_kind")
        )
        backtest_prediction_file = backtest.get("prediction_file")
        prediction_file_display = display_path(pred_path)

        summary = {
            "profile": profile_name,
            "prediction_file": prediction_file_display,
            "prediction_summary_file": display_path(pred_summary_path) if prediction_summary is not None else None,
            "backtest_file": display_path(backtest_path),
            "train_metrics_file": display_path(train_metrics_path),
            "prediction_profile": prediction_summary.get("profile") if prediction_summary else None,
            "backtest_profile": backtest.get("profile"),
            "train_profile": train_run_context.get("profile") if isinstance(train_run_context, dict) else None,
            "prediction_target_date": prediction_summary.get("target_date") if prediction_summary else None,
            "score_source": summary_score_source,
            "score_source_model_config": prediction_summary.get("score_source_model_config") if prediction_summary else None,
            "policy_name": policy_name,
            "policy_strategy_kind": policy_strategy_kind,
            "manifest_file": prediction_summary.get("manifest_file") if prediction_summary else None,
            "backtest_prediction_file": backtest_prediction_file,
            "backtest_prediction_matches": backtest_prediction_file == prediction_file_display if isinstance(backtest_prediction_file, str) else None,
            "rows": int(len(pred_df)),
            "races": int(pred_df["race_id"].nunique()) if "race_id" in pred_df.columns else None,
            "top1_hit_rate": backtest.get("top1_hit_rate"),
            "top3_hit_rate": backtest.get("top3_hit_rate"),
            "top5_hit_rate": backtest.get("top5_hit_rate"),
            "simple_top1_win_roi": backtest.get("simple_top1_win_roi"),
            "policy_roi": backtest.get("policy_roi"),
            "policy_bets": backtest.get("policy_bets"),
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
        if profile_name is not None:
            print(f"[dashboard] profile: {profile_name}")
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
