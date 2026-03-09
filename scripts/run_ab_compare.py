import argparse
import json
from pathlib import Path
import sys
import time
import traceback

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import resolve_output_artifacts
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.policy import add_market_signals, evaluate_fixed_stake_summary
from racing_ml.evaluation.scoring import generate_prediction_outputs, prepare_scored_frame, resolve_odds_column, topk_hit_rate
from racing_ml.features.builder import build_features
from racing_ml.features.selection import FeatureSelection, prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[ab {now}] {message}", flush=True)


def evaluate_model(name: str, model_cfg: dict, frame: pd.DataFrame, fallback_selection: FeatureSelection, label_col: str) -> dict:
    output_cfg = model_cfg.get("output", {})
    output_artifacts = resolve_output_artifacts(output_cfg)
    model_path = output_artifacts.model_path if output_artifacts.model_path.is_absolute() else (ROOT / output_artifacts.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    feature_selection = resolve_model_feature_selection(model, fallback_selection)
    eval_frame = frame.copy()
    x_eval = prepare_model_input_frame(eval_frame, feature_selection.feature_columns, feature_selection.categorical_columns)
    odds_col = resolve_odds_column(eval_frame)
    outputs = generate_prediction_outputs(model, x_eval, race_ids=eval_frame["race_id"])
    eval_frame = prepare_scored_frame(eval_frame, outputs.score, odds_col=odds_col, score_col="score")
    if odds_col is not None:
        eval_frame = add_market_signals(eval_frame, score_col="score", odds_col=odds_col)

    policy_summary = evaluate_fixed_stake_summary(eval_frame, odds_col=odds_col, score_col="score", stake=100.0)
    metrics = {
        "model": name,
        "model_file": str(model_path.relative_to(ROOT)),
        "manifest_file": output_artifacts.manifest_path.as_posix() if (ROOT / output_artifacts.manifest_path).exists() else None,
        "feature_count": int(len(feature_selection.feature_columns)),
        "categorical_feature_count": int(len(feature_selection.categorical_columns)),
        "n_rows": int(len(eval_frame)),
        "n_races": int(eval_frame["race_id"].nunique()),
        "top1_hit_rate": topk_hit_rate(eval_frame, 1),
        "top3_hit_rate": topk_hit_rate(eval_frame, 3),
        "top1_roi": policy_summary.get("top1_roi"),
    }

    if label_col in eval_frame.columns:
        y_true = eval_frame[label_col].astype(int).to_numpy()
        if len(np.unique(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, eval_frame["score"].to_numpy()))
        else:
            metrics["auc"] = None

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/model.yaml")
    parser.add_argument("--challenger-config", default="configs/model_ranker.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--max-rows", type=int, default=30000)
    args = parser.parse_args()

    try:
        base_cfg = load_yaml(ROOT / args.base_config)
        challenger_cfg = load_yaml(ROOT / args.challenger_config)
        data_cfg = load_yaml(ROOT / args.data_config)
        feature_cfg = load_yaml(ROOT / args.feature_config)
        progress = ProgressBar(total=7, prefix="[ab]", logger=log_progress, min_interval_sec=0.0)
        progress.start("configs loaded")

        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        with Heartbeat("[ab]", "loading training table", logger=log_progress):
            frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
        progress.update(message=f"training table loaded rows={len(frame):,}")

        with Heartbeat("[ab]", "building features", logger=log_progress):
            frame = build_features(frame)
        progress.update(message=f"features built columns={len(frame.columns):,}")

        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()
        progress.update(message=f"evaluation slice ready rows={len(frame):,}")

        label_col = str(base_cfg.get("label", "is_win"))
        fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
        if not fallback_selection.feature_columns:
            raise RuntimeError("No configured feature columns found for comparison")
        progress.update(
            message=(
                f"feature selection ready features={len(fallback_selection.feature_columns):,} "
                f"categorical={len(fallback_selection.categorical_columns):,}"
            )
        )

        with Heartbeat("[ab]", "evaluating baseline model", logger=log_progress):
            base_metrics = evaluate_model("baseline", base_cfg, frame, fallback_selection, label_col)
        progress.update(message="baseline evaluated")

        with Heartbeat("[ab]", "evaluating challenger model", logger=log_progress):
            challenger_metrics = evaluate_model("challenger", challenger_cfg, frame, fallback_selection, label_col)
        progress.update(message="challenger evaluated")

        summary = {
            "base_config": args.base_config,
            "challenger_config": args.challenger_config,
            "max_rows": int(len(frame)),
            "baseline": base_metrics,
            "challenger": challenger_metrics,
            "delta": {
                "auc": (
                    (challenger_metrics.get("auc") or 0.0) - (base_metrics.get("auc") or 0.0)
                    if challenger_metrics.get("auc") is not None and base_metrics.get("auc") is not None
                    else None
                ),
                "top1_hit_rate": (
                    (challenger_metrics.get("top1_hit_rate") or 0.0) - (base_metrics.get("top1_hit_rate") or 0.0)
                    if challenger_metrics.get("top1_hit_rate") is not None and base_metrics.get("top1_hit_rate") is not None
                    else None
                ),
                "top1_roi": (
                    (challenger_metrics.get("top1_roi") or 0.0) - (base_metrics.get("top1_roi") or 0.0)
                    if challenger_metrics.get("top1_roi") is not None and base_metrics.get("top1_roi") is not None
                    else None
                ),
            },
        }

        out_path = ROOT / "artifacts/reports/ab_compare_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Heartbeat("[ab]", "writing comparison summary", logger=log_progress):
            with out_path.open("w", encoding="utf-8") as file:
                json.dump(summary, file, ensure_ascii=False, indent=2)
        progress.complete(message="comparison summary written")

        print(f"[ab] summary saved: {out_path}")
        print(f"[ab] baseline: {base_metrics}")
        print(f"[ab] challenger: {challenger_metrics}")
        print(f"[ab] delta: {summary['delta']}")
        return 0
    except KeyboardInterrupt:
        print("[ab] interrupted by user")
        return 130
    except Exception as error:
        print(f"[ab] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())