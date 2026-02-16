import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.probability import normalize_position_probabilities
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.features.builder import build_features


def predict_score(model: Any, features: pd.DataFrame, race_ids: pd.Series | None = None) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(features), dtype=float)
    if isinstance(model, dict) and model.get("kind") == "multi_position_top3":
        prep = model.get("prep")
        models = model.get("models", {})
        if prep is None or not isinstance(models, dict):
            raise RuntimeError("Invalid multi_position model bundle")
        rank1_model = models.get("p_rank1")
        if rank1_model is None:
            raise RuntimeError("Missing p_rank1 model in bundle")
        if race_ids is None:
            raise RuntimeError("race_ids are required for multi_position model scoring")
        transformed = prep.transform(features)
        raw = rank1_model.predict_proba(transformed)[:, 1]
        work = pd.DataFrame({"race_id": race_ids.to_numpy(copy=False)})
        work["p_rank1_raw"] = raw
        work = normalize_position_probabilities(
            work,
            raw_columns=["p_rank1_raw"],
            race_id_col="race_id",
            output_prefix="",
        )
        return work["p_rank1_raw"].to_numpy(dtype=float)
    raise RuntimeError("Loaded model does not support predict/predict_proba")


def rank_by_score(frame: pd.DataFrame, score_col: str, out_col: str = "pred_rank") -> pd.DataFrame:
    ranked = frame.copy()
    ranked[out_col] = (
        ranked.groupby("race_id")[score_col]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return ranked


def topk_hit_rate(frame: pd.DataFrame, k: int) -> float | None:
    if "rank" not in frame.columns:
        return None
    hits: list[int] = []
    for _, group in frame.groupby("race_id"):
        picks = group[group["pred_rank"] <= k]
        rank_values = pd.to_numeric(picks["rank"], errors="coerce")
        hits.append(int((rank_values == 1).any()))
    return float(np.mean(hits)) if hits else None


def top1_roi(frame: pd.DataFrame, stake: float = 100.0) -> float | None:
    if "rank" not in frame.columns or "odds" not in frame.columns:
        return None

    total_bet = 0.0
    total_return = 0.0
    for _, group in frame.groupby("race_id"):
        pick = group.sort_values("pred_rank").iloc[0]
        total_bet += stake
        rank = pd.to_numeric(pd.Series([pick.get("rank")]), errors="coerce").iloc[0]
        if pd.notna(rank) and int(rank) == 1:
            odds = pd.to_numeric(pd.Series([pick.get("odds")]), errors="coerce").iloc[0]
            if pd.notna(odds) and float(odds) > 0:
                total_return += float(stake * float(odds))
    if total_bet == 0:
        return None
    return float(total_return / total_bet)


def evaluate_model(name: str, model_cfg: dict, frame: pd.DataFrame, feature_columns: list[str], label_col: str) -> dict:
    output_cfg = model_cfg.get("output", {})
    model_path = ROOT / output_cfg.get("model_dir", "artifacts/models") / output_cfg.get("model_file", "baseline_model.joblib")
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    eval_frame = frame.copy()
    x_eval = eval_frame[feature_columns]
    eval_frame["score"] = predict_score(model, x_eval, eval_frame["race_id"])
    eval_frame = rank_by_score(eval_frame, score_col="score")

    metrics = {
        "model": name,
        "model_file": str(model_path.relative_to(ROOT)),
        "n_rows": int(len(eval_frame)),
        "n_races": int(eval_frame["race_id"].nunique()),
        "top1_hit_rate": topk_hit_rate(eval_frame, 1),
        "top3_hit_rate": topk_hit_rate(eval_frame, 3),
        "top1_roi": top1_roi(eval_frame),
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

        raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
        frame = load_training_table(str(ROOT / raw_dir))
        frame = build_features(frame)
        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()

        features_cfg = feature_cfg.get("features", {})
        feature_columns = features_cfg.get("base", []) + features_cfg.get("history", [])
        available_features = [column for column in feature_columns if column in frame.columns]
        if not available_features:
            raise RuntimeError("No configured feature columns found for comparison")

        odds_col = next((column for column in ["odds", "単勝"] if column in frame.columns), None)
        if odds_col and odds_col != "odds":
            frame = frame.rename(columns={odds_col: "odds"})

        label_col = str(base_cfg.get("label", "is_win"))
        base_metrics = evaluate_model("baseline", base_cfg, frame, available_features, label_col)
        challenger_metrics = evaluate_model("challenger", challenger_cfg, frame, available_features, label_col)

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
        with out_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

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
