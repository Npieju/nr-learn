import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.features.builder import build_features
from racing_ml.models.trainer import train_and_evaluate


def objective_score(metrics: dict[str, float]) -> float:
    rank1 = float(metrics.get("auc_rank1", 0.0))
    rank2 = float(metrics.get("auc_rank2", 0.0))
    rank3 = float(metrics.get("auc_rank3", 0.0))
    return (0.80 * rank1) + (0.15 * rank2) + (0.05 * rank3)


def build_candidates(base_params: dict) -> list[dict]:
    candidates = [
        {},
        {"learning_rate": 0.02, "num_leaves": 96, "min_data_in_leaf": 40, "feature_fraction": 0.85},
        {"learning_rate": 0.02, "num_leaves": 128, "min_data_in_leaf": 30, "feature_fraction": 0.9, "bagging_fraction": 0.9},
        {"learning_rate": 0.04, "num_leaves": 64, "min_data_in_leaf": 60, "feature_fraction": 0.8},
        {"learning_rate": 0.03, "num_leaves": 80, "min_data_in_leaf": 35, "feature_fraction": 0.9, "bagging_fraction": 0.85},
        {"learning_rate": 0.015, "num_leaves": 160, "min_data_in_leaf": 25, "feature_fraction": 0.95, "bagging_fraction": 0.9},
        {"learning_rate": 0.02, "num_leaves": 96, "min_data_in_leaf": 30, "feature_fraction": 0.9, "bagging_fraction": 0.9, "lambda_l2": 1.0},
        {"learning_rate": 0.02, "num_leaves": 96, "min_data_in_leaf": 25, "feature_fraction": 0.9, "bagging_fraction": 0.9, "lambda_l2": 3.0},
        {"learning_rate": 0.025, "num_leaves": 96, "min_data_in_leaf": 30, "feature_fraction": 0.88, "bagging_fraction": 0.9, "lambda_l2": 1.0},
        {"learning_rate": 0.02, "num_leaves": 80, "min_data_in_leaf": 20, "feature_fraction": 0.92, "bagging_fraction": 0.9, "lambda_l2": 1.0},
        {"learning_rate": 0.02, "num_leaves": 64, "min_data_in_leaf": 25, "feature_fraction": 0.92, "bagging_fraction": 0.9, "lambda_l2": 2.0},
        {"learning_rate": 0.03, "num_leaves": 96, "min_data_in_leaf": 30, "feature_fraction": 0.88, "bagging_fraction": 0.85, "lambda_l2": 1.0},
    ]
    merged: list[dict] = []
    for patch in candidates:
        params = dict(base_params)
        params.update(patch)
        merged.append(params)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_top3.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--max-candidates", type=int, default=6)
    args = parser.parse_args()

    model_cfg = load_yaml(ROOT / args.config)
    data_cfg = load_yaml(ROOT / args.data_config)
    feature_cfg = load_yaml(ROOT / args.feature_config)

    raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
    split_cfg = data_cfg.get("split", {})
    frame = load_training_table(str(ROOT / raw_dir))
    frame = build_features(frame)

    feature_columns = feature_cfg.get("features", {}).get("base", []) + feature_cfg.get("features", {}).get("history", [])
    label_column = model_cfg.get("label", "is_win")
    task = str(model_cfg.get("task", "multi_position"))

    base_params = model_cfg.get("model", {}).get("params", {})
    training_cfg = model_cfg.get("training", {})
    output_cfg = model_cfg.get("output", {})
    model_name = model_cfg.get("model", {}).get("name", "lightgbm")

    candidates = build_candidates(base_params)[: max(1, args.max_candidates)]

    run_rows: list[dict] = []
    best_row: dict | None = None
    best_score = -1.0

    for idx, params in enumerate(candidates, start=1):
        model_file = f"tune_top3_{idx}.joblib"
        report_file = f"tune_top3_{idx}.json"
        print(f"[tune] candidate {idx}/{len(candidates)}: {params}")

        result = train_and_evaluate(
            frame=frame,
            feature_columns=feature_columns,
            label_column=label_column,
            task=task,
            model_name=model_name,
            model_params=params,
            train_end=split_cfg.get("train_end", "2022-12-31"),
            valid_start=split_cfg.get("valid_start", "2023-01-01"),
            valid_end=split_cfg.get("valid_end", "2023-12-31"),
            max_train_rows=training_cfg.get("max_train_rows", 300000),
            max_valid_rows=training_cfg.get("max_valid_rows", 100000),
            early_stopping_rounds=training_cfg.get("early_stopping_rounds", 120),
            allow_fallback=bool(training_cfg.get("allow_fallback_model", False)),
            model_dir=output_cfg.get("model_dir", "artifacts/models"),
            report_dir=output_cfg.get("report_dir", "artifacts/reports"),
            model_file_name=model_file,
            report_file_name=report_file,
        )

        score = objective_score(result.metrics)
        row = {
            "candidate": idx,
            "score": score,
            "params": params,
            "metrics": result.metrics,
            "model_file": model_file,
            "report_file": report_file,
        }
        run_rows.append(row)
        print(f"[tune] candidate={idx} score={score:.6f} auc_rank1={result.metrics.get('auc_rank1')}")

        if score > best_score:
            best_score = score
            best_row = row

    if best_row is None:
        raise RuntimeError("No tuning candidate was evaluated")

    summary = {
        "base_config": args.config,
        "objective": "0.80*auc_rank1 + 0.15*auc_rank2 + 0.05*auc_rank3",
        "n_candidates": len(run_rows),
        "best": best_row,
        "runs": run_rows,
    }

    out_path = ROOT / "artifacts/reports/tune_top3_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(f"[tune] summary saved: {out_path}")
    print(f"[tune] best candidate: {best_row['candidate']}")
    print(f"[tune] best score: {best_score:.6f}")
    print(f"[tune] best params: {best_row['params']}")
    print(f"[tune] best metrics: {best_row['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
