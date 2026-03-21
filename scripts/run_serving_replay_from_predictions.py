from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.evaluation.policy import run_policy_strategy, simulate_annotated_runtime_policy
from racing_ml.evaluation.scoring import resolve_odds_column
from racing_ml.pipeline.backtest_pipeline import _ev_top1_roi, _plot_backtest, _simple_win_roi, _topk_hit_rate
from racing_ml.serving.runtime_policy import annotate_runtime_policy, resolve_runtime_policy


DATE_PATTERN = re.compile(r"predictions_(\d{8})")


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


def _compute_metrics(frame: pd.DataFrame, model_config: dict[str, Any], config_path: Path, prediction_path: Path) -> dict[str, Any]:
    working = frame.copy()
    working["pred_rank"] = pd.to_numeric(working["pred_rank"], errors="coerce")
    working["score"] = pd.to_numeric(working["score"], errors="coerce")
    if "odds" in working.columns:
        working["odds"] = pd.to_numeric(working["odds"], errors="coerce")
    if "expected_value" not in working.columns and "odds" in working.columns:
        working["expected_value"] = working["score"] * working["odds"]
    working = working.dropna(subset=["pred_rank", "score"])

    metrics: dict[str, Any] = {
        "prediction_file": _relative(prediction_path),
        "config_file": _relative(config_path),
        "num_rows": int(len(working)),
        "num_races": int(working["race_id"].nunique()),
        "top1_hit_rate": _topk_hit_rate(working, 1),
        "top3_hit_rate": _topk_hit_rate(working, 3),
        "top5_hit_rate": _topk_hit_rate(working, 5),
        "simple_top1_win_roi": _simple_win_roi(working),
        "ev_top1_win_roi": _ev_top1_roi(working),
    }

    odds_col = resolve_odds_column(working)
    policy_resolution = resolve_runtime_policy(model_config, frame=working)
    if odds_col is None or policy_resolution is None:
        return metrics

    policy_name, policy_config = policy_resolution
    policy_frame = annotate_runtime_policy(
        working,
        odds_col=odds_col,
        policy_name=policy_name,
        policy_config=policy_config,
        score_col="score",
    )
    policy_strategy_kind = str(policy_config.get("strategy_kind", "")).strip().lower()
    if policy_strategy_kind == "staged":
        policy_metrics = simulate_annotated_runtime_policy(policy_frame, odds_col)
    else:
        policy_metrics = run_policy_strategy(
            policy_frame,
            prob_col="policy_prob",
            odds_col=odds_col,
            params=policy_config,
        )

    selected_mask = policy_frame["policy_selected"].fillna(False).astype(bool)
    metrics["policy_name"] = policy_name
    metrics["policy_strategy_kind"] = policy_strategy_kind
    metrics["policy_blend_weight"] = None if policy_strategy_kind == "staged" else float(policy_config.get("blend_weight", 1.0))
    metrics["policy_selected_rows"] = int(selected_mask.sum())
    metrics["policy_selected_races"] = int(policy_frame.loc[selected_mask, "race_id"].nunique()) if selected_mask.any() else 0
    if policy_strategy_kind == "portfolio":
        metrics["policy_roi"] = policy_metrics.get("portfolio_roi")
        metrics["policy_bets"] = int(policy_metrics.get("portfolio_bets") or 0)
        metrics["policy_hit_rate"] = policy_metrics.get("portfolio_hit_rate")
        metrics["policy_final_bankroll"] = policy_metrics.get("portfolio_final_bankroll")
        metrics["policy_max_drawdown"] = policy_metrics.get("portfolio_max_drawdown")
        metrics["policy_avg_synthetic_odds"] = policy_metrics.get("portfolio_avg_synthetic_odds")
    elif policy_strategy_kind == "staged":
        metrics["policy_roi"] = policy_metrics.get("policy_roi")
        metrics["policy_bets"] = int(policy_metrics.get("policy_bets") or 0)
        metrics["policy_hit_rate"] = policy_metrics.get("policy_hit_rate")
        metrics["policy_final_bankroll"] = policy_metrics.get("policy_final_bankroll")
        metrics["policy_max_drawdown"] = policy_metrics.get("policy_max_drawdown")
        metrics["policy_avg_synthetic_odds"] = policy_metrics.get("policy_avg_synthetic_odds")
        selected_stage_values = policy_frame.loc[selected_mask, "policy_stage_name"].dropna().astype(str).tolist() if "policy_stage_name" in policy_frame.columns else []
        metrics["policy_stage_names"] = sorted(set(selected_stage_values))
    else:
        metrics["policy_roi"] = policy_metrics.get("kelly_roi")
        metrics["policy_bets"] = int(policy_metrics.get("kelly_bets") or 0)
        metrics["policy_hit_rate"] = policy_metrics.get("kelly_hit_rate")
        metrics["policy_final_bankroll"] = policy_metrics.get("kelly_final_bankroll")
        metrics["policy_max_drawdown"] = policy_metrics.get("kelly_max_drawdown")

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--prediction-files", nargs="+", required=True)
    parser.add_argument("--artifact-suffix", required=True)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--output-file", default=None)
    args = parser.parse_args()

    config_path = _normalize_path(args.config)
    model_config = load_yaml(config_path)
    prediction_paths = [_normalize_path(path) for path in args.prediction_files]
    output_file = _normalize_path(args.output_file) if args.output_file else ROOT / "artifacts" / "reports" / f"serving_smoke_{args.artifact_suffix}.json"
    report_dir = ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    cases: list[dict[str, Any]] = []
    for prediction_path in prediction_paths:
        frame = pd.read_csv(prediction_path)
        metrics = _compute_metrics(frame, model_config, config_path, prediction_path)
        stem = prediction_path.stem.replace("predictions_", "")
        archived_backtest_json = report_dir / f"backtest_{stem}_{args.artifact_suffix}.json"
        archived_backtest_png = report_dir / f"backtest_{stem}_{args.artifact_suffix}.png"
        with archived_backtest_json.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)
        _plot_backtest(frame, archived_backtest_png)

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

    payload = {
        "profile": args.profile or args.artifact_suffix,
        "config_file": _relative(config_path),
        "artifact_suffix": args.artifact_suffix,
        "cases": cases,
    }
    with output_file.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print(f"saved replay summary to {_relative(output_file)}")
    print(f"processed_cases={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())