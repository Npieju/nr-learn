from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from racing_ml.evaluation.policy import run_policy_strategy, simulate_annotated_runtime_policy
from racing_ml.evaluation.scoring import resolve_odds_column
from racing_ml.pipeline.backtest_pipeline import _ev_top1_roi, _plot_backtest, _simple_win_roi, _topk_hit_rate
from racing_ml.serving.runtime_policy import annotate_runtime_policy, resolve_runtime_policy


def compute_prediction_backtest_metrics(
    frame: pd.DataFrame,
    *,
    model_config: dict[str, Any],
    config_path: Path,
    prediction_path: Path,
    workspace_root: Path,
) -> dict[str, Any]:
    working = frame.copy()
    working["pred_rank"] = pd.to_numeric(working["pred_rank"], errors="coerce")
    working["score"] = pd.to_numeric(working["score"], errors="coerce")
    if "odds" in working.columns:
        working["odds"] = pd.to_numeric(working["odds"], errors="coerce")
    if "expected_value" not in working.columns and "odds" in working.columns:
        working["expected_value"] = working["score"] * working["odds"]
    working = working.dropna(subset=["pred_rank", "score"])

    metrics: dict[str, Any] = {
        "prediction_file": str(prediction_path.relative_to(workspace_root)),
        "config_file": str(config_path.relative_to(workspace_root)),
        "num_rows": int(len(working)),
        "num_races": int(working["race_id"].nunique()),
        "top1_hit_rate": _topk_hit_rate(working, 1),
        "top3_hit_rate": _topk_hit_rate(working, 3),
        "top5_hit_rate": _topk_hit_rate(working, 5),
        "simple_top1_win_roi": _simple_win_roi(working),
        "ev_top1_win_roi": _ev_top1_roi(working),
    }
    if "score_source" in working.columns:
        score_source_counts = working["score_source"].fillna("default").astype(str).value_counts().to_dict()
        metrics["score_source_count"] = int(len(score_source_counts))
        metrics["score_sources"] = {str(key): int(value) for key, value in score_source_counts.items()}

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


def write_prediction_backtest_artifacts(
    frame: pd.DataFrame,
    *,
    model_config: dict[str, Any],
    config_path: Path,
    prediction_path: Path,
    workspace_root: Path,
    output_json_path: Path,
    output_png_path: Path,
) -> dict[str, Any]:
    metrics = compute_prediction_backtest_metrics(
        frame,
        model_config=model_config,
        config_path=config_path,
        prediction_path=prediction_path,
        workspace_root=workspace_root,
    )
    output_json_path.parent.mkdir(parents=True, exist_ok=True)
    with output_json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)
    _plot_backtest(frame, output_png_path)
    return metrics