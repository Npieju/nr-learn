from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path
import sys
import time
import traceback

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import append_suffix_to_file_name
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import resolve_output_artifacts
from racing_ml.common.artifacts import write_csv_file, write_text_file
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table_for_feature_build
from racing_ml.evaluation.policy import (
    PolicyConstraints,
    add_market_signals,
    apply_selection_mode,
    blend_prob,
    compute_market_prob,
    evaluate_candidate_gate,
    run_policy_strategy,
)
from racing_ml.evaluation.scoring import generate_prediction_outputs, prepare_scored_frame, resolve_odds_column
from racing_ml.evaluation.stability import build_stability_guardrail
from racing_ml.evaluation.walk_forward import (
    _resolve_regime_search_config,
    _resolve_search_candidate_values,
    build_nested_wf_slices,
    fit_isotonic,
)
from racing_ml.features.builder import build_features
from racing_ml.features.selection import prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[wf-feasibility {now}] {message}", flush=True)


def _count_strategy_evals_per_outer_step(
    *,
    kelly_frac_candidates: list[float],
    max_frac_candidates: list[float],
    top_k_candidates: list[int],
    min_ev_candidates: list[float],
) -> int:
    return (
        len(kelly_frac_candidates) * len(max_frac_candidates)
        + len(top_k_candidates) * len(min_ev_candidates)
    )


def _count_outer_search_steps(
    *,
    blend_candidates: list[float],
    edge_candidates: list[float],
    min_prob_candidates: list[float],
    odds_min_candidates: list[float],
    odds_max_candidates: list[float],
) -> int:
    return (
        len(blend_candidates)
        * len(edge_candidates)
        * len(min_prob_candidates)
        * len(odds_min_candidates)
        * len(odds_max_candidates)
    )


def _should_emit_checkpoint(
    *,
    processed: int,
    total: int,
    checkpoint_interval: int,
    now_monotonic: float,
    last_progress_at: float,
    max_silent_seconds: float | None,
) -> bool:
    if processed <= 0:
        return False
    if processed == 1 or processed == total or processed % checkpoint_interval == 0:
        return True
    if max_silent_seconds is not None and max_silent_seconds > 0 and (now_monotonic - last_progress_at) >= max_silent_seconds:
        return True
    return False


def _sanitize_output_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def _derive_date_window_slug(start_date: str | None, end_date: str | None) -> str:
    if not start_date and not end_date:
        return ""

    start_token = _sanitize_output_slug((start_date or "start_auto").replace("-", ""))
    end_token = _sanitize_output_slug((end_date or "end_auto").replace("-", ""))
    return f"_{start_token}_{end_token}"


def _derive_wf_slug(wf_mode: str, wf_scheme: str) -> str:
    normalized_mode = _sanitize_output_slug(wf_mode)
    normalized_scheme = _sanitize_output_slug(wf_scheme)
    return f"_wf_{normalized_mode}_{normalized_scheme}"


def _derive_output_slug(config_path: str, model_path: Path) -> str:
    candidates = [Path(config_path).stem, model_path.stem]
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        if text.endswith("_model"):
            text = text[: -len("_model")]
        if text.startswith("model_"):
            text = text[len("model_") :]
        slug = _sanitize_output_slug(text)
        if slug:
            return slug
    return "model"


def _resolve_feasibility_output_paths(
    *,
    report_dir: Path,
    config_path: str,
    model_path: Path,
    wf_mode: str,
    wf_scheme: str,
    start_date: str | None,
    end_date: str | None,
    summary_output: str | None,
    detail_output: str | None,
) -> tuple[Path, Path]:
    if summary_output:
        summary_path = ROOT / summary_output if not Path(summary_output).is_absolute() else Path(summary_output)
    else:
        output_slug = _derive_output_slug(config_path, model_path)
        date_slug = _derive_date_window_slug(start_date, end_date)
        wf_slug = _derive_wf_slug(wf_mode, wf_scheme)
        summary_path = report_dir / f"wf_feasibility_diag_{output_slug}{date_slug}{wf_slug}.json"

    if detail_output:
        detail_path = ROOT / detail_output if not Path(detail_output).is_absolute() else Path(detail_output)
    else:
        if summary_path.suffix == ".json":
            detail_path = summary_path.with_suffix(".csv")
        else:
            detail_path = summary_path.parent / f"{summary_path.name}.csv"

    return summary_path, detail_path


def _filter_frame_by_date_window(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return frame

    if "date" not in frame.columns:
        raise RuntimeError("Date filtering requires a 'date' column")

    date_series = pd.to_datetime(frame["date"], errors="coerce")
    mask = date_series.notna()
    if start_date:
        mask &= date_series >= pd.Timestamp(start_date)
    if end_date:
        mask &= date_series <= pd.Timestamp(end_date)

    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise RuntimeError(
            f"No rows found in date window: {start_date or '-inf'} .. {end_date or '+inf'}"
        )
    return filtered


def _serialize_candidate(row: dict[str, object] | None) -> dict[str, object] | None:
    if row is None:
        return None

    strategy_kind = str(row.get("strategy_kind") or "")
    params: dict[str, object] = {}
    for key in ["blend_weight", "min_prob", "odds_min", "odds_max"]:
        if key in row:
            params[key] = row.get(key)
    if strategy_kind == "kelly":
        for key in ["min_edge", "fractional_kelly", "max_fraction"]:
            if key in row:
                params[key] = row.get(key)
    if strategy_kind == "portfolio":
        for key in ["top_k", "min_expected_value"]:
            if key in row:
                params[key] = row.get(key)

    return {
        "strategy_kind": row.get("strategy_kind"),
        "params": params,
        "roi": row.get("roi"),
        "bets": row.get("bets"),
        "hit_rate": row.get("hit_rate"),
        "final_bankroll": row.get("final_bankroll"),
        "max_drawdown": row.get("max_drawdown"),
        "gate_failures": row.get("gate_failures"),
        "is_feasible": row.get("is_feasible"),
        "base_score": row.get("base_score"),
        "selection_score": row.get("selection_score"),
    }


def _extract_strategy_params(row: dict[str, object] | None) -> dict[str, float | int | str]:
    if not isinstance(row, dict):
        return {}
    strategy_kind = str(row.get("strategy_kind") or "")
    params: dict[str, float | int | str] = {
        "strategy_kind": strategy_kind,
        "blend_weight": float(row.get("blend_weight") or 0.0),
        "min_prob": float(row.get("min_prob") or 0.0),
        "odds_min": float(row.get("odds_min") or 0.0),
        "odds_max": float(row.get("odds_max") or 0.0),
    }
    if strategy_kind == "kelly":
        params.update(
            {
                "min_edge": float(row.get("min_edge") or 0.0),
                "fractional_kelly": float(row.get("fractional_kelly") or 0.0),
                "max_fraction": float(row.get("max_fraction") or 0.0),
            }
        )
    elif strategy_kind == "portfolio":
        params.update(
            {
                "top_k": int(row.get("top_k") or 0),
                "min_expected_value": float(row.get("min_expected_value") or 0.0),
            }
        )
    return params


def _build_candidate_snapshot_from_metrics(
    *,
    strategy_kind: str,
    params: dict[str, float | int | str],
    metrics: dict[str, float | int | None],
) -> dict[str, object]:
    if strategy_kind == "kelly":
        row = {
            "strategy_kind": strategy_kind,
            **params,
            "roi": metrics.get("kelly_roi"),
            "bets": int(metrics.get("kelly_bets") or 0),
            "hit_rate": metrics.get("kelly_hit_rate"),
            "final_bankroll": metrics.get("kelly_final_bankroll"),
            "max_drawdown": metrics.get("kelly_max_drawdown"),
        }
    elif strategy_kind == "portfolio":
        row = {
            "strategy_kind": strategy_kind,
            **params,
            "roi": metrics.get("portfolio_roi"),
            "bets": int(metrics.get("portfolio_bets") or 0),
            "hit_rate": metrics.get("portfolio_hit_rate"),
            "final_bankroll": metrics.get("portfolio_final_bankroll"),
            "max_drawdown": metrics.get("portfolio_max_drawdown"),
        }
    else:
        row = {"strategy_kind": strategy_kind, **params, "roi": None, "bets": 0, "hit_rate": None, "final_bankroll": None, "max_drawdown": None}
    return _serialize_candidate(row) or {}


def _summarize_fold_candidates(
    *,
    fold_index: int,
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame,
    label_col: str,
    odds_col: str,
    constraints: PolicyConstraints,
    search_config: dict[str, object] | None,
    mode: str,
    search_space_callback: Callable[[dict[str, object]], None] | None = None,
    progress_callback: Callable[[int, int, dict[str, object]], None] | None = None,
    max_silent_seconds: float | None = None,
    max_fold_elapsed_seconds: float | None = None,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    resolved_search_config = _resolve_regime_search_config(search_config, frame=valid_df)

    blend_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="blend_weights",
        default=[0.2, 0.4, 0.6, 0.8],
        cast=float,
    )
    edge_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="min_edges",
        default=[0.01, 0.03, 0.05],
        cast=float,
    )
    min_prob_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="min_probabilities",
        default=[0.03, 0.05],
        cast=float,
    )
    kelly_frac_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="fractional_kelly_values",
        default=[0.25, 0.5],
        cast=float,
    )
    max_frac_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="max_fraction_values",
        default=[0.02, 0.05],
        cast=float,
    )
    odds_min_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="odds_mins",
        default=[1.0],
        cast=float,
    )
    odds_max_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="odds_maxs",
        default=[25.0, 40.0, 80.0],
        cast=float,
    )
    top_k_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="top_ks",
        default=[1, 2],
        cast=int,
    )
    min_ev_candidates = _resolve_search_candidate_values(
        resolved_search_config,
        mode=mode,
        key="min_expected_values",
        default=[1.0, 1.05, 1.10],
        cast=float,
    )

    train_scores = train_df["score"].to_numpy()
    train_labels = train_df[label_col].astype(int).to_numpy()
    valid_df = valid_df.copy()
    valid_df["iso_prob"] = fit_isotonic(train_scores, train_labels, valid_df["score"].to_numpy())
    valid_df["market_prob"] = compute_market_prob(valid_df, odds_col=odds_col)

    valid_races = int(valid_df["race_id"].nunique())
    min_bets_required = constraints.min_bets_required(valid_races)
    candidate_rows: list[dict[str, object]] = []
    best_feasible: dict[str, object] | None = None
    best_feasible_score = float("-inf")
    best_fallback: dict[str, object] | None = None
    best_fallback_score = float("-inf")
    outer_search_steps = _count_outer_search_steps(
        blend_candidates=blend_candidates,
        edge_candidates=edge_candidates,
        min_prob_candidates=min_prob_candidates,
        odds_min_candidates=odds_min_candidates,
        odds_max_candidates=odds_max_candidates,
    )
    strategy_evals_per_outer_step = _count_strategy_evals_per_outer_step(
        kelly_frac_candidates=kelly_frac_candidates,
        max_frac_candidates=max_frac_candidates,
        top_k_candidates=top_k_candidates,
        min_ev_candidates=min_ev_candidates,
    )
    total_search_steps = outer_search_steps * strategy_evals_per_outer_step
    search_space = {
        "blend_weights": int(len(blend_candidates)),
        "min_edges": int(len(edge_candidates)),
        "min_probabilities": int(len(min_prob_candidates)),
        "odds_mins": int(len(odds_min_candidates)),
        "odds_maxs": int(len(odds_max_candidates)),
        "fractional_kelly_values": int(len(kelly_frac_candidates)),
        "max_fraction_values": int(len(max_frac_candidates)),
        "top_ks": int(len(top_k_candidates)),
        "min_expected_values": int(len(min_ev_candidates)),
        "outer_search_steps": int(outer_search_steps),
        "strategy_evals_per_outer_step": int(strategy_evals_per_outer_step),
        "total_search_steps": int(total_search_steps),
    }
    processed_search_steps = 0
    checkpoint_interval = max(1, total_search_steps // 16)
    fold_started_at = time.monotonic()
    last_progress_at = fold_started_at
    if search_space_callback is not None:
        search_space_callback(search_space)

    for blend_weight in blend_candidates:
        valid_df["blend_prob"] = blend_prob(valid_df["iso_prob"], valid_df["market_prob"], float(blend_weight))
        for min_edge in edge_candidates:
            for min_prob in min_prob_candidates:
                for odds_min in odds_min_candidates:
                    for odds_max in odds_max_candidates:
                        for frac in kelly_frac_candidates:
                            for max_frac in max_frac_candidates:
                                params = {
                                    "strategy_kind": "kelly",
                                    "blend_weight": float(blend_weight),
                                    "min_edge": float(min_edge),
                                    "min_prob": float(min_prob),
                                    "fractional_kelly": float(frac),
                                    "max_fraction": float(max_frac),
                                    "odds_min": float(odds_min),
                                    "odds_max": float(odds_max),
                                }
                                metrics = run_policy_strategy(valid_df, prob_col="blend_prob", odds_col=odds_col, params=params)
                                processed_search_steps += 1
                                roi = metrics.get("kelly_roi")
                                hit_rate = float(metrics.get("kelly_hit_rate") or 0.0)
                                base_score = float(roi) + 0.10 * hit_rate if roi is not None else None
                                gate_result = evaluate_candidate_gate(
                                    bets=int(metrics.get("kelly_bets") or 0),
                                    max_drawdown=float(metrics.get("kelly_max_drawdown")) if metrics.get("kelly_max_drawdown") is not None else None,
                                    final_bankroll=float(metrics.get("kelly_final_bankroll")) if metrics.get("kelly_final_bankroll") is not None else None,
                                    constraints=constraints,
                                    n_races=valid_races,
                                )
                                selection_score = (
                                    apply_selection_mode(
                                        float(base_score),
                                        gate_result,
                                        constraints,
                                        bets=int(metrics.get("kelly_bets") or 0),
                                        max_drawdown=float(metrics.get("kelly_max_drawdown")) if metrics.get("kelly_max_drawdown") is not None else None,
                                        final_bankroll=float(metrics.get("kelly_final_bankroll")) if metrics.get("kelly_final_bankroll") is not None else None,
                                    )
                                    if base_score is not None
                                    else None
                                )
                                row = {
                                    "fold": int(fold_index),
                                    "strategy_kind": "kelly",
                                    "roi": metrics.get("kelly_roi"),
                                    "bets": int(metrics.get("kelly_bets") or 0),
                                    "hit_rate": metrics.get("kelly_hit_rate"),
                                    "final_bankroll": metrics.get("kelly_final_bankroll"),
                                    "max_drawdown": metrics.get("kelly_max_drawdown"),
                                    "base_score": base_score,
                                    "selection_score": selection_score,
                                    "is_feasible": bool(gate_result.get("is_feasible")),
                                    "gate_failures": list(gate_result.get("gate_failures", [])),
                                    "min_bets_required": int(gate_result.get("min_bets_required", min_bets_required)),
                                    **params,
                                }
                                candidate_rows.append(row)
                                if base_score is not None and base_score > best_fallback_score:
                                    best_fallback_score = float(base_score)
                                    best_fallback = row
                                if selection_score is not None and selection_score > best_feasible_score:
                                    best_feasible_score = float(selection_score)
                                    best_feasible = row
                                now_monotonic = time.monotonic()
                                if progress_callback is not None and _should_emit_checkpoint(
                                    processed=processed_search_steps,
                                    total=total_search_steps,
                                    checkpoint_interval=checkpoint_interval,
                                    now_monotonic=now_monotonic,
                                    last_progress_at=last_progress_at,
                                    max_silent_seconds=max_silent_seconds,
                                ):
                                    progress_callback(
                                        processed_search_steps,
                                        total_search_steps,
                                        {
                                            "fold": int(fold_index),
                                            "strategy_kind": "kelly",
                                            "blend_weight": float(blend_weight),
                                            "min_edge": float(min_edge),
                                            "min_prob": float(min_prob),
                                            "odds_min": float(odds_min),
                                            "odds_max": float(odds_max),
                                            "candidate_rows": int(len(candidate_rows)),
                                        },
                                    )
                                    last_progress_at = now_monotonic
                                if max_fold_elapsed_seconds is not None and max_fold_elapsed_seconds > 0 and (now_monotonic - fold_started_at) >= max_fold_elapsed_seconds:
                                    raise RuntimeError(
                                        f"wf feasibility fold {fold_index} exceeded max fold elapsed seconds "
                                        f"({max_fold_elapsed_seconds:.0f}s); interrupted for operator review"
                                    )

                        for top_k in top_k_candidates:
                            for min_ev in min_ev_candidates:
                                params = {
                                    "strategy_kind": "portfolio",
                                    "blend_weight": float(blend_weight),
                                    "min_prob": float(min_prob),
                                    "odds_min": float(odds_min),
                                    "odds_max": float(odds_max),
                                    "top_k": int(top_k),
                                    "min_expected_value": float(min_ev),
                                }
                                metrics = run_policy_strategy(valid_df, prob_col="blend_prob", odds_col=odds_col, params=params)
                                processed_search_steps += 1
                                roi = metrics.get("portfolio_roi")
                                hit_rate = float(metrics.get("portfolio_hit_rate") or 0.0)
                                base_score = float(roi) + 0.20 * hit_rate if roi is not None else None
                                gate_result = evaluate_candidate_gate(
                                    bets=int(metrics.get("portfolio_bets") or 0),
                                    max_drawdown=float(metrics.get("portfolio_max_drawdown")) if metrics.get("portfolio_max_drawdown") is not None else None,
                                    final_bankroll=float(metrics.get("portfolio_final_bankroll")) if metrics.get("portfolio_final_bankroll") is not None else None,
                                    constraints=constraints,
                                    n_races=valid_races,
                                )
                                selection_score = (
                                    apply_selection_mode(
                                        float(base_score),
                                        gate_result,
                                        constraints,
                                        bets=int(metrics.get("portfolio_bets") or 0),
                                        max_drawdown=float(metrics.get("portfolio_max_drawdown")) if metrics.get("portfolio_max_drawdown") is not None else None,
                                        final_bankroll=float(metrics.get("portfolio_final_bankroll")) if metrics.get("portfolio_final_bankroll") is not None else None,
                                    )
                                    if base_score is not None
                                    else None
                                )
                                row = {
                                    "fold": int(fold_index),
                                    "strategy_kind": "portfolio",
                                    "roi": metrics.get("portfolio_roi"),
                                    "bets": int(metrics.get("portfolio_bets") or 0),
                                    "hit_rate": metrics.get("portfolio_hit_rate"),
                                    "final_bankroll": metrics.get("portfolio_final_bankroll"),
                                    "max_drawdown": metrics.get("portfolio_max_drawdown"),
                                    "base_score": base_score,
                                    "selection_score": selection_score,
                                    "is_feasible": bool(gate_result.get("is_feasible")),
                                    "gate_failures": list(gate_result.get("gate_failures", [])),
                                    "min_bets_required": int(gate_result.get("min_bets_required", min_bets_required)),
                                    **params,
                                }
                                candidate_rows.append(row)
                                if base_score is not None and base_score > best_fallback_score:
                                    best_fallback_score = float(base_score)
                                    best_fallback = row
                                if selection_score is not None and selection_score > best_feasible_score:
                                    best_feasible_score = float(selection_score)
                                    best_feasible = row
                                now_monotonic = time.monotonic()
                                if progress_callback is not None and _should_emit_checkpoint(
                                    processed=processed_search_steps,
                                    total=total_search_steps,
                                    checkpoint_interval=checkpoint_interval,
                                    now_monotonic=now_monotonic,
                                    last_progress_at=last_progress_at,
                                    max_silent_seconds=max_silent_seconds,
                                ):
                                    progress_callback(
                                        processed_search_steps,
                                        total_search_steps,
                                        {
                                            "fold": int(fold_index),
                                            "strategy_kind": "portfolio",
                                            "blend_weight": float(blend_weight),
                                            "min_edge": float(min_edge),
                                            "min_prob": float(min_prob),
                                            "odds_min": float(odds_min),
                                            "odds_max": float(odds_max),
                                            "candidate_rows": int(len(candidate_rows)),
                                        },
                                    )
                                    last_progress_at = now_monotonic
                                if max_fold_elapsed_seconds is not None and max_fold_elapsed_seconds > 0 and (now_monotonic - fold_started_at) >= max_fold_elapsed_seconds:
                                    raise RuntimeError(
                                        f"wf feasibility fold {fold_index} exceeded max fold elapsed seconds "
                                        f"({max_fold_elapsed_seconds:.0f}s); interrupted for operator review"
                                    )

    failure_reason_counts = Counter(
        reason for row in candidate_rows for reason in row.get("gate_failures", [])
    )
    failure_combo_counts = Counter(
        ",".join(row.get("gate_failures", []))
        for row in candidate_rows
        if row.get("gate_failures")
    )
    feasible_by_strategy = Counter(
        str(row.get("strategy_kind"))
        for row in candidate_rows
        if bool(row.get("is_feasible"))
    )

    def closest_key(row: dict[str, object]) -> tuple[float, float, float, float, float]:
        bets_gap = max(0, min_bets_required - int(row.get("bets") or 0))
        drawdown_gap = max(0.0, float(row.get("max_drawdown") or 0.0) - constraints.max_drawdown)
        bankroll_gap = max(0.0, constraints.min_final_bankroll - float(row.get("final_bankroll") or 0.0))
        base_score = float(row.get("base_score")) if row.get("base_score") is not None else float("-inf")
        return (
            float(len(row.get("gate_failures", []))),
            float(bets_gap),
            float(drawdown_gap),
            float(bankroll_gap),
            -base_score,
        )

    closest_infeasible = [
        _serialize_candidate(row)
        for row in sorted(
            [row for row in candidate_rows if not bool(row.get("is_feasible"))],
            key=closest_key,
        )[:3]
    ]

    best_feasible_test: dict[str, object] | None = None
    if isinstance(best_feasible, dict):
        best_feasible_params = _extract_strategy_params(best_feasible)
        strategy_kind = str(best_feasible_params.get("strategy_kind") or "")
        if strategy_kind in {"kelly", "portfolio"}:
            test_eval = test_df.copy()
            test_eval["iso_prob"] = fit_isotonic(train_scores, train_labels, test_eval["score"].to_numpy())
            test_eval["market_prob"] = compute_market_prob(test_eval, odds_col=odds_col)
            test_eval["blend_prob"] = blend_prob(
                test_eval["iso_prob"],
                test_eval["market_prob"],
                float(best_feasible_params.get("blend_weight") or 0.0),
            )
            test_metrics = run_policy_strategy(
                test_eval,
                prob_col="blend_prob",
                odds_col=odds_col,
                params=best_feasible_params,
            )
            best_feasible_test = _build_candidate_snapshot_from_metrics(
                strategy_kind=strategy_kind,
                params=best_feasible_params,
                metrics=test_metrics,
            )

    summary = {
        "fold": int(fold_index),
        "train_dates": [
            str(pd.to_datetime(train_df["date"], errors="coerce").min().date()),
            str(pd.to_datetime(train_df["date"], errors="coerce").max().date()),
        ],
        "valid_dates": [
            str(pd.to_datetime(valid_df["date"], errors="coerce").min().date()),
            str(pd.to_datetime(valid_df["date"], errors="coerce").max().date()),
        ],
        "test_dates": [
            str(pd.to_datetime(test_df["date"], errors="coerce").min().date()),
            str(pd.to_datetime(test_df["date"], errors="coerce").max().date()),
        ],
        "train_rows": int(len(train_df)),
        "valid_rows": int(len(valid_df)),
        "test_rows": int(len(test_df)),
        "train_races": int(train_df["race_id"].nunique()),
        "valid_races": int(valid_races),
        "test_races": int(test_df["race_id"].nunique()),
        "min_bets_required": int(min_bets_required),
        "total_candidates": int(len(candidate_rows)),
        "search_space": search_space,
        "feasible_candidates": int(sum(1 for row in candidate_rows if bool(row.get("is_feasible")))),
        "feasible_by_strategy": dict(feasible_by_strategy),
        "failure_reason_counts": dict(failure_reason_counts),
        "failure_combo_counts": dict(failure_combo_counts),
        "valid_stability_guardrail": build_stability_guardrail(frame=valid_df),
        "test_stability_guardrail": build_stability_guardrail(frame=test_df),
        "best_feasible": _serialize_candidate(best_feasible),
        "best_feasible_test": best_feasible_test,
        "best_fallback": _serialize_candidate(best_fallback),
        "closest_infeasible": closest_infeasible,
    }
    summary["valid_stability_assessment"] = summary["valid_stability_guardrail"]["assessment"]
    summary["test_stability_assessment"] = summary["test_stability_guardrail"]["assessment"]
    return summary, candidate_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_diag.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features_catboost_rich_high_coverage_diag.yaml")
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--model-artifact-suffix", default=None)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--detail-output", default=None)
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--wf-mode", choices=["fast", "full"], default="full")
    parser.add_argument("--wf-scheme", choices=["nested"], default="nested")
    parser.add_argument("--max-silent-seconds", type=float, default=60.0)
    parser.add_argument("--max-fold-elapsed-seconds", type=float, default=None)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=6, prefix="[wf-feasibility]", logger=log_progress, min_interval_sec=0.0)
        model_cfg = load_yaml(ROOT / args.config)
        data_cfg = load_yaml(ROOT / args.data_config)
        feature_cfg = load_yaml(ROOT / args.feature_config)
        dataset_cfg = data_cfg.get("dataset", {})
        label_col = str(model_cfg.get("label", "is_win"))
        evaluation_cfg = model_cfg.get("evaluation", {})
        search_config = evaluation_cfg.get("policy_search", {})
        constraints = PolicyConstraints.from_config(evaluation_cfg)
        progress.start(message="configs loaded")

        print("[wf-feasibility] loading training table")
        with Heartbeat("[wf-feasibility]", "loading training table", logger=log_progress):
            load_result = load_training_table_for_feature_build(
                dataset_cfg.get("raw_dir", "data/raw"),
                pre_feature_max_rows=int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
                dataset_config=dataset_cfg,
                base_dir=ROOT,
            )
        frame = load_result.frame
        loaded_rows = load_result.loaded_rows
        pre_feature_rows = load_result.pre_feature_rows
        data_load_strategy = load_result.data_load_strategy
        primary_source_rows_total = load_result.primary_source_rows_total
        load_message = f"[wf-feasibility] training table loaded rows={loaded_rows:,} strategy={data_load_strategy}"
        if primary_source_rows_total is not None:
            load_message += f" primary_source_rows_total={primary_source_rows_total:,}"
        print(load_message)
        print(f"[wf-feasibility] pre-feature slice ready rows={pre_feature_rows:,} loaded_rows={loaded_rows:,}")
        progress.update(message=f"training table loaded rows={loaded_rows:,}")
        print("[wf-feasibility] building features")
        with Heartbeat("[wf-feasibility]", "building features", logger=log_progress):
            frame = build_features(frame)
        frame = _filter_frame_by_date_window(frame, start_date=args.start_date, end_date=args.end_date)
        progress.update(message=f"features ready rows={len(frame):,}")

        output_cfg = dict(model_cfg.get("output", {}))
        if args.artifact_suffix:
            output_cfg["model_file"] = append_suffix_to_file_name(
                str(output_cfg.get("model_file", "baseline_model.joblib")),
                str(args.artifact_suffix),
            )
        output_artifacts = resolve_output_artifacts(output_cfg)

        load_output_cfg = dict(model_cfg.get("output", {}))
        load_artifact_suffix = args.model_artifact_suffix or args.artifact_suffix
        if load_artifact_suffix:
            load_output_cfg["model_file"] = append_suffix_to_file_name(
                str(load_output_cfg.get("model_file", "baseline_model.joblib")),
                str(load_artifact_suffix),
            )
        load_artifacts = resolve_output_artifacts(load_output_cfg)
        model_path = ROOT / load_artifacts.model_path
        with Heartbeat("[wf-feasibility]", f"loading model {model_path.name}", logger=log_progress):
            model = joblib.load(model_path)
        fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
        feature_selection = resolve_model_feature_selection(model, fallback_selection)
        x_eval = prepare_model_input_frame(frame, feature_selection.feature_columns, feature_selection.categorical_columns)
        odds_col = resolve_odds_column(frame)
        if odds_col is None:
            raise RuntimeError("Odds column is required for feasibility diagnostic")

        print("[wf-feasibility] running model inference")
        with Heartbeat("[wf-feasibility]", "running model inference", logger=log_progress):
            outputs = generate_prediction_outputs(model, x_eval, race_ids=frame["race_id"])
        pred = prepare_scored_frame(frame, outputs.score, odds_col=odds_col, score_col="score")
        pred = add_market_signals(pred, score_col="score", odds_col=odds_col)
        stability_guardrail = build_stability_guardrail(frame=pred)
        if stability_guardrail["assessment"] != "representative":
            print(
                "[wf-feasibility] stability guardrail="
                f"{stability_guardrail['assessment']}: "
                f"{'; '.join(stability_guardrail.get('warnings', [])[:2])}"
            )

        n_folds = 5 if args.wf_mode == "full" else 3
        nested_slices = build_nested_wf_slices(
            pred,
            date_col="date",
            n_folds=n_folds,
            valid_ratio=0.15,
            test_ratio=0.15,
            min_train_rows=1000,
            min_valid_rows=500,
            min_test_rows=500,
        )
        if not nested_slices:
            raise RuntimeError("No nested walk-forward slices available for the requested window")
        progress.update(message=f"wf slices ready folds={len(nested_slices)}")

        fold_summaries: list[dict[str, object]] = []
        detail_rows: list[dict[str, object]] = []
        fold_progress = ProgressBar(total=max(len(nested_slices), 1), prefix="[wf-feasibility folds]", logger=log_progress, min_interval_sec=0.0)
        fold_progress.start(message="analyzing folds")
        for fold_index, (train_df, valid_df, test_df) in enumerate(nested_slices, start=1):
            print(f"[wf-feasibility] analyzing fold {fold_index}/{len(nested_slices)}")
            fold_started_at = time.monotonic()

            def report_fold_search_space(search_space: dict[str, object]) -> None:
                log_progress(
                    "[wf-feasibility] "
                    f"fold={fold_index}/{len(nested_slices)} "
                    f"search_space outer={search_space.get('outer_search_steps')} "
                    f"strategy_evals_per_outer={search_space.get('strategy_evals_per_outer_step')} "
                    f"total={search_space.get('total_search_steps')}"
                )

            def report_fold_checkpoint(processed: int, total: int, checkpoint: dict[str, object]) -> None:
                log_progress(
                    "[wf-feasibility] "
                    f"fold={fold_index}/{len(nested_slices)} "
                    f"strategy={checkpoint['strategy_kind']} "
                    f"search_step={processed}/{total} "
                    f"blend_weight={checkpoint['blend_weight']:.2f} "
                    f"min_edge={checkpoint['min_edge']:.2f} "
                    f"min_prob={checkpoint['min_prob']:.2f} "
                    f"odds={checkpoint['odds_min']:.1f}-{checkpoint['odds_max']:.1f} "
                    f"candidate_rows={checkpoint['candidate_rows']}"
                )

            fold_summary, fold_detail = _summarize_fold_candidates(
                fold_index=fold_index,
                train_df=train_df,
                valid_df=valid_df,
                test_df=test_df,
                label_col=label_col,
                odds_col=odds_col,
                constraints=constraints,
                search_config=search_config,
                mode=args.wf_mode,
                search_space_callback=report_fold_search_space,
                progress_callback=report_fold_checkpoint,
                max_silent_seconds=args.max_silent_seconds,
                max_fold_elapsed_seconds=args.max_fold_elapsed_seconds,
            )
            fold_summaries.append(fold_summary)
            detail_rows.extend(fold_detail)
            elapsed_fold_sec = int(time.monotonic() - fold_started_at)
            fold_progress.update(
                message=(
                    f"fold={fold_index} candidates={len(fold_detail)} "
                    f"feasible={fold_summary['feasible_candidates']} "
                    f"elapsed={elapsed_fold_sec}s"
                )
            )
        progress.update(message="fold analysis completed")

        report_dir = ROOT / "artifacts" / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path, detail_path = _resolve_feasibility_output_paths(
            report_dir=report_dir,
            config_path=args.config,
            model_path=model_path,
            wf_mode=args.wf_mode,
            wf_scheme=args.wf_scheme,
            start_date=args.start_date,
            end_date=args.end_date,
            summary_output=args.summary_output,
            detail_output=args.detail_output,
        )
        artifact_ensure_output_file_path(summary_path, label="summary output", workspace_root=ROOT)
        artifact_ensure_output_file_path(detail_path, label="detail output", workspace_root=ROOT)

        summary_payload = {
            "run_context": {
                "config": str(args.config),
                "data_config": str(args.data_config),
                "feature_config": str(args.feature_config),
                "loaded_rows": loaded_rows,
                "data_load_strategy": data_load_strategy,
                "primary_source_rows_total": int(primary_source_rows_total) if primary_source_rows_total is not None else None,
                "pre_feature_max_rows": int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
                "pre_feature_rows": pre_feature_rows,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "wf_mode": args.wf_mode,
                "wf_scheme": args.wf_scheme,
                "rows": int(len(pred)),
                "races": int(pred["race_id"].nunique()),
                "feature_count": int(len(feature_selection.feature_columns)),
                "categorical_feature_count": int(len(feature_selection.categorical_columns)),
                "artifact_suffix": str(args.artifact_suffix or ""),
                "model_artifact_suffix": str(args.model_artifact_suffix or args.artifact_suffix or ""),
                "model_path": artifact_display_path(model_path, workspace_root=ROOT),
            },
            "policy_constraints": constraints.to_dict(),
            "policy_search": search_config,
            "stability_assessment": stability_guardrail["assessment"],
            "stability_guardrail": stability_guardrail,
            "folds": fold_summaries,
        }
        with Heartbeat("[wf-feasibility]", "writing feasibility outputs", logger=log_progress):
            write_text_file(summary_path, json.dumps(summary_payload, ensure_ascii=False, indent=2), label="summary output")

            detail_frame = pd.DataFrame(detail_rows)
            if not detail_frame.empty:
                detail_frame = detail_frame.copy()
                detail_frame["gate_failures"] = detail_frame["gate_failures"].apply(lambda values: ",".join(values) if isinstance(values, list) else str(values or ""))
                write_csv_file(detail_path, detail_frame, index=False, label="detail output")
            else:
                write_csv_file(detail_path, pd.DataFrame(columns=["fold", "strategy_kind", "gate_failures"]), index=False, label="detail output")

        print(f"[wf-feasibility] summary saved: {summary_path}")
        print(f"[wf-feasibility] detail saved: {detail_path}")
        print(json.dumps(fold_summaries, ensure_ascii=False, indent=2))
        progress.complete(message="feasibility diagnostic completed")
        return 0
    except KeyboardInterrupt:
        print("[wf-feasibility] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[wf-feasibility] failed: {error}")
        return 1
    except Exception as error:
        print(f"[wf-feasibility] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
