import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
NO_MODEL_ARTIFACT_SUFFIX = "__NO_MODEL_ARTIFACT_SUFFIX__"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import append_suffix_to_file_name
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import resolve_output_artifacts, utc_now_iso, write_json, write_text_file
from racing_ml.common.local_nankan_trust import require_local_nankan_trust_ready
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles, resolve_model_run_profile
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.common.regime import resolve_regime_name
from racing_ml.data.dataset_loader import load_training_table_for_feature_build
from racing_ml.evaluation.benchmark import (
    count_winner_only_races,
    combine_benter_prob,
    fit_benter_combiner,
    pseudo_r2,
    race_winner_logloss,
    supports_winner_only_benchmark,
)
from racing_ml.evaluation.leakage import run_leakage_audit
from racing_ml.evaluation.policy import (
    PolicyConstraints,
    add_market_signals,
    blend_prob,
    compute_market_prob,
    ev_top1_roi_from_prob,
    evaluate_fixed_stake_summary,
    run_policy_strategy,
    simulate_fractional_kelly,
)
from racing_ml.evaluation.scoring import (
    generate_prediction_outputs,
    predict_target_values,
    prepare_scored_frame,
    resolve_odds_column,
    topk_hit_rate,
)
from racing_ml.evaluation.walk_forward import (
    build_nested_wf_slices,
    fit_isotonic,
    fit_platt,
    optimize_blend_weight,
    optimize_roi_strategy,
    split_for_calibration,
    split_three_way_time,
)
from racing_ml.evaluation.stability import build_stability_guardrail
from racing_ml.features.builder import build_features
from racing_ml.features.selection import prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection, summarize_feature_coverage


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[evaluate {now}] {message}", flush=True)


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
    start_ts = pd.Timestamp(start_date) if start_date else None
    end_ts = pd.Timestamp(end_date) if end_date else None
    if start_ts is not None and end_ts is not None and end_ts < start_ts:
        raise ValueError(f"end_date must be >= start_date: {start_date} .. {end_date}")
    if start_ts is not None:
        mask &= date_series >= start_ts
    if end_ts is not None:
        mask &= date_series <= end_ts

    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise RuntimeError(
            f"No rows found in date window: {start_date or '-inf'} .. {end_date or '+inf'}"
        )
    return filtered


def _resolve_score_source_name(
    overrides: list[dict[str, Any]] | None,
    *,
    frame: pd.DataFrame,
    default_name: str = "default",
) -> str:
    return resolve_regime_name(overrides, frame=frame, default_name=default_name)


def _build_scored_prediction_frame(
    *,
    model: object,
    frame: pd.DataFrame,
    fallback_selection: Any,
    odds_col: str | None,
) -> tuple[pd.DataFrame, Any, Any]:
    feature_selection = resolve_model_feature_selection(model, fallback_selection)
    if not feature_selection.feature_columns:
        raise RuntimeError("No features available for evaluation")

    x_eval = prepare_model_input_frame(
        frame,
        feature_selection.feature_columns,
        feature_selection.categorical_columns,
    )
    outputs = generate_prediction_outputs(model, x_eval, race_ids=frame["race_id"])
    pred = prepare_scored_frame(frame, outputs.score, odds_col=odds_col, score_col="score")
    if odds_col is not None:
        pred = add_market_signals(pred, score_col="score", odds_col=odds_col)
    return pred, feature_selection, outputs


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _base_summary(
    pred: pd.DataFrame,
    odds_col: str | None,
    score_col: str = "score",
    include_ev_metrics: bool = True,
) -> dict[str, float | int | None]:
    policy_summary = evaluate_fixed_stake_summary(pred, odds_col=odds_col, score_col=score_col, stake=100.0)
    if not include_ev_metrics:
        policy_summary["ev_top1_roi"] = None
        policy_summary["ev_threshold_1_0_roi"] = None
        policy_summary["ev_threshold_1_0_bets"] = 0
        policy_summary["ev_threshold_1_2_roi"] = None
        policy_summary["ev_threshold_1_2_bets"] = 0
    return {
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()),
        "top1_hit_rate": topk_hit_rate(pred, 1),
        "top3_hit_rate": topk_hit_rate(pred, 3),
        "top5_hit_rate": topk_hit_rate(pred, 5),
        **policy_summary,
    }


def _append_top3_auc(summary: dict[str, object], pred: pd.DataFrame, top3_probs: dict[str, np.ndarray] | None) -> None:
    if top3_probs is None or "rank" not in pred.columns:
        return

    rank_series = pd.to_numeric(pred["rank"], errors="coerce")
    for pos, key in [(1, "p_rank1"), (2, "p_rank2"), (3, "p_rank3")]:
        y_pos = (rank_series == pos).astype(int).to_numpy()
        auc_value = _safe_auc(y_pos, top3_probs[key])
        if auc_value is not None:
            summary[f"auc_rank{pos}"] = auc_value


def _safe_regression_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(finite_mask)) == 0:
        return None
    return float(np.sqrt(np.mean(np.square(y_true[finite_mask] - y_pred[finite_mask]))))


def _safe_regression_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(finite_mask)) == 0:
        return None
    return float(np.mean(np.abs(y_true[finite_mask] - y_pred[finite_mask])))


def _safe_regression_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(finite_mask)) < 2:
        return None
    return float(np.corrcoef(y_true[finite_mask], y_pred[finite_mask])[0, 1])


def _compute_market_deviation_metrics(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    label_col: str,
) -> dict[str, float | None | str]:
    required_columns = {"race_id", "odds", label_col}
    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        return {
            "alpha_target_mean": None,
            "alpha_target_std": None,
            "pred_mean": None,
            "pred_std": None,
            "alpha_pred_corr": None,
            "positive_signal_rate": None,
            "market_deviation_metrics_skipped_reason": f"missing_columns:{','.join(sorted(missing_columns))}",
        }

    odds = pd.to_numeric(frame["odds"], errors="coerce").clip(lower=1.01)
    implied = 1.0 / odds.replace(0, np.nan)
    denom = implied.groupby(frame["race_id"]).transform("sum")
    market_prob = (implied / denom.replace(0, np.nan)).clip(lower=1e-4, upper=1.0 - 1e-4)
    labels = pd.to_numeric(frame[label_col], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    observed_prob = labels.clip(lower=1e-4, upper=1.0 - 1e-4)
    observed_logit = np.log(observed_prob / (1.0 - observed_prob))
    market_logit = np.log(market_prob / (1.0 - market_prob))
    alpha_target = (observed_logit - market_logit).replace([np.inf, -np.inf], np.nan)
    pred = pd.Series(np.asarray(scores, dtype=float).reshape(-1), index=frame.index)

    finite_target = alpha_target.to_numpy(dtype=float)
    finite_pred = pred.to_numpy(dtype=float)
    return {
        "alpha_target_mean": float(np.nanmean(finite_target)) if np.isfinite(finite_target).any() else None,
        "alpha_target_std": float(np.nanstd(finite_target)) if np.isfinite(finite_target).any() else None,
        "pred_mean": float(np.nanmean(finite_pred)) if np.isfinite(finite_pred).any() else None,
        "pred_std": float(np.nanstd(finite_pred)) if np.isfinite(finite_pred).any() else None,
        "alpha_pred_corr": _safe_regression_corr(finite_target, finite_pred),
        "positive_signal_rate": float(np.nanmean(finite_pred > 0.0)) if np.isfinite(finite_pred).any() else None,
        "market_deviation_metrics_skipped_reason": None,
    }


def _date_window_payload(frame: pd.DataFrame) -> dict[str, str | int | None]:
    if "date" not in frame.columns:
        return {
            "start_date": None,
            "end_date": None,
            "start_month": None,
            "end_month": None,
        }

    date_series = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if date_series.empty:
        return {
            "start_date": None,
            "end_date": None,
            "start_month": None,
            "end_month": None,
        }

    start_ts = pd.Timestamp(date_series.iloc[0])
    end_ts = pd.Timestamp(date_series.iloc[-1])
    return {
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "start_month": int(start_ts.month),
        "end_month": int(end_ts.month),
    }


def _prefix_window_payload(
    prefix: str,
    window_payload: dict[str, str | int | None] | None,
) -> dict[str, str | int | None]:
    keys = ("start_date", "end_date", "start_month", "end_month")
    if not isinstance(window_payload, dict):
        return {f"{prefix}_{key}": None for key in keys}
    return {f"{prefix}_{key}": window_payload.get(key) for key in keys}


def _record_single_wf_summary(
    summary: dict[str, object],
    best_params: dict[str, float | str],
    best_valid_metrics: dict[str, float | int | None],
    wf_test_metrics: dict[str, float | int | None],
    wf_family_diagnostics: dict[str, object] | None = None,
    score_source: str = "default",
) -> None:
    strategy_kind = str(best_params.get("strategy_kind", "kelly"))
    policy_params: dict[str, float | int | str] = {
        "strategy_kind": strategy_kind,
        "blend_weight": float(best_params.get("blend_weight", 0.0)),
        "min_prob": float(best_params.get("min_prob", 0.05)),
        "odds_min": float(best_params.get("odds_min", 1.0)),
        "odds_max": float(best_params.get("odds_max", 999.0)),
    }
    if strategy_kind == "kelly":
        policy_params.update(
            {
                "min_edge": float(best_params.get("min_edge", 0.03)),
                "fractional_kelly": float(best_params.get("fractional_kelly", 0.5)),
                "max_fraction": float(best_params.get("max_fraction", 0.05)),
            }
        )
    elif strategy_kind == "portfolio":
        policy_params.update(
            {
                "top_k": int(best_params.get("top_k", 2)),
                "min_expected_value": float(best_params.get("min_expected_value", 1.0)),
            }
        )

    summary["wf_strategy_kind"] = str(best_params.get("strategy_kind", "kelly"))
    summary["wf_score_source"] = str(score_source)
    summary["wf_best_blend_weight"] = float(best_params.get("blend_weight", 0.0))
    summary["wf_best_min_prob"] = float(best_params.get("min_prob", 0.05))
    summary["wf_best_odds_min"] = float(best_params.get("odds_min", 1.0))
    summary["wf_best_odds_max"] = float(best_params.get("odds_max", 999.0))
    summary["wf_policy_params"] = policy_params
    selection_reason = best_params.get("selection_reason")
    summary["wf_selection_reason"] = str(selection_reason) if selection_reason is not None else None
    if wf_family_diagnostics is not None:
        summary["wf_family_diagnostics"] = wf_family_diagnostics

    if summary["wf_strategy_kind"] == "kelly":
        summary["wf_best_min_edge"] = float(best_params.get("min_edge", 0.03))
        summary["wf_best_fractional_kelly"] = float(best_params.get("fractional_kelly", 0.5))
        summary["wf_best_max_fraction"] = float(best_params.get("max_fraction", 0.05))
        summary["wf_valid_roi"] = best_valid_metrics.get("kelly_roi")
        summary["wf_valid_bets"] = best_valid_metrics.get("kelly_bets")
        summary["wf_valid_hit_rate"] = best_valid_metrics.get("kelly_hit_rate")
        summary["wf_test_roi"] = wf_test_metrics.get("kelly_roi")
        summary["wf_test_bets"] = wf_test_metrics.get("kelly_bets")
        summary["wf_test_hit_rate"] = wf_test_metrics.get("kelly_hit_rate")
        summary["wf_test_final_bankroll"] = wf_test_metrics.get("kelly_final_bankroll")
        summary["wf_valid_max_drawdown"] = best_valid_metrics.get("kelly_max_drawdown")
        summary["wf_test_max_drawdown"] = wf_test_metrics.get("kelly_max_drawdown")
    elif summary["wf_strategy_kind"] == "portfolio":
        summary["wf_best_top_k"] = int(best_params.get("top_k", 2))
        summary["wf_best_min_expected_value"] = float(best_params.get("min_expected_value", 1.0))
        summary["wf_valid_roi"] = best_valid_metrics.get("portfolio_roi")
        summary["wf_valid_bets"] = best_valid_metrics.get("portfolio_bets")
        summary["wf_valid_hit_rate"] = best_valid_metrics.get("portfolio_hit_rate")
        summary["wf_test_roi"] = wf_test_metrics.get("portfolio_roi")
        summary["wf_test_bets"] = wf_test_metrics.get("portfolio_bets")
        summary["wf_test_hit_rate"] = wf_test_metrics.get("portfolio_hit_rate")
        summary["wf_test_avg_synthetic_odds"] = wf_test_metrics.get("portfolio_avg_synthetic_odds")
        summary["wf_valid_final_bankroll"] = best_valid_metrics.get("portfolio_final_bankroll")
        summary["wf_test_final_bankroll"] = wf_test_metrics.get("portfolio_final_bankroll")
        summary["wf_valid_max_drawdown"] = best_valid_metrics.get("portfolio_max_drawdown")
        summary["wf_test_max_drawdown"] = wf_test_metrics.get("portfolio_max_drawdown")
    else:
        summary["wf_valid_roi"] = None
        summary["wf_valid_bets"] = 0
        summary["wf_valid_hit_rate"] = None
        summary["wf_test_roi"] = None
        summary["wf_test_bets"] = 0
        summary["wf_test_hit_rate"] = None
        summary["wf_valid_final_bankroll"] = 1.0
        summary["wf_test_final_bankroll"] = 1.0
        summary["wf_valid_max_drawdown"] = 0.0
        summary["wf_test_max_drawdown"] = 0.0


def _fold_row(
    strategy_kind: str,
    best_params: dict[str, float | str],
    best_valid_metrics: dict[str, float | int | None],
    fold_test_metrics: dict[str, float | int | None],
    fold_index: int,
    family_diagnostics: dict[str, object] | None = None,
    score_source: str = "default",
    train_window: dict[str, str | int | None] | None = None,
    valid_window: dict[str, str | int | None] | None = None,
    test_window: dict[str, str | int | None] | None = None,
) -> dict[str, float | int | str | None]:
    window_fields = {
        **_prefix_window_payload("train", train_window),
        **_prefix_window_payload("valid", valid_window),
        **_prefix_window_payload("test", test_window),
    }
    if strategy_kind == "kelly":
        row = {
            "fold": int(fold_index),
            "strategy_kind": strategy_kind,
            "score_source": str(score_source),
            "selection_reason": str(best_params.get("selection_reason")) if best_params.get("selection_reason") is not None else None,
            "valid_roi": best_valid_metrics.get("kelly_roi"),
            "valid_bets": best_valid_metrics.get("kelly_bets"),
            "valid_hit_rate": best_valid_metrics.get("kelly_hit_rate"),
            "valid_final_bankroll": best_valid_metrics.get("kelly_final_bankroll"),
            "valid_max_drawdown": best_valid_metrics.get("kelly_max_drawdown"),
            "test_roi": fold_test_metrics.get("kelly_roi"),
            "test_bets": fold_test_metrics.get("kelly_bets"),
            "test_hit_rate": fold_test_metrics.get("kelly_hit_rate"),
            "test_final_bankroll": fold_test_metrics.get("kelly_final_bankroll"),
            "test_max_drawdown": fold_test_metrics.get("kelly_max_drawdown"),
            "blend_weight": float(best_params.get("blend_weight", 0.0)),
            "min_prob": float(best_params.get("min_prob", 0.05)),
            "odds_min": float(best_params.get("odds_min", 1.0)),
            "odds_max": float(best_params.get("odds_max", 999.0)),
            "min_edge": float(best_params.get("min_edge", 0.03)),
            "fractional_kelly": float(best_params.get("fractional_kelly", 0.5)),
            "max_fraction": float(best_params.get("max_fraction", 0.05)),
            **window_fields,
        }
        if family_diagnostics is not None:
            row["family_diagnostics"] = family_diagnostics
        return row
    if strategy_kind == "portfolio":
        row = {
            "fold": int(fold_index),
            "strategy_kind": strategy_kind,
            "score_source": str(score_source),
            "selection_reason": str(best_params.get("selection_reason")) if best_params.get("selection_reason") is not None else None,
            "valid_roi": best_valid_metrics.get("portfolio_roi"),
            "valid_bets": best_valid_metrics.get("portfolio_bets"),
            "valid_hit_rate": best_valid_metrics.get("portfolio_hit_rate"),
            "valid_final_bankroll": best_valid_metrics.get("portfolio_final_bankroll"),
            "valid_max_drawdown": best_valid_metrics.get("portfolio_max_drawdown"),
            "test_roi": fold_test_metrics.get("portfolio_roi"),
            "test_bets": fold_test_metrics.get("portfolio_bets"),
            "test_hit_rate": fold_test_metrics.get("portfolio_hit_rate"),
            "test_final_bankroll": fold_test_metrics.get("portfolio_final_bankroll"),
            "test_max_drawdown": fold_test_metrics.get("portfolio_max_drawdown"),
            "blend_weight": float(best_params.get("blend_weight", 0.0)),
            "min_prob": float(best_params.get("min_prob", 0.05)),
            "odds_min": float(best_params.get("odds_min", 1.0)),
            "odds_max": float(best_params.get("odds_max", 999.0)),
            "top_k": int(best_params.get("top_k", 2)),
            "min_expected_value": float(best_params.get("min_expected_value", 1.0)),
            **window_fields,
        }
        if family_diagnostics is not None:
            row["family_diagnostics"] = family_diagnostics
        return row
    row = {
        "fold": int(fold_index),
        "strategy_kind": strategy_kind,
        "score_source": str(score_source),
        "selection_reason": str(best_params.get("selection_reason")) if best_params.get("selection_reason") is not None else None,
        "valid_roi": None,
        "valid_bets": 0,
        "valid_hit_rate": None,
        "valid_final_bankroll": 1.0,
        "valid_max_drawdown": 0.0,
        "test_roi": None,
        "test_bets": 0,
        "test_hit_rate": None,
        "test_final_bankroll": 1.0,
        "test_max_drawdown": 0.0,
        "blend_weight": float(best_params.get("blend_weight", 0.0)),
        "min_prob": None,
        "odds_min": None,
        "odds_max": None,
        **window_fields,
    }
    if family_diagnostics is not None:
        row["family_diagnostics"] = family_diagnostics
    return row


def _sanitize_output_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def _shorten_output_slug(slug: str, *, max_length: int = 96) -> str:
    normalized = _sanitize_output_slug(slug)
    if len(normalized) <= max_length:
        return normalized
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]
    head_length = max(max_length - len(digest) - 1, 16)
    return f"{normalized[:head_length].rstrip('_')}_{digest}"


def _derive_date_window_slug(start_date: str | None, end_date: str | None) -> str:
    if not start_date and not end_date:
        return ""

    start_token = _sanitize_output_slug((start_date or "start_auto").replace("-", ""))
    end_token = _sanitize_output_slug((end_date or "end_auto").replace("-", ""))
    return f"_{start_token}_{end_token}"


def _derive_wf_slug(wf_mode: str, wf_scheme: str) -> str:
    normalized_mode = _sanitize_output_slug(wf_mode)
    normalized_scheme = _sanitize_output_slug(wf_scheme)
    if normalized_mode == "fast" and normalized_scheme == "nested":
        return ""
    return f"_wf_{normalized_mode}_{normalized_scheme}"


def _derive_evaluation_output_slug(config_path: str, model_path: Path, *, prefer_config: bool = False) -> str:
    candidates = [Path(config_path).stem, model_path.stem] if prefer_config else [model_path.stem, Path(config_path).stem]
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        if text.endswith("_model"):
            text = text[: -len("_model")]
        if text.startswith("model_"):
            text = text[len("model_") :]
        slug = _shorten_output_slug(text)
        if slug:
            return slug
    return "model"


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _build_evaluation_output_manifest(
    *,
    summary: dict[str, Any],
    by_date: pd.DataFrame,
    latest_summary_path: Path,
    latest_by_date_path: Path | None,
    latest_manifest_path: Path,
    versioned_summary_path: Path,
    versioned_by_date_path: Path | None,
    versioned_manifest_path: Path,
    summary_sha256: str,
    by_date_sha256: str | None,
    latest_wf_progress_path: Path | None,
    versioned_wf_progress_path: Path | None,
) -> dict[str, Any]:
    run_context = summary.get("run_context") if isinstance(summary.get("run_context"), dict) else {}
    by_date_present = latest_by_date_path is not None and not by_date.empty
    by_date_start = str(by_date["date"].iloc[0]) if by_date_present and "date" in by_date.columns else None
    by_date_end = str(by_date["date"].iloc[-1]) if by_date_present and "date" in by_date.columns else None
    n_dates_value = summary.get("n_dates")
    expected_n_dates = int(n_dates_value) if isinstance(n_dates_value, int) else None
    actual_by_date_rows = int(len(by_date)) if by_date_present else 0

    return {
        "created_at": utc_now_iso(),
        "profile": run_context.get("profile"),
        "config": run_context.get("config"),
        "data_config": run_context.get("data_config"),
        "feature_config": run_context.get("feature_config"),
        "task": run_context.get("task"),
        "loaded_rows": run_context.get("loaded_rows"),
        "data_load_strategy": run_context.get("data_load_strategy"),
        "primary_source_rows_total": run_context.get("primary_source_rows_total"),
        "pre_feature_max_rows": run_context.get("pre_feature_max_rows"),
        "pre_feature_rows": run_context.get("pre_feature_rows"),
        "requested_max_rows": run_context.get("requested_max_rows"),
        "date_window": summary.get("date_window"),
        "wf_mode": run_context.get("wf_mode"),
        "wf_scheme": run_context.get("wf_scheme"),
        "n_rows": summary.get("n_rows"),
        "n_dates": n_dates_value,
        "stability_assessment": summary.get("stability_assessment"),
        "stability_guardrail": summary.get("stability_guardrail"),
        "by_date_rows": actual_by_date_rows,
        "by_date_present": by_date_present,
        "consistency": {
            "n_dates_matches_by_date_rows": (
                expected_n_dates == actual_by_date_rows
                if expected_n_dates is not None and by_date_present
                else None
            ),
            "by_date_start": by_date_start,
            "by_date_end": by_date_end,
        },
        "source_model_manifest": run_context.get("artifact_manifest"),
        "files": {
            "latest_summary": artifact_display_path(latest_summary_path, workspace_root=ROOT),
            "latest_by_date": artifact_display_path(latest_by_date_path, workspace_root=ROOT) if latest_by_date_path is not None else None,
            "latest_manifest": artifact_display_path(latest_manifest_path, workspace_root=ROOT),
            "latest_wf_progress": artifact_display_path(latest_wf_progress_path, workspace_root=ROOT) if latest_wf_progress_path is not None else None,
            "versioned_summary": artifact_display_path(versioned_summary_path, workspace_root=ROOT),
            "versioned_by_date": artifact_display_path(versioned_by_date_path, workspace_root=ROOT) if versioned_by_date_path is not None else None,
            "versioned_manifest": artifact_display_path(versioned_manifest_path, workspace_root=ROOT),
            "versioned_wf_progress": artifact_display_path(versioned_wf_progress_path, workspace_root=ROOT) if versioned_wf_progress_path is not None else None,
        },
        "checksums": {
            "summary_sha256": summary_sha256,
            "by_date_sha256": by_date_sha256,
        },
    }


def _build_wf_progress_payload(
    *,
    summary: dict[str, Any],
    output_slug: str,
    total_folds: int,
    completed_folds: int,
    status: str,
    current_fold: int | None = None,
    current_fold_state: str | None = None,
    current_score_source: str | None = None,
    current_train_window: dict[str, str | int | None] | None = None,
    current_valid_window: dict[str, str | int | None] | None = None,
    current_test_window: dict[str, str | int | None] | None = None,
    latest_completed_fold: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_context = summary.get("run_context") if isinstance(summary.get("run_context"), dict) else {}
    payload: dict[str, Any] = {
        "updated_at": utc_now_iso(),
        "status": status,
        "output_slug": output_slug,
        "profile": run_context.get("profile"),
        "config": run_context.get("config"),
        "data_config": run_context.get("data_config"),
        "feature_config": run_context.get("feature_config"),
        "artifact_suffix": run_context.get("artifact_suffix"),
        "wf_mode": run_context.get("wf_mode"),
        "wf_scheme": run_context.get("wf_scheme"),
        "target_folds": int(total_folds),
        "completed_folds": int(completed_folds),
        "current_fold": int(current_fold) if current_fold is not None else None,
        "current_fold_state": current_fold_state,
        "current_score_source": current_score_source,
        "current_train_window": current_train_window,
        "current_valid_window": current_valid_window,
        "current_test_window": current_test_window,
        "latest_completed_fold": latest_completed_fold,
    }
    if status == "completed":
        payload["wf_nested_test_roi_weighted"] = summary.get("wf_nested_test_roi_weighted")
        payload["wf_nested_test_bets_total"] = summary.get("wf_nested_test_bets_total")
    return payload


def _write_wf_progress_outputs(
    *,
    latest_path: Path,
    versioned_path: Path,
    payload: dict[str, Any],
) -> None:
    write_json(latest_path, payload)
    write_json(versioned_path, payload)


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")

    parser = argparse.ArgumentParser()
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--model-artifact-suffix", default=None)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--progress-interval-sec", type=float, default=5.0)
    parser.add_argument("--allow-diagnostic-local-nankan", action="store_true")
    args = parser.parse_args()

    try:
        if args.list_profiles:
            print(format_model_run_profiles())
            return 0

        if args.profile and any(value is not None for value in (args.config, args.data_config, args.feature_config)):
            raise ValueError("--profile cannot be combined with --config, --data-config, or --feature-config")

        resolved_profile, model_config_path, data_config_path, feature_config_path = resolve_model_run_profile(
            args.profile,
            default_model_config=args.config or "configs/model.yaml",
            default_data_config=args.data_config or "configs/data.yaml",
            default_feature_config=args.feature_config or "configs/features.yaml",
        )
        profile_default_suffix = None
        if resolved_profile is not None:
            profile_default_suffix = MODEL_RUN_PROFILES[resolved_profile].default_model_artifact_suffix
        model_cfg = load_yaml(ROOT / model_config_path)
        data_cfg = load_yaml(ROOT / data_config_path)
        feature_cfg = load_yaml(ROOT / feature_config_path)
        require_local_nankan_trust_ready(
            workspace_root=ROOT,
            data_config=data_cfg,
            data_config_path=data_config_path,
            allow_diagnostic_override=bool(args.allow_diagnostic_local_nankan),
            command_name="evaluate",
            profile_name=resolved_profile,
        )
        explicit_no_model_artifact_suffix = args.model_artifact_suffix == NO_MODEL_ARTIFACT_SUFFIX
        resolved_model_artifact_suffix = None if explicit_no_model_artifact_suffix else (args.model_artifact_suffix or profile_default_suffix)

        progress = ProgressBar(total=9, prefix="[evaluate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"configs loaded profile={resolved_profile or 'custom'} config={model_config_path} "
                f"data_config={data_config_path} feature_config={feature_config_path} "
                f"artifact_suffix={args.artifact_suffix or 'none'} "
                f"model_artifact_suffix={resolved_model_artifact_suffix if not explicit_no_model_artifact_suffix else 'none'} "
                f"allow_diagnostic_local_nankan={args.allow_diagnostic_local_nankan}"
            )
        )

        task = str(model_cfg.get("task", "classification")).strip().lower()
        evaluation_cfg = model_cfg.get("evaluation", {})
        policy_search_cfg = evaluation_cfg.get("policy_search", {})
        score_regime_overrides_cfg = evaluation_cfg.get("score_regime_overrides", [])
        output_cfg = dict(model_cfg.get("output", {}))
        if args.artifact_suffix:
            output_cfg["model_file"] = append_suffix_to_file_name(
                str(output_cfg.get("model_file", "baseline_model.joblib")),
                args.artifact_suffix,
            )
            output_cfg["report_file"] = append_suffix_to_file_name(
                str(output_cfg.get("report_file", "train_metrics.json")),
                args.artifact_suffix,
            )
            output_cfg["manifest_file"] = append_suffix_to_file_name(
                str(output_cfg.get("manifest_file", output_cfg.get("model_file", "baseline_model.joblib"))),
                args.artifact_suffix,
            )
        output_artifacts = resolve_output_artifacts(output_cfg)

        load_output_cfg = dict(model_cfg.get("output", {}))
        load_artifact_suffix = None if explicit_no_model_artifact_suffix else (resolved_model_artifact_suffix or args.artifact_suffix)
        if load_artifact_suffix:
            load_output_cfg["model_file"] = append_suffix_to_file_name(
                str(load_output_cfg.get("model_file", "baseline_model.joblib")),
                load_artifact_suffix,
            )
            load_output_cfg["report_file"] = append_suffix_to_file_name(
                str(load_output_cfg.get("report_file", "train_metrics.json")),
                load_artifact_suffix,
            )
            load_output_cfg["manifest_file"] = append_suffix_to_file_name(
                str(load_output_cfg.get("manifest_file", load_output_cfg.get("model_file", "baseline_model.joblib"))),
                load_artifact_suffix,
            )
        load_artifacts = resolve_output_artifacts(load_output_cfg)
        policy_constraints = PolicyConstraints.from_config(evaluation_cfg)
        leakage_cfg = evaluation_cfg.get("leakage_audit", {})
        leakage_enabled = bool(leakage_cfg.get("enabled", True))

        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        label_col = model_cfg.get("label", "is_win")
        with Heartbeat("[evaluate]", "loading training table", logger=log_progress):
            load_result = load_training_table_for_feature_build(
                raw_dir,
                pre_feature_max_rows=int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
                dataset_config=dataset_cfg,
                base_dir=ROOT,
            )
        frame = load_result.frame
        loaded_rows = load_result.loaded_rows
        pre_feature_rows = load_result.pre_feature_rows
        data_load_strategy = load_result.data_load_strategy
        primary_source_rows_total = load_result.primary_source_rows_total
        load_message = f"training table loaded rows={loaded_rows:,} strategy={data_load_strategy}"
        if primary_source_rows_total is not None:
            load_message += f" primary_source_rows_total={primary_source_rows_total:,}"
        progress.update(message=load_message)
        progress.update(message=f"pre-feature slice ready rows={pre_feature_rows:,}")
        with Heartbeat("[evaluate]", "building features", logger=log_progress):
            frame = build_features(frame)
        progress.update(message=f"features built columns={len(frame.columns):,}")
        if args.start_date or args.end_date:
            frame = _filter_frame_by_date_window(
                frame,
                start_date=args.start_date,
                end_date=args.end_date,
            )
            progress.update(
                message=(
                    f"date filter applied rows={len(frame):,} "
                    f"start={args.start_date or '-inf'} end={args.end_date or '+inf'}"
                )
            )
        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()
            print(f"[evaluate] using tail rows: {len(frame)}")
        if label_col not in frame.columns:
            raise RuntimeError(f"Missing label column: {label_col}")

        loaded_model_path = load_artifacts.model_path if load_artifacts.model_path.is_absolute() else (ROOT / load_artifacts.model_path)
        model = joblib.load(loaded_model_path)
        fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
        feature_selection = resolve_model_feature_selection(model, fallback_selection)
        feature_coverage = summarize_feature_coverage(frame, feature_cfg, feature_selection)
        if not feature_selection.feature_columns:
            raise RuntimeError("No features available for evaluation")
        progress.update(
            message=(
                f"model ready features={len(feature_selection.feature_columns):,} "
                f"categorical={len(feature_selection.categorical_columns):,}"
            )
        )
        if feature_coverage["missing_force_include_features"]:
            print(f"[evaluate] missing force-include features: {feature_coverage['missing_force_include_features']}")
        if feature_coverage["low_coverage_force_include_features"]:
            print(f"[evaluate] low-coverage force-include features: {feature_coverage['low_coverage_force_include_features']}")

        x_eval = prepare_model_input_frame(frame, feature_selection.feature_columns, feature_selection.categorical_columns)
        y_eval = frame[label_col].astype(int).to_numpy()
        odds_col = resolve_odds_column(frame)
        include_ev_metrics = task not in {"time_regression", "time_deviation"}

        with Heartbeat("[evaluate]", "running model inference", logger=log_progress):
            outputs = generate_prediction_outputs(model, x_eval, race_ids=frame["race_id"])
            pred = prepare_scored_frame(frame, outputs.score, odds_col=odds_col, score_col="score")
        if odds_col is not None:
            pred = add_market_signals(pred, score_col="score", odds_col=odds_col)
        progress.update(message=f"inference complete rows={len(pred):,} races={pred['race_id'].nunique():,}")
        post_inference_progress = ProgressBar(
            total=3,
            prefix="[evaluate post]",
            logger=log_progress,
            min_interval_sec=0.0,
        )
        post_inference_progress.start("post-inference phases started")

        prediction_sources: dict[str, dict[str, Any]] = {
            "default": {
                "pred": pred,
                "model_config": str(model_config_path),
                "model_path": load_artifacts.model_path.as_posix(),
                "feature_count": int(len(feature_selection.feature_columns)),
                "categorical_feature_count": int(len(feature_selection.categorical_columns)),
            }
        }
        if isinstance(score_regime_overrides_cfg, list) and score_regime_overrides_cfg:
            for override_index, override_cfg in enumerate(score_regime_overrides_cfg, start=1):
                if not isinstance(override_cfg, dict):
                    continue
                override_name = str(override_cfg.get("name") or f"override_{override_index}").strip()
                override_model_config = str(override_cfg.get("model_config", "")).strip()
                if not override_name or not override_model_config:
                    continue

                override_config_path = Path(override_model_config)
                if not override_config_path.is_absolute():
                    override_config_path = ROOT / override_config_path
                override_model_cfg = load_yaml(override_config_path)
                override_output_artifacts = resolve_output_artifacts(override_model_cfg.get("output", {}))
                override_model_path = (
                    override_output_artifacts.model_path
                    if override_output_artifacts.model_path.is_absolute()
                    else ROOT / override_output_artifacts.model_path
                )
                override_model = joblib.load(override_model_path)
                with Heartbeat(
                    "[evaluate]",
                    f"running override inference {override_name}",
                    logger=log_progress,
                ):
                    override_pred, override_feature_selection, _ = _build_scored_prediction_frame(
                        model=override_model,
                        frame=frame,
                        fallback_selection=fallback_selection,
                        odds_col=odds_col,
                    )
                prediction_sources[override_name] = {
                    "pred": override_pred,
                    "model_config": str(override_config_path.relative_to(ROOT)),
                    "model_path": override_output_artifacts.model_path.as_posix(),
                    "feature_count": int(len(override_feature_selection.feature_columns)),
                    "categorical_feature_count": int(len(override_feature_selection.categorical_columns)),
                }
                log_progress(
                    f"Loaded override source '{override_name}' from {override_config_path.relative_to(ROOT)} "
                    f"features={len(override_feature_selection.feature_columns):,}"
                )
        post_inference_progress.update(message=f"score sources ready count={len(prediction_sources):,}")

        score_is_prob = bool(np.nanmin(outputs.score) >= 0.0 and np.nanmax(outputs.score) <= 1.0)
        compute_prob_metrics = task in {"classification", "ranking", "multi_position"}
        probabilistic_flow = bool(compute_prob_metrics and score_is_prob)

        summary = {
            **_base_summary(pred, odds_col=odds_col, score_col="score", include_ev_metrics=include_ev_metrics),
            "loaded_rows": loaded_rows,
            "data_load_strategy": data_load_strategy,
            "primary_source_rows_total": int(primary_source_rows_total) if primary_source_rows_total is not None else None,
            "pre_feature_max_rows": int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
            "pre_feature_rows": pre_feature_rows,
            "requested_max_rows": int(args.max_rows),
            "n_dates": int(pred["date"].nunique()) if "date" in pred.columns else None,
            "auc": _safe_auc(y_eval, outputs.score) if compute_prob_metrics else None,
            "logloss": float(log_loss(y_eval, np.clip(outputs.score, 1e-12, 1 - 1e-12), labels=[0, 1])) if (compute_prob_metrics and score_is_prob) else None,
            "score_is_probability": score_is_prob,
            "task": task,
            "evaluation_flow": "probability_market" if probabilistic_flow else "roi_direct",
        }

        if task in {"time_regression", "time_deviation"}:
            raw_prediction = predict_target_values(model, x_eval)
            if task == "time_regression":
                raw_target = pd.to_numeric(frame.get("finish_time_sec"), errors="coerce").to_numpy(dtype=float)
                summary["time_rmse_sec"] = _safe_regression_rmse(raw_target, raw_prediction)
                summary["time_mae_sec"] = _safe_regression_mae(raw_target, raw_prediction)
                summary["time_pred_corr"] = _safe_regression_corr(raw_target, raw_prediction)
            else:
                raw_target = pd.to_numeric(frame.get("time_deviation"), errors="coerce").to_numpy(dtype=float)
                summary["time_deviation_rmse"] = _safe_regression_rmse(raw_target, raw_prediction)
                summary["time_deviation_mae"] = _safe_regression_mae(raw_target, raw_prediction)
                summary["time_deviation_pred_corr"] = _safe_regression_corr(raw_target, raw_prediction)
        elif task == "market_deviation":
            summary.update(
                _compute_market_deviation_metrics(
                    frame,
                    outputs.score,
                    label_col=label_col,
                )
            )

        summary["run_context"] = {
            "profile": resolved_profile,
            "config": str(model_config_path),
            "data_config": str(data_config_path),
            "feature_config": str(feature_config_path),
            "artifact_suffix": str(args.artifact_suffix or ""),
            "task": task,
            "label_column": label_col,
            "max_rows": int(args.max_rows),
            "requested_max_rows": int(args.max_rows),
            "loaded_rows": loaded_rows,
            "data_load_strategy": data_load_strategy,
            "primary_source_rows_total": int(primary_source_rows_total) if primary_source_rows_total is not None else None,
            "pre_feature_max_rows": int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
            "pre_feature_rows": pre_feature_rows,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "progress_interval_sec": float(args.progress_interval_sec),
            "rows_total_after_tail": int(len(frame)),
            "feature_selection_mode": feature_selection.mode,
            "feature_count": int(len(feature_selection.feature_columns)),
            "categorical_feature_count": int(len(feature_selection.categorical_columns)),
            "score_source_count": int(len(prediction_sources)),
            "artifact_suffix": args.artifact_suffix,
            "model_artifact_suffix": None if explicit_no_model_artifact_suffix else (resolved_model_artifact_suffix or args.artifact_suffix),
            "artifact_manifest": load_artifacts.manifest_path.as_posix() if (ROOT / load_artifacts.manifest_path).exists() else None,
        }
        summary["feature_coverage"] = feature_coverage
        summary["policy_constraints"] = policy_constraints.to_dict()
        if isinstance(policy_search_cfg, dict) and policy_search_cfg:
            summary["policy_search"] = policy_search_cfg
        if len(prediction_sources) > 1:
            summary["score_regime_overrides"] = score_regime_overrides_cfg
            summary["score_sources"] = {
                name: {
                    key: value
                    for key, value in source.items()
                    if key != "pred"
                }
                for name, source in prediction_sources.items()
            }
        summary["date_window"] = {
            "start": args.start_date,
            "end": args.end_date,
        }
        report_dir = ROOT / "artifacts/reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        output_slug = _derive_evaluation_output_slug(
            model_config_path,
            output_artifacts.model_path,
            prefer_config=bool(isinstance(score_regime_overrides_cfg, list) and score_regime_overrides_cfg),
        )
        date_window_slug = _derive_date_window_slug(args.start_date, args.end_date)
        wf_slug = _derive_wf_slug(args.wf_mode, args.wf_scheme)
        wf_progress_enabled = args.wf_mode != "off" and args.wf_scheme == "nested"
        latest_wf_progress_path = report_dir / "evaluation_wf_progress.json"
        versioned_wf_progress_path = report_dir / f"evaluation_wf_progress_{output_slug}{date_window_slug}{wf_slug}.json"
        if wf_progress_enabled:
            artifact_ensure_output_file_path(latest_wf_progress_path, label="latest wf progress output", workspace_root=ROOT)
            artifact_ensure_output_file_path(versioned_wf_progress_path, label="versioned wf progress output", workspace_root=ROOT)
        post_inference_progress.update(
            message=(
                f"summary payload ready auc={summary.get('auc')} "
                f"top1_roi={summary.get('top1_roi')}"
            )
        )
        post_inference_progress.complete(message="post-inference phases finished")

        if leakage_enabled:
            with Heartbeat("[evaluate]", "running leakage audit", logger=log_progress):
                summary["leakage_audit"] = run_leakage_audit(frame=frame, feature_columns=feature_selection.feature_columns, label_column=label_col)
            progress.update(message="leakage audit complete")
        else:
            summary["leakage_audit"] = {"enabled": False}
            progress.update(message="leakage audit skipped")

        if odds_col is not None and probabilistic_flow:
            market_prob = compute_market_prob(pred, odds_col=odds_col).to_numpy(dtype=float)
            model_prob = np.clip(np.asarray(outputs.score, dtype=float), 1e-6, 1 - 1e-6)
            market_prob_clip = np.clip(market_prob, 1e-6, 1 - 1e-6)
            odds_values = pd.to_numeric(pred[odds_col], errors="coerce").to_numpy(dtype=float)
            edge = model_prob * odds_values - 1.0
            delta_logit = np.log(model_prob / (1.0 - model_prob)) - np.log(market_prob_clip / (1.0 - market_prob_clip))
            summary["market_prob_corr"] = float(np.corrcoef(model_prob, market_prob_clip)[0, 1]) if len(model_prob) > 1 else None
            summary["edge_mean"] = float(np.nanmean(edge)) if np.isfinite(np.nanmean(edge)) else None
            summary["delta_logit_mean"] = float(np.nanmean(delta_logit)) if np.isfinite(np.nanmean(delta_logit)) else None

        _append_top3_auc(summary, pred, outputs.top3_probs)

        with Heartbeat("[evaluate]", "running calibration and walk-forward", logger=log_progress):
            calib_train, calib_test = split_for_calibration(pred, date_col="date", train_ratio=0.7)
            if probabilistic_flow and len(calib_train) >= 1000 and len(calib_test) >= 1000:
                train_scores = calib_train["score"].to_numpy()
                train_labels = calib_train[label_col].astype(int).to_numpy()
                test_scores = calib_test["score"].to_numpy()

                platt_scores = fit_platt(train_scores, train_labels, test_scores)
                isotonic_scores = fit_isotonic(train_scores, train_labels, test_scores)

                platt_df = prepare_scored_frame(calib_test, platt_scores, odds_col=odds_col, score_col="calibrated_score")
                isotonic_df = prepare_scored_frame(calib_test, isotonic_scores, odds_col=odds_col, score_col="calibrated_score")
                if odds_col is not None:
                    platt_df = add_market_signals(platt_df, score_col="calibrated_score", odds_col=odds_col)
                    isotonic_df = add_market_signals(isotonic_df, score_col="calibrated_score", odds_col=odds_col)

                platt_metrics = evaluate_fixed_stake_summary(platt_df, odds_col=odds_col, score_col="calibrated_score", stake=100.0)
                isotonic_metrics = evaluate_fixed_stake_summary(isotonic_df, odds_col=odds_col, score_col="calibrated_score", stake=100.0)

                summary["calibration_eval_rows"] = int(len(calib_test))
                summary["calibration_eval_races"] = int(calib_test["race_id"].nunique())
                summary["platt_top1_roi"] = platt_metrics.get("top1_roi")
                summary["platt_ev_top1_roi"] = platt_metrics.get("ev_top1_roi")
                summary["isotonic_top1_roi"] = isotonic_metrics.get("top1_roi")
                summary["isotonic_ev_top1_roi"] = isotonic_metrics.get("ev_top1_roi")
                if summary.get("top1_roi") is not None and summary.get("platt_top1_roi") is not None:
                    summary["platt_top1_roi_lift"] = float(summary["platt_top1_roi"] - summary["top1_roi"])
                if summary.get("top1_roi") is not None and summary.get("isotonic_top1_roi") is not None:
                    summary["isotonic_top1_roi_lift"] = float(summary["isotonic_top1_roi"] - summary["top1_roi"])

                if odds_col is not None:
                    calib_train_b = calib_train.copy()
                    calib_test_b = calib_test.copy()
                    calib_train_b["isotonic_prob"] = fit_isotonic(train_scores, train_labels, train_scores)
                    calib_test_b["isotonic_prob"] = isotonic_scores
                    calib_train_b["market_prob"] = compute_market_prob(calib_train_b, odds_col=odds_col)
                    calib_test_b["market_prob"] = compute_market_prob(calib_test_b, odds_col=odds_col)

                    blend_weight = optimize_blend_weight(
                        frame=calib_train_b,
                        label_col=label_col,
                        model_prob_col="isotonic_prob",
                        market_prob_col="market_prob",
                    )
                    calib_test_b["linear_blend_prob"] = blend_prob(
                        calib_test_b["isotonic_prob"],
                        calib_test_b["market_prob"],
                        weight=blend_weight,
                    )

                    summary["market_blend_weight"] = float(blend_weight)
                    summary["linear_blend_ev_top1_roi"] = ev_top1_roi_from_prob(
                        calib_test_b,
                        prob_col="linear_blend_prob",
                        odds_col=odds_col,
                    )
                    linear_blend_kelly = simulate_fractional_kelly(
                        calib_test_b,
                        prob_col="linear_blend_prob",
                        odds_col=odds_col,
                        min_edge=0.03,
                        min_prob=0.05,
                        fractional_kelly=0.5,
                        max_fraction=0.05,
                        initial_bankroll=1.0,
                    )
                    summary["linear_blend_kelly_roi"] = linear_blend_kelly["kelly_roi"]
                    summary["linear_blend_kelly_bets"] = linear_blend_kelly["kelly_bets"]
                    summary["linear_blend_kelly_hit_rate"] = linear_blend_kelly["kelly_hit_rate"]
                    summary["linear_blend_kelly_final_bankroll"] = linear_blend_kelly["kelly_final_bankroll"]
                    summary["linear_blend_kelly_max_drawdown"] = linear_blend_kelly.get("kelly_max_drawdown")
                    summary["benter_blend_weight"] = float(blend_weight)

                    benchmark_compatible = supports_winner_only_benchmark(
                        calib_test_b[label_col].astype(int).to_numpy(),
                        calib_test_b["race_id"].to_numpy(),
                    )
                    summary["benchmark_eval_rows"] = int(len(calib_test_b))
                    summary["benchmark_eval_races"] = int(calib_test_b["race_id"].nunique())
                    summary["benchmark_single_winner_races"] = bool(benchmark_compatible)
                    summary["benchmark_usable_races"] = count_winner_only_races(
                        calib_test_b[label_col].astype(int).to_numpy(),
                        calib_test_b["race_id"].to_numpy(),
                    )

                    if benchmark_compatible:
                        benter_combiner = fit_benter_combiner(
                            calib_train_b,
                            label_col=label_col,
                            model_prob_col="isotonic_prob",
                            market_prob_col="market_prob",
                        )

                        public_prob = calib_test_b["market_prob"].to_numpy(dtype=float)
                        model_prob = calib_test_b["isotonic_prob"].to_numpy(dtype=float)
                        labels_test = calib_test_b[label_col].astype(int).to_numpy()
                        race_ids_test = calib_test_b["race_id"].to_numpy()

                        summary["public_race_logloss"] = race_winner_logloss(public_prob, labels_test, race_ids_test)
                        summary["model_race_logloss"] = race_winner_logloss(model_prob, labels_test, race_ids_test)
                        summary["public_pseudo_r2"] = pseudo_r2(public_prob, labels_test, race_ids_test)
                        summary["model_pseudo_r2"] = pseudo_r2(model_prob, labels_test, race_ids_test)
                        summary["linear_blend_pseudo_r2"] = pseudo_r2(
                            calib_test_b["linear_blend_prob"].to_numpy(dtype=float),
                            labels_test,
                            race_ids_test,
                        )

                        if benter_combiner is not None:
                            calib_test_b["benter_prob"] = combine_benter_prob(
                                model_prob,
                                public_prob,
                                race_ids_test,
                                alpha=benter_combiner.alpha,
                                beta=benter_combiner.beta,
                            )
                            summary["benter_alpha"] = float(benter_combiner.alpha)
                            summary["benter_beta"] = float(benter_combiner.beta)
                            summary["benter_train_race_logloss"] = benter_combiner.train_race_logloss
                            summary["benter_race_logloss"] = race_winner_logloss(
                                calib_test_b["benter_prob"].to_numpy(dtype=float),
                                labels_test,
                                race_ids_test,
                            )
                            summary["benter_combined_pseudo_r2"] = pseudo_r2(
                                calib_test_b["benter_prob"].to_numpy(dtype=float),
                                labels_test,
                                race_ids_test,
                            )
                            if summary.get("public_pseudo_r2") is not None and summary.get("benter_combined_pseudo_r2") is not None:
                                summary["benter_delta_pseudo_r2"] = float(summary["benter_combined_pseudo_r2"] - summary["public_pseudo_r2"])
                            summary["benter_ev_top1_roi"] = ev_top1_roi_from_prob(
                                calib_test_b,
                                prob_col="benter_prob",
                                odds_col=odds_col,
                            )
                            benter_kelly = simulate_fractional_kelly(
                                calib_test_b,
                                prob_col="benter_prob",
                                odds_col=odds_col,
                                min_edge=0.03,
                                min_prob=0.05,
                                fractional_kelly=0.5,
                                max_fraction=0.05,
                                initial_bankroll=1.0,
                            )
                            summary["benter_kelly_roi"] = benter_kelly["kelly_roi"]
                            summary["benter_kelly_bets"] = benter_kelly["kelly_bets"]
                            summary["benter_kelly_hit_rate"] = benter_kelly["kelly_hit_rate"]
                            summary["benter_kelly_final_bankroll"] = benter_kelly["kelly_final_bankroll"]
                            summary["benter_kelly_max_drawdown"] = benter_kelly.get("kelly_max_drawdown")

                    wf_train, wf_valid, wf_test = split_three_way_time(pred, date_col="date", train_ratio=0.5, valid_ratio=0.25)
                    summary["wf_train_rows"] = int(len(wf_train))
                    summary["wf_valid_rows"] = int(len(wf_valid))
                    summary["wf_test_rows"] = int(len(wf_test))
                    summary["wf_train_window"] = _date_window_payload(wf_train)
                    summary["wf_valid_window"] = _date_window_payload(wf_valid)
                    summary["wf_test_window"] = _date_window_payload(wf_test)
                    summary["wf_mode"] = args.wf_mode
                    summary["wf_scheme"] = args.wf_scheme
                    summary["wf_constraints_min_bet_ratio"] = float(policy_constraints.min_bet_ratio)
                    summary["wf_constraints_min_bets_abs"] = int(policy_constraints.min_bets_abs)
                    summary["wf_constraints_max_drawdown"] = float(policy_constraints.max_drawdown)
                    summary["wf_constraints_min_final_bankroll"] = float(policy_constraints.min_final_bankroll)
                    summary["wf_enabled"] = bool(args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500)

                    if args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500:
                        if args.wf_scheme == "single":
                            log_progress("Walk-forward optimization started (single split)...")
                            wf_score_source = _resolve_score_source_name(
                                score_regime_overrides_cfg,
                                frame=wf_valid,
                            )
                            wf_pred_source = prediction_sources.get(wf_score_source, prediction_sources["default"])
                            wf_train_source = wf_pred_source["pred"].loc[wf_train.index].copy()
                            wf_valid_source = wf_pred_source["pred"].loc[wf_valid.index].copy()
                            wf_test_source = wf_pred_source["pred"].loc[wf_test.index].copy()
                            log_progress(f"Walk-forward single split using score_source={wf_score_source}")
                            wf_policy_constraints = PolicyConstraints.from_config(
                                evaluation_cfg,
                                frame=wf_valid_source,
                            )
                            best_params, best_valid_metrics, wf_family_diagnostics = optimize_roi_strategy(
                                train_df=wf_train_source,
                                valid_df=wf_valid_source,
                                label_col=label_col,
                                odds_col=odds_col,
                                constraints=wf_policy_constraints,
                                mode=args.wf_mode,
                                search_config=policy_search_cfg,
                                progress_interval_sec=float(args.progress_interval_sec),
                                logger=log_progress,
                            )

                            wf_train_scores = wf_train_source["score"].to_numpy()
                            wf_train_labels = wf_train_source[label_col].astype(int).to_numpy()
                            wf_test_source["iso_prob"] = fit_isotonic(wf_train_scores, wf_train_labels, wf_test_source["score"].to_numpy())
                            wf_test_source["market_prob"] = compute_market_prob(wf_test_source, odds_col=odds_col)
                            wf_test_source["blend_prob"] = blend_prob(
                                wf_test_source["iso_prob"],
                                wf_test_source["market_prob"],
                                weight=float(best_params.get("blend_weight", 0.0)),
                            )
                            wf_test_metrics = run_policy_strategy(wf_test_source, prob_col="blend_prob", odds_col=odds_col, params=best_params)
                            log_progress("Walk-forward optimization finished (single split).")
                            _record_single_wf_summary(
                                summary,
                                best_params,
                                best_valid_metrics,
                                wf_test_metrics,
                                wf_family_diagnostics=wf_family_diagnostics,
                                score_source=wf_score_source,
                            )
                        else:
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
                            summary["wf_nested_target_folds"] = int(n_folds)
                            summary["wf_nested_actual_folds"] = int(len(nested_slices))

                            if nested_slices:
                                fold_rows: list[dict[str, float | int | str | None]] = []
                                fold_progress = ProgressBar(total=len(nested_slices), prefix="[evaluate wf]", logger=log_progress, min_interval_sec=0.0)
                                fold_progress.start(message="nested folds started")
                                log_progress(f"Nested WF started: folds={len(nested_slices)}")
                                if wf_progress_enabled:
                                    _write_wf_progress_outputs(
                                        latest_path=latest_wf_progress_path,
                                        versioned_path=versioned_wf_progress_path,
                                        payload=_build_wf_progress_payload(
                                            summary=summary,
                                            output_slug=output_slug,
                                            total_folds=len(nested_slices),
                                            completed_folds=0,
                                            status="running",
                                            current_fold=None,
                                            current_fold_state="starting",
                                        ),
                                    )
                                for fold_index, (fold_train, fold_valid, fold_test) in enumerate(nested_slices, start=1):
                                    fold_train_window = _date_window_payload(fold_train)
                                    fold_valid_window = _date_window_payload(fold_valid)
                                    fold_test_window = _date_window_payload(fold_test)
                                    fold_score_source = _resolve_score_source_name(
                                        score_regime_overrides_cfg,
                                        frame=fold_valid,
                                    )
                                    fold_pred_source = prediction_sources.get(fold_score_source, prediction_sources["default"])
                                    fold_train_source = fold_pred_source["pred"].loc[fold_train.index].copy()
                                    fold_valid_source = fold_pred_source["pred"].loc[fold_valid.index].copy()
                                    fold_test_source = fold_pred_source["pred"].loc[fold_test.index].copy()
                                    log_progress(
                                        f"Nested WF fold {fold_index}/{len(nested_slices)}: optimizing on inner valid "
                                        f"with score_source={fold_score_source}..."
                                    )
                                    if wf_progress_enabled:
                                        _write_wf_progress_outputs(
                                            latest_path=latest_wf_progress_path,
                                            versioned_path=versioned_wf_progress_path,
                                            payload=_build_wf_progress_payload(
                                                summary=summary,
                                                output_slug=output_slug,
                                                total_folds=len(nested_slices),
                                                completed_folds=len(fold_rows),
                                                status="running",
                                                current_fold=fold_index,
                                                current_fold_state="optimizing",
                                                current_score_source=fold_score_source,
                                                current_train_window=fold_train_window,
                                                current_valid_window=fold_valid_window,
                                                current_test_window=fold_test_window,
                                                latest_completed_fold=fold_rows[-1] if fold_rows else None,
                                            ),
                                        )
                                    fold_policy_constraints = PolicyConstraints.from_config(
                                        evaluation_cfg,
                                        frame=fold_valid_source,
                                    )
                                    best_params, best_valid_metrics, fold_family_diagnostics = optimize_roi_strategy(
                                        train_df=fold_train_source,
                                        valid_df=fold_valid_source,
                                        label_col=label_col,
                                        odds_col=odds_col,
                                        constraints=fold_policy_constraints,
                                        mode=args.wf_mode,
                                        search_config=policy_search_cfg,
                                        progress_interval_sec=float(args.progress_interval_sec),
                                        logger=log_progress,
                                    )

                                    fold_train_scores = fold_train_source["score"].to_numpy()
                                    fold_train_labels = fold_train_source[label_col].astype(int).to_numpy()
                                    fold_test_source["iso_prob"] = fit_isotonic(fold_train_scores, fold_train_labels, fold_test_source["score"].to_numpy())
                                    fold_test_source["market_prob"] = compute_market_prob(fold_test_source, odds_col=odds_col)
                                    fold_test_source["blend_prob"] = blend_prob(
                                        fold_test_source["iso_prob"],
                                        fold_test_source["market_prob"],
                                        weight=float(best_params.get("blend_weight", 0.0)),
                                    )
                                    fold_test_metrics = run_policy_strategy(fold_test_source, prob_col="blend_prob", odds_col=odds_col, params=best_params)
                                    strategy_kind = str(best_params.get("strategy_kind", "kelly"))
                                    fold_rows.append(
                                        _fold_row(
                                            strategy_kind,
                                            best_params,
                                            best_valid_metrics,
                                            fold_test_metrics,
                                            fold_index,
                                            family_diagnostics=fold_family_diagnostics,
                                            score_source=fold_score_source,
                                            train_window=fold_train_window,
                                            valid_window=fold_valid_window,
                                            test_window=fold_test_window,
                                        )
                                    )
                                    fold_progress.update(message=f"fold {fold_index} complete")
                                    if wf_progress_enabled:
                                        _write_wf_progress_outputs(
                                            latest_path=latest_wf_progress_path,
                                            versioned_path=versioned_wf_progress_path,
                                            payload=_build_wf_progress_payload(
                                                summary=summary,
                                                output_slug=output_slug,
                                                total_folds=len(nested_slices),
                                                completed_folds=len(fold_rows),
                                                status="running",
                                                current_fold=fold_index,
                                                current_fold_state="completed",
                                                current_score_source=fold_score_source,
                                                current_train_window=fold_train_window,
                                                current_valid_window=fold_valid_window,
                                                current_test_window=fold_test_window,
                                                latest_completed_fold=fold_rows[-1],
                                            ),
                                        )

                                fold_progress.complete(message="nested folds finished")
                                summary["wf_nested_folds"] = fold_rows
                                test_roi_values = [float(row["test_roi"]) for row in fold_rows if row.get("test_roi") is not None]
                                test_bets_values = [int(row["test_bets"] or 0) for row in fold_rows]
                                weighted_numerator = 0.0
                                weighted_denominator = 0.0
                                for row in fold_rows:
                                    row_roi = row.get("test_roi")
                                    row_bets = int(row.get("test_bets") or 0)
                                    if row_roi is None or row_bets <= 0:
                                        continue
                                    weighted_numerator += float(row_roi) * row_bets
                                    weighted_denominator += row_bets

                                summary["wf_nested_test_roi_mean"] = float(np.mean(test_roi_values)) if test_roi_values else None
                                summary["wf_nested_test_roi_weighted"] = float(weighted_numerator / weighted_denominator) if weighted_denominator > 0 else None
                                summary["wf_nested_test_bets_total"] = int(np.sum(test_bets_values))
                                summary["wf_nested_test_bets_mean"] = float(np.mean(test_bets_values)) if test_bets_values else None
                                summary["wf_nested_completed"] = True
                                log_progress("Nested WF finished.")
                                if wf_progress_enabled:
                                    _write_wf_progress_outputs(
                                        latest_path=latest_wf_progress_path,
                                        versioned_path=versioned_wf_progress_path,
                                        payload=_build_wf_progress_payload(
                                            summary=summary,
                                            output_slug=output_slug,
                                            total_folds=len(nested_slices),
                                            completed_folds=len(fold_rows),
                                            status="completed",
                                            current_fold=None,
                                            current_fold_state="completed",
                                            latest_completed_fold=fold_rows[-1] if fold_rows else None,
                                        ),
                                    )
                            else:
                                summary["wf_nested_completed"] = False
                                summary["wf_nested_reason"] = "insufficient_data_for_nested_folds"
                                if wf_progress_enabled:
                                    _write_wf_progress_outputs(
                                        latest_path=latest_wf_progress_path,
                                        versioned_path=versioned_wf_progress_path,
                                        payload=_build_wf_progress_payload(
                                            summary=summary,
                                            output_slug=output_slug,
                                            total_folds=0,
                                            completed_folds=0,
                                            status="skipped",
                                            current_fold_state="insufficient_data_for_nested_folds",
                                        ),
                                    )
            else:
                if not probabilistic_flow:
                    summary["calibration_skipped_reason"] = "non_probability_task_or_score"
                    summary["wf_enabled"] = False
                    summary["wf_mode"] = args.wf_mode
                    summary["wf_scheme"] = args.wf_scheme
                    summary["wf_skipped_reason"] = "non_probability_task_or_score"
                else:
                    summary["calibration_skipped_reason"] = "insufficient_calibration_rows"
                    summary["wf_enabled"] = False
                    summary["wf_mode"] = args.wf_mode
                    summary["wf_scheme"] = args.wf_scheme
                    summary["wf_skipped_reason"] = "insufficient_calibration_rows"
        progress.update(message="calibration and walk-forward complete")

        by_date_rows: list[dict[str, object]] = []
        if "date" in pred.columns:
            date_series = pd.to_datetime(pred["date"], errors="coerce")
            by_date_progress = ProgressBar(
                total=max(int(date_series.dt.date.nunique()), 1),
                prefix="[evaluate by-date]",
                logger=log_progress,
                min_interval_sec=2.0,
            )
            by_date_progress.start(message="daily aggregation started")
            for date_value, date_df in pred.groupby(date_series.dt.date):
                date_summary = _base_summary(date_df, odds_col=odds_col, score_col="score")
                if not include_ev_metrics:
                    date_summary["ev_top1_roi"] = None
                    date_summary["ev_threshold_1_0_roi"] = None
                    date_summary["ev_threshold_1_0_bets"] = 0
                    date_summary["ev_threshold_1_2_roi"] = None
                    date_summary["ev_threshold_1_2_bets"] = 0
                if task == "market_deviation":
                    date_summary.update(
                        _compute_market_deviation_metrics(
                            date_df,
                            date_df["score"].to_numpy(dtype=float),
                            label_col=label_col,
                        )
                    )
                by_date_rows.append({"date": str(date_value), **date_summary})
                by_date_progress.update(message=f"date={date_value}")
            by_date_progress.complete(message="daily aggregation finished")
        progress.update(message=f"by-date aggregation complete rows={len(by_date_rows):,}")

        by_date = pd.DataFrame(by_date_rows).sort_values("date") if by_date_rows else pd.DataFrame()
        summary["stability_guardrail"] = build_stability_guardrail(frame=pred, by_date=by_date, summary=summary)
        summary["stability_assessment"] = summary["stability_guardrail"]["assessment"]
        if summary["stability_assessment"] != "representative":
            warning_preview = "; ".join(summary["stability_guardrail"].get("warnings", [])[:2])
            log_progress(
                f"Stability guardrail={summary['stability_assessment']}: {warning_preview}"
            )

        summary_path = report_dir / "evaluation_summary.json"
        by_date_path = report_dir / "evaluation_by_date.csv"
        manifest_path = report_dir / "evaluation_manifest.json"
        versioned_summary_path = report_dir / f"evaluation_summary_{output_slug}{date_window_slug}{wf_slug}.json"
        versioned_by_date_path = report_dir / f"evaluation_by_date_{output_slug}{date_window_slug}{wf_slug}.csv"
        versioned_manifest_path = report_dir / f"evaluation_manifest_{output_slug}{date_window_slug}{wf_slug}.json"
        artifact_ensure_output_file_path(summary_path, label="latest summary output", workspace_root=ROOT)
        artifact_ensure_output_file_path(manifest_path, label="latest manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(versioned_summary_path, label="versioned summary output", workspace_root=ROOT)
        artifact_ensure_output_file_path(versioned_manifest_path, label="versioned manifest output", workspace_root=ROOT)
        if not by_date.empty:
            artifact_ensure_output_file_path(by_date_path, label="latest by-date output", workspace_root=ROOT)
            artifact_ensure_output_file_path(versioned_by_date_path, label="versioned by-date output", workspace_root=ROOT)

        latest_by_date_output_path = by_date_path if not by_date.empty else None
        versioned_by_date_output_path = versioned_by_date_path if not by_date.empty else None

        summary["output_files"] = {
            "latest_summary": artifact_display_path(summary_path, workspace_root=ROOT),
            "latest_by_date": artifact_display_path(latest_by_date_output_path, workspace_root=ROOT) if latest_by_date_output_path is not None else None,
            "latest_manifest": artifact_display_path(manifest_path, workspace_root=ROOT),
            "latest_wf_progress": artifact_display_path(latest_wf_progress_path, workspace_root=ROOT) if wf_progress_enabled else None,
            "versioned_summary": artifact_display_path(versioned_summary_path, workspace_root=ROOT),
            "versioned_by_date": artifact_display_path(versioned_by_date_output_path, workspace_root=ROOT) if versioned_by_date_output_path is not None else None,
            "versioned_manifest": artifact_display_path(versioned_manifest_path, workspace_root=ROOT),
            "versioned_wf_progress": artifact_display_path(versioned_wf_progress_path, workspace_root=ROOT) if wf_progress_enabled else None,
        }

        summary_text = json.dumps(summary, ensure_ascii=False, indent=2)
        summary_sha256 = _sha256_text(summary_text)
        by_date_text = by_date.to_csv(index=False) if not by_date.empty else None
        by_date_sha256 = _sha256_text(by_date_text) if by_date_text is not None else None
        evaluation_manifest = _build_evaluation_output_manifest(
            summary=summary,
            by_date=by_date,
            latest_summary_path=summary_path,
            latest_by_date_path=latest_by_date_output_path,
            latest_manifest_path=manifest_path,
            versioned_summary_path=versioned_summary_path,
            versioned_by_date_path=versioned_by_date_output_path,
            versioned_manifest_path=versioned_manifest_path,
            summary_sha256=summary_sha256,
            by_date_sha256=by_date_sha256,
            latest_wf_progress_path=latest_wf_progress_path if wf_progress_enabled else None,
            versioned_wf_progress_path=versioned_wf_progress_path if wf_progress_enabled else None,
        )

        with Heartbeat("[evaluate]", "writing evaluation outputs", logger=log_progress):
            write_text_file(summary_path, summary_text, label="latest summary output")
            write_text_file(versioned_summary_path, summary_text, label="versioned summary output")
            write_json(manifest_path, evaluation_manifest)
            write_json(versioned_manifest_path, evaluation_manifest)
            if by_date_text is not None:
                write_text_file(by_date_path, by_date_text, label="latest by-date output")
                write_text_file(versioned_by_date_path, by_date_text, label="versioned by-date output")
        progress.complete(message="evaluation outputs written")

        print(f"[evaluate] summary saved: {summary_path}")
        print(f"[evaluate] versioned summary saved: {versioned_summary_path}")
        print(f"[evaluate] manifest saved: {manifest_path}")
        print(f"[evaluate] versioned manifest saved: {versioned_manifest_path}")
        if by_date_text is not None:
            print(f"[evaluate] by-date saved: {by_date_path}")
            print(f"[evaluate] versioned by-date saved: {versioned_by_date_path}")
            print(by_date.tail(5).to_string(index=False))
        print(f"[evaluate] summary: {json.dumps(summary, ensure_ascii=False)}")
        return 0
    except KeyboardInterrupt:
        print("[evaluate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[evaluate] failed: {error}")
        return 1
    except Exception as error:
        print(f"[evaluate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
