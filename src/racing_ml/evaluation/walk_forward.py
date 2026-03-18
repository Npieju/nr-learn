from __future__ import annotations

import time
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from racing_ml.common.regime import resolve_regime_override
from racing_ml.evaluation.policy import (
    PolicyConstraints,
    apply_selection_mode,
    blend_prob,
    compute_market_prob,
    evaluate_candidate_gate,
    run_policy_strategy,
)


Logger = Callable[[str], None] | None


def _resolve_search_candidate_values(
    search_config: dict[str, Any] | None,
    *,
    mode: str,
    key: str,
    default: list[float | int],
    cast: Callable[[Any], float | int],
) -> list[float | int]:
    if not isinstance(search_config, dict):
        return list(default)

    merged: dict[str, Any] = {}
    for config_key, config_value in search_config.items():
        if isinstance(config_value, dict):
            continue
        merged[config_key] = config_value

    mode_value = search_config.get(str(mode).strip().lower())
    if isinstance(mode_value, dict):
        merged.update(mode_value)

    raw_values = merged.get(key)
    if not isinstance(raw_values, (list, tuple)):
        return list(default)

    resolved: list[float | int] = []
    for raw_value in raw_values:
        try:
            resolved_value = cast(raw_value)
        except Exception:
            continue
        if resolved_value in resolved:
            continue
        resolved.append(resolved_value)

    return resolved or list(default)


def _resolve_regime_search_config(
    search_config: dict[str, Any] | None,
    *,
    frame: pd.DataFrame,
    date_col: str = "date",
) -> dict[str, Any] | None:
    if not isinstance(search_config, dict):
        return search_config

    overrides = search_config.get("regime_overrides")
    if not isinstance(overrides, list) or not overrides:
        return search_config

    resolved = {key: value for key, value in search_config.items() if key != "regime_overrides"}
    override = resolve_regime_override(overrides, frame=frame, date_col=date_col)
    if not isinstance(override, dict):
        return resolved

    for key, value in override.items():
        if key in {"when", "name"}:
            continue
        resolved[key] = value

    return resolved


def split_for_calibration(frame: pd.DataFrame, date_col: str = "date", train_ratio: float = 0.7) -> tuple[pd.DataFrame, pd.DataFrame]:
    if date_col not in frame.columns:
        midpoint = int(len(frame) * train_ratio)
        return frame.iloc[:midpoint].copy(), frame.iloc[midpoint:].copy()

    date_series = pd.to_datetime(frame[date_col], errors="coerce")
    valid_dates = pd.Series(date_series.dropna().unique()).sort_values()
    if len(valid_dates) < 4:
        midpoint = int(len(frame) * train_ratio)
        return frame.iloc[:midpoint].copy(), frame.iloc[midpoint:].copy()

    cutoff_idx = max(int(len(valid_dates) * train_ratio), 1)
    cutoff_date = pd.to_datetime(valid_dates.iloc[cutoff_idx - 1])
    train_df = frame[date_series <= cutoff_date].copy()
    test_df = frame[date_series > cutoff_date].copy()

    if train_df.empty or test_df.empty:
        midpoint = int(len(frame) * train_ratio)
        train_df = frame.iloc[:midpoint].copy()
        test_df = frame.iloc[midpoint:].copy()
    return train_df, test_df


def split_three_way_time(
    frame: pd.DataFrame,
    date_col: str = "date",
    train_ratio: float = 0.5,
    valid_ratio: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if date_col not in frame.columns:
        n = len(frame)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        return frame.iloc[:train_end].copy(), frame.iloc[train_end:valid_end].copy(), frame.iloc[valid_end:].copy()

    date_series = pd.to_datetime(frame[date_col], errors="coerce")
    valid_dates = pd.Series(date_series.dropna().unique()).sort_values().reset_index(drop=True)
    if len(valid_dates) < 12:
        n = len(frame)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        return frame.iloc[:train_end].copy(), frame.iloc[train_end:valid_end].copy(), frame.iloc[valid_end:].copy()

    train_idx = max(int(len(valid_dates) * train_ratio), 1)
    valid_idx = max(int(len(valid_dates) * (train_ratio + valid_ratio)), train_idx + 1)
    if valid_idx >= len(valid_dates):
        valid_idx = len(valid_dates) - 1

    train_cut = pd.to_datetime(valid_dates.iloc[train_idx - 1])
    valid_cut = pd.to_datetime(valid_dates.iloc[valid_idx - 1])

    train_df = frame[date_series <= train_cut].copy()
    valid_df = frame[(date_series > train_cut) & (date_series <= valid_cut)].copy()
    test_df = frame[date_series > valid_cut].copy()

    if train_df.empty or valid_df.empty or test_df.empty:
        n = len(frame)
        train_end = int(n * train_ratio)
        valid_end = int(n * (train_ratio + valid_ratio))
        train_df = frame.iloc[:train_end].copy()
        valid_df = frame.iloc[train_end:valid_end].copy()
        test_df = frame.iloc[valid_end:].copy()
    return train_df, valid_df, test_df


def build_nested_wf_slices(
    frame: pd.DataFrame,
    date_col: str = "date",
    n_folds: int = 3,
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    min_train_rows: int = 1000,
    min_valid_rows: int = 500,
    min_test_rows: int = 500,
) -> list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    if date_col not in frame.columns:
        return []

    date_series = pd.to_datetime(frame[date_col], errors="coerce")
    valid_dates = pd.Series(date_series.dropna().unique()).sort_values().reset_index(drop=True)
    if len(valid_dates) < 20:
        return []

    n_folds = max(int(n_folds), 1)
    valid_span = max(int(len(valid_dates) * valid_ratio), 1)
    test_span = max(int(len(valid_dates) * test_ratio), 1)
    min_train_span = max(int(len(valid_dates) * 0.3), 3)

    latest_start = len(valid_dates) - (valid_span + test_span)
    earliest_start = min_train_span
    if latest_start <= earliest_start:
        return []

    if n_folds == 1:
        train_end_indices = [latest_start]
    else:
        train_end_indices = np.linspace(earliest_start, latest_start, num=n_folds, dtype=int).tolist()

    slices: list[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]] = []
    for train_end_idx in train_end_indices:
        train_cut = pd.to_datetime(valid_dates.iloc[train_end_idx])
        valid_end_idx = min(train_end_idx + valid_span, len(valid_dates) - 2)
        test_end_idx = min(valid_end_idx + test_span, len(valid_dates) - 1)
        valid_cut = pd.to_datetime(valid_dates.iloc[valid_end_idx])
        test_cut = pd.to_datetime(valid_dates.iloc[test_end_idx])

        train_df = frame[date_series <= train_cut].copy()
        valid_df = frame[(date_series > train_cut) & (date_series <= valid_cut)].copy()
        test_df = frame[(date_series > valid_cut) & (date_series <= test_cut)].copy()

        if len(train_df) < min_train_rows or len(valid_df) < min_valid_rows or len(test_df) < min_test_rows:
            continue
        slices.append((train_df, valid_df, test_df))

    return slices


def fit_platt(train_scores: np.ndarray, train_labels: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    model = LogisticRegression(max_iter=1000)
    model.fit(train_scores.reshape(-1, 1), train_labels)
    return model.predict_proba(test_scores.reshape(-1, 1))[:, 1]


def fit_isotonic(train_scores: np.ndarray, train_labels: np.ndarray, test_scores: np.ndarray) -> np.ndarray:
    model = IsotonicRegression(out_of_bounds="clip")
    model.fit(train_scores, train_labels)
    return model.transform(test_scores)


def optimize_blend_weight(
    frame: pd.DataFrame,
    label_col: str,
    model_prob_col: str,
    market_prob_col: str,
) -> float:
    candidates = np.linspace(0.1, 0.9, 9)
    best_weight = 0.7
    best_loss = float("inf")

    y_true = frame[label_col].astype(int).to_numpy()
    for weight in candidates:
        probs = blend_prob(frame[model_prob_col], frame[market_prob_col], float(weight)).to_numpy()
        loss = float(log_loss(y_true, probs, labels=[0, 1]))
        if loss < best_loss:
            best_loss = loss
            best_weight = float(weight)
    return best_weight


def _emit(logger: Logger, message: str) -> None:
    if logger is not None:
        logger(message)


def optimize_roi_strategy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_col: str,
    odds_col: str,
    constraints: PolicyConstraints,
    mode: str = "fast",
    search_config: dict[str, Any] | None = None,
    progress_interval_sec: float = 5.0,
    logger: Logger = None,
) -> tuple[dict[str, float | str], dict[str, float | int | None]]:
    search_config = _resolve_regime_search_config(search_config, frame=valid_df)
    train_scores = train_df["score"].to_numpy()
    train_labels = train_df[label_col].astype(int).to_numpy()

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["iso_prob"] = fit_isotonic(train_scores, train_labels, train_scores)
    valid_df["iso_prob"] = fit_isotonic(train_scores, train_labels, valid_df["score"].to_numpy())
    train_df["market_prob"] = compute_market_prob(train_df, odds_col=odds_col)
    valid_df["market_prob"] = compute_market_prob(valid_df, odds_col=odds_col)

    if mode == "full":
        blend_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="blend_weights",
            default=[0.2, 0.4, 0.6, 0.8],
            cast=float,
        )
        edge_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="min_edges",
            default=[0.01, 0.03, 0.05],
            cast=float,
        )
        min_prob_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="min_probabilities",
            default=[0.03, 0.05],
            cast=float,
        )
        kelly_frac_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="fractional_kelly_values",
            default=[0.25, 0.5],
            cast=float,
        )
        max_frac_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="max_fraction_values",
            default=[0.02, 0.05],
            cast=float,
        )
        odds_min_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="odds_mins",
            default=[1.0],
            cast=float,
        )
        odds_max_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="odds_maxs",
            default=[25.0, 40.0, 80.0],
            cast=float,
        )
        top_k_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="top_ks",
            default=[1, 2],
            cast=int,
        )
        min_ev_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="min_expected_values",
            default=[1.0, 1.05, 1.10],
            cast=float,
        )
    else:
        blend_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="blend_weights",
            default=[0.2, 0.4, 0.6],
            cast=float,
        )
        edge_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="min_edges",
            default=[0.01, 0.03],
            cast=float,
        )
        min_prob_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="min_probabilities",
            default=[0.05],
            cast=float,
        )
        kelly_frac_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="fractional_kelly_values",
            default=[0.25, 0.5],
            cast=float,
        )
        max_frac_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="max_fraction_values",
            default=[0.02],
            cast=float,
        )
        odds_min_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="odds_mins",
            default=[1.0],
            cast=float,
        )
        odds_max_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="odds_maxs",
            default=[25.0, 40.0],
            cast=float,
        )
        top_k_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="top_ks",
            default=[1, 2],
            cast=int,
        )
        min_ev_candidates = _resolve_search_candidate_values(
            search_config,
            mode=mode,
            key="min_expected_values",
            default=[1.0, 1.05],
            cast=float,
        )

    best_params: dict[str, float | str] = {
        "strategy_kind": "kelly",
        "blend_weight": 0.7,
        "min_edge": 0.03,
        "min_prob": 0.05,
        "fractional_kelly": 0.5,
        "max_fraction": 0.05,
        "odds_min": 1.0,
        "odds_max": 50.0,
        "top_k": 2,
        "min_expected_value": 1.0,
    }
    best_score = float("-inf")
    best_metrics: dict[str, float | int | None] = {}
    fallback_params: dict[str, float | str] = dict(best_params)
    fallback_score = float("-inf")
    fallback_metrics: dict[str, float | int | None] = {}

    n_races = int(valid_df["race_id"].nunique())
    total_trials = (
        len(blend_candidates) * len(edge_candidates) * len(min_prob_candidates) * len(odds_min_candidates) * len(odds_max_candidates) * len(kelly_frac_candidates) * len(max_frac_candidates)
        + len(blend_candidates) * len(edge_candidates) * len(min_prob_candidates) * len(odds_min_candidates) * len(odds_max_candidates) * len(top_k_candidates) * len(min_ev_candidates)
    )
    completed = 0
    started_at = time.perf_counter()
    last_logged_at = started_at

    _emit(
        logger,
        "WF strategy search started: "
        f"mode={mode}, total_trials={total_trials}, "
        f"selection_mode={constraints.selection_mode}, min_bets>={constraints.min_bets_required(n_races)}",
    )

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal last_logged_at
        now = time.perf_counter()
        if not force and (now - last_logged_at) < max(progress_interval_sec, 0.5):
            return
        elapsed = now - started_at
        ratio = (completed / total_trials) if total_trials > 0 else 1.0
        rate = (completed / elapsed) if elapsed > 0 else 0.0
        remaining = total_trials - completed
        eta = (remaining / rate) if rate > 0 else float("inf")
        eta_text = "--" if eta == float("inf") else f"{max(int(eta), 0)}s"
        _emit(
            logger,
            "WF strategy search progress: "
            f"{completed}/{total_trials} ({ratio:.1%}), eta={eta_text}, best_score={best_score:.4f}",
        )
        last_logged_at = now

    def update_candidate(
        params: dict[str, float],
        metrics: dict[str, float | int | None],
        roi_key: str,
        bets_key: str,
        hit_key: str,
        bankroll_key: str,
        drawdown_key: str,
        hit_weight: float,
    ) -> None:
        nonlocal best_score, best_params, best_metrics, fallback_score, fallback_params, fallback_metrics

        roi = metrics.get(roi_key)
        if roi is None:
            return

        roi_value = float(roi)
        hit_value = float(metrics.get(hit_key) or 0.0)
        base_score = roi_value + hit_weight * hit_value
        bets = int(metrics.get(bets_key) or 0)
        final_bankroll = metrics.get(bankroll_key)
        max_drawdown = metrics.get(drawdown_key)
        gate_result = evaluate_candidate_gate(
            bets=bets,
            max_drawdown=float(max_drawdown) if max_drawdown is not None else None,
            final_bankroll=float(final_bankroll) if final_bankroll is not None else None,
            constraints=constraints,
            n_races=n_races,
        )

        if base_score > fallback_score:
            fallback_score = base_score
            fallback_params = dict(params)
            fallback_metrics = dict(metrics)

        selection_score = apply_selection_mode(
            base_score,
            gate_result,
            constraints,
            bets=bets,
            max_drawdown=float(max_drawdown) if max_drawdown is not None else None,
            final_bankroll=float(final_bankroll) if final_bankroll is not None else None,
        )
        if selection_score is not None and selection_score > best_score:
            best_score = float(selection_score)
            best_params = dict(params)
            best_metrics = dict(metrics)

    for blend_weight in blend_candidates:
        valid_df["blend_prob"] = blend_prob(valid_df["iso_prob"], valid_df["market_prob"], blend_weight)
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
                                completed += 1
                                update_candidate(
                                    params,
                                    metrics,
                                    roi_key="kelly_roi",
                                    bets_key="kelly_bets",
                                    hit_key="kelly_hit_rate",
                                    bankroll_key="kelly_final_bankroll",
                                    drawdown_key="kelly_max_drawdown",
                                    hit_weight=0.10,
                                )
                                maybe_log_progress()

                        for top_k in top_k_candidates:
                            for min_ev in min_ev_candidates:
                                params = {
                                    "strategy_kind": "portfolio",
                                    "blend_weight": float(blend_weight),
                                    "min_prob": float(min_prob),
                                    "odds_min": float(odds_min),
                                    "odds_max": float(odds_max),
                                    "top_k": float(top_k),
                                    "min_expected_value": float(min_ev),
                                }
                                metrics = run_policy_strategy(valid_df, prob_col="blend_prob", odds_col=odds_col, params=params)
                                completed += 1
                                update_candidate(
                                    params,
                                    metrics,
                                    roi_key="portfolio_roi",
                                    bets_key="portfolio_bets",
                                    hit_key="portfolio_hit_rate",
                                    bankroll_key="portfolio_final_bankroll",
                                    drawdown_key="portfolio_max_drawdown",
                                    hit_weight=0.20,
                                )
                                maybe_log_progress()

    if best_score == float("-inf"):
        selection_mode = constraints.selection_mode.strip().lower()
        if selection_mode in {"gate_then_roi", "risk_first"}:
            best_params = {
                "strategy_kind": "no_bet",
                "blend_weight": 0.0,
                "initial_bankroll": 1.0,
                "selection_reason": "no_feasible_candidate",
            }
            best_metrics = run_policy_strategy(valid_df, prob_col="blend_prob", odds_col=odds_col, params=best_params)
            best_score = 0.0
            _emit(logger, "WF strategy search found no feasible candidate; selecting no_bet.")
        else:
            best_params = dict(fallback_params)
            best_metrics = dict(fallback_metrics)
            best_score = float(fallback_score)

    maybe_log_progress(force=True)
    _emit(
        logger,
        "WF strategy search finished: "
        f"best_strategy={best_params.get('strategy_kind', 'unknown')}, best_score={best_score:.4f}",
    )

    return best_params, best_metrics