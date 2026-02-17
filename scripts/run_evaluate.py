import argparse
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
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.probability import normalize_position_probabilities
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.leakage import run_leakage_audit
from racing_ml.features.builder import build_features


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[evaluate {now}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds_int = max(int(seconds), 0)
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def predict_score(model: object, frame: pd.DataFrame, race_ids: pd.Series | None = None) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(frame), dtype=float)
    if isinstance(model, dict) and model.get("kind") == "multi_position_top3":
        probs = predict_top3_probs(model, frame, race_ids=race_ids)
        if probs is None:
            raise RuntimeError("Invalid multi_position model bundle")
        return probs["p_rank1"]
    raise RuntimeError("Loaded model does not support predict/predict_proba")


def predict_top3_probs(
    model: Any,
    frame: pd.DataFrame,
    race_ids: pd.Series | None = None,
) -> dict[str, np.ndarray] | None:
    if not (isinstance(model, dict) and model.get("kind") == "multi_position_top3"):
        return None
    prep = model.get("prep")
    models = model.get("models", {})
    if prep is None or not isinstance(models, dict):
        return None
    transformed = prep.transform(frame)
    output: dict[str, np.ndarray] = {}
    for key in ["p_rank1", "p_rank2", "p_rank3"]:
        model_obj = models.get(key)
        if model_obj is None:
            return None
        output[key] = model_obj.predict_proba(transformed)[:, 1]

    if race_ids is None:
        if "race_id" not in frame.columns:
            raise RuntimeError("race_ids are required for multi_position probability normalization")
        race_ids = frame["race_id"]

    work = pd.DataFrame({"race_id": race_ids.to_numpy(copy=False)})
    work["p_rank1_raw"] = output["p_rank1"]
    work["p_rank2_raw"] = output["p_rank2"]
    work["p_rank3_raw"] = output["p_rank3"]
    work = normalize_position_probabilities(
        work,
        raw_columns=["p_rank1_raw", "p_rank2_raw", "p_rank3_raw"],
        race_id_col="race_id",
        output_prefix="",
    )
    return {
        "p_rank1": work["p_rank1_raw"].to_numpy(dtype=float),
        "p_rank2": work["p_rank2_raw"].to_numpy(dtype=float),
        "p_rank3": work["p_rank3_raw"].to_numpy(dtype=float),
    }


def topk_hit_rate(frame: pd.DataFrame, k: int) -> float:
    hits: list[int] = []
    for _, group in frame.groupby("race_id"):
        picks = group[group["pred_rank"] <= k]
        if "rank" in picks.columns:
            rank_values = pd.to_numeric(picks["rank"], errors="coerce")
        else:
            rank_values = pd.Series([], dtype=float)
        hits.append(int((rank_values == 1).any()))
    return float(np.mean(hits)) if hits else float("nan")


def rank_by_score(frame: pd.DataFrame, score_col: str, out_col: str = "pred_rank") -> pd.DataFrame:
    ranked = frame.copy()
    ranked[out_col] = (
        ranked.groupby("race_id")[score_col]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return ranked


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(converted):
            return None
        return float(converted)
    except Exception:
        return None


def top1_roi(frame: pd.DataFrame, odds_col: str | None, stake: float = 100.0) -> float | None:
    if odds_col is None or "rank" not in frame.columns:
        return None

    total_bet = 0.0
    total_return = 0.0
    for _, group in frame.groupby("race_id"):
        pick = group.sort_values("pred_rank").iloc[0]
        total_bet += stake
        rank = to_float(pick.get("rank"))
        if rank is not None and int(rank) == 1:
            odds = to_float(pick.get(odds_col))
            if odds is not None and odds > 0:
                total_return += float(stake * odds)
            else:
                total_return += stake

    return float(total_return / total_bet) if total_bet > 0 else None


def ev_top1_roi(frame: pd.DataFrame, odds_col: str | None, stake: float = 100.0) -> float | None:
    if odds_col is None or "rank" not in frame.columns:
        return None

    if "expected_value" not in frame.columns:
        return None

    total_bet = 0.0
    total_return = 0.0
    for _, group in frame.groupby("race_id"):
        pick = group.sort_values("expected_value", ascending=False).iloc[0]
        total_bet += stake
        rank = to_float(pick.get("rank"))
        if rank is not None and int(rank) == 1:
            odds = to_float(pick.get(odds_col))
            if odds is not None and odds > 0:
                total_return += float(stake * odds)
            else:
                total_return += stake

    return float(total_return / total_bet) if total_bet > 0 else None


def ev_threshold_roi(
    frame: pd.DataFrame,
    odds_col: str | None,
    threshold: float = 1.0,
    stake: float = 100.0,
) -> tuple[float | None, int]:
    if odds_col is None or "rank" not in frame.columns:
        return None, 0
    if "expected_value" not in frame.columns:
        return None, 0

    total_bet = 0.0
    total_return = 0.0
    bets = 0

    for _, group in frame.groupby("race_id"):
        candidates = group[group["expected_value"] >= threshold]
        if candidates.empty:
            continue
        pick = candidates.sort_values("expected_value", ascending=False).iloc[0]
        bets += 1
        total_bet += stake
        rank = to_float(pick.get("rank"))
        if rank is not None and int(rank) == 1:
            odds = to_float(pick.get(odds_col))
            if odds is not None and odds > 0:
                total_return += float(stake * odds)
            else:
                total_return += stake

    if total_bet == 0:
        return None, bets
    return float(total_return / total_bet), bets


def evaluate_frame(frame: pd.DataFrame, score_col: str, odds_col: str | None) -> dict[str, float | int | None]:
    scored = rank_by_score(frame, score_col=score_col, out_col="pred_rank")
    if odds_col is not None:
        scored["expected_value"] = scored[score_col] * scored[odds_col]

    threshold_1_0_roi, threshold_1_0_bets = ev_threshold_roi(scored, odds_col=odds_col, threshold=1.0)
    threshold_1_2_roi, threshold_1_2_bets = ev_threshold_roi(scored, odds_col=odds_col, threshold=1.2)

    return {
        "n_rows": int(len(scored)),
        "n_races": int(scored["race_id"].nunique()),
        "top1_hit_rate": topk_hit_rate(scored, 1) if "rank" in scored.columns else None,
        "top3_hit_rate": topk_hit_rate(scored, 3) if "rank" in scored.columns else None,
        "top5_hit_rate": topk_hit_rate(scored, 5) if "rank" in scored.columns else None,
        "top1_roi": top1_roi(scored, odds_col=odds_col),
        "ev_top1_roi": ev_top1_roi(scored, odds_col=odds_col),
        "ev_threshold_1_0_roi": threshold_1_0_roi,
        "ev_threshold_1_0_bets": int(threshold_1_0_bets),
        "ev_threshold_1_2_roi": threshold_1_2_roi,
        "ev_threshold_1_2_bets": int(threshold_1_2_bets),
    }


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


def optimize_roi_strategy(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    label_col: str,
    odds_col: str,
    mode: str = "fast",
    progress_interval_sec: float = 5.0,
    max_drawdown: float = 0.35,
    min_final_bankroll: float = 0.85,
    min_bets: int = 50,
) -> tuple[dict[str, float], dict[str, float | int | None]]:
    train_scores = train_df["score"].to_numpy()
    train_labels = train_df[label_col].astype(int).to_numpy()

    train_df = train_df.copy()
    valid_df = valid_df.copy()
    train_df["iso_prob"] = fit_isotonic(train_scores, train_labels, train_scores)
    valid_df["iso_prob"] = fit_isotonic(train_scores, train_labels, valid_df["score"].to_numpy())
    train_df["market_prob"] = compute_market_prob(train_df, odds_col=odds_col)
    valid_df["market_prob"] = compute_market_prob(valid_df, odds_col=odds_col)

    if mode == "full":
        blend_candidates = [0.2, 0.4, 0.6, 0.8]
        edge_candidates = [0.01, 0.03, 0.05]
        min_prob_candidates = [0.03, 0.05]
        kelly_frac_candidates = [0.25, 0.5]
        max_frac_candidates = [0.02, 0.05]
        odds_min_candidates = [1.0]
        odds_max_candidates = [25.0, 40.0, 80.0]
        top_k_candidates = [1, 2]
        min_ev_candidates = [1.0, 1.05, 1.10]
    else:
        blend_candidates = [0.2, 0.4, 0.6]
        edge_candidates = [0.01, 0.03]
        min_prob_candidates = [0.05]
        kelly_frac_candidates = [0.25, 0.5]
        max_frac_candidates = [0.02]
        odds_min_candidates = [1.0]
        odds_max_candidates = [25.0, 40.0]
        top_k_candidates = [1, 2]
        min_ev_candidates = [1.0, 1.05]

    best_params: dict[str, float] = {
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
    best_score = -1e9
    best_metrics: dict[str, float | int | None] = {}
    total_trials = (
        len(blend_candidates) * len(edge_candidates) * len(min_prob_candidates) * len(odds_min_candidates) * len(odds_max_candidates) * len(kelly_frac_candidates) * len(max_frac_candidates)
        + len(blend_candidates) * len(edge_candidates) * len(min_prob_candidates) * len(odds_min_candidates) * len(odds_max_candidates) * len(top_k_candidates) * len(min_ev_candidates)
    )
    completed = 0
    started_at = time.perf_counter()
    last_logged_at = started_at

    log_progress(
        "WF strategy search started: "
        f"mode={mode}, total_trials={total_trials}, "
        f"max_drawdown<={max_drawdown:.3f}, min_final_bankroll>={min_final_bankroll:.3f}, min_bets>={min_bets}"
    )

    def maybe_log_progress(force: bool = False) -> None:
        nonlocal last_logged_at
        now = time.perf_counter()
        if not force and (now - last_logged_at) < max(progress_interval_sec, 0.5):
            return
        elapsed = now - started_at
        ratio = (completed / total_trials) if total_trials > 0 else 1.0
        rate = (completed / elapsed) if elapsed > 0 else 0.0
        remaining = (total_trials - completed)
        eta = (remaining / rate) if rate > 0 else float("inf")
        eta_text = format_duration(eta) if eta != float("inf") else "--"
        log_progress(
            "WF strategy search progress: "
            f"{completed}/{total_trials} ({ratio:.1%}), "
            f"elapsed={format_duration(elapsed)}, eta={eta_text}, best_score={best_score:.4f}"
        )
        last_logged_at = now

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
                                metrics = run_strategy(valid_df, prob_col="blend_prob", odds_col=odds_col, params=params)
                                completed += 1
                                roi = metrics.get("kelly_roi")
                                bets = int(metrics.get("kelly_bets") or 0)
                                hit = float(metrics.get("kelly_hit_rate") or 0.0)
                                drawdown = float(metrics.get("kelly_max_drawdown") or 1.0)
                                final_bankroll = float(metrics.get("kelly_final_bankroll") or 0.0)
                                if roi is None:
                                    maybe_log_progress()
                                    continue
                                if drawdown > max_drawdown or final_bankroll < min_final_bankroll:
                                    maybe_log_progress()
                                    continue
                                score = float(roi) + 0.10 * hit
                                if bets < int(min_bets):
                                    score -= 0.15
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_metrics = dict(metrics)
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
                                metrics = run_strategy(valid_df, prob_col="blend_prob", odds_col=odds_col, params=params)
                                completed += 1
                                roi = metrics.get("portfolio_roi")
                                bets = int(metrics.get("portfolio_bets") or 0)
                                hit = float(metrics.get("portfolio_hit_rate") or 0.0)
                                drawdown = float(metrics.get("portfolio_max_drawdown") or 1.0)
                                final_bankroll = float(metrics.get("portfolio_final_bankroll") or 0.0)
                                if roi is None:
                                    maybe_log_progress()
                                    continue
                                if drawdown > max_drawdown or final_bankroll < min_final_bankroll:
                                    maybe_log_progress()
                                    continue
                                score = float(roi) + 0.20 * hit
                                if bets < int(min_bets):
                                    score -= 0.15
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_metrics = dict(metrics)
                                maybe_log_progress()

    maybe_log_progress(force=True)
    log_progress(
        "WF strategy search finished: "
        f"best_strategy={best_params.get('strategy_kind', 'unknown')}, "
        f"best_score={best_score:.4f}"
    )

    return best_params, best_metrics


def compute_market_prob(frame: pd.DataFrame, odds_col: str) -> pd.Series:
    odds = pd.to_numeric(frame[odds_col], errors="coerce")
    implied = 1.0 / odds.replace(0, np.nan)
    denom = implied.groupby(frame["race_id"]).transform("sum")
    market_prob = implied / denom.replace(0, np.nan)
    return market_prob.fillna(0.0)


def blend_prob(model_prob: pd.Series, market_prob: pd.Series, weight: float) -> pd.Series:
    blended = weight * model_prob + (1.0 - weight) * market_prob
    return blended.clip(1e-6, 1 - 1e-6)


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


def ev_top1_roi_from_prob(
    frame: pd.DataFrame,
    prob_col: str,
    odds_col: str | None,
    stake: float = 100.0,
) -> float | None:
    if odds_col is None:
        return None
    tmp = frame.copy()
    tmp["expected_value"] = pd.to_numeric(tmp[prob_col], errors="coerce") * pd.to_numeric(tmp[odds_col], errors="coerce")
    return ev_top1_roi(tmp, odds_col=odds_col, stake=stake)


def simulate_fractional_kelly(
    frame: pd.DataFrame,
    prob_col: str,
    odds_col: str | None,
    min_edge: float = 0.03,
    min_prob: float = 0.05,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.05,
    odds_min: float = 1.0,
    odds_max: float = 999.0,
    initial_bankroll: float = 1.0,
) -> dict[str, float | int | None]:
    if odds_col is None or "rank" not in frame.columns:
        return {
            "kelly_roi": None,
            "kelly_bets": 0,
            "kelly_hit_rate": None,
            "kelly_final_bankroll": initial_bankroll,
            "kelly_max_drawdown": None,
        }

    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    total_bet = 0.0
    total_return = 0.0
    hits = 0
    bets = 0

    for _, group in frame.groupby("race_id"):
        work = group.copy()
        work["prob"] = pd.to_numeric(work[prob_col], errors="coerce")
        work["odds_num"] = pd.to_numeric(work[odds_col], errors="coerce")
        work = work.dropna(subset=["prob", "odds_num"])
        work = work[(work["odds_num"] > odds_min) & (work["odds_num"] <= odds_max)]
        if work.empty:
            continue

        work["edge"] = work["prob"] * work["odds_num"] - 1.0
        pick = work.sort_values("edge", ascending=False).iloc[0]

        prob = float(pick["prob"])
        odds = float(pick["odds_num"])
        edge = float(pick["edge"])
        if edge < min_edge or prob < min_prob:
            continue

        b = odds - 1.0
        raw_kelly = ((b * prob) - (1.0 - prob)) / b
        if raw_kelly <= 0:
            continue
        frac = min(max(raw_kelly * fractional_kelly, 0.0), max_fraction)
        stake = bankroll * frac
        if stake <= 0:
            continue

        bets += 1
        total_bet += stake

        rank = to_float(pick.get("rank"))
        payout = 0.0
        if rank is not None and int(rank) == 1:
            hits += 1
            payout = stake * odds
        total_return += payout
        bankroll = bankroll - stake + payout
        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, drawdown)

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (hits / bets) if bets > 0 else None
    return {
        "kelly_roi": float(roi) if roi is not None else None,
        "kelly_bets": int(bets),
        "kelly_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "kelly_final_bankroll": float(bankroll),
        "kelly_max_drawdown": float(max_drawdown),
    }


def simulate_ev_portfolio(
    frame: pd.DataFrame,
    prob_col: str,
    odds_col: str | None,
    top_k: int = 2,
    min_prob: float = 0.05,
    min_expected_value: float = 1.0,
    odds_min: float = 1.0,
    odds_max: float = 50.0,
    stake_per_race: float = 1.0,
    initial_bankroll: float = 1.0,
) -> dict[str, float | int | None]:
    if odds_col is None or "rank" not in frame.columns:
        return {
            "portfolio_roi": None,
            "portfolio_bets": 0,
            "portfolio_hit_rate": None,
            "portfolio_avg_synthetic_odds": None,
            "portfolio_final_bankroll": initial_bankroll,
            "portfolio_max_drawdown": None,
        }

    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    total_bet = 0.0
    total_return = 0.0
    race_bets = 0
    race_hits = 0
    synthetic_odds_hits: list[float] = []

    for _, group in frame.groupby("race_id"):
        work = group.copy()
        work["prob"] = pd.to_numeric(work[prob_col], errors="coerce")
        work["odds_num"] = pd.to_numeric(work[odds_col], errors="coerce")
        work = work.dropna(subset=["prob", "odds_num"])
        work = work[(work["odds_num"] > odds_min) & (work["odds_num"] <= odds_max)]
        if work.empty:
            continue

        work["expected_value"] = work["prob"] * work["odds_num"]
        work = work[(work["prob"] >= min_prob) & (work["expected_value"] >= min_expected_value)]
        if work.empty:
            continue

        picks = work.sort_values("expected_value", ascending=False).head(max(top_k, 1))
        if picks.empty:
            continue

        race_bets += 1
        stake_each = stake_per_race / len(picks)
        total_bet += stake_per_race

        payout = 0.0
        for _, pick in picks.iterrows():
            rank = to_float(pick.get("rank"))
            if rank is not None and int(rank) == 1:
                payout += float(stake_each * float(pick["odds_num"]))

        if payout > 0:
            race_hits += 1
            synthetic_odds_hits.append(float(payout / stake_per_race))
        total_return += payout
        bankroll = bankroll - stake_per_race + payout
        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, drawdown)

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (race_hits / race_bets) if race_bets > 0 else None
    avg_synth_odds = float(np.mean(synthetic_odds_hits)) if synthetic_odds_hits else None
    return {
        "portfolio_roi": float(roi) if roi is not None else None,
        "portfolio_bets": int(race_bets),
        "portfolio_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "portfolio_avg_synthetic_odds": avg_synth_odds,
        "portfolio_final_bankroll": float(bankroll),
        "portfolio_max_drawdown": float(max_drawdown),
    }


def run_strategy(frame: pd.DataFrame, prob_col: str, odds_col: str, params: dict[str, float]) -> dict[str, float | int | None]:
    strategy_kind = str(params.get("strategy_kind", "kelly"))
    if strategy_kind == "portfolio":
        return simulate_ev_portfolio(
            frame=frame,
            prob_col=prob_col,
            odds_col=odds_col,
            top_k=int(params.get("top_k", 2)),
            min_prob=float(params.get("min_prob", 0.05)),
            min_expected_value=float(params.get("min_expected_value", 1.0)),
            odds_min=float(params.get("odds_min", 1.0)),
            odds_max=float(params.get("odds_max", 50.0)),
            stake_per_race=1.0,
            initial_bankroll=float(params.get("initial_bankroll", 1.0)),
        )

    return simulate_fractional_kelly(
        frame=frame,
        prob_col=prob_col,
        odds_col=odds_col,
        min_edge=float(params.get("min_edge", 0.03)),
        min_prob=float(params.get("min_prob", 0.05)),
        fractional_kelly=float(params.get("fractional_kelly", 0.5)),
        max_fraction=float(params.get("max_fraction", 0.05)),
        odds_min=float(params.get("odds_min", 1.0)),
        odds_max=float(params.get("odds_max", 50.0)),
        initial_bankroll=float(params.get("initial_bankroll", 1.0)),
    )


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--progress-interval-sec", type=float, default=5.0)
    args = parser.parse_args()

    try:
        model_cfg = load_yaml(ROOT / args.config)
        data_cfg = load_yaml(ROOT / args.data_config)
        feature_cfg = load_yaml(ROOT / args.feature_config)
        task = str(model_cfg.get("task", "classification")).strip().lower()
        evaluation_cfg = model_cfg.get("evaluation", {})
        leakage_cfg = evaluation_cfg.get("leakage_audit", {})
        leakage_enabled = bool(leakage_cfg.get("enabled", True))

        raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
        label_col = model_cfg.get("label", "is_win")
        feature_columns = feature_cfg.get("features", {}).get("base", []) + feature_cfg.get("features", {}).get("history", [])

        log_progress("Loading training table...")
        frame = load_training_table(str(ROOT / raw_dir))
        log_progress("Building features...")
        frame = build_features(frame)
        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()
            print(f"[evaluate] using tail rows: {len(frame)}")

        available_features = [column for column in feature_columns if column in frame.columns]
        if not available_features:
            raise RuntimeError("No features available for evaluation")
        if label_col not in frame.columns:
            raise RuntimeError(f"Missing label column: {label_col}")

        output_cfg = model_cfg.get("output", {})
        model_path = ROOT / output_cfg.get("model_dir", "artifacts/models") / output_cfg.get("model_file", "baseline_model.joblib")
        model = joblib.load(model_path)

        x_eval = frame[available_features]
        y_eval = frame[label_col].astype(int).to_numpy()
        log_progress("Running model inference...")
        y_score = predict_score(model, x_eval, race_ids=frame["race_id"])
        top3_probs = predict_top3_probs(model, x_eval, race_ids=frame["race_id"])

        pred = frame.copy()
        pred["score"] = y_score
        pred = rank_by_score(pred, score_col="score", out_col="pred_rank")

        odds_col = next((column for column in ["odds", "単勝"] if column in pred.columns), None)
        if odds_col is not None:
            pred[odds_col] = pd.to_numeric(pred[odds_col], errors="coerce")
            pred["expected_value"] = pred["score"] * pred[odds_col]
            pred["ev_rank"] = pred.groupby("race_id")["expected_value"].rank(method="first", ascending=False).astype("Int64")

        base_metrics = evaluate_frame(pred, score_col="score", odds_col=odds_col)

        score_is_prob = bool(np.nanmin(y_score) >= 0.0 and np.nanmax(y_score) <= 1.0)
        compute_prob_metrics = task in {"classification", "ranking", "multi_position"}
        probabilistic_flow = bool(compute_prob_metrics and score_is_prob)
        summary = {
            "n_rows": int(len(pred)),
            "n_races": int(pred["race_id"].nunique()),
            "n_dates": int(pred["date"].nunique()) if "date" in pred.columns else None,
            "auc": float(roc_auc_score(y_eval, y_score)) if (compute_prob_metrics and len(np.unique(y_eval)) > 1) else None,
            "logloss": float(log_loss(y_eval, np.clip(y_score, 1e-12, 1 - 1e-12), labels=[0, 1])) if (compute_prob_metrics and score_is_prob) else None,
            "score_is_probability": score_is_prob,
            "task": task,
            "evaluation_flow": ("probability_market" if probabilistic_flow else "roi_direct"),
            **base_metrics,
        }

        summary["run_context"] = {
            "config": str(args.config),
            "data_config": str(args.data_config),
            "feature_config": str(args.feature_config),
            "task": task,
            "label_column": label_col,
            "max_rows": int(args.max_rows),
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "progress_interval_sec": float(args.progress_interval_sec),
            "rows_total_after_tail": int(len(frame)),
        }

        summary["leakage_audit"] = (
            run_leakage_audit(frame=frame, feature_columns=available_features, label_column=label_col)
            if leakage_enabled
            else {"enabled": False}
        )

        if odds_col is not None and probabilistic_flow:
            market_prob = compute_market_prob(pred, odds_col=odds_col).to_numpy(dtype=float)
            model_prob = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)
            market_prob_clip = np.clip(market_prob, 1e-6, 1 - 1e-6)
            edge = model_prob * pd.to_numeric(pred[odds_col], errors="coerce").to_numpy(dtype=float) - 1.0
            delta_logit = np.log(model_prob / (1.0 - model_prob)) - np.log(market_prob_clip / (1.0 - market_prob_clip))

            summary["market_prob_corr"] = float(np.corrcoef(model_prob, market_prob_clip)[0, 1]) if len(model_prob) > 1 else None
            summary["edge_mean"] = float(np.nanmean(edge)) if np.isfinite(np.nanmean(edge)) else None
            summary["delta_logit_mean"] = float(np.nanmean(delta_logit)) if np.isfinite(np.nanmean(delta_logit)) else None

        if top3_probs is not None and "rank" in pred.columns:
            rank_series = pd.to_numeric(pred["rank"], errors="coerce")
            for pos, key in [(1, "p_rank1"), (2, "p_rank2"), (3, "p_rank3")]:
                y_pos = (rank_series == pos).astype(int).to_numpy()
                if len(np.unique(y_pos)) > 1:
                    summary[f"auc_rank{pos}"] = float(roc_auc_score(y_pos, top3_probs[key]))

        log_progress("Starting calibration/evaluation block...")
        calib_train, calib_test = split_for_calibration(pred, date_col="date", train_ratio=0.7)
        if probabilistic_flow and len(calib_train) >= 1000 and len(calib_test) >= 1000:
            train_scores = calib_train["score"].to_numpy()
            train_labels = calib_train[label_col].astype(int).to_numpy()
            test_scores = calib_test["score"].to_numpy()

            platt_scores = fit_platt(train_scores, train_labels, test_scores)
            isotonic_scores = fit_isotonic(train_scores, train_labels, test_scores)

            platt_df = calib_test.copy()
            platt_df["calibrated_score"] = platt_scores
            isotonic_df = calib_test.copy()
            isotonic_df["calibrated_score"] = isotonic_scores

            platt_metrics = evaluate_frame(platt_df, score_col="calibrated_score", odds_col=odds_col)
            isotonic_metrics = evaluate_frame(isotonic_df, score_col="calibrated_score", odds_col=odds_col)

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
                strategy_constraints_cfg = evaluation_cfg.get("strategy_constraints", {})
                wf_max_drawdown = float(strategy_constraints_cfg.get("max_drawdown", 0.45))
                wf_min_final_bankroll = float(strategy_constraints_cfg.get("min_final_bankroll", 0.85))
                wf_min_bets = int(strategy_constraints_cfg.get("min_bets", 50))

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
                calib_test_b["benter_prob"] = blend_prob(
                    calib_test_b["isotonic_prob"],
                    calib_test_b["market_prob"],
                    weight=blend_weight,
                )

                summary["benter_blend_weight"] = float(blend_weight)
                summary["benter_ev_top1_roi"] = ev_top1_roi_from_prob(
                    calib_test_b,
                    prob_col="benter_prob",
                    odds_col=odds_col,
                )
                kelly = simulate_fractional_kelly(
                    calib_test_b,
                    prob_col="benter_prob",
                    odds_col=odds_col,
                    min_edge=0.03,
                    min_prob=0.05,
                    fractional_kelly=0.5,
                    max_fraction=0.05,
                    initial_bankroll=1.0,
                )
                summary["benter_kelly_roi"] = kelly["kelly_roi"]
                summary["benter_kelly_bets"] = kelly["kelly_bets"]
                summary["benter_kelly_hit_rate"] = kelly["kelly_hit_rate"]
                summary["benter_kelly_final_bankroll"] = kelly["kelly_final_bankroll"]
                summary["benter_kelly_max_drawdown"] = kelly.get("kelly_max_drawdown")

                wf_train, wf_valid, wf_test = split_three_way_time(pred, date_col="date", train_ratio=0.5, valid_ratio=0.25)
                summary["wf_train_rows"] = int(len(wf_train))
                summary["wf_valid_rows"] = int(len(wf_valid))
                summary["wf_test_rows"] = int(len(wf_test))
                summary["wf_mode"] = args.wf_mode
                summary["wf_scheme"] = args.wf_scheme
                summary["wf_constraints_max_drawdown"] = wf_max_drawdown
                summary["wf_constraints_min_final_bankroll"] = wf_min_final_bankroll
                summary["wf_constraints_min_bets"] = wf_min_bets
                summary["wf_enabled"] = bool(args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500)
                if args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500:
                    if args.wf_scheme == "single":
                        log_progress("Walk-forward optimization started (single split)...")
                        best_params, best_valid_metrics = optimize_roi_strategy(
                            train_df=wf_train,
                            valid_df=wf_valid,
                            label_col=label_col,
                            odds_col=odds_col,
                            mode=args.wf_mode,
                            progress_interval_sec=float(args.progress_interval_sec),
                            max_drawdown=wf_max_drawdown,
                            min_final_bankroll=wf_min_final_bankroll,
                            min_bets=wf_min_bets,
                        )

                        wf_train_scores = wf_train["score"].to_numpy()
                        wf_train_labels = wf_train[label_col].astype(int).to_numpy()
                        wf_test = wf_test.copy()
                        wf_test["iso_prob"] = fit_isotonic(wf_train_scores, wf_train_labels, wf_test["score"].to_numpy())
                        wf_test["market_prob"] = compute_market_prob(wf_test, odds_col=odds_col)
                        wf_test["blend_prob"] = blend_prob(
                            wf_test["iso_prob"],
                            wf_test["market_prob"],
                            weight=float(best_params["blend_weight"]),
                        )
                        wf_test_metrics = run_strategy(wf_test, prob_col="blend_prob", odds_col=odds_col, params=best_params)
                        log_progress("Walk-forward optimization finished (single split).")

                        summary["wf_strategy_kind"] = str(best_params.get("strategy_kind", "kelly"))
                        summary["wf_best_blend_weight"] = float(best_params["blend_weight"])
                        summary["wf_best_min_prob"] = float(best_params["min_prob"])
                        summary["wf_best_odds_min"] = float(best_params.get("odds_min", 1.0))
                        summary["wf_best_odds_max"] = float(best_params.get("odds_max", 999.0))

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
                        else:
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
                            log_progress(f"Nested WF started: folds={len(nested_slices)}")
                            for fold_index, (fold_train, fold_valid, fold_test) in enumerate(nested_slices, start=1):
                                log_progress(f"Nested WF fold {fold_index}/{len(nested_slices)}: optimizing on inner valid...")
                                best_params, best_valid_metrics = optimize_roi_strategy(
                                    train_df=fold_train,
                                    valid_df=fold_valid,
                                    label_col=label_col,
                                    odds_col=odds_col,
                                    mode=args.wf_mode,
                                    progress_interval_sec=float(args.progress_interval_sec),
                                    max_drawdown=wf_max_drawdown,
                                    min_final_bankroll=wf_min_final_bankroll,
                                    min_bets=wf_min_bets,
                                )

                                fold_train_scores = fold_train["score"].to_numpy()
                                fold_train_labels = fold_train[label_col].astype(int).to_numpy()
                                fold_test = fold_test.copy()
                                fold_test["iso_prob"] = fit_isotonic(fold_train_scores, fold_train_labels, fold_test["score"].to_numpy())
                                fold_test["market_prob"] = compute_market_prob(fold_test, odds_col=odds_col)
                                fold_test["blend_prob"] = blend_prob(
                                    fold_test["iso_prob"],
                                    fold_test["market_prob"],
                                    weight=float(best_params["blend_weight"]),
                                )
                                fold_test_metrics = run_strategy(fold_test, prob_col="blend_prob", odds_col=odds_col, params=best_params)

                                strategy_kind = str(best_params.get("strategy_kind", "kelly"))
                                if strategy_kind == "kelly":
                                    fold_valid_roi = best_valid_metrics.get("kelly_roi")
                                    fold_valid_bets = best_valid_metrics.get("kelly_bets")
                                    fold_valid_hit_rate = best_valid_metrics.get("kelly_hit_rate")
                                    fold_test_roi = fold_test_metrics.get("kelly_roi")
                                    fold_test_bets = fold_test_metrics.get("kelly_bets")
                                    fold_test_hit_rate = fold_test_metrics.get("kelly_hit_rate")
                                    fold_valid_final_bankroll = best_valid_metrics.get("kelly_final_bankroll")
                                    fold_valid_max_drawdown = best_valid_metrics.get("kelly_max_drawdown")
                                    fold_test_final_bankroll = fold_test_metrics.get("kelly_final_bankroll")
                                    fold_test_max_drawdown = fold_test_metrics.get("kelly_max_drawdown")
                                else:
                                    fold_valid_roi = best_valid_metrics.get("portfolio_roi")
                                    fold_valid_bets = best_valid_metrics.get("portfolio_bets")
                                    fold_valid_hit_rate = best_valid_metrics.get("portfolio_hit_rate")
                                    fold_test_roi = fold_test_metrics.get("portfolio_roi")
                                    fold_test_bets = fold_test_metrics.get("portfolio_bets")
                                    fold_test_hit_rate = fold_test_metrics.get("portfolio_hit_rate")
                                    fold_valid_final_bankroll = best_valid_metrics.get("portfolio_final_bankroll")
                                    fold_valid_max_drawdown = best_valid_metrics.get("portfolio_max_drawdown")
                                    fold_test_final_bankroll = fold_test_metrics.get("portfolio_final_bankroll")
                                    fold_test_max_drawdown = fold_test_metrics.get("portfolio_max_drawdown")

                                fold_row: dict[str, float | int | str | None] = {
                                    "fold": int(fold_index),
                                    "strategy_kind": strategy_kind,
                                    "valid_roi": fold_valid_roi,
                                    "valid_bets": fold_valid_bets,
                                    "valid_hit_rate": fold_valid_hit_rate,
                                    "valid_final_bankroll": fold_valid_final_bankroll,
                                    "valid_max_drawdown": fold_valid_max_drawdown,
                                    "test_roi": fold_test_roi,
                                    "test_bets": fold_test_bets,
                                    "test_hit_rate": fold_test_hit_rate,
                                    "test_final_bankroll": fold_test_final_bankroll,
                                    "test_max_drawdown": fold_test_max_drawdown,
                                    "blend_weight": float(best_params.get("blend_weight", 0.0)),
                                }
                                fold_rows.append(fold_row)

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
                        else:
                            summary["wf_nested_completed"] = False
                            summary["wf_nested_reason"] = "insufficient_data_for_nested_folds"
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

        by_date_rows: list[dict] = []
        if "date" in pred.columns:
            date_series = pd.to_datetime(pred["date"], errors="coerce")
            for date, date_df in pred.groupby(date_series.dt.date):
                date_metrics = evaluate_frame(date_df, score_col="score", odds_col=odds_col)
                by_date_rows.append({"date": str(date), **date_metrics})

        by_date = pd.DataFrame(by_date_rows).sort_values("date") if by_date_rows else pd.DataFrame()

        report_dir = ROOT / "artifacts/reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "evaluation_summary.json"
        by_date_path = report_dir / "evaluation_by_date.csv"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        if not by_date.empty:
            by_date.to_csv(by_date_path, index=False)

        print(f"[evaluate] summary saved: {summary_path}")
        if not by_date.empty:
            print(f"[evaluate] by-date saved: {by_date_path}")
            print(by_date.tail(5).to_string(index=False))
        print(f"[evaluate] summary: {json.dumps(summary, ensure_ascii=False)}")
        return 0
    except KeyboardInterrupt:
        print("[evaluate] interrupted by user")
        return 130
    except Exception as error:
        print(f"[evaluate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
