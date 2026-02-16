import argparse
import json
from pathlib import Path
import sys
import traceback
from typing import Any

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
from racing_ml.features.builder import build_features


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
                                roi = metrics.get("kelly_roi")
                                bets = int(metrics.get("kelly_bets") or 0)
                                hit = float(metrics.get("kelly_hit_rate") or 0.0)
                                if roi is None:
                                    continue
                                score = float(roi) + 0.10 * hit
                                if bets < 50:
                                    score -= 0.15
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_metrics = dict(metrics)

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
                                roi = metrics.get("portfolio_roi")
                                bets = int(metrics.get("portfolio_bets") or 0)
                                hit = float(metrics.get("portfolio_hit_rate") or 0.0)
                                if roi is None:
                                    continue
                                score = float(roi) + 0.20 * hit
                                if bets < 50:
                                    score -= 0.15
                                if score > best_score:
                                    best_score = score
                                    best_params = params
                                    best_metrics = dict(metrics)

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
        }

    bankroll = float(initial_bankroll)
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

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (hits / bets) if bets > 0 else None
    return {
        "kelly_roi": float(roi) if roi is not None else None,
        "kelly_bets": int(bets),
        "kelly_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "kelly_final_bankroll": float(bankroll),
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
) -> dict[str, float | int | None]:
    if odds_col is None or "rank" not in frame.columns:
        return {
            "portfolio_roi": None,
            "portfolio_bets": 0,
            "portfolio_hit_rate": None,
            "portfolio_avg_synthetic_odds": None,
        }

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

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (race_hits / race_bets) if race_bets > 0 else None
    avg_synth_odds = float(np.mean(synthetic_odds_hits)) if synthetic_odds_hits else None
    return {
        "portfolio_roi": float(roi) if roi is not None else None,
        "portfolio_bets": int(race_bets),
        "portfolio_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "portfolio_avg_synthetic_odds": avg_synth_odds,
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
        initial_bankroll=1.0,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="fast")
    args = parser.parse_args()

    try:
        model_cfg = load_yaml(ROOT / args.config)
        data_cfg = load_yaml(ROOT / args.data_config)
        feature_cfg = load_yaml(ROOT / args.feature_config)

        raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
        label_col = model_cfg.get("label", "is_win")
        feature_columns = feature_cfg.get("features", {}).get("base", []) + feature_cfg.get("features", {}).get("history", [])

        frame = load_training_table(str(ROOT / raw_dir))
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
        summary = {
            "n_rows": int(len(pred)),
            "n_races": int(pred["race_id"].nunique()),
            "n_dates": int(pred["date"].nunique()) if "date" in pred.columns else None,
            "auc": float(roc_auc_score(y_eval, y_score)) if len(np.unique(y_eval)) > 1 else None,
            "logloss": float(log_loss(y_eval, np.clip(y_score, 1e-12, 1 - 1e-12), labels=[0, 1])) if score_is_prob else None,
            "score_is_probability": score_is_prob,
            **base_metrics,
        }

        if top3_probs is not None and "rank" in pred.columns:
            rank_series = pd.to_numeric(pred["rank"], errors="coerce")
            for pos, key in [(1, "p_rank1"), (2, "p_rank2"), (3, "p_rank3")]:
                y_pos = (rank_series == pos).astype(int).to_numpy()
                if len(np.unique(y_pos)) > 1:
                    summary[f"auc_rank{pos}"] = float(roc_auc_score(y_pos, top3_probs[key]))

        calib_train, calib_test = split_for_calibration(pred, date_col="date", train_ratio=0.7)
        if len(calib_train) >= 1000 and len(calib_test) >= 1000:
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

                wf_train, wf_valid, wf_test = split_three_way_time(pred, date_col="date", train_ratio=0.5, valid_ratio=0.25)
                summary["wf_train_rows"] = int(len(wf_train))
                summary["wf_valid_rows"] = int(len(wf_valid))
                summary["wf_test_rows"] = int(len(wf_test))
                summary["wf_mode"] = args.wf_mode
                summary["wf_enabled"] = bool(args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500)
                if args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500:
                    best_params, best_valid_metrics = optimize_roi_strategy(
                        train_df=wf_train,
                        valid_df=wf_valid,
                        label_col=label_col,
                        odds_col=odds_col,
                        mode=args.wf_mode,
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
