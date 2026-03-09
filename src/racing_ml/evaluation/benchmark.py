from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


PROB_FLOOR = 1e-9


@dataclass(frozen=True)
class BenterCombiner:
    alpha: float
    beta: float
    train_race_logloss: float | None = None


def _winner_frame(
    probabilities: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    race_ids: np.ndarray | pd.Series,
) -> pd.DataFrame | None:
    frame = pd.DataFrame(
        {
            "probability": np.asarray(probabilities, dtype=float),
            "label": np.asarray(labels, dtype=int),
            "race_id": np.asarray(race_ids),
        }
    )
    if frame.empty:
        return None

    winner_counts = frame.groupby("race_id", sort=False)["label"].sum()
    if winner_counts.empty:
        return None

    valid_race_ids = winner_counts[winner_counts == 1].index
    if len(valid_race_ids) == 0:
        return None

    frame = frame[frame["race_id"].isin(valid_race_ids)].copy()
    winner_counts = frame.groupby("race_id", sort=False)["label"].sum()

    winners = frame[frame["label"] == 1].copy()
    winners = winners.drop_duplicates(subset=["race_id"], keep="first")
    winners = winners.set_index("race_id").reindex(winner_counts.index)
    if winners["probability"].isna().any():
        return None

    winners["field_size"] = frame.groupby("race_id", sort=False).size().reindex(winner_counts.index).to_numpy(dtype=int)
    return winners.reset_index()


def supports_winner_only_benchmark(labels: np.ndarray | pd.Series, race_ids: np.ndarray | pd.Series) -> bool:
    return _winner_frame(np.ones(len(np.asarray(labels))), labels, race_ids) is not None


def count_winner_only_races(labels: np.ndarray | pd.Series, race_ids: np.ndarray | pd.Series) -> int:
    frame = pd.DataFrame(
        {
            "label": np.asarray(labels, dtype=int),
            "race_id": np.asarray(race_ids),
        }
    )
    if frame.empty:
        return 0
    winner_counts = frame.groupby("race_id", sort=False)["label"].sum()
    return int((winner_counts == 1).sum())


def normalize_within_race(values: np.ndarray | pd.Series, race_ids: np.ndarray | pd.Series) -> np.ndarray:
    value_series = pd.Series(np.asarray(values, dtype=float))
    race_series = pd.Series(np.asarray(race_ids))

    denom = value_series.groupby(race_series, sort=False).transform("sum").to_numpy(dtype=float)
    denom = np.where(denom > 0.0, denom, 1.0)
    normalized = value_series.to_numpy(dtype=float) / denom
    normalized = np.clip(normalized, PROB_FLOOR, None)

    renorm = pd.Series(normalized).groupby(race_series, sort=False).transform("sum").to_numpy(dtype=float)
    renorm = np.where(renorm > 0.0, renorm, 1.0)
    return normalized / renorm


def combine_benter_prob(
    model_prob: np.ndarray | pd.Series,
    market_prob: np.ndarray | pd.Series,
    race_ids: np.ndarray | pd.Series,
    *,
    alpha: float,
    beta: float,
) -> np.ndarray:
    model = np.clip(np.asarray(model_prob, dtype=float), PROB_FLOOR, 1.0)
    market = np.clip(np.asarray(market_prob, dtype=float), PROB_FLOOR, 1.0)
    combined_strength = np.power(model, float(alpha)) * np.power(market, float(beta))
    return normalize_within_race(combined_strength, race_ids)


def race_winner_logloss(
    probabilities: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    race_ids: np.ndarray | pd.Series,
) -> float | None:
    winners = _winner_frame(probabilities, labels, race_ids)
    if winners is None:
        return None

    win_prob = np.clip(winners["probability"].to_numpy(dtype=float), PROB_FLOOR, 1.0)
    return float((-np.log(win_prob)).mean())


def pseudo_r2(
    probabilities: np.ndarray | pd.Series,
    labels: np.ndarray | pd.Series,
    race_ids: np.ndarray | pd.Series,
) -> float | None:
    winners = _winner_frame(probabilities, labels, race_ids)
    if winners is None:
        return None

    win_prob = np.clip(winners["probability"].to_numpy(dtype=float), PROB_FLOOR, 1.0)
    field_sizes = winners["field_size"].to_numpy(dtype=float)
    random_nll = float(np.log(field_sizes).sum())
    if random_nll <= 0.0:
        return None

    model_nll = float((-np.log(win_prob)).sum())
    return float(1.0 - (model_nll / random_nll))


def fit_benter_combiner(
    frame: pd.DataFrame,
    *,
    label_col: str,
    model_prob_col: str,
    market_prob_col: str,
) -> BenterCombiner | None:
    required_columns = {"race_id", label_col, model_prob_col, market_prob_col}
    if not required_columns.issubset(frame.columns):
        return None

    labels = frame[label_col].astype(int).to_numpy()
    race_ids = frame["race_id"].to_numpy()
    if not supports_winner_only_benchmark(labels, race_ids):
        return None

    model_prob = frame[model_prob_col].to_numpy(dtype=float)
    market_prob = frame[market_prob_col].to_numpy(dtype=float)

    def evaluate(alpha: float, beta: float) -> float:
        combined = combine_benter_prob(model_prob, market_prob, race_ids, alpha=alpha, beta=beta)
        winners = _winner_frame(combined, labels, race_ids)
        if winners is None:
            return float("inf")
        win_prob = np.clip(winners["probability"].to_numpy(dtype=float), PROB_FLOOR, 1.0)
        return float((-np.log(win_prob)).sum())

    def search(alpha_candidates: np.ndarray, beta_candidates: np.ndarray, seed_alpha: float, seed_beta: float) -> tuple[float, float, float]:
        best_alpha = float(seed_alpha)
        best_beta = float(seed_beta)
        best_loss = evaluate(best_alpha, best_beta)

        for alpha in alpha_candidates:
            for beta in beta_candidates:
                loss = evaluate(float(alpha), float(beta))
                if loss < best_loss:
                    best_alpha = float(alpha)
                    best_beta = float(beta)
                    best_loss = loss
        return best_alpha, best_beta, best_loss

    coarse = np.round(np.arange(0.0, 2.01, 0.2), 4)
    best_alpha, best_beta, best_loss = search(coarse, coarse, seed_alpha=1.0, seed_beta=1.0)

    fine_alpha = np.round(np.arange(max(0.0, best_alpha - 0.2), best_alpha + 0.201, 0.05), 4)
    fine_beta = np.round(np.arange(max(0.0, best_beta - 0.2), best_beta + 0.201, 0.05), 4)
    best_alpha, best_beta, best_loss = search(np.unique(fine_alpha), np.unique(fine_beta), seed_alpha=best_alpha, seed_beta=best_beta)

    winners = _winner_frame(
        combine_benter_prob(model_prob, market_prob, race_ids, alpha=best_alpha, beta=best_beta),
        labels,
        race_ids,
    )
    train_race_logloss = None
    if winners is not None:
        train_race_logloss = float((-np.log(np.clip(winners["probability"].to_numpy(dtype=float), PROB_FLOOR, 1.0))).mean())

    return BenterCombiner(alpha=best_alpha, beta=best_beta, train_race_logloss=train_race_logloss)