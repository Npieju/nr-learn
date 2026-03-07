from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from racing_ml.common.probability import normalize_position_probabilities


@dataclass(frozen=True)
class PredictionOutputs:
    score: np.ndarray
    top3_probs: dict[str, np.ndarray] | None = None


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


def predict_score(model: object, frame: pd.DataFrame, race_ids: pd.Series | None = None) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(frame), dtype=float)

    probs = predict_top3_probs(model, frame, race_ids=race_ids)
    if probs is not None:
        return probs["p_rank1"]

    raise RuntimeError("Loaded model does not support predict/predict_proba")


def generate_prediction_outputs(
    model: object,
    frame: pd.DataFrame,
    race_ids: pd.Series | None = None,
) -> PredictionOutputs:
    top3_probs = predict_top3_probs(model, frame, race_ids=race_ids)
    if top3_probs is not None:
        return PredictionOutputs(score=top3_probs["p_rank1"], top3_probs=top3_probs)
    return PredictionOutputs(score=predict_score(model, frame, race_ids=race_ids), top3_probs=None)


def rank_by_score(frame: pd.DataFrame, score_col: str, out_col: str = "pred_rank") -> pd.DataFrame:
    ranked = frame.copy()
    ranked[out_col] = (
        ranked.groupby("race_id")[score_col]
        .rank(method="first", ascending=False)
        .astype("Int64")
    )
    return ranked


def topk_hit_rate(frame: pd.DataFrame, k: int, rank_col: str = "pred_rank") -> float | None:
    if "rank" not in frame.columns:
        return None

    hits: list[int] = []
    for _, group in frame.groupby("race_id"):
        picks = group[group[rank_col] <= k]
        rank_values = pd.to_numeric(picks["rank"], errors="coerce")
        hits.append(int((rank_values == 1).any()))
    return float(np.mean(hits)) if hits else None


def resolve_odds_column(frame: pd.DataFrame) -> str | None:
    return next((column for column in ["odds", "単勝"] if column in frame.columns), None)


def prepare_scored_frame(
    frame: pd.DataFrame,
    score: np.ndarray,
    odds_col: str | None,
    score_col: str = "score",
    rank_col: str = "pred_rank",
) -> pd.DataFrame:
    scored = frame.copy()
    scored[score_col] = np.asarray(score, dtype=float)
    scored = rank_by_score(scored, score_col=score_col, out_col=rank_col)

    if odds_col is not None:
        scored[odds_col] = pd.to_numeric(scored[odds_col], errors="coerce")
        scored["expected_value"] = scored[score_col] * scored[odds_col]
        scored["ev_rank"] = (
            scored.groupby("race_id")["expected_value"]
            .rank(method="first", ascending=False)
            .astype("Int64")
        )

    return scored