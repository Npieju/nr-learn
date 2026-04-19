from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from racing_ml.common.probability import normalize_position_probabilities
from racing_ml.evaluation.policy import blend_prob, compute_market_prob
from racing_ml.features.selection import prepare_model_input_frame


@dataclass(frozen=True)
class PredictionOutputs:
    score: np.ndarray
    top3_probs: dict[str, np.ndarray] | None = None


TIME_REGRESSION_TASKS = {"time_regression", "time_deviation"}


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def _logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def compose_value_blend_probabilities(
    *,
    win_prob: np.ndarray,
    params: dict[str, Any] | None = None,
    alpha_raw: np.ndarray | None = None,
    roi_raw: np.ndarray | None = None,
    time_raw: np.ndarray | None = None,
    market_prob: np.ndarray | None = None,
) -> np.ndarray:
    params = params or {}
    win_prob = np.clip(np.asarray(win_prob, dtype=float).reshape(-1), 1e-6, 1.0 - 1e-6)
    probability_path_mode = str(params.get("probability_path_mode", "legacy_blend")).strip().lower() or "legacy_blend"
    win_logit = _logit(win_prob)
    combined_logit = win_logit

    alpha_signal: np.ndarray | None = None
    if alpha_raw is not None:
        alpha_scale = max(float(params.get("alpha_scale", 2.0)), 1e-6)
        alpha_signal = np.tanh(np.asarray(alpha_raw, dtype=float).reshape(-1) / alpha_scale)
        if bool(params.get("alpha_positive_only", False)):
            alpha_signal = np.maximum(alpha_signal, 0.0)

    if probability_path_mode == "market_aware_alpha_branch" and market_prob is not None:
        market_logit = _logit(np.asarray(market_prob, dtype=float).reshape(-1))
        market_weight = float(params.get("market_blend_weight", 1.0))
        alpha_branch = market_logit
        if alpha_signal is not None:
            alpha_branch = alpha_branch + float(params.get("alpha_weight", 0.0)) * alpha_signal
        combined_logit = ((1.0 - market_weight) * win_logit) + (market_weight * alpha_branch)

    elif alpha_signal is not None:
        combined_logit = combined_logit + float(params.get("alpha_weight", 0.0)) * alpha_signal

    if roi_raw is not None:
        roi_scale = max(float(params.get("roi_scale", 2.0)), 1e-6)
        roi_signal = np.tanh(np.asarray(roi_raw, dtype=float).reshape(-1) / roi_scale)
        if bool(params.get("roi_positive_only", False)):
            roi_signal = np.maximum(roi_signal, 0.0)
        combined_logit = combined_logit + float(params.get("roi_weight", 0.0)) * roi_signal

    if time_raw is not None:
        time_scale = max(float(params.get("time_scale", 3.0)), 1e-6)
        time_signal = np.tanh((-np.asarray(time_raw, dtype=float).reshape(-1)) / time_scale)
        if bool(params.get("time_positive_only", True)):
            time_signal = np.maximum(time_signal, 0.0)
        combined_logit = combined_logit + float(params.get("time_weight", 0.0)) * time_signal

    blended_prob = _sigmoid(combined_logit)

    if market_prob is not None and probability_path_mode != "market_aware_alpha_branch":
        blended_prob = blend_prob(
            pd.Series(blended_prob),
            pd.Series(np.asarray(market_prob, dtype=float).reshape(-1)),
            weight=float(params.get("market_blend_weight", 1.0)),
        ).to_numpy(dtype=float)

    return np.clip(np.asarray(blended_prob, dtype=float), 1e-6, 1.0 - 1e-6)


def _resolve_bundle_input(model_bundle: dict[str, Any], frame: pd.DataFrame) -> Any:
    feature_columns = [str(column) for column in model_bundle.get("feature_columns", []) if str(column).strip()]
    categorical_columns = [str(column) for column in model_bundle.get("categorical_columns", []) if str(column).strip()]
    prepared = prepare_model_input_frame(frame, feature_columns, categorical_columns) if feature_columns else frame
    prep = model_bundle.get("prep")
    return prep.transform(prepared) if prep is not None else prepared


def _predict_binary_scores(model: object, frame: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(frame)[:, 1], dtype=float)
    if hasattr(model, "predict"):
        return np.asarray(model.predict(frame), dtype=float).reshape(-1)
    raise RuntimeError("Loaded model does not support predict/predict_proba")


def _normalize_lower_better_predictions(values: np.ndarray, race_ids: pd.Series | np.ndarray) -> np.ndarray:
    race_series = pd.Series(np.asarray(race_ids))
    value_series = pd.Series(np.asarray(values, dtype=float))
    centered = value_series - value_series.groupby(race_series, sort=False).transform("min")
    centered = centered.clip(lower=0.0, upper=50.0)
    strength = np.exp(-centered.to_numpy(dtype=float))
    denom = pd.Series(strength).groupby(race_series, sort=False).transform("sum").to_numpy(dtype=float)
    denom = np.where(denom > 0.0, denom, 1.0)
    return strength / denom


def _predict_value_blend_scores(
    model_bundle: dict[str, Any],
    frame: pd.DataFrame,
    race_ids: pd.Series | None = None,
) -> np.ndarray:
    components = model_bundle.get("components", {})
    if not isinstance(components, dict):
        raise RuntimeError("Invalid value blend model bundle")

    win_model = components.get("win")
    if win_model is None:
        raise RuntimeError("Value blend model requires a win component")

    params = model_bundle.get("params", {})
    win_prob = np.asarray(predict_score(win_model, frame, race_ids=race_ids), dtype=float).reshape(-1)
    alpha_raw: np.ndarray | None = None
    roi_raw: np.ndarray | None = None
    time_raw: np.ndarray | None = None

    alpha_model = components.get("alpha")
    if alpha_model is not None:
        alpha_raw = np.asarray(predict_score(alpha_model, frame, race_ids=race_ids), dtype=float).reshape(-1)

    roi_model = components.get("roi")
    if roi_model is not None:
        roi_raw = np.asarray(predict_score(roi_model, frame, race_ids=race_ids), dtype=float).reshape(-1)

    time_model = components.get("time")
    if time_model is not None:
        time_raw = np.asarray(predict_target_values(time_model, frame), dtype=float).reshape(-1)

    market_prob: np.ndarray | None = None
    market_blend_weight = float(params.get("market_blend_weight", 1.0))
    odds_column = str(params.get("odds_column", "odds"))
    if market_blend_weight < 0.999 and odds_column in frame.columns:
        if race_ids is None and "race_id" not in frame.columns:
            raise RuntimeError("race_ids are required for value blend market anchoring")
        race_values = frame["race_id"].to_numpy(copy=False) if "race_id" in frame.columns else pd.Series(race_ids).to_numpy(copy=False)
        market_frame = pd.DataFrame(
            {
                "race_id": race_values,
                odds_column: frame[odds_column].to_numpy(copy=False),
            }
        )
        market_prob = compute_market_prob(market_frame, odds_col=odds_column).to_numpy(dtype=float)

    return compose_value_blend_probabilities(
        win_prob=win_prob,
        params=params,
        alpha_raw=alpha_raw,
        roi_raw=roi_raw,
        time_raw=time_raw,
        market_prob=market_prob,
    )


def predict_top3_probs(
    model: Any,
    frame: pd.DataFrame,
    race_ids: pd.Series | None = None,
) -> dict[str, np.ndarray] | None:
    if not (isinstance(model, dict) and model.get("kind") == "multi_position_top3"):
        return None

    models = model.get("models", {})
    if not isinstance(models, dict):
        return None

    transformed = _resolve_bundle_input(model, frame)
    output: dict[str, np.ndarray] = {}
    for key in ["p_rank1", "p_rank2", "p_rank3"]:
        model_obj = models.get(key)
        if model_obj is None:
            return None
        output[key] = _predict_binary_scores(model_obj, transformed)

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


def predict_target_values(model: object, frame: pd.DataFrame) -> np.ndarray:
    if isinstance(model, dict) and model.get("kind") == "tabular_model":
        estimator = model.get("model")
        if estimator is None:
            raise RuntimeError("Invalid tabular model bundle")
        transformed = _resolve_bundle_input(model, frame)
        return _predict_binary_scores(estimator, transformed)

    return predict_score(model, frame)


def predict_score(model: object, frame: pd.DataFrame, race_ids: pd.Series | None = None) -> np.ndarray:
    probs = predict_top3_probs(model, frame, race_ids=race_ids)
    if probs is not None:
        return probs["p_rank1"]

    if isinstance(model, dict) and model.get("kind") == "value_blend_model":
        return _predict_value_blend_scores(model, frame, race_ids=race_ids)

    if isinstance(model, dict) and model.get("kind") == "tabular_model":
        raw_prediction = predict_target_values(model, frame)
        task = str(model.get("task", "")).strip().lower()
        if task in TIME_REGRESSION_TASKS:
            if race_ids is None:
                if "race_id" not in frame.columns:
                    return -np.asarray(raw_prediction, dtype=float)
                race_ids = frame["race_id"]
            return _normalize_lower_better_predictions(np.asarray(raw_prediction, dtype=float), race_ids)
        return np.asarray(raw_prediction, dtype=float)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(frame)[:, 1]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(frame), dtype=float)

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