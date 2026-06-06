from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _resolve_glob(pattern: str, *, workspace_root: Path) -> list[Path]:
    paths = sorted(workspace_root.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No score calibration files matched: {pattern}")
    return paths


class _SafeFormatDict(dict[str, Any]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _format_config_string(value: str, context: dict[str, Any] | None) -> str:
    if not context:
        return value
    return value.format_map(_SafeFormatDict(context))


def _format_context(calibration_config: dict[str, Any], format_context: dict[str, Any] | None) -> dict[str, Any]:
    context = dict(format_context or {})
    configured_artifact_suffix = calibration_config.get("artifact_suffix")
    if configured_artifact_suffix is not None and str(configured_artifact_suffix).strip():
        context["artifact_suffix"] = str(configured_artifact_suffix).strip()
    return context


def _read_prediction_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = path.as_posix()
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _fit_calibrator(method: str, train_score: np.ndarray, train_label: np.ndarray) -> Any:
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(train_score, train_label)
        return model
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(train_score.reshape(-1, 1), train_label)
        return model
    raise ValueError(f"Unsupported score calibration method: {method}")


def _predict_calibrated(method: str, model: Any, score: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return np.asarray(model.transform(score), dtype=float)
    if method == "platt":
        return np.asarray(model.predict_proba(score.reshape(-1, 1))[:, 1], dtype=float)
    raise ValueError(f"Unsupported score calibration method: {method}")


def _apply_exposure_guards(
    frame: pd.DataFrame,
    calibrated: np.ndarray,
    *,
    base_col: str,
    top_popularity_max: int | None,
    non_top_max_lift: float,
    shrinkage: float,
) -> np.ndarray:
    base = _numeric(frame, base_col).to_numpy(dtype=float)
    raw = np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1.0 - 1e-6)
    blended = ((1.0 - shrinkage) * base) + (shrinkage * raw)

    popularity = _numeric(frame, "popularity")
    if top_popularity_max is not None:
        top_mask = (popularity <= top_popularity_max).fillna(False).to_numpy(dtype=bool)
        guarded = base.copy()
        guarded[top_mask] = blended[top_mask]
        lift_cap = base * (1.0 + max(non_top_max_lift, 0.0))
        guarded[~top_mask] = np.minimum(blended[~top_mask], lift_cap[~top_mask])
        return np.clip(guarded, 1e-6, 1.0 - 1e-6)

    return np.clip(blended, 1e-6, 1.0 - 1e-6)


def _apply_race_closure(
    frame: pd.DataFrame,
    score: np.ndarray,
    *,
    base_col: str,
    mode: str,
    softmax_temperature: float,
) -> np.ndarray:
    if mode == "none":
        return np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    if "race_id" not in frame.columns:
        raise ValueError("score calibration race closure requires race_id column")

    working = frame[["race_id"]].copy()
    working["score"] = np.clip(np.asarray(score, dtype=float), 1e-6, 1.0 - 1e-6)
    if mode == "softmax":
        temperature = float(softmax_temperature)
        if not np.isfinite(temperature) or temperature <= 0.0:
            raise ValueError("score calibration softmax temperature must be positive")
        logits = np.log(working["score"].to_numpy(dtype=float)) / temperature
        max_logit = pd.Series(logits, index=working.index).groupby(working["race_id"], sort=False).transform("max")
        stabilized = np.exp(logits - max_logit.to_numpy(dtype=float))
        denom = pd.Series(stabilized, index=working.index).groupby(working["race_id"], sort=False).transform("sum")
        adjusted = stabilized / denom.replace(0.0, np.nan).to_numpy(dtype=float)
        return pd.Series(adjusted, index=working.index).fillna(working["score"]).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)
    if mode == "normalize_score_sum":
        target_sum = pd.Series(1.0, index=working.index)
    elif mode == "preserve_base_score_sum":
        base = _numeric(frame, base_col).clip(1e-6, 1.0 - 1e-6)
        target_sum = base.groupby(frame["race_id"], sort=False).transform("sum")
    else:
        raise ValueError(f"Unsupported score calibration race closure mode: {mode}")

    current_sum = working.groupby("race_id", sort=False)["score"].transform("sum")
    adjusted = working["score"] * (target_sum / current_sum.replace(0.0, np.nan))
    return adjusted.fillna(working["score"]).clip(1e-6, 1.0 - 1e-6).to_numpy(dtype=float)


def apply_score_calibration(
    frame: pd.DataFrame,
    calibration_config: dict[str, Any] | None,
    *,
    workspace_root: Path,
    score_col: str = "score",
    label_col: str = "rank",
    format_context: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any] | None]:
    if not isinstance(calibration_config, dict) or not calibration_config.get("enabled", False):
        return frame, None

    effective_format_context = _format_context(calibration_config, format_context)
    train_glob_template = str(calibration_config.get("train_glob", "")).strip()
    train_glob = _format_config_string(train_glob_template, effective_format_context).strip()
    if not train_glob:
        raise ValueError("serving.score_calibration.enabled requires train_glob")
    calibration_label_col = str(calibration_config.get("label_col", label_col)).strip() or label_col

    method = str(calibration_config.get("method", "isotonic")).strip().lower()
    top_popularity_max_raw = calibration_config.get("top_popularity_max", 3)
    top_popularity_max = int(top_popularity_max_raw) if top_popularity_max_raw is not None else None
    non_top_max_lift = float(calibration_config.get("non_top_max_lift", 0.0))
    shrinkage = float(calibration_config.get("shrinkage", 1.0))
    race_closure_mode = str(calibration_config.get("race_closure_mode", "none")).strip().lower() or "none"
    race_softmax_temperature = float(calibration_config.get("race_softmax_temperature", 1.0))
    min_calibration_rows = int(calibration_config.get("min_calibration_rows", 100))

    train_paths = _resolve_glob(train_glob, workspace_root=workspace_root)
    train = _read_prediction_files(train_paths)
    if score_col not in train.columns:
        raise ValueError(f"score calibration train files missing score column: {score_col}")
    if calibration_label_col not in train.columns:
        raise ValueError(f"score calibration train files missing label column: {calibration_label_col}")

    train_score = _numeric(train, score_col)
    if calibration_label_col == "rank":
        train_label = (_numeric(train, calibration_label_col) == 1).astype(int)
    else:
        train_label = _numeric(train, calibration_label_col).astype(int)
    valid = train_score.notna() & train_label.notna()
    if int(valid.sum()) < min_calibration_rows:
        raise ValueError(f"Not enough score calibration rows: {int(valid.sum())}")

    calibrator = _fit_calibrator(method, train_score.loc[valid].to_numpy(dtype=float), train_label.loc[valid].to_numpy(dtype=int))
    calibrated = _predict_calibrated(method, calibrator, _numeric(frame, score_col).to_numpy(dtype=float))

    result = frame.copy()
    result["score_calibration_method"] = method
    result["score_calibration_train_rows"] = int(valid.sum())
    result["score_calibrated_raw"] = calibrated
    result["score_before_calibration"] = _numeric(result, score_col)
    result["score_calibrated_before_race_closure"] = _apply_exposure_guards(
        result,
        calibrated,
        base_col=score_col,
        top_popularity_max=top_popularity_max,
        non_top_max_lift=non_top_max_lift,
        shrinkage=shrinkage,
    )
    result[score_col] = _apply_race_closure(
        result,
        result["score_calibrated_before_race_closure"].to_numpy(dtype=float),
        base_col="score_before_calibration",
        mode=race_closure_mode,
        softmax_temperature=race_softmax_temperature,
    )

    summary = {
        "method": method,
        "train_glob": train_glob,
        "train_glob_template": train_glob_template,
        "artifact_suffix": effective_format_context.get("artifact_suffix"),
        "train_files": [path.as_posix() for path in train_paths],
        "calibration_rows": int(valid.sum()),
        "label_col": calibration_label_col,
        "top_popularity_max": top_popularity_max,
        "non_top_max_lift": non_top_max_lift,
        "shrinkage": shrinkage,
        "race_closure_mode": race_closure_mode,
        "race_softmax_temperature": race_softmax_temperature,
    }
    return result, summary
