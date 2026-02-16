from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


@dataclass
class TrainResult:
    model_path: Path
    report_path: Path
    metrics: dict[str, float]
    used_features: list[str]


def _is_gpu_requested(model_name: str, params: dict) -> bool:
    if model_name.lower() != "lightgbm":
        return False

    device_type = str(params.get("device_type", "")).strip().lower()
    return device_type in {"gpu", "cuda"}


def _validate_lightgbm_gpu_runtime(params: dict) -> None:
    from lightgbm import LGBMClassifier
    device_type = str(params.get("device_type", "gpu")).strip().lower() or "gpu"

    test_params = {k: v for k, v in params.items() if k != "objective"}
    test_params["n_estimators"] = min(int(test_params.get("n_estimators", 10)), 10)
    test_params.setdefault("num_leaves", 8)
    test_params.setdefault("min_data_in_leaf", 1)
    test_params.setdefault("verbosity", -1)

    x = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.2, 0.8],
            [0.8, 0.2],
        ],
        dtype=np.float32,
    )
    y = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    try:
        model = LGBMClassifier(**test_params)
        model.fit(x, y)
    except Exception as error:
        if device_type == "cuda":
            raise RuntimeError(
                "LightGBM CUDA validation failed. "
                "Current LightGBM wheel is likely not built with CUDA support. "
                "Rebuild/install LightGBM with CUDA enabled (-DUSE_CUDA=1), "
                "or switch model.params.device_type to 'gpu' (OpenCL) / CPU. "
                f"Original error: {error}"
            ) from error

        raise RuntimeError(
            "LightGBM GPU validation failed. "
            "Set container GPU allocation (e.g. --gpus all), "
            "install OpenCL runtime in the container (e.g. ocl-icd-libopencl1), "
            "confirm OpenCL device visibility, "
            "or remove model.params.device_type='gpu' to run on CPU. "
            f"Original error: {error}"
        ) from error


def _build_model(model_name: str, params: dict) -> object:
    if model_name.lower() == "lightgbm":
        try:
            from lightgbm import LGBMClassifier

            clean_params = {k: v for k, v in params.items() if k != "objective"}
            return LGBMClassifier(**clean_params)
        except Exception as error:
            raise RuntimeError(f"Failed to initialize LightGBM: {error}") from error

    raise ValueError(f"Unsupported model name: {model_name}")


def _build_model_with_optional_fallback(model_name: str, params: dict, allow_fallback: bool) -> object:
    try:
        return _build_model(model_name, params)
    except Exception:
        if not allow_fallback:
            raise

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )


def _time_split(
    frame: pd.DataFrame,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_end_ts = pd.to_datetime(train_end)
    valid_start_ts = pd.to_datetime(valid_start)
    valid_end_ts = pd.to_datetime(valid_end)

    train = frame[frame["date"] <= train_end_ts]
    valid = frame[(frame["date"] >= valid_start_ts) & (frame["date"] <= valid_end_ts)]

    if train.empty or valid.empty:
        unique_dates = frame["date"].dropna().sort_values().unique()
        if len(unique_dates) < 3:
            raise ValueError("Time split is empty. Dataset has too few dated samples.")

        cutoff_index = max(int(len(unique_dates) * 0.8), 1)
        cutoff_date = pd.to_datetime(unique_dates[cutoff_index - 1])
        train = frame[frame["date"] <= cutoff_date]
        valid = frame[frame["date"] > cutoff_date]

        if train.empty or valid.empty:
            raise ValueError("Time split is empty after automatic fallback split.")

    return train, valid


def _safe_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred_proba))


def _safe_logloss(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    eps = 1e-12
    y_pred_proba = np.clip(y_pred_proba, eps, 1 - eps)
    return float(log_loss(y_true, y_pred_proba, labels=[0, 1]))


def train_and_evaluate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    model_name: str,
    model_params: dict,
    train_end: str,
    valid_start: str,
    valid_end: str,
    max_train_rows: int | None,
    max_valid_rows: int | None,
    early_stopping_rounds: int | None,
    allow_fallback: bool,
    model_dir: str,
    report_dir: str,
) -> TrainResult:
    available_features = [column for column in feature_columns if column in frame.columns]
    if not available_features:
        raise ValueError("No configured feature columns found in dataset")

    if label_column not in frame.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    if _is_gpu_requested(model_name, model_params):
        _validate_lightgbm_gpu_runtime(model_params)

    train, valid = _time_split(frame, train_end, valid_start, valid_end)

    if max_train_rows and len(train) > max_train_rows:
        train = train.tail(max_train_rows).copy()
    if max_valid_rows and len(valid) > max_valid_rows:
        valid = valid.tail(max_valid_rows).copy()

    x_train = train[available_features].copy()
    y_train = train[label_column].astype(int).to_numpy()
    x_valid = valid[available_features].copy()
    y_valid = valid[label_column].astype(int).to_numpy()

    numeric_columns = [column for column in available_features if pd.api.types.is_numeric_dtype(x_train[column])]
    categorical_columns = [column for column in available_features if column not in numeric_columns]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_columns,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("encoder", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )

    model = _build_model_with_optional_fallback(model_name, model_params, allow_fallback=allow_fallback)

    try:
        preprocessor.fit(x_train)
        x_train_processed = preprocessor.transform(x_train)
        x_valid_processed = preprocessor.transform(x_valid)
    except Exception as error:
        raise RuntimeError(f"Feature preprocessing failed: {error}") from error

    if model_name.lower() == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        callbacks = [log_evaluation(period=200)]
        if early_stopping_rounds and early_stopping_rounds > 0:
            callbacks.insert(0, early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False))

        try:
            model.fit(
                x_train_processed,
                y_train,
                eval_set=[(x_valid_processed, y_valid)],
                eval_metric="auc",
                callbacks=callbacks,
            )
        except Exception as error:
            raise RuntimeError(
                "LightGBM training failed. If using GPU, verify CUDA/OpenCL runtime and container GPU access. "
                f"Original error: {error}"
            ) from error
    else:
        try:
            model.fit(x_train_processed, y_train)
        except Exception as error:
            raise RuntimeError(
                "Model training failed. "
                f"Original error: {error}"
            ) from error

    pipeline = Pipeline(steps=[("prep", preprocessor), ("model", model)])

    try:
        valid_proba = model.predict_proba(x_valid_processed)[:, 1]
    except Exception as error:
        raise RuntimeError(f"Validation prediction failed: {error}") from error

    metrics = {
        "auc": _safe_auc(y_valid, valid_proba),
        "logloss": _safe_logloss(y_valid, valid_proba),
        "valid_samples": float(len(y_valid)),
        "positive_rate": float(np.mean(y_valid)),
        "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
    }

    if hasattr(model, "best_iteration_") and getattr(model, "best_iteration_", None) is not None:
        metrics["best_iteration"] = float(getattr(model, "best_iteration_"))

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    model_file = model_path / "baseline_model.joblib"
    report_file = report_path / "train_metrics.json"

    joblib.dump(pipeline, model_file)
    with report_file.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    return TrainResult(
        model_path=model_file,
        report_path=report_file,
        metrics=metrics,
        used_features=available_features,
    )
