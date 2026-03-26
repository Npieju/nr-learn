from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, ndcg_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from racing_ml.common.artifacts import dump_joblib_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.features.selection import prepare_model_input_frame


@dataclass
class TrainResult:
    model_path: Path
    report_path: Path
    metrics: dict[str, float]
    used_features: list[str]
    categorical_features: list[str]


REGRESSION_TASKS = {"roi_regression", "market_deviation", "time_regression", "time_deviation"}
TIME_REGRESSION_TASKS = {"time_regression", "time_deviation"}


def _is_gpu_requested(model_name: str, params: dict[str, Any]) -> bool:
    normalized_name = model_name.lower()
    if normalized_name == "lightgbm":
        device_type = str(params.get("device_type", "")).strip().lower()
        return device_type in {"gpu", "cuda"}
    if normalized_name == "catboost":
        task_type = str(params.get("task_type", params.get("device_type", "cpu"))).strip().lower()
        return task_type == "gpu"
    return False


def _validate_lightgbm_gpu_runtime(params: dict[str, Any], task: str) -> None:
    from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

    device_type = str(params.get("device_type", "gpu")).strip().lower() or "gpu"
    test_params = {
        key: value
        for key, value in params.items()
        if key not in {"objective", "odds_clip", "market_prob_floor", "target_clip"}
    }
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
        if task == "ranking":
            model = LGBMRanker(**test_params)
            model.fit(x, y, group=[3, 3])
        elif task in REGRESSION_TASKS:
            y_reg = np.array([0.2, -0.3, 0.1, -0.1, 0.4, -0.2], dtype=np.float32)
            model = LGBMRegressor(**test_params)
            model.fit(x, y_reg)
        else:
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


def _normalize_catboost_params(params: dict[str, Any], task: str) -> dict[str, Any]:
    clean_params = {
        key: value
        for key, value in params.items()
        if key not in {"objective", "device_type", "odds_clip", "market_prob_floor", "target_clip"}
    }

    if "n_estimators" in clean_params and "iterations" not in clean_params:
        clean_params["iterations"] = clean_params.pop("n_estimators")
    task_type = str(params.get("task_type", params.get("device_type", "CPU"))).strip().upper() or "CPU"
    clean_params["task_type"] = task_type
    clean_params.setdefault("iterations", 800)
    clean_params.setdefault("random_seed", 42)
    clean_params.setdefault("allow_writing_files", False)
    clean_params.setdefault("verbose", False)

    if task == "ranking":
        clean_params.setdefault("loss_function", "YetiRankPairwise")
        if str(clean_params.get("task_type", "CPU")).strip().upper() != "GPU":
            clean_params["one_hot_max_size"] = 1
    elif task in REGRESSION_TASKS:
        clean_params.setdefault("loss_function", "RMSE")
    else:
        clean_params.setdefault("loss_function", "Logloss")
        clean_params.setdefault("eval_metric", "AUC")

    return clean_params


def _validate_catboost_gpu_runtime(params: dict[str, Any], task: str) -> None:
    from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor, Pool

    test_params = _normalize_catboost_params(params, task)
    test_params["task_type"] = "GPU"
    test_params["iterations"] = min(int(test_params.get("iterations", 10)), 10)

    frame = pd.DataFrame(
        {
            "num_a": [0.0, 1.0, 0.0, 1.0, 0.2, 0.8],
            "cat_a": ["a", "b", "a", "b", "a", "b"],
        }
    )
    y_cls = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
    y_reg = np.array([0.2, -0.3, 0.1, -0.1, 0.4, -0.2], dtype=np.float32)

    try:
        if task == "ranking":
            model = CatBoostRanker(**test_params)
            pool = Pool(frame, label=y_cls, cat_features=["cat_a"], group_id=[0, 0, 0, 1, 1, 1])
            model.fit(pool, verbose=False)
        elif task in REGRESSION_TASKS:
            model = CatBoostRegressor(**test_params)
            pool = Pool(frame, label=y_reg, cat_features=["cat_a"])
            model.fit(pool, verbose=False)
        else:
            model = CatBoostClassifier(**test_params)
            pool = Pool(frame, label=y_cls, cat_features=["cat_a"])
            model.fit(pool, verbose=False)
    except Exception as error:
        raise RuntimeError(
            "CatBoost GPU validation failed. "
            "Confirm NVIDIA device visibility in the container or switch model.params.task_type to 'CPU'. "
            f"Original error: {error}"
        ) from error


def _build_model(model_name: str, params: dict[str, Any], task: str) -> object:
    normalized_name = model_name.lower()
    if normalized_name == "lightgbm":
        try:
            from lightgbm import LGBMClassifier, LGBMRanker, LGBMRegressor

            clean_params = {
                key: value
                for key, value in params.items()
                if key not in {"objective", "odds_clip", "market_prob_floor", "target_clip"}
            }
            if task == "ranking":
                return LGBMRanker(**clean_params)
            if task in REGRESSION_TASKS:
                return LGBMRegressor(**clean_params)
            return LGBMClassifier(**clean_params)
        except Exception as error:
            raise RuntimeError(f"Failed to initialize LightGBM: {error}") from error

    if normalized_name == "catboost":
        try:
            from catboost import CatBoostClassifier, CatBoostRanker, CatBoostRegressor

            clean_params = _normalize_catboost_params(params, task)
            if task == "ranking":
                return CatBoostRanker(**clean_params)
            if task in REGRESSION_TASKS:
                return CatBoostRegressor(**clean_params)
            return CatBoostClassifier(**clean_params)
        except Exception as error:
            raise RuntimeError(f"Failed to initialize CatBoost: {error}") from error

    raise ValueError(f"Unsupported model name: {model_name}")


def _build_model_with_optional_fallback(model_name: str, params: dict[str, Any], task: str, allow_fallback: bool) -> object:
    try:
        return _build_model(model_name, params, task=task)
    except Exception:
        if not allow_fallback:
            raise

    if task in REGRESSION_TASKS:
        return RandomForestRegressor(
            n_estimators=300,
            max_depth=10,
            n_jobs=-1,
            random_state=42,
        )

    return RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        n_jobs=-1,
        random_state=42,
    )


def _time_split(
    frame: pd.DataFrame,
    train_start: str | None,
    train_end: str,
    valid_start: str,
    valid_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_start_ts = pd.to_datetime(train_start) if train_start else None
    train_end_ts = pd.to_datetime(train_end)
    valid_start_ts = pd.to_datetime(valid_start)
    valid_end_ts = pd.to_datetime(valid_end)

    if train_start_ts is not None and train_start_ts > train_end_ts:
        raise ValueError(f"train_start must be <= train_end: {train_start} .. {train_end}")
    if valid_start_ts > valid_end_ts:
        raise ValueError(f"valid_start must be <= valid_end: {valid_start} .. {valid_end}")

    train_mask = frame["date"] <= train_end_ts
    if train_start_ts is not None:
        train_mask &= frame["date"] >= train_start_ts
    train = frame[train_mask]
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


def _group_sizes(group_ids: pd.Series) -> list[int]:
    return [int(size) for size in group_ids.groupby(group_ids).size().tolist()]


def _mean_ndcg_at_k(y_true: np.ndarray, y_score: np.ndarray, group_ids: pd.Series, k: int) -> float:
    values: list[float] = []
    for group_id in pd.unique(group_ids):
        mask = (group_ids == group_id).to_numpy()
        truth = y_true[mask]
        pred = y_score[mask]
        if len(truth) <= 1:
            continue
        values.append(float(ndcg_score([truth], [pred], k=k)))
    return float(np.mean(values)) if values else float("nan")


def _top1_hit_rate_by_group(y_true: np.ndarray, y_score: np.ndarray, group_ids: pd.Series) -> float:
    hits: list[int] = []
    for group_id in pd.unique(group_ids):
        mask = (group_ids == group_id).to_numpy()
        truth = y_true[mask]
        pred = y_score[mask]
        if len(pred) == 0:
            continue
        top_index = int(np.argmax(pred))
        hits.append(int(truth[top_index] >= 1))
    return float(np.mean(hits)) if hits else float("nan")


def _safe_auc_binary(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_pred))


def _compute_roi_target(frame: pd.DataFrame, odds_clip: float = 30.0) -> np.ndarray:
    if "rank" not in frame.columns or "odds" not in frame.columns:
        raise ValueError("roi_regression task requires 'rank' and 'odds' columns")

    rank = pd.to_numeric(frame["rank"], errors="coerce")
    odds = pd.to_numeric(frame["odds"], errors="coerce")
    odds = odds.clip(lower=0.0, upper=float(odds_clip))
    target = np.where(rank.to_numpy() == 1, odds.to_numpy() - 1.0, -1.0)
    target = np.nan_to_num(target, nan=-1.0, posinf=float(odds_clip - 1.0), neginf=-1.0)
    return target.astype(float)


def _compute_market_deviation_target(
    frame: pd.DataFrame,
    label_column: str,
    odds_clip: float = 30.0,
    market_prob_floor: float = 1e-4,
    target_clip: float = 8.0,
) -> np.ndarray:
    if "odds" not in frame.columns:
        raise ValueError("market_deviation task requires 'odds' column")

    if "rank" in frame.columns:
        win_label = (pd.to_numeric(frame["rank"], errors="coerce") == 1).astype(float)
    elif label_column in frame.columns:
        win_label = pd.to_numeric(frame[label_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        raise ValueError("market_deviation task requires 'rank' or label column")

    odds = pd.to_numeric(frame["odds"], errors="coerce").clip(lower=1.01, upper=float(odds_clip))
    implied = 1.0 / odds.replace(0, np.nan)
    denom = implied.groupby(frame["race_id"]).transform("sum")
    market_prob = (implied / denom.replace(0, np.nan)).clip(lower=float(market_prob_floor), upper=1.0 - float(market_prob_floor))

    eps = float(market_prob_floor)
    observed_prob = win_label.clip(lower=eps, upper=1.0 - eps)
    observed_logit = np.log(observed_prob / (1.0 - observed_prob))
    market_logit = np.log(market_prob / (1.0 - market_prob))
    target = (observed_logit - market_logit).astype(float)
    target = target.clip(lower=-float(target_clip), upper=float(target_clip))
    target = target.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return target.to_numpy(dtype=float)


def _compute_time_regression_target(frame: pd.DataFrame) -> np.ndarray:
    if "finish_time_sec" not in frame.columns:
        raise ValueError("time_regression task requires 'finish_time_sec' column")
    return pd.to_numeric(frame["finish_time_sec"], errors="coerce").to_numpy(dtype=float)


def _compute_time_deviation_target(frame: pd.DataFrame) -> np.ndarray:
    if "time_deviation" not in frame.columns:
        raise ValueError("time_deviation task requires 'time_deviation' column")
    return pd.to_numeric(frame["time_deviation"], errors="coerce").to_numpy(dtype=float)


def _top1_roi_by_group(rank: np.ndarray, odds: np.ndarray, score: np.ndarray, group_ids: pd.Series) -> float:
    total_bet = 0.0
    total_return = 0.0
    for group_id in pd.unique(group_ids):
        mask = (group_ids == group_id).to_numpy()
        if not np.any(mask):
            continue
        rank_group = rank[mask]
        odds_group = odds[mask]
        score_group = score[mask]
        top_index = int(np.argmax(score_group))
        total_bet += 1.0
        if not np.isnan(rank_group[top_index]) and int(rank_group[top_index]) == 1 and not np.isnan(odds_group[top_index]) and odds_group[top_index] > 0:
            total_return += float(odds_group[top_index])
    if total_bet == 0:
        return float("nan")
    return float(total_return / total_bet)


def _safe_regression_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(finite_mask)) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(y_true[finite_mask] - y_pred[finite_mask]))))


def _safe_regression_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(finite_mask)) == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[finite_mask] - y_pred[finite_mask])))


def _safe_regression_corr(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    finite_mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if int(np.sum(finite_mask)) < 2:
        return float("nan")
    return float(np.corrcoef(y_true[finite_mask], y_pred[finite_mask])[0, 1])


def _get_best_iteration(model: object) -> float | None:
    if hasattr(model, "best_iteration_") and getattr(model, "best_iteration_", None) is not None:
        return float(getattr(model, "best_iteration_"))
    if hasattr(model, "get_best_iteration"):
        try:
            best_iteration = model.get_best_iteration()
            if best_iteration is not None and int(best_iteration) >= 0:
                return float(best_iteration)
        except Exception:
            return None
    return None


def _fit_catboost_model(
    model: object,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    y_valid: np.ndarray,
    categorical_columns: list[str],
    early_stopping_rounds: int | None,
    group_train: pd.Series | None = None,
    group_valid: pd.Series | None = None,
) -> None:
    from catboost import Pool

    train_pool = Pool(
        x_train,
        label=y_train,
        cat_features=categorical_columns or None,
        group_id=group_train.astype(str).to_numpy() if group_train is not None else None,
    )
    valid_pool = Pool(
        x_valid,
        label=y_valid,
        cat_features=categorical_columns or None,
        group_id=group_valid.astype(str).to_numpy() if group_valid is not None else None,
    )

    fit_kwargs: dict[str, Any] = {
        "eval_set": valid_pool,
        "use_best_model": bool(early_stopping_rounds),
        "verbose": 200,
    }
    if early_stopping_rounds and int(early_stopping_rounds) > 0:
        fit_kwargs["early_stopping_rounds"] = int(early_stopping_rounds)

    model.fit(train_pool, **fit_kwargs)


def _make_model_bundle(
    *,
    model: object,
    backend: str,
    task: str,
    feature_columns: list[str],
    categorical_columns: list[str],
    preprocessor: object | None = None,
) -> dict[str, Any]:
    return {
        "kind": "tabular_model",
        "backend": backend,
        "task": task,
        "feature_columns": list(feature_columns),
        "categorical_columns": list(categorical_columns),
        "prep": preprocessor,
        "model": model,
    }


def train_and_evaluate(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    task: str,
    model_name: str,
    model_params: dict[str, Any],
    train_start: str | None,
    train_end: str,
    valid_start: str,
    valid_end: str,
    max_train_rows: int | None,
    max_valid_rows: int | None,
    early_stopping_rounds: int | None,
    allow_fallback: bool,
    model_dir: str,
    report_dir: str,
    model_file_name: str,
    report_file_name: str,
    categorical_features: list[str] | None = None,
) -> TrainResult:
    available_features = [column for column in feature_columns if column in frame.columns]
    if not available_features:
        raise ValueError("No configured feature columns found in dataset")

    task = str(task).strip().lower() or "classification"
    if task not in {"classification", "ranking", "multi_position", "roi_regression", "market_deviation", "time_regression", "time_deviation"}:
        raise ValueError(f"Unsupported task: {task}")
    if task not in REGRESSION_TASKS and label_column not in frame.columns:
        raise ValueError(f"Label column '{label_column}' not found")

    fit_progress = ProgressBar(total=5, prefix="[train-fit]", min_interval_sec=0.0)
    fit_progress.start(message=f"model={model_name.lower()} task={task}")

    normalized_model_name = model_name.lower()
    if _is_gpu_requested(model_name, model_params):
        if normalized_model_name == "lightgbm":
            _validate_lightgbm_gpu_runtime(model_params, task=task)
        elif normalized_model_name == "catboost":
            gpu_task = "classification" if task == "multi_position" else task
            _validate_catboost_gpu_runtime(model_params, task=gpu_task)

    train, valid = _time_split(frame, train_start, train_end, valid_start, valid_end)
    if max_train_rows and len(train) > max_train_rows:
        train = train.tail(max_train_rows).copy()
    if max_valid_rows and len(valid) > max_valid_rows:
        valid = valid.tail(max_valid_rows).copy()

    available_categorical_features = [column for column in (categorical_features or []) if column in available_features]

    if task == "roi_regression":
        odds_clip = float(model_params.get("odds_clip", 30.0))
        y_train = _compute_roi_target(train, odds_clip=odds_clip)
        y_valid = _compute_roi_target(valid, odds_clip=odds_clip)
    elif task == "market_deviation":
        y_train = _compute_market_deviation_target(
            train,
            label_column=label_column,
            odds_clip=float(model_params.get("odds_clip", 30.0)),
            market_prob_floor=float(model_params.get("market_prob_floor", 1e-4)),
            target_clip=float(model_params.get("target_clip", 8.0)),
        )
        y_valid = _compute_market_deviation_target(
            valid,
            label_column=label_column,
            odds_clip=float(model_params.get("odds_clip", 30.0)),
            market_prob_floor=float(model_params.get("market_prob_floor", 1e-4)),
            target_clip=float(model_params.get("target_clip", 8.0)),
        )
    elif task == "time_regression":
        y_train = _compute_time_regression_target(train)
        y_valid = _compute_time_regression_target(valid)
    elif task == "time_deviation":
        y_train = _compute_time_deviation_target(train)
        y_valid = _compute_time_deviation_target(valid)
    else:
        y_train = train[label_column].astype(int).to_numpy()
        y_valid = valid[label_column].astype(int).to_numpy()

    if task in REGRESSION_TASKS:
        train_mask = np.isfinite(y_train)
        valid_mask = np.isfinite(y_valid)
        train = train.loc[train_mask].copy()
        valid = valid.loc[valid_mask].copy()
        y_train = y_train[train_mask]
        y_valid = y_valid[valid_mask]
        if train.empty or valid.empty:
            raise ValueError(f"Task '{task}' has no finite target rows after filtering")
    fit_progress.update(message=f"split ready train={len(train):,} valid={len(valid):,} features={len(available_features):,}")

    x_train_native = prepare_model_input_frame(train, available_features, available_categorical_features)
    x_valid_native = prepare_model_input_frame(valid, available_features, available_categorical_features)

    model = _build_model_with_optional_fallback(
        model_name,
        model_params,
        task=("classification" if task == "multi_position" else task),
        allow_fallback=allow_fallback,
    )

    x_train_processed = None
    x_valid_processed = None
    preprocessor = None
    if normalized_model_name != "catboost":
        numeric_columns = [column for column in available_features if column not in available_categorical_features]
        categorical_columns = [column for column in available_features if column in available_categorical_features]
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
        try:
            with Heartbeat("[train-fit]", "preprocessing feature matrix"):
                preprocessor.fit(x_train_native)
                x_train_processed = preprocessor.transform(x_train_native)
                x_valid_processed = preprocessor.transform(x_valid_native)
        except Exception as error:
            raise RuntimeError(f"Feature preprocessing failed: {error}") from error
    fit_progress.update(message="model input prepared")

    if task == "multi_position":
        if "rank" not in train.columns or "rank" not in valid.columns:
            raise ValueError("multi_position task requires 'rank' column")

        position_models: dict[str, Any] = {}
        rank_train = pd.to_numeric(train["rank"], errors="coerce")
        rank_valid = pd.to_numeric(valid["rank"], errors="coerce")
        pred_valid: dict[str, np.ndarray] = {}
        position_progress = ProgressBar(total=3, prefix="[train-fit positions]", min_interval_sec=0.0)
        position_progress.start(message="position models started")

        for position in (1, 2, 3):
            pos_label = f"p_rank{position}"
            y_train_pos = (rank_train == position).astype(int).to_numpy()
            y_valid_pos = (rank_valid == position).astype(int).to_numpy()
            model_pos = _build_model_with_optional_fallback(
                model_name,
                model_params,
                task="classification",
                allow_fallback=allow_fallback,
            )

            if normalized_model_name == "lightgbm":
                from lightgbm import early_stopping, log_evaluation

                callbacks = [log_evaluation(period=200)]
                if early_stopping_rounds and int(early_stopping_rounds) > 0:
                    callbacks.insert(0, early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False))

                with Heartbeat("[train-fit]", f"fitting position model p_rank{position}"):
                    model_pos.fit(
                        x_train_processed,
                        y_train_pos,
                        eval_set=[(x_valid_processed, y_valid_pos)],
                        eval_metric="auc",
                        callbacks=callbacks,
                    )
                pred_valid[pos_label] = model_pos.predict_proba(x_valid_processed)[:, 1]
            elif normalized_model_name == "catboost":
                with Heartbeat("[train-fit]", f"fitting position model p_rank{position}"):
                    _fit_catboost_model(
                        model=model_pos,
                        x_train=x_train_native,
                        y_train=y_train_pos,
                        x_valid=x_valid_native,
                        y_valid=y_valid_pos,
                        categorical_columns=available_categorical_features,
                        early_stopping_rounds=early_stopping_rounds,
                    )
                pred_valid[pos_label] = np.asarray(model_pos.predict_proba(x_valid_native)[:, 1], dtype=float)
            else:
                with Heartbeat("[train-fit]", f"fitting position model p_rank{position}"):
                    model_pos.fit(x_train_processed, y_train_pos)
                pred_valid[pos_label] = model_pos.predict_proba(x_valid_processed)[:, 1]

            position_models[pos_label] = model_pos
            position_progress.update(message=f"p_rank{position} ready")
        position_progress.complete(message="all position models ready")

        metrics = {
            "auc_rank1": _safe_auc_binary((rank_valid == 1).astype(int).to_numpy(), pred_valid["p_rank1"]),
            "auc_rank2": _safe_auc_binary((rank_valid == 2).astype(int).to_numpy(), pred_valid["p_rank2"]),
            "auc_rank3": _safe_auc_binary((rank_valid == 3).astype(int).to_numpy(), pred_valid["p_rank3"]),
            "logloss_rank1": _safe_logloss((rank_valid == 1).astype(int).to_numpy(), pred_valid["p_rank1"]),
            "logloss_rank2": _safe_logloss((rank_valid == 2).astype(int).to_numpy(), pred_valid["p_rank2"]),
            "logloss_rank3": _safe_logloss((rank_valid == 3).astype(int).to_numpy(), pred_valid["p_rank3"]),
            "valid_samples": float(len(rank_valid)),
            "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
            "avg_top3_mass": float(np.mean(pred_valid["p_rank1"] + pred_valid["p_rank2"] + pred_valid["p_rank3"])),
        }
        top1_truth = (rank_valid == 1).astype(int).to_numpy()
        metrics["top1_hit_rate"] = _top1_hit_rate_by_group(top1_truth, pred_valid["p_rank1"], valid["race_id"].astype(str))

        best_iterations = [
            best_iteration
            for best_iteration in (_get_best_iteration(model_obj) for model_obj in position_models.values())
            if best_iteration is not None
        ]
        if best_iterations:
            metrics["best_iteration_mean"] = float(np.mean(best_iterations))

        trained_model: Any = {
            "kind": "multi_position_top3",
            "backend": normalized_model_name,
            "prep": preprocessor,
            "models": position_models,
            "feature_columns": available_features,
            "categorical_columns": available_categorical_features,
        }
    elif normalized_model_name == "lightgbm":
        from lightgbm import early_stopping, log_evaluation

        callbacks = [log_evaluation(period=200)]
        if early_stopping_rounds and int(early_stopping_rounds) > 0:
            callbacks.insert(0, early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False))

        try:
            with Heartbeat("[train-fit]", f"fitting {normalized_model_name} {task} model"):
                if task == "ranking":
                    train_groups = _group_sizes(train["race_id"])
                    valid_groups = _group_sizes(valid["race_id"])
                    model.fit(
                        x_train_processed,
                        y_train,
                        group=train_groups,
                        eval_set=[(x_valid_processed, y_valid)],
                        eval_group=[valid_groups],
                        eval_metric="ndcg",
                        callbacks=callbacks,
                    )
                elif task in REGRESSION_TASKS:
                    model.fit(
                        x_train_processed,
                        y_train,
                        eval_set=[(x_valid_processed, y_valid)],
                        eval_metric="l2",
                        callbacks=callbacks,
                    )
                else:
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

        trained_model = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    elif normalized_model_name == "catboost":
        try:
            with Heartbeat("[train-fit]", f"fitting {normalized_model_name} {task} model"):
                _fit_catboost_model(
                    model=model,
                    x_train=x_train_native,
                    y_train=y_train,
                    x_valid=x_valid_native,
                    y_valid=y_valid,
                    categorical_columns=available_categorical_features,
                    early_stopping_rounds=early_stopping_rounds,
                    group_train=train["race_id"] if task == "ranking" else None,
                    group_valid=valid["race_id"] if task == "ranking" else None,
                )
        except Exception as error:
            raise RuntimeError(
                "CatBoost training failed. Check task_type, categorical feature handling, and runtime availability. "
                f"Original error: {error}"
            ) from error

        trained_model = _make_model_bundle(
            model=model,
            backend=normalized_model_name,
            task=task,
            feature_columns=available_features,
            categorical_columns=available_categorical_features,
        )
    else:
        try:
            with Heartbeat("[train-fit]", f"fitting {normalized_model_name} {task} model"):
                model.fit(x_train_processed, y_train)
        except Exception as error:
            raise RuntimeError(
                "Model training failed. "
                f"Original error: {error}"
            ) from error

        trained_model = Pipeline(steps=[("prep", preprocessor), ("model", model)])
    fit_progress.update(message="model fit complete")

    if task == "multi_position":
        pass
    elif task == "ranking":
        try:
            if normalized_model_name == "catboost":
                valid_score = np.asarray(model.predict(x_valid_native), dtype=float).reshape(-1)
            else:
                valid_score = np.asarray(model.predict(x_valid_processed), dtype=float).reshape(-1)
        except Exception as error:
            raise RuntimeError(f"Validation ranking prediction failed: {error}") from error

        valid_groups = valid["race_id"].astype(str)
        metrics = {
            "ndcg_at_1": _mean_ndcg_at_k(y_valid, valid_score, valid_groups, k=1),
            "ndcg_at_3": _mean_ndcg_at_k(y_valid, valid_score, valid_groups, k=3),
            "ndcg_at_5": _mean_ndcg_at_k(y_valid, valid_score, valid_groups, k=5),
            "top1_hit_rate": _top1_hit_rate_by_group(y_valid, valid_score, valid_groups),
            "valid_samples": float(len(y_valid)),
            "positive_rate": float(np.mean(y_valid)),
            "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
        }
    elif task == "classification":
        try:
            if normalized_model_name == "catboost":
                valid_proba = np.asarray(model.predict_proba(x_valid_native)[:, 1], dtype=float)
            else:
                valid_proba = np.asarray(model.predict_proba(x_valid_processed)[:, 1], dtype=float)
        except Exception as error:
            raise RuntimeError(f"Validation prediction failed: {error}") from error

        metrics = {
            "auc": _safe_auc(y_valid, valid_proba),
            "logloss": _safe_logloss(y_valid, valid_proba),
            "valid_samples": float(len(y_valid)),
            "positive_rate": float(np.mean(y_valid)),
            "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
        }
    elif task == "roi_regression":
        try:
            if normalized_model_name == "catboost":
                valid_score = np.asarray(model.predict(x_valid_native), dtype=float).reshape(-1)
            else:
                valid_score = np.asarray(model.predict(x_valid_processed), dtype=float).reshape(-1)
        except Exception as error:
            raise RuntimeError(f"Validation ROI prediction failed: {error}") from error

        rank_valid = pd.to_numeric(valid["rank"], errors="coerce").to_numpy(dtype=float)
        odds_valid = pd.to_numeric(valid["odds"], errors="coerce").to_numpy(dtype=float)
        group_ids = valid["race_id"].astype(str)
        top1_truth = (rank_valid == 1).astype(int)
        metrics = {
            "roi_target_mean": float(np.mean(y_valid)),
            "pred_mean": float(np.mean(valid_score)),
            "top1_hit_rate": _top1_hit_rate_by_group(top1_truth, valid_score, group_ids),
            "top1_roi": _top1_roi_by_group(rank_valid, odds_valid, valid_score, group_ids),
            "valid_samples": float(len(y_valid)),
            "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
        }
    elif task == "market_deviation":
        try:
            if normalized_model_name == "catboost":
                valid_score = np.asarray(model.predict(x_valid_native), dtype=float).reshape(-1)
            else:
                valid_score = np.asarray(model.predict(x_valid_processed), dtype=float).reshape(-1)
        except Exception as error:
            raise RuntimeError(f"Validation market deviation prediction failed: {error}") from error

        rank_valid = pd.to_numeric(valid["rank"], errors="coerce").to_numpy(dtype=float) if "rank" in valid.columns else np.full(len(valid), np.nan)
        odds_valid = pd.to_numeric(valid["odds"], errors="coerce").to_numpy(dtype=float) if "odds" in valid.columns else np.full(len(valid), np.nan)
        group_ids = valid["race_id"].astype(str)
        top1_truth = (rank_valid == 1).astype(int)
        finite_mask = np.isfinite(y_valid) & np.isfinite(valid_score)
        corr = float(np.corrcoef(y_valid[finite_mask], valid_score[finite_mask])[0, 1]) if int(np.sum(finite_mask)) >= 2 else float("nan")

        metrics = {
            "alpha_target_mean": float(np.mean(y_valid)),
            "pred_mean": float(np.mean(valid_score)),
            "alpha_target_std": float(np.std(y_valid)),
            "pred_std": float(np.std(valid_score)),
            "alpha_pred_corr": corr,
            "positive_signal_rate": float(np.mean(valid_score > 0.0)),
            "top1_hit_rate": _top1_hit_rate_by_group(top1_truth, valid_score, group_ids),
            "top1_roi": _top1_roi_by_group(rank_valid, odds_valid, valid_score, group_ids),
            "valid_samples": float(len(y_valid)),
            "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
        }
    elif task in TIME_REGRESSION_TASKS:
        try:
            if normalized_model_name == "catboost":
                valid_pred = np.asarray(model.predict(x_valid_native), dtype=float).reshape(-1)
            else:
                valid_pred = np.asarray(model.predict(x_valid_processed), dtype=float).reshape(-1)
        except Exception as error:
            raise RuntimeError(f"Validation time prediction failed: {error}") from error

        rank_valid = pd.to_numeric(valid["rank"], errors="coerce").to_numpy(dtype=float) if "rank" in valid.columns else np.full(len(valid), np.nan)
        odds_valid = pd.to_numeric(valid["odds"], errors="coerce").to_numpy(dtype=float) if "odds" in valid.columns else np.full(len(valid), np.nan)
        group_ids = valid["race_id"].astype(str)
        ranking_score = -valid_pred
        top1_truth = (rank_valid == 1).astype(int)

        rmse_key = "time_rmse_sec" if task == "time_regression" else "time_deviation_rmse"
        mae_key = "time_mae_sec" if task == "time_regression" else "time_deviation_mae"
        corr_key = "time_pred_corr" if task == "time_regression" else "time_deviation_pred_corr"
        target_mean_key = "finish_time_target_mean" if task == "time_regression" else "time_deviation_target_mean"
        pred_mean_key = "finish_time_pred_mean" if task == "time_regression" else "time_deviation_pred_mean"

        metrics = {
            target_mean_key: float(np.mean(y_valid)),
            pred_mean_key: float(np.mean(valid_pred)),
            rmse_key: _safe_regression_rmse(y_valid, valid_pred),
            mae_key: _safe_regression_mae(y_valid, valid_pred),
            corr_key: _safe_regression_corr(y_valid, valid_pred),
            "top1_hit_rate": _top1_hit_rate_by_group(top1_truth, ranking_score, group_ids),
            "top1_roi": _top1_roi_by_group(rank_valid, odds_valid, ranking_score, group_ids),
            "valid_samples": float(len(y_valid)),
            "gpu_enabled": float(_is_gpu_requested(model_name, model_params)),
        }
    else:
        metrics = {}
    fit_progress.update(message="validation metrics ready")

    if task in {"classification", "ranking", "roi_regression", "market_deviation", "time_regression", "time_deviation"}:
        best_iteration = _get_best_iteration(model)
        if best_iteration is not None:
            metrics["best_iteration"] = best_iteration

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    report_path = Path(report_dir)
    report_path.mkdir(parents=True, exist_ok=True)

    model_file = model_path / model_file_name
    report_file = report_path / report_file_name
    with Heartbeat("[train-fit]", "writing model/report files"):
        dump_joblib_file(model_file, trained_model, label="model output")
        write_json(report_file, metrics)
    fit_progress.complete(message="model/report files written")

    return TrainResult(
        model_path=model_file,
        report_path=report_file,
        metrics=metrics,
        used_features=available_features,
        categorical_features=available_categorical_features,
    )