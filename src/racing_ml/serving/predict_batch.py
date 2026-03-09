from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from racing_ml.common.artifacts import resolve_output_artifacts
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.scoring import generate_prediction_outputs, predict_target_values
from racing_ml.features.builder import build_features
from racing_ml.features.selection import prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[predict {now}] {message}", flush=True)


def _predict_score(model: object, frame: pd.DataFrame, race_ids: pd.Series | None) -> np.ndarray:
    return generate_prediction_outputs(model, frame, race_ids=race_ids).score


def _predict_multi_position(model_bundle: dict[str, Any], frame: pd.DataFrame, race_ids: pd.Series | None) -> dict[str, np.ndarray]:
    outputs = generate_prediction_outputs(model_bundle, frame, race_ids=race_ids)
    if outputs.top3_probs is None:
        raise RuntimeError("Invalid multi_position model bundle")
    return outputs.top3_probs


def _resolve_target_date(frame: pd.DataFrame, race_date: str | None) -> pd.Timestamp:
    available_dates = frame["date"].dropna().sort_values().unique()
    if len(available_dates) == 0:
        raise ValueError("No dated rows found in dataset")

    if race_date is None:
        return pd.to_datetime(available_dates[-1])

    target = pd.to_datetime(race_date)
    candidates = [date for date in available_dates if pd.to_datetime(date) <= target]
    if candidates:
        return pd.to_datetime(candidates[-1])
    return pd.to_datetime(available_dates[0])


def _plot_predictions(predictions: pd.DataFrame, out_path: Path) -> None:
    if predictions.empty:
        return

    race_counts = predictions["race_id"].value_counts()
    target_race_id = race_counts.index[0]
    race_frame = predictions[predictions["race_id"] == target_race_id].copy()
    race_frame = race_frame.sort_values("score", ascending=False).head(12)

    labels = race_frame["horse_id"].astype(str)
    scores = race_frame["score"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(labels.iloc[::-1], scores.iloc[::-1], color="#3b82f6")
    axes[0].set_title(f"Top Scores: race_id={target_race_id}")
    axes[0].set_xlabel("Predicted win score")

    axes[1].hist(predictions["score"].astype(float), bins=20, color="#10b981", alpha=0.85)
    axes[1].set_title("Score Distribution")
    axes[1].set_xlabel("Predicted win score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_predict(
    model_config_path: str,
    data_config_path: str,
    feature_config_path: str,
    race_date: str | None = None,
) -> None:
    model_config = load_yaml(Path(model_config_path))
    data_config = load_yaml(Path(data_config_path))
    feature_config = load_yaml(Path(feature_config_path))
    progress = ProgressBar(total=6, prefix="[predict]", logger=log_progress, min_interval_sec=0.0)
    progress.start("configs loaded")

    dataset_cfg = data_config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    with Heartbeat("[predict]", "loading training table", logger=log_progress):
        frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=Path.cwd())
    progress.update(message=f"training table loaded rows={len(frame):,}")

    with Heartbeat("[predict]", "building features", logger=log_progress):
        frame = build_features(frame)
    progress.update(message=f"features built columns={len(frame.columns):,}")

    target_date = _resolve_target_date(frame, race_date)
    pred_frame = frame[frame["date"] == target_date].copy()
    if pred_frame.empty:
        raise ValueError(f"No races found for target date: {target_date}")
    progress.update(
        message=f"target date resolved date={pd.Timestamp(target_date).date()} races={pred_frame['race_id'].nunique():,}"
    )

    with Heartbeat("[predict]", "loading model and preparing inputs", logger=log_progress):
        output_artifacts = resolve_output_artifacts(model_config.get("output", {}))
        workspace_root = Path.cwd()
        model_path = output_artifacts.model_path if output_artifacts.model_path.is_absolute() else (workspace_root / output_artifacts.model_path)
        manifest_path = output_artifacts.manifest_path if output_artifacts.manifest_path.is_absolute() else (workspace_root / output_artifacts.manifest_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        fallback_selection = resolve_feature_selection(pred_frame, feature_config, label_column=model_config.get("label", "is_win"))
        feature_selection = resolve_model_feature_selection(model, fallback_selection)
        model_input = prepare_model_input_frame(pred_frame, feature_selection.feature_columns, feature_selection.categorical_columns)
        model_task = str(model.get("task", "")) if isinstance(model, dict) else ""
    progress.update(
        message=(
            f"model ready features={len(feature_selection.feature_columns):,} "
            f"categorical={len(feature_selection.categorical_columns):,}"
        )
    )

    with Heartbeat("[predict]", "running inference", logger=log_progress):
        if isinstance(model, dict) and model.get("kind") == "multi_position_top3":
            outputs = _predict_multi_position(model, model_input, pred_frame.get("race_id"))
            pred_frame["p_rank1_raw"] = outputs["p_rank1"]
            pred_frame["p_rank2_raw"] = outputs["p_rank2"]
            pred_frame["p_rank3_raw"] = outputs["p_rank3"]
            pred_frame = pred_frame.rename(
                columns={
                    "p_rank1_raw": "p_rank1",
                    "p_rank2_raw": "p_rank2",
                    "p_rank3_raw": "p_rank3",
                }
            )
            pred_frame["p_top3"] = (pred_frame["p_rank1"] + pred_frame["p_rank2"] + pred_frame["p_rank3"]).clip(0.0, 1.0)
            pred_frame["score"] = pred_frame["p_rank1"]
        else:
            pred_frame["score"] = _predict_score(model, model_input, pred_frame.get("race_id"))
            if model_task == "time_regression":
                pred_frame["pred_finish_time_sec"] = predict_target_values(model, model_input)
            elif model_task == "time_deviation":
                pred_frame["pred_time_deviation"] = predict_target_values(model, model_input)
    pred_frame["pred_rank"] = (
        pred_frame.groupby("race_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    progress.update(message=f"inference complete rows={len(pred_frame):,}")

    if "odds" in pred_frame.columns:
        pred_frame["odds"] = pd.to_numeric(pred_frame["odds"], errors="coerce")
        pred_frame["expected_value"] = pred_frame["score"] * pred_frame["odds"]
        ev_rank = pred_frame.groupby("race_id")["expected_value"].rank(method="first", ascending=False)
        pred_frame["ev_rank"] = ev_rank.astype("Int64")

    columns = ["date", "race_id", "horse_id"]
    if "horse_name" in pred_frame.columns:
        columns.append("horse_name")
    if "rank" in pred_frame.columns:
        columns.append("rank")
    if "odds" in pred_frame.columns:
        columns.append("odds")
    columns += ["score", "pred_rank"]
    for extra_col in ["p_rank1", "p_rank2", "p_rank3", "p_top3"]:
        if extra_col in pred_frame.columns:
            columns.append(extra_col)
    for extra_col in ["pred_finish_time_sec", "pred_time_deviation"]:
        if extra_col in pred_frame.columns:
            columns.append(extra_col)
    if "expected_value" in pred_frame.columns:
        columns += ["expected_value", "ev_rank"]

    out_dir = Path("artifacts/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    date_tag = pd.Timestamp(target_date).strftime("%Y%m%d")
    csv_path = out_dir / f"predictions_{date_tag}.csv"
    png_path = out_dir / f"predictions_{date_tag}.png"

    with Heartbeat("[predict]", "writing prediction outputs", logger=log_progress):
        output = pred_frame[columns].sort_values(["race_id", "pred_rank"]).reset_index(drop=True)
        output.to_csv(csv_path, index=False)
        _plot_predictions(output, png_path)
    progress.complete(message="prediction outputs written")

    print(f"[predict] target date: {pd.Timestamp(target_date).date()}")
    print(f"[predict] predictions saved: {csv_path}")
    print(f"[predict] chart saved: {png_path}")
    if manifest_path.exists():
        print(f"[predict] manifest: {manifest_path}")
    print(f"[predict] records: {len(output)}")
