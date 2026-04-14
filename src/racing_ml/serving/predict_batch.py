from __future__ import annotations

from pathlib import Path
import time
from typing import Any

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from racing_ml.common.artifacts import append_suffix_to_file_name, resolve_output_artifacts, save_figure, write_csv_file, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.common.regime import resolve_regime_override
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.scoring import generate_prediction_outputs, predict_target_values, prepare_scored_frame, resolve_odds_column
from racing_ml.features.builder import build_features
from racing_ml.features.selection import prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection
from racing_ml.serving.runtime_policy import annotate_runtime_policy, resolve_runtime_policy, summarize_policy_diagnostics


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


def _resolve_workspace_path(path_value: str | Path, workspace_root: Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return workspace_root / path


def _display_path(path: Path, workspace_root: Path) -> str:
    try:
        return path.relative_to(workspace_root).as_posix()
    except ValueError:
        return path.as_posix()


def _resolve_chart_labels(race_frame: pd.DataFrame) -> pd.Series:
    horse_ids = race_frame["horse_id"].astype(str) if "horse_id" in race_frame.columns else race_frame.index.astype(str)
    if "horse_name" not in race_frame.columns:
        return pd.Series(horse_ids, index=race_frame.index)

    horse_names = race_frame["horse_name"].fillna("").astype(str)
    non_empty_names = horse_names[horse_names.str.len() > 0]
    if not non_empty_names.empty and non_empty_names.map(str.isascii).all():
        return horse_names.where(horse_names.str.len() > 0, horse_ids)
    return pd.Series(horse_ids, index=race_frame.index)


def _resolve_prediction_source(
    model_config: dict[str, Any],
    *,
    model_config_path: Path,
    pred_frame: pd.DataFrame,
    workspace_root: Path,
) -> tuple[str, Path, dict[str, Any]]:
    serving_cfg = model_config.get("serving", {})
    evaluation_cfg = model_config.get("evaluation", {})
    score_regime_overrides = []
    if isinstance(serving_cfg, dict):
        score_regime_overrides = serving_cfg.get("score_regime_overrides", [])
    if not isinstance(score_regime_overrides, list) or not score_regime_overrides:
        score_regime_overrides = evaluation_cfg.get("score_regime_overrides", [])
    override = resolve_regime_override(score_regime_overrides, frame=pred_frame)
    if not isinstance(override, dict):
        return "default", model_config_path, model_config

    override_model_config = str(override.get("model_config", "")).strip()
    if not override_model_config:
        return "default", model_config_path, model_config

    override_name = str(override.get("name", "")).strip() or "default"
    override_config_path = _resolve_workspace_path(override_model_config, workspace_root)
    override_model_config_data = load_yaml(override_config_path)
    return override_name, override_config_path, override_model_config_data


def _plot_predictions(predictions: pd.DataFrame, out_path: Path) -> None:
    if predictions.empty:
        return

    race_counts = predictions["race_id"].value_counts()
    target_race_id = race_counts.index[0]
    race_frame = predictions[predictions["race_id"] == target_race_id].copy()
    race_frame = race_frame.sort_values("score", ascending=False).head(12)

    labels = _resolve_chart_labels(race_frame)
    scores = race_frame["score"].astype(float)
    y_positions = np.arange(len(race_frame), dtype=float)
    display_labels = labels.iloc[::-1].tolist()
    display_scores = scores.iloc[::-1].to_numpy(dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(y_positions, display_scores, color="#3b82f6")
    axes[0].set_yticks(y_positions)
    axes[0].set_yticklabels(display_labels)
    axes[0].set_title(f"Top Scores: race_id={target_race_id}")
    axes[0].set_xlabel("Predicted win score")

    axes[1].hist(predictions["score"].astype(float), bins=20, color="#10b981", alpha=0.85)
    axes[1].set_title("Score Distribution")
    axes[1].set_xlabel("Predicted win score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    save_figure(out_path, fig, label="chart output", dpi=140)
    plt.close(fig)


def prepare_prediction_frame(data_config_path: str | Path) -> pd.DataFrame:
    workspace_root = Path.cwd()
    resolved_data_config_path = _resolve_workspace_path(data_config_path, workspace_root)
    data_config = load_yaml(resolved_data_config_path)

    dataset_cfg = data_config.get("dataset", {})
    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    with Heartbeat("[predict]", "loading training table", logger=log_progress):
        frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=Path.cwd())

    with Heartbeat("[predict]", "building features", logger=log_progress):
        frame = build_features(frame)
    return frame


def run_predict_from_frame(
    model_config_path: str | Path,
    feature_config_path: str | Path,
    frame: pd.DataFrame,
    race_date: str | None = None,
    profile_name: str | None = None,
    model_artifact_suffix: str | None = None,
    output_file_suffix: str | None = None,
) -> dict[str, Any]:
    workspace_root = Path.cwd()
    resolved_model_config_path = _resolve_workspace_path(model_config_path, workspace_root)
    resolved_feature_config_path = _resolve_workspace_path(feature_config_path, workspace_root)
    model_config = load_yaml(resolved_model_config_path)
    feature_config = load_yaml(resolved_feature_config_path)
    progress = ProgressBar(total=5, prefix="[predict]", logger=log_progress, min_interval_sec=0.0)
    progress.start(f"configs loaded frame_rows={len(frame):,} frame_columns={len(frame.columns):,}")

    target_date = _resolve_target_date(frame, race_date)
    pred_frame = frame[frame["date"] == target_date].copy()
    if pred_frame.empty:
        raise ValueError(f"No races found for target date: {target_date}")
    progress.update(
        message=f"target date resolved date={pd.Timestamp(target_date).date()} races={pred_frame['race_id'].nunique():,}"
    )

    with Heartbeat("[predict]", "loading model and preparing inputs", logger=log_progress):
        score_source_name, active_model_config_path, active_model_config = _resolve_prediction_source(
            model_config,
            model_config_path=resolved_model_config_path,
            pred_frame=pred_frame,
            workspace_root=workspace_root,
        )
        output_cfg = dict(active_model_config.get("output", {}))
        if model_artifact_suffix:
            output_cfg["model_file"] = append_suffix_to_file_name(
                str(output_cfg.get("model_file", "baseline_model.joblib")),
                model_artifact_suffix,
            )
            output_cfg["manifest_file"] = append_suffix_to_file_name(
                str(output_cfg.get("manifest_file", output_cfg.get("model_file", "baseline_model.joblib"))),
                model_artifact_suffix,
            )
        output_artifacts = resolve_output_artifacts(output_cfg)
        model_path = output_artifacts.model_path if output_artifacts.model_path.is_absolute() else (workspace_root / output_artifacts.model_path)
        manifest_path = output_artifacts.manifest_path if output_artifacts.manifest_path.is_absolute() else (workspace_root / output_artifacts.manifest_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        fallback_selection = resolve_feature_selection(
            pred_frame,
            feature_config,
            label_column=active_model_config.get("label", model_config.get("label", "is_win")),
        )
        feature_selection = resolve_model_feature_selection(model, fallback_selection)
        model_input = prepare_model_input_frame(pred_frame, feature_selection.feature_columns, feature_selection.categorical_columns)
        model_task = str(model.get("task", "")) if isinstance(model, dict) else ""
    progress.update(
        message=(
            f"model ready score_source={score_source_name} features={len(feature_selection.feature_columns):,} "
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
    odds_col = resolve_odds_column(pred_frame)
    pred_frame = prepare_scored_frame(pred_frame, pred_frame["score"].to_numpy(), odds_col=odds_col, score_col="score")
    pred_frame["score_source"] = score_source_name
    pred_frame["score_source_model_config"] = _display_path(active_model_config_path, workspace_root)
    policy_name: str | None = None
    policy_strategy_kind: str | None = None
    policy_resolution = resolve_runtime_policy(model_config, frame=pred_frame)
    if policy_resolution is not None:
        policy_name, policy_config = policy_resolution
        policy_strategy_kind = str(policy_config.get("strategy_kind", "")).strip().lower() or None
        pred_frame = annotate_runtime_policy(
            pred_frame,
            odds_col=odds_col,
            policy_name=policy_name,
            policy_config=policy_config,
            score_col="score",
        )
    progress.update(message=f"inference complete rows={len(pred_frame):,}")

    columns = ["date", "race_id"]
    for extra_col in ["race_no", "headline", "track", "distance"]:
        if extra_col in pred_frame.columns:
            columns.append(extra_col)
    columns.append("horse_id")
    if "horse_name" in pred_frame.columns:
        columns.append("horse_name")
    if "rank" in pred_frame.columns:
        columns.append("rank")
    if odds_col is not None and odds_col in pred_frame.columns:
        columns.append(odds_col)
    if "popularity" in pred_frame.columns:
        columns.append("popularity")
    columns += ["score_source", "score_source_model_config", "score", "pred_rank"]
    for extra_col in [
        "policy_name",
        "policy_strategy_kind",
        "policy_stage_name",
        "policy_stage_index",
        "policy_stage_trace",
        "policy_stage_fallback_reasons",
        "policy_reject_reason_primary",
        "policy_reject_reasons",
        "policy_selected_strategy_kind",
        "policy_selected",
        "policy_selection_rank",
        "policy_weight",
        "policy_prob",
        "policy_market_prob",
        "policy_expected_value",
        "policy_edge",
        "policy_blend_weight",
        "policy_min_prob",
        "policy_odds_min",
        "policy_odds_max",
        "policy_min_edge",
        "policy_fractional_kelly",
        "policy_max_fraction",
        "policy_top_k",
        "policy_min_expected_value",
    ]:
        if extra_col in pred_frame.columns:
            columns.append(extra_col)
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
    normalized_output_suffix = str(output_file_suffix or "").strip()
    file_suffix = f"_{normalized_output_suffix}" if normalized_output_suffix else ""
    csv_path = out_dir / f"predictions_{date_tag}{file_suffix}.csv"
    png_path = out_dir / f"predictions_{date_tag}{file_suffix}.png"
    summary_path = out_dir / f"predictions_{date_tag}{file_suffix}.summary.json"

    with Heartbeat("[predict]", "writing prediction outputs", logger=log_progress):
        output = pred_frame[columns].sort_values(["race_id", "pred_rank"]).reset_index(drop=True)
        write_csv_file(csv_path, output, index=False, label="prediction output")
        _plot_predictions(output, png_path)
        policy_diagnostics = summarize_policy_diagnostics(output) if "policy_selected" in output.columns else None
        prediction_summary = {
            "profile": profile_name,
            "target_date": str(pd.Timestamp(target_date).date()),
            "prediction_file": _display_path(csv_path, workspace_root),
            "chart_file": _display_path(png_path, workspace_root),
            "summary_file": _display_path(summary_path, workspace_root),
            "score_source": score_source_name,
            "score_source_model_config": _display_path(active_model_config_path, workspace_root),
            "policy_name": policy_name,
            "policy_strategy_kind": policy_strategy_kind,
            "policy_selected_rows": int(output["policy_selected"].fillna(False).sum()) if "policy_selected" in output.columns else 0,
            "policy_diagnostics": policy_diagnostics,
            "records": int(len(output)),
            "num_races": int(output["race_id"].nunique()),
            "manifest_file": _display_path(manifest_path, workspace_root) if manifest_path.exists() else None,
        }
        write_json(summary_path, prediction_summary)
    progress.complete(message="prediction outputs written")

    print(f"[predict] target date: {pd.Timestamp(target_date).date()}")
    print(f"[predict] score source: {score_source_name}")
    print(f"[predict] score config: {_display_path(active_model_config_path, workspace_root)}")
    if policy_name is not None:
        selected_rows = int(output["policy_selected"].fillna(False).sum()) if "policy_selected" in output.columns else 0
        print(f"[predict] policy: {policy_name}")
        print(f"[predict] policy strategy: {policy_strategy_kind}")
        print(f"[predict] policy selected rows: {selected_rows}")
    print(f"[predict] predictions saved: {csv_path}")
    print(f"[predict] chart saved: {png_path}")
    print(f"[predict] summary saved: {summary_path}")
    if manifest_path.exists():
        print(f"[predict] manifest: {manifest_path}")
    print(f"[predict] records: {len(output)}")
    return prediction_summary


def run_predict(
    model_config_path: str,
    data_config_path: str,
    feature_config_path: str,
    race_date: str | None = None,
    profile_name: str | None = None,
    model_artifact_suffix: str | None = None,
    output_file_suffix: str | None = None,
) -> dict[str, Any]:
    frame = prepare_prediction_frame(data_config_path)
    return run_predict_from_frame(
        model_config_path=model_config_path,
        feature_config_path=feature_config_path,
        frame=frame,
        race_date=race_date,
        profile_name=profile_name,
        model_artifact_suffix=model_artifact_suffix,
        output_file_suffix=output_file_suffix,
    )
