from pathlib import Path

from racing_ml.common.artifacts import (
    build_model_manifest,
    build_training_report_payload,
    resolve_output_artifacts,
    write_json,
)
from racing_ml.evaluation.leakage import run_leakage_audit
from racing_ml.evaluation.policy import PolicyConstraints

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.features.builder import build_features
from racing_ml.features.selection import resolve_feature_selection, summarize_feature_coverage
from racing_ml.models.trainer import train_and_evaluate


def _resolve_model_device_label(model_name: str, model_params: dict[str, object]) -> str:
    normalized_name = str(model_name).strip().lower()
    if normalized_name == "catboost":
        return str(model_params.get("task_type", model_params.get("device_type", "cpu"))).strip().lower() or "cpu"
    return str(model_params.get("device_type", model_params.get("task_type", "cpu"))).strip().lower() or "cpu"


def run_train(model_config_path: str, data_config_path: str, feature_config_path: str) -> None:
    model_path = Path(model_config_path)
    data_path = Path(data_config_path)
    feature_path = Path(feature_config_path)

    model_config = load_yaml(model_path)
    data_config = load_yaml(data_path)
    feature_config = load_yaml(feature_path)

    dataset_cfg = data_config.get("dataset", {})
    split_cfg = data_config.get("split", {})
    progress = ProgressBar(total=6, prefix="[train]", min_interval_sec=0.0)
    progress.start("configs loaded")

    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    with Heartbeat("[train]", "loading training table"):
        frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=Path.cwd())
    progress.update(message=f"training table loaded rows={len(frame):,}")

    with Heartbeat("[train]", "building features"):
        frame = build_features(frame)
    progress.update(message=f"features built columns={len(frame.columns):,}")

    label_column = model_config.get("label", "is_win")
    feature_selection = resolve_feature_selection(frame, feature_config, label_column=label_column)
    feature_coverage = summarize_feature_coverage(frame, feature_config, feature_selection)
    feature_columns = feature_selection.feature_columns
    progress.update(
        message=(
            f"feature selection ready features={len(feature_selection.feature_columns):,} "
            f"categorical={len(feature_selection.categorical_columns):,}"
        )
    )
    task = str(model_config.get("task", "classification"))
    model_cfg = model_config.get("model", {})
    output_cfg = model_config.get("output", {})
    output_artifacts = resolve_output_artifacts(output_cfg)
    model_name = model_cfg.get("name", "lightgbm")
    model_params = model_cfg.get("params", {})
    device_type = _resolve_model_device_label(model_name, model_params)
    training_cfg = model_config.get("training", {})
    evaluation_cfg = model_config.get("evaluation", {})
    allow_fallback = bool(training_cfg.get("allow_fallback_model", False))
    early_stopping_rounds = training_cfg.get("early_stopping_rounds")
    leakage_cfg = evaluation_cfg.get("leakage_audit", {})
    leakage_enabled = bool(leakage_cfg.get("enabled", True))

    print(f"[train] model: {model_name}")
    print(f"[train] task: {task}")
    print(f"[train] device_type: {device_type}")
    print(f"[train] allow_fallback_model: {allow_fallback}")
    print(f"[train] early_stopping_rounds: {early_stopping_rounds}")
    if feature_coverage["missing_force_include_features"]:
        print(f"[train] missing force-include features: {feature_coverage['missing_force_include_features']}")
    if feature_coverage["low_coverage_force_include_features"]:
        print(f"[train] low-coverage force-include features: {feature_coverage['low_coverage_force_include_features']}")

    if leakage_enabled:
        with Heartbeat("[train]", "running leakage audit"):
            leakage_report = run_leakage_audit(frame=frame, feature_columns=feature_columns, label_column=label_column)
        progress.update(message="leakage audit complete")
    else:
        leakage_report = {"enabled": False}
        progress.update(message="leakage audit skipped")

    result = train_and_evaluate(
        frame=frame,
        feature_columns=feature_columns,
        label_column=label_column,
        task=task,
        model_name=model_name,
        model_params=model_params,
        train_end=split_cfg.get("train_end", "2022-12-31"),
        valid_start=split_cfg.get("valid_start", "2023-01-01"),
        valid_end=split_cfg.get("valid_end", "2023-12-31"),
        max_train_rows=training_cfg.get("max_train_rows"),
        max_valid_rows=training_cfg.get("max_valid_rows"),
        early_stopping_rounds=early_stopping_rounds,
        allow_fallback=allow_fallback,
        model_dir=output_cfg.get("model_dir", "artifacts/models"),
        report_dir=output_cfg.get("report_dir", "artifacts/reports"),
        model_file_name=output_cfg.get("model_file", "baseline_model.joblib"),
        report_file_name=output_cfg.get("report_file", "train_metrics.json"),
        categorical_features=feature_selection.categorical_columns,
    )
    progress.update(message="model training complete")

    with Heartbeat("[train]", "writing training artifacts"):
        policy_constraints = PolicyConstraints.from_config(evaluation_cfg).to_dict()
        run_context = {
            "model_config": str(model_path),
            "data_config": str(data_path),
            "feature_config": str(feature_path),
            "task": task,
            "label_column": label_column,
            "model_name": model_name,
            "device_type": device_type,
            "raw_dir": str(raw_dir),
            "rows_total": int(len(frame)),
            "feature_selection_mode": feature_selection.mode,
            "feature_count": int(len(feature_selection.feature_columns)),
            "categorical_feature_count": int(len(feature_selection.categorical_columns)),
            "rows_train_max": training_cfg.get("max_train_rows"),
            "rows_valid_max": training_cfg.get("max_valid_rows"),
            "split_train_end": split_cfg.get("train_end", "2022-12-31"),
            "split_valid_start": split_cfg.get("valid_start", "2023-01-01"),
            "split_valid_end": split_cfg.get("valid_end", "2023-12-31"),
            "artifact_model": output_artifacts.model_path.as_posix(),
            "artifact_report": output_artifacts.report_path.as_posix(),
            "artifact_manifest": output_artifacts.manifest_path.as_posix(),
        }
        report_payload = build_training_report_payload(
            metrics=result.metrics,
            run_context=run_context,
            leakage_audit=leakage_report,
            policy_constraints=policy_constraints,
            extra_metadata={"feature_coverage": feature_coverage},
        )
        write_json(result.report_path, report_payload)

        workspace_root = Path.cwd()
        manifest_path = output_artifacts.manifest_path if output_artifacts.manifest_path.is_absolute() else (workspace_root / output_artifacts.manifest_path)
        manifest_payload = build_model_manifest(
            workspace_root=workspace_root,
            model_config_path=model_path,
            data_config_path=data_path,
            feature_config_path=feature_path,
            model_path=result.model_path,
            report_path=result.report_path,
            task=task,
            label_column=label_column,
            model_name=model_name,
            used_features=result.used_features,
            categorical_columns=result.categorical_features,
            metrics=result.metrics,
            run_context=run_context,
            leakage_audit=leakage_report,
            policy_constraints=policy_constraints,
            extra_metadata={"feature_coverage": feature_coverage},
        )
        write_json(manifest_path, manifest_payload)
    progress.complete(message="artifacts written")

    print(f"[train] model saved: {result.model_path}")
    print(f"[train] report saved: {result.report_path}")
    print(f"[train] manifest saved: {manifest_path}")
    print(f"[train] metrics: {result.metrics}")
    print(f"[train] leakage_audit: {leakage_report}")
    print(f"[train] used features: {result.used_features}")
    print(f"[train] categorical features: {result.categorical_features}")
