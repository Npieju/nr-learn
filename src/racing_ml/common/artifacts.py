from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from racing_ml.common.config import load_yaml


@dataclass(frozen=True)
class OutputArtifacts:
    model_dir: Path
    report_dir: Path
    model_path: Path
    report_path: Path
    manifest_path: Path


def derive_manifest_file_name(model_file_name: str) -> str:
    model_path = Path(model_file_name)
    return f"{model_path.stem}.manifest.json"


def resolve_output_artifacts(output_cfg: dict[str, Any] | None = None) -> OutputArtifacts:
    output_cfg = output_cfg or {}
    model_dir = Path(output_cfg.get("model_dir", "artifacts/models"))
    report_dir = Path(output_cfg.get("report_dir", "artifacts/reports"))
    model_file_name = str(output_cfg.get("model_file", "baseline_model.joblib"))
    report_file_name = str(output_cfg.get("report_file", "train_metrics.json"))
    manifest_file_name = str(output_cfg.get("manifest_file", derive_manifest_file_name(model_file_name)))
    return OutputArtifacts(
        model_dir=model_dir,
        report_dir=report_dir,
        model_path=model_dir / model_file_name,
        report_path=report_dir / report_file_name,
        manifest_path=model_dir / manifest_file_name,
    )


def absolutize_path(path: str | Path, workspace_root: Path) -> Path:
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else (workspace_root / path_obj)


def relativize_path(path: str | Path | None, workspace_root: Path) -> str | None:
    if path is None:
        return None

    path_obj = Path(path)
    if not path_obj.is_absolute():
        return path_obj.as_posix()

    try:
        return path_obj.relative_to(workspace_root).as_posix()
    except ValueError:
        return path_obj.as_posix()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: str | Path, payload: Any) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def build_training_report_payload(
    *,
    metrics: dict[str, Any],
    run_context: dict[str, Any],
    leakage_audit: dict[str, Any] | None,
    policy_constraints: dict[str, Any] | None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(metrics)
    payload["run_context"] = dict(run_context)
    if leakage_audit is not None:
        payload["leakage_audit"] = leakage_audit
    if policy_constraints is not None:
        payload["policy_constraints"] = policy_constraints
    if extra_metadata:
        payload["metadata"] = extra_metadata
    return payload


def build_model_manifest(
    *,
    workspace_root: Path,
    model_config_path: str | Path,
    data_config_path: str | Path | None,
    feature_config_path: str | Path | None,
    model_path: str | Path,
    report_path: str | Path,
    task: str,
    label_column: str,
    model_name: str,
    used_features: list[str],
    categorical_columns: list[str] | None = None,
    metrics: dict[str, Any],
    run_context: dict[str, Any],
    leakage_audit: dict[str, Any] | None,
    policy_constraints: dict[str, Any] | None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    model_abs = absolutize_path(model_path, workspace_root)
    report_abs = absolutize_path(report_path, workspace_root)

    return {
        "artifact_schema_version": 1,
        "artifact_type": "trained_model",
        "created_at": utc_now_iso(),
        "model": {
            "path": relativize_path(model_abs, workspace_root),
            "report_path": relativize_path(report_abs, workspace_root),
            "task": str(task),
            "label_column": str(label_column),
            "model_name": str(model_name),
        },
        "configs": {
            "model_config": relativize_path(absolutize_path(model_config_path, workspace_root), workspace_root),
            "data_config": relativize_path(absolutize_path(data_config_path, workspace_root), workspace_root) if data_config_path else None,
            "feature_config": relativize_path(absolutize_path(feature_config_path, workspace_root), workspace_root) if feature_config_path else None,
        },
        "features": {
            "count": int(len(used_features)),
            "columns": list(used_features),
            "categorical_count": int(len(categorical_columns or [])),
            "categorical_columns": list(categorical_columns or []),
        },
        "metrics": dict(metrics),
        "run_context": dict(run_context),
        "leakage_audit": leakage_audit,
        "policy_constraints": policy_constraints,
        "metadata": extra_metadata or {},
    }


def _extract_metrics(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    excluded = {"run_context", "leakage_audit", "policy_constraints", "metadata"}
    return {key: value for key, value in payload.items() if key not in excluded}


def resolve_component_from_config(
    *,
    workspace_root: Path,
    component_name: str,
    config_path: str | Path,
) -> dict[str, Any]:
    config_abs = absolutize_path(config_path, workspace_root)
    config = load_yaml(config_abs)
    output_cfg = config.get("output", {})
    artifacts = resolve_output_artifacts(output_cfg)

    model_abs = absolutize_path(artifacts.model_path, workspace_root)
    report_abs = absolutize_path(artifacts.report_path, workspace_root)
    manifest_abs = absolutize_path(artifacts.manifest_path, workspace_root)

    if not model_abs.exists():
        raise FileNotFoundError(f"Component '{component_name}' model file not found: {model_abs}")

    manifest_payload = read_json(manifest_abs) if manifest_abs.exists() else None
    report_payload = read_json(report_abs) if report_abs.exists() else None
    metrics = _extract_metrics(manifest_payload.get("metrics", {})) if isinstance(manifest_payload, dict) else _extract_metrics(report_payload)

    return {
        "name": component_name,
        "config_path": relativize_path(config_abs, workspace_root),
        "task": str(config.get("task", "classification")),
        "label_column": str(config.get("label", "is_win")),
        "model_name": str(config.get("model", {}).get("name", "lightgbm")),
        "model_path": relativize_path(model_abs, workspace_root),
        "report_path": relativize_path(report_abs, workspace_root) if report_abs.exists() else None,
        "manifest_path": relativize_path(manifest_abs, workspace_root) if manifest_abs.exists() else None,
        "metrics": metrics,
        "evaluation_policy": dict(config.get("evaluation", {}).get("policy", config.get("evaluation", {}).get("strategy_constraints", {}))),
    }


def build_bundle_manifest(
    *,
    bundle_name: str,
    bundle_kind: str,
    primary_component: str,
    components: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "artifact_schema_version": 1,
        "artifact_type": "model_bundle",
        "bundle_kind": str(bundle_kind),
        "bundle_name": str(bundle_name),
        "created_at": utc_now_iso(),
        "primary_component": str(primary_component),
        "components": components,
        "metadata": {
            "component_count": int(len(components)),
            "component_names": list(components.keys()),
        },
    }