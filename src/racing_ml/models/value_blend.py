from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from racing_ml.common.artifacts import append_suffix_to_file_name, absolutize_path, read_json, relativize_path, resolve_output_artifacts
from racing_ml.common.config import load_yaml


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if not text or text in seen:
            continue
        seen.add(text)
        output.append(text)
    return output


def load_component_from_config(
    *,
    workspace_root: Path,
    config_path: str | Path,
    artifact_suffix: str | None = None,
) -> dict[str, Any]:
    config_abs = absolutize_path(config_path, workspace_root)
    config = load_yaml(config_abs)
    output_cfg = dict(config.get("output", {}))
    if artifact_suffix:
        output_cfg["model_file"] = append_suffix_to_file_name(
            str(output_cfg.get("model_file", "baseline_model.joblib")),
            str(artifact_suffix),
        )
        output_cfg["report_file"] = append_suffix_to_file_name(
            str(output_cfg.get("report_file", "train_metrics.json")),
            str(artifact_suffix),
        )
        output_cfg["manifest_file"] = append_suffix_to_file_name(
            str(output_cfg.get("manifest_file", output_cfg.get("model_file", "baseline_model.joblib"))),
            str(artifact_suffix),
        )
    artifacts = resolve_output_artifacts(output_cfg)

    model_abs = absolutize_path(artifacts.model_path, workspace_root)
    report_abs = absolutize_path(artifacts.report_path, workspace_root)
    manifest_abs = absolutize_path(artifacts.manifest_path, workspace_root)

    if not model_abs.exists():
        raise FileNotFoundError(f"Component model not found: {model_abs}")

    model = joblib.load(model_abs)
    manifest_payload = read_json(manifest_abs) if manifest_abs.exists() else None
    report_payload = read_json(report_abs) if report_abs.exists() else None

    feature_columns: list[str] = []
    categorical_columns: list[str] = []
    if isinstance(model, dict):
        feature_columns = [str(column) for column in model.get("feature_columns", []) if str(column).strip()]
        categorical_columns = [str(column) for column in model.get("categorical_columns", []) if str(column).strip()]

    if not feature_columns and isinstance(manifest_payload, dict):
        feature_meta = manifest_payload.get("features", {})
        feature_columns = [str(column) for column in feature_meta.get("columns", []) if str(column).strip()]
        categorical_columns = [str(column) for column in feature_meta.get("categorical_columns", []) if str(column).strip()]

    return {
        "config": config,
        "config_path": relativize_path(config_abs, workspace_root),
        "model": model,
        "model_path": relativize_path(model_abs, workspace_root),
        "report_path": relativize_path(report_abs, workspace_root) if report_abs.exists() else None,
        "manifest_path": relativize_path(manifest_abs, workspace_root) if manifest_abs.exists() else None,
        "manifest": manifest_payload,
        "report": report_payload,
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "task": str(config.get("task", "classification")),
        "model_name": str(config.get("model", {}).get("name", "unknown")),
        "artifact_suffix": str(artifact_suffix or ""),
    }


def build_value_blend_bundle(
    *,
    win_component: dict[str, Any],
    alpha_component: dict[str, Any] | None,
    roi_component: dict[str, Any] | None,
    time_component: dict[str, Any] | None,
    params: dict[str, Any],
) -> dict[str, Any]:
    feature_columns = _dedupe_preserve_order(
        [
            *win_component.get("feature_columns", []),
            *(alpha_component or {}).get("feature_columns", []),
            *(roi_component or {}).get("feature_columns", []),
            *(time_component or {}).get("feature_columns", []),
        ]
    )
    categorical_columns = _dedupe_preserve_order(
        [
            *win_component.get("categorical_columns", []),
            *(alpha_component or {}).get("categorical_columns", []),
            *(roi_component or {}).get("categorical_columns", []),
            *(time_component or {}).get("categorical_columns", []),
        ]
    )

    components: dict[str, Any] = {
        "win": win_component["model"],
    }
    component_metadata: dict[str, Any] = {
        "win": {
            "task": win_component.get("task"),
            "model_name": win_component.get("model_name"),
            "config_path": win_component.get("config_path"),
            "model_path": win_component.get("model_path"),
            "report_path": win_component.get("report_path"),
            "manifest_path": win_component.get("manifest_path"),
        },
    }

    if alpha_component is not None:
        components["alpha"] = alpha_component["model"]
        component_metadata["alpha"] = {
            "task": alpha_component.get("task"),
            "model_name": alpha_component.get("model_name"),
            "config_path": alpha_component.get("config_path"),
            "model_path": alpha_component.get("model_path"),
            "report_path": alpha_component.get("report_path"),
            "manifest_path": alpha_component.get("manifest_path"),
        }

    if roi_component is not None:
        components["roi"] = roi_component["model"]
        component_metadata["roi"] = {
            "task": roi_component.get("task"),
            "model_name": roi_component.get("model_name"),
            "config_path": roi_component.get("config_path"),
            "model_path": roi_component.get("model_path"),
            "report_path": roi_component.get("report_path"),
            "manifest_path": roi_component.get("manifest_path"),
        }

    if time_component is not None:
        components["time"] = time_component["model"]
        component_metadata["time"] = {
            "task": time_component.get("task"),
            "model_name": time_component.get("model_name"),
            "config_path": time_component.get("config_path"),
            "model_path": time_component.get("model_path"),
            "report_path": time_component.get("report_path"),
            "manifest_path": time_component.get("manifest_path"),
        }

    return {
        "kind": "value_blend_model",
        "backend": "stacked",
        "task": "classification",
        "feature_columns": feature_columns,
        "categorical_columns": categorical_columns,
        "components": components,
        "params": dict(params),
        "component_metadata": component_metadata,
    }