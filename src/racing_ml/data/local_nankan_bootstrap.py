from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml

from racing_ml.common.config import load_yaml


def resolve_python_executable(*, workspace_root: Path, fallback: str | None = None) -> str:
    venv_python = workspace_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    if fallback:
        return str(fallback)
    return str(sys.executable)


def _suffix_filename(filename: str, revision: str) -> str:
    path = Path(str(filename))
    stem = path.stem
    suffixes = "".join(path.suffixes)
    return f"{stem}_{revision}{suffixes}"


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)


def _display_config_path(path: Path, *, workspace_root: Path) -> str:
    try:
        return str(path.relative_to(workspace_root))
    except ValueError:
        return str(path)


def materialize_local_nankan_bootstrap_runtime_configs(
    *,
    workspace_root: Path,
    revision: str,
    win_config: str,
    roi_config: str,
    stack_config: str,
    output_dir: str | Path,
) -> dict[str, str]:
    runtime_dir = Path(output_dir)
    if not runtime_dir.is_absolute():
        runtime_dir = workspace_root / runtime_dir
    runtime_dir.mkdir(parents=True, exist_ok=True)

    win_payload = load_yaml(workspace_root / win_config)
    roi_payload = load_yaml(workspace_root / roi_config)
    stack_payload = load_yaml(workspace_root / stack_config)

    win_output = dict(win_payload.get("output", {}))
    roi_output = dict(roi_payload.get("output", {}))
    stack_output = dict(stack_payload.get("output", {}))
    if not win_output or not roi_output or not stack_output:
        raise ValueError("bootstrap runtime config materialization requires output sections on win, roi, and stack configs")

    for key in ("model_file", "report_file", "manifest_file"):
        if key in win_output:
            win_output[key] = _suffix_filename(str(win_output[key]), revision)
        if key in roi_output:
            roi_output[key] = _suffix_filename(str(roi_output[key]), revision)
        if key in stack_output:
            stack_output[key] = _suffix_filename(str(stack_output[key]), revision)

    win_payload["output"] = win_output
    roi_payload["output"] = roi_output
    stack_payload["output"] = stack_output

    win_runtime_path = runtime_dir / f"model_catboost_win_local_nankan_value_blend_bootstrap_{revision}.yaml"
    roi_runtime_path = runtime_dir / f"model_lightgbm_roi_local_nankan_value_blend_bootstrap_{revision}.yaml"
    stack_runtime_path = runtime_dir / f"model_value_stack_local_nankan_value_blend_bootstrap_{revision}.yaml"

    stack_components = dict(stack_payload.get("components", {}))
    stack_components["win"] = _display_config_path(win_runtime_path, workspace_root=workspace_root)
    stack_components["roi"] = _display_config_path(roi_runtime_path, workspace_root=workspace_root)
    stack_payload["components"] = stack_components

    _write_yaml(win_runtime_path, win_payload)
    _write_yaml(roi_runtime_path, roi_payload)
    _write_yaml(stack_runtime_path, stack_payload)

    return {
        "win_config": _display_config_path(win_runtime_path, workspace_root=workspace_root),
        "roi_config": _display_config_path(roi_runtime_path, workspace_root=workspace_root),
        "stack_config": _display_config_path(stack_runtime_path, workspace_root=workspace_root),
    }


def build_value_blend_bootstrap_command_plan(
    *,
    workspace_root: Path,
    python_executable: str,
    data_config: str,
    feature_config: str,
    win_config: str,
    roi_config: str,
    stack_config: str,
    revision: str,
    evaluation_pointer_output: str | None = None,
    universe: str = "local_nankan",
    source_scope: str = "local_only",
    baseline_reference: str = "local_nankan_pre_race_ready",
) -> list[dict[str, Any]]:
    run_train = str(workspace_root / "scripts" / "run_train.py")
    run_build_stack = str(workspace_root / "scripts" / "run_build_value_stack.py")
    run_revision_gate = str(workspace_root / "scripts" / "run_revision_gate.py")
    run_local_evaluate = str(workspace_root / "scripts" / "run_local_evaluate.py")

    pointer_output = evaluation_pointer_output or f"artifacts/reports/evaluation_{revision}_pointer.json"

    return [
        {
            "label": "train_win_component",
            "command": [
                python_executable,
                run_train,
                "--config",
                win_config,
                "--data-config",
                data_config,
                "--feature-config",
                feature_config,
            ],
        },
        {
            "label": "train_roi_component",
            "command": [
                python_executable,
                run_train,
                "--config",
                roi_config,
                "--data-config",
                data_config,
                "--feature-config",
                feature_config,
            ],
        },
        {
            "label": "build_value_stack",
            "command": [
                python_executable,
                run_build_stack,
                "--config",
                stack_config,
                "--data-config",
                data_config,
                "--feature-config",
                feature_config,
            ],
        },
        {
            "label": "run_revision_gate",
            "command": [
                python_executable,
                run_revision_gate,
                "--config",
                stack_config,
                "--data-config",
                data_config,
                "--feature-config",
                feature_config,
                "--revision",
                revision,
                "--skip-train",
            ],
        },
        {
            "label": "write_local_evaluation_pointer",
            "command": [
                python_executable,
                run_local_evaluate,
                "--config",
                stack_config,
                "--data-config",
                data_config,
                "--feature-config",
                feature_config,
                "--output",
                pointer_output,
                "--universe",
                universe,
                "--source-scope",
                source_scope,
                "--baseline-reference",
                baseline_reference,
            ],
        },
    ]
