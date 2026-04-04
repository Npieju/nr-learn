from __future__ import annotations

from pathlib import Path
import sys
from typing import Any


def resolve_python_executable(*, workspace_root: Path, fallback: str | None = None) -> str:
    venv_python = workspace_root / ".venv" / "bin" / "python"
    if venv_python.exists():
        return str(venv_python)
    if fallback:
        return str(fallback)
    return str(sys.executable)


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
) -> list[dict[str, Any]]:
    run_train = str(workspace_root / "scripts" / "run_train.py")
    run_build_stack = str(workspace_root / "scripts" / "run_build_value_stack.py")
    run_revision_gate = str(workspace_root / "scripts" / "run_revision_gate.py")

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
    ]
