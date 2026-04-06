from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import sys

from racing_ml.data.local_nankan_bootstrap import (
    build_value_blend_bootstrap_command_plan,
    materialize_local_nankan_bootstrap_runtime_configs,
    resolve_python_executable,
)
from racing_ml.common.config import load_yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


bootstrap_handoff_script = _load_script_module(
    "test_run_local_nankan_result_ready_bootstrap_handoff",
    "scripts/run_local_nankan_result_ready_bootstrap_handoff.py",
)


class LocalNankanBootstrapTest(unittest.TestCase):
    def test_resolve_python_executable_prefers_venv(self) -> None:
        root = Path("/workspaces/nr-learn")

        resolved = resolve_python_executable(workspace_root=root, fallback="/usr/bin/python3")

        self.assertTrue(resolved.endswith(".venv/bin/python"))

    def test_build_value_blend_bootstrap_command_plan_has_expected_steps(self) -> None:
        root = Path("/workspaces/nr-learn")

        plan = build_value_blend_bootstrap_command_plan(
            workspace_root=root,
            python_executable="/workspaces/nr-learn/.venv/bin/python",
            data_config="configs/data_local_nankan_pre_race_ready.yaml",
            feature_config="configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
            win_config="configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
            roi_config="configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
            stack_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml",
            revision="r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1",
        )

        self.assertEqual([step["label"] for step in plan], [
            "train_win_component",
            "train_roi_component",
            "build_value_stack",
            "run_revision_gate",
        ])
        self.assertIn("configs/data_local_nankan_pre_race_ready.yaml", plan[0]["command"])
        self.assertIn("configs/data_local_nankan_pre_race_ready.yaml", plan[1]["command"])
        self.assertIn("--skip-train", plan[3]["command"])
        self.assertIn("r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1", plan[3]["command"])

    def test_materialize_local_nankan_bootstrap_runtime_configs_suffixes_outputs(self) -> None:
        root = Path("/workspaces/nr-learn")

        with tempfile.TemporaryDirectory() as tmpdir:
            resolved = materialize_local_nankan_bootstrap_runtime_configs(
                workspace_root=root,
                revision="r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1",
                win_config="configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
                roi_config="configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml",
                stack_config="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml",
                output_dir=tmpdir,
            )

            win_cfg = load_yaml(root / resolved["win_config"])
            roi_cfg = load_yaml(root / resolved["roi_config"])
            stack_cfg = load_yaml(root / resolved["stack_config"])

            self.assertIn("r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1", win_cfg["output"]["model_file"])
            self.assertIn("r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1", roi_cfg["output"]["model_file"])
            self.assertIn("r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1", stack_cfg["output"]["model_file"])
            self.assertEqual(stack_cfg["components"]["win"], resolved["win_config"])
            self.assertEqual(stack_cfg["components"]["roi"], resolved["roi_config"])

    def test_bootstrap_handoff_returns_not_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_result_ready_bootstrap_handoff_manifest.json"
            handoff_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"

            with patch.object(bootstrap_handoff_script, "resolve_python_executable", return_value="/usr/bin/python3"), patch.object(
                bootstrap_handoff_script,
                "materialize_local_nankan_bootstrap_runtime_configs",
                return_value={
                    "win_config": "artifacts/runtime_configs/win.yaml",
                    "roi_config": "artifacts/runtime_configs/roi.yaml",
                    "stack_config": "artifacts/runtime_configs/stack.yaml",
                },
            ), patch.object(
                bootstrap_handoff_script,
                "build_value_blend_bootstrap_command_plan",
                return_value=[{"label": "train_win_component", "command": ["python", "train_win"]}],
            ), patch.object(
                bootstrap_handoff_script, "_run_command", return_value=2
            ), patch.object(
                bootstrap_handoff_script,
                "_read_json_dict",
                return_value={"status": "not_ready", "current_phase": "await_result_arrival"},
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_result_ready_bootstrap_handoff.py",
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--handoff-manifest-output",
                    str(handoff_manifest),
                ],
            ):
                exit_code = bootstrap_handoff_script.main()

            self.assertEqual(exit_code, 2)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "not_ready")
            self.assertEqual(manifest["current_phase"], "await_result_arrival")
            self.assertEqual(manifest["recommended_action"], "wait_for_result_ready_pre_race_races")
            self.assertEqual(manifest["runtime_configs"]["stack_config"], "artifacts/runtime_configs/stack.yaml")
            self.assertEqual(manifest["bootstrap_command_plan"][0]["label"], "train_win_component")

    def test_bootstrap_handoff_returns_benchmark_ready_without_running_children(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_result_ready_bootstrap_handoff_manifest.json"
            handoff_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"

            with patch.object(bootstrap_handoff_script, "resolve_python_executable", return_value="/usr/bin/python3"), patch.object(
                bootstrap_handoff_script,
                "materialize_local_nankan_bootstrap_runtime_configs",
                return_value={
                    "win_config": "artifacts/runtime_configs/win.yaml",
                    "roi_config": "artifacts/runtime_configs/roi.yaml",
                    "stack_config": "artifacts/runtime_configs/stack.yaml",
                },
            ), patch.object(
                bootstrap_handoff_script,
                "build_value_blend_bootstrap_command_plan",
                return_value=[{"label": "train_win_component", "command": ["python", "train_win"]}],
            ), patch.object(
                bootstrap_handoff_script, "_run_command", return_value=0
            ) as run_command, patch.object(
                bootstrap_handoff_script,
                "_read_json_dict",
                return_value={"status": "completed", "current_phase": "benchmark_handoff_completed"},
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_result_ready_bootstrap_handoff.py",
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--handoff-manifest-output",
                    str(handoff_manifest),
                ],
            ):
                exit_code = bootstrap_handoff_script.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_command.call_count, 1)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "benchmark_ready")
            self.assertEqual(manifest["current_phase"], "bootstrap_pending")
            self.assertEqual(manifest["recommended_action"], "run_bootstrap_command_plan")

    def test_bootstrap_handoff_reports_failed_child_phase(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_result_ready_bootstrap_handoff_manifest.json"
            handoff_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            command_plan = [
                {"label": "train_win_component", "command": ["python", "train_win"]},
                {"label": "train_roi_component", "command": ["python", "train_roi"]},
            ]

            with patch.object(bootstrap_handoff_script, "resolve_python_executable", return_value="/usr/bin/python3"), patch.object(
                bootstrap_handoff_script,
                "materialize_local_nankan_bootstrap_runtime_configs",
                return_value={
                    "win_config": "artifacts/runtime_configs/win.yaml",
                    "roi_config": "artifacts/runtime_configs/roi.yaml",
                    "stack_config": "artifacts/runtime_configs/stack.yaml",
                },
            ), patch.object(
                bootstrap_handoff_script,
                "build_value_blend_bootstrap_command_plan",
                return_value=command_plan,
            ), patch.object(
                bootstrap_handoff_script, "_run_command", side_effect=[0, 0, 1]
            ), patch.object(
                bootstrap_handoff_script,
                "_read_json_dict",
                return_value={"status": "completed", "current_phase": "benchmark_handoff_completed"},
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_result_ready_bootstrap_handoff.py",
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--handoff-manifest-output",
                    str(handoff_manifest),
                    "--run-bootstrap",
                ],
            ):
                exit_code = bootstrap_handoff_script.main()

            self.assertEqual(exit_code, 1)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_phase"], "train_roi_component")
            self.assertEqual(manifest["recommended_action"], "inspect_bootstrap_child_command")
            self.assertEqual([run["label"] for run in manifest["bootstrap_runs"]], ["train_win_component", "train_roi_component"])
            self.assertEqual(manifest["bootstrap_runs"][1]["exit_code"], 1)


if __name__ == "__main__":
    unittest.main()
