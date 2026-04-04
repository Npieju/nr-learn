from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from racing_ml.data.local_nankan_bootstrap import (
    build_value_blend_bootstrap_command_plan,
    materialize_local_nankan_bootstrap_runtime_configs,
    resolve_python_executable,
)
from racing_ml.common.config import load_yaml


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


if __name__ == "__main__":
    unittest.main()
