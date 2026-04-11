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
wait_then_cycle_script = _load_script_module(
    "test_run_local_nankan_future_only_wait_then_cycle",
    "scripts/run_local_nankan_future_only_wait_then_cycle.py",
)
followup_oneshot_script = _load_script_module(
    "test_run_local_nankan_future_only_followup_oneshot",
    "scripts/run_local_nankan_future_only_followup_oneshot.py",
)


class LocalNankanBootstrapTest(unittest.TestCase):
    def test_followup_oneshot_blocks_without_fresh_upstream_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "followup_output.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_followup_oneshot.py",
                    "--upstream-manifest",
                    str(tmp_path / "missing_upstream.json"),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = followup_oneshot_script.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "not_ready")
            self.assertEqual(payload["current_phase"], "await_external_refresh_completion")
            self.assertEqual(payload["recommended_action"], "produce_fresh_upstream_refresh_artifact_before_followup_oneshot")
            self.assertEqual(payload["execution_role"], "readiness_followup_gate")
            self.assertEqual(payload["trigger_contract"], "external_refresh_completed_only")
            self.assertFalse(payload["upstream_fresh"])
            self.assertFalse(payload["child_launch_allowed"])
            self.assertFalse(payload["upstream_refresh"]["exists"])
            self.assertFalse(payload["upstream_refresh"]["upstream_fresh"])
            self.assertFalse(payload["upstream_refresh"]["contract_valid"])
            self.assertEqual(payload["read_order"][0], "status")
            self.assertIn("upstream_fresh=false", payload["highlights"])
            self.assertIn("--oneshot", payload["followup_command"]["command"])

    def test_followup_oneshot_blocks_invalid_upstream_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            upstream_path = tmp_path / "capture_cycle.json"
            output_path = tmp_path / "followup_output.json"
            upstream_path.write_text(
                json.dumps(
                    {
                        "status": "capturing",
                        "current_phase": "capturing_pre_race_pool",
                        "recommended_action": "wait",
                        "finished_at": followup_oneshot_script.utc_now_iso(),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_followup_oneshot.py",
                    "--upstream-manifest",
                    str(upstream_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = followup_oneshot_script.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["current_phase"], "invalid_upstream_refresh_contract")
            self.assertEqual(payload["recommended_action"], "rerun_capture_refresh_with_self_describing_manifest_before_followup_oneshot")
            self.assertFalse(payload["upstream_refresh"]["contract_valid"])
            self.assertIn("execution_role must be pre_race_capture_refresh_loop", payload["upstream_refresh"]["contract_errors"])

    def test_followup_oneshot_blocks_stale_upstream_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            upstream_path = tmp_path / "capture_cycle.json"
            output_path = tmp_path / "followup_output.json"
            upstream_path.write_text(
                json.dumps(
                    {
                        "status": "capturing",
                        "current_phase": "capturing_pre_race_pool",
                        "recommended_action": "wait",
                        "execution_role": "pre_race_capture_refresh_loop",
                        "data_update_mode": "capture_refresh_only",
                        "execution_mode": "bounded_pass_loop",
                        "trigger_contract": "direct_capture_refresh",
                        "finished_at": "2024-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_followup_oneshot.py",
                    "--upstream-manifest",
                    str(upstream_path),
                    "--max-upstream-age-seconds",
                    "60",
                    "--output",
                    str(output_path),
                ],
            ), patch.object(
                followup_oneshot_script,
                "utc_now_iso",
                return_value="2024-01-01T00:02:01+00:00",
            ):
                exit_code = followup_oneshot_script.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 2)
            self.assertEqual(payload["status"], "not_ready")
            self.assertEqual(payload["current_phase"], "await_fresh_external_refresh_completion")
            self.assertEqual(payload["recommended_action"], "refresh_upstream_artifact_then_rerun_followup_oneshot")
            self.assertEqual(payload["observed_at"], "2024-01-01T00:02:01+00:00")
            self.assertFalse(payload["upstream_fresh"])
            self.assertFalse(payload["child_launch_allowed"])
            self.assertFalse(payload["upstream_refresh"]["upstream_fresh"])
            self.assertTrue(payload["upstream_refresh"]["contract_valid"])
            self.assertEqual(payload["upstream_refresh"]["age_seconds"], 121)

    def test_followup_oneshot_dry_run_plans_child_without_launching_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            upstream_path = tmp_path / "capture_cycle.json"
            output_path = tmp_path / "followup_output.json"
            wait_cycle_manifest = tmp_path / "wait_then_cycle.json"
            upstream_path.write_text(
                json.dumps(
                    {
                        "status": "capturing",
                        "current_phase": "capturing_pre_race_pool",
                        "recommended_action": "wait",
                        "execution_role": "pre_race_capture_refresh_loop",
                        "data_update_mode": "capture_refresh_only",
                        "execution_mode": "bounded_pass_loop",
                        "trigger_contract": "direct_capture_refresh",
                        "finished_at": "2024-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(
                followup_oneshot_script,
                "_run_command",
                side_effect=AssertionError("dry-run must not launch child command"),
            ), patch.object(
                followup_oneshot_script,
                "utc_now_iso",
                side_effect=["2024-01-01T00:00:30+00:00", "2024-01-01T00:00:30+00:00", "2024-01-01T00:00:30+00:00"],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_followup_oneshot.py",
                    "--upstream-manifest",
                    str(upstream_path),
                    "--wait-cycle-manifest-output",
                    str(wait_cycle_manifest),
                    "--output",
                    str(output_path),
                    "--artifact-prefix",
                    "unit_followup_oneshot",
                    "--run-id",
                    "run02",
                    "--dry-run",
                ],
            ):
                exit_code = followup_oneshot_script.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["current_phase"], "followup_plan_ready")
            self.assertEqual(payload["recommended_action"], "run_followup_oneshot")
            self.assertEqual(payload["observed_at"], "2024-01-01T00:00:30+00:00")
            self.assertTrue(payload["dry_run"])
            self.assertTrue(payload["upstream_fresh"])
            self.assertTrue(payload["child_launch_allowed"])
            self.assertEqual(payload["upstream_refresh"]["age_seconds"], 30)
            self.assertNotIn("exit_code", payload["followup_command"])
            self.assertIn("followup_exit_code=planned", payload["highlights"])

    def test_followup_oneshot_runs_wait_cycle_for_fresh_upstream_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            upstream_path = tmp_path / "capture_cycle.json"
            output_path = tmp_path / "followup_output.json"
            wait_cycle_manifest = tmp_path / "wait_then_cycle.json"
            upstream_path.write_text(
                json.dumps(
                    {
                        "status": "capturing",
                        "current_phase": "capturing_pre_race_pool",
                        "recommended_action": "wait",
                        "execution_role": "pre_race_capture_refresh_loop",
                        "data_update_mode": "capture_refresh_only",
                        "trigger_contract": "direct_capture_refresh",
                        "finished_at": "2024-01-01T00:00:00+00:00",
                    }
                ),
                encoding="utf-8",
            )
            commands: list[list[str]] = []

            def fake_run_command(*, label: str, command: list[str]) -> int:
                commands.append(list(command))
                return 0

            def fake_read_json_dict(path: Path | None) -> dict[str, object]:
                if path is None:
                    return {}
                if path == upstream_path:
                    return json.loads(upstream_path.read_text(encoding="utf-8"))
                if path == wait_cycle_manifest:
                    return {
                        "status": "partial",
                        "current_phase": "future_only_readiness_track",
                        "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
                    }
                return {}

            with patch.object(followup_oneshot_script, "_run_command", side_effect=fake_run_command), patch.object(
                followup_oneshot_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                followup_oneshot_script,
                "utc_now_iso",
                return_value="2024-01-01T00:00:45+00:00",
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_followup_oneshot.py",
                    "--upstream-manifest",
                    str(upstream_path),
                    "--wait-cycle-manifest-output",
                    str(wait_cycle_manifest),
                    "--output",
                    str(output_path),
                    "--artifact-prefix",
                    "unit_followup_oneshot",
                    "--run-id",
                    "run01",
                    "--run-bootstrap-on-ready",
                ],
            ):
                exit_code = followup_oneshot_script.main()

            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(exit_code, 0)
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["current_phase"], "future_only_readiness_track")
            self.assertEqual(payload["execution_role"], "readiness_followup_gate")
            self.assertEqual(payload["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(payload["run_id"], "run01")
            self.assertEqual(payload["observed_at"], "2024-01-01T00:00:45+00:00")
            self.assertTrue(payload["upstream_fresh"])
            self.assertTrue(payload["upstream_refresh"]["upstream_fresh"])
            self.assertTrue(payload["upstream_refresh"]["contract_valid"])
            self.assertTrue(payload["child_launch_allowed"])
            self.assertTrue(payload["upstream_refresh"]["exists"])
            self.assertEqual(payload["upstream_refresh"]["observed_at"], "2024-01-01T00:00:00+00:00")
            self.assertEqual(payload["upstream_refresh"]["age_seconds"], 45)
            self.assertEqual(payload["read_order"][3], "upstream_refresh.upstream_fresh")
            self.assertIn("followup_exit_code=0", payload["highlights"])
            self.assertEqual(len(commands), 1)
            self.assertIn("--oneshot", commands[0])
            self.assertIn("--run-id", commands[0])
            self.assertIn("run01", commands[0])
            self.assertIn("--artifact-prefix", commands[0])
            self.assertIn("unit_followup_oneshot", commands[0])
            self.assertIn("--manifest-output", commands[0])
            self.assertIn(str(wait_cycle_manifest), commands[0])
            self.assertIn("--run-bootstrap-on-ready", commands[0])

    def test_bootstrap_log_path_compacts_duplicate_revision_and_length(self) -> None:
        long_prefix = "local_nankan_future_only_wait_then_cycle_issue122_live_auto_resume_20260407_20260407T120132Z_cycle_001_bootstrap_handoff"

        log_path = bootstrap_handoff_script._build_log_path(
            log_dir=Path("artifacts/logs"),
            log_prefix=long_prefix,
            revision=long_prefix,
            label="pre_race_benchmark_handoff",
        )

        self.assertEqual(log_path.parent, Path("artifacts/logs"))
        self.assertLessEqual(len(log_path.name), 240)
        self.assertTrue(log_path.name.endswith(".log"))
        self.assertIn("pre_race_benchmark_handoff", log_path.name)
        self.assertEqual(log_path.name.count(long_prefix), 1)

    def test_future_readiness_cycle_passes_cycle_scoped_bootstrap_log_prefix(self) -> None:
        readiness_cycle_script = _load_script_module(
            "test_run_local_nankan_future_only_readiness_cycle",
            "scripts/run_local_nankan_future_only_readiness_cycle.py",
        )

        commands: list[list[str]] = []

        def fake_run_command(*, label: str, command: list[str]) -> int:
            commands.append(list(command))
            return 0 if label != "readiness_watcher" else 2

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            expected_probe_summary = str(tmp_path / "watcher_readiness_probe_summary.json")
            with patch.object(readiness_cycle_script, "_run_command", side_effect=fake_run_command), patch.object(
                readiness_cycle_script,
                "_read_json_dict",
                side_effect=[
                    {"status": "capturing", "current_phase": "capturing_pre_race_pool", "recommended_action": "wait"},
                    {"status": "not_ready", "current_phase": "future_only_readiness_track", "recommended_action": "wait"},
                    {"status": "not_ready", "current_phase": "await_result_arrival", "recommended_action": "wait"},
                    {"status": "partial", "current_phase": "future_only_readiness_track", "recommended_action": "wait"},
                ],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_readiness_cycle.py",
                    "--wrapper-manifest-output",
                    str(tmp_path / "wrapper.json"),
                    "--capture-loop-manifest-output",
                    str(tmp_path / "capture.json"),
                    "--watcher-manifest-output",
                    str(tmp_path / "watcher.json"),
                    "--bootstrap-manifest-output",
                    "artifacts/reports/custom_bootstrap_manifest.json",
                    "--status-board-output",
                    str(tmp_path / "board.json"),
                ],
            ):
                exit_code = readiness_cycle_script.main()
                wrapper_manifest = json.loads((tmp_path / "wrapper.json").read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(wrapper_manifest["execution_role"], "readiness_cycle_wrapper")
        self.assertEqual(wrapper_manifest["data_update_mode"], "capture_refresh_with_readiness")
        self.assertEqual(wrapper_manifest["execution_mode"], "single_cycle")
        self.assertEqual(wrapper_manifest["trigger_contract"], "direct_refresh_plus_readiness")
        capture_command = commands[0]
        watcher_command = commands[1]
        bootstrap_command = commands[2]
        status_board_command = commands[3]
        self.assertIn("--snapshot-dir", capture_command)
        self.assertIn(str(tmp_path / "capture_pre_race_capture_snapshots"), capture_command)
        self.assertIn("--probe-summary-output", watcher_command)
        self.assertIn(expected_probe_summary, watcher_command)
        self.assertIn("--log-prefix", bootstrap_command)
        self.assertIn("custom_bootstrap_manifest", bootstrap_command)
        self.assertIn("--bootstrap-revision", bootstrap_command)
        self.assertIn("custom_bootstrap_manifest", bootstrap_command)
        self.assertIn("--handoff-manifest-output", bootstrap_command)
        self.assertIn("artifacts/reports/custom_pre_race_benchmark_handoff.json", bootstrap_command)
        self.assertIn("--filtered-race-card-output", bootstrap_command)
        self.assertTrue(any(item.endswith("data/local_nankan_pre_race_ready/raw/custom_race_card.csv") for item in bootstrap_command))
        self.assertIn("--filtered-race-result-output", bootstrap_command)
        self.assertTrue(any(item.endswith("data/local_nankan_pre_race_ready/raw/custom_race_result.csv") for item in bootstrap_command))
        self.assertIn("--primary-output-file", bootstrap_command)
        self.assertTrue(any(item.endswith("data/local_nankan_pre_race_ready/raw/custom_primary.csv") for item in bootstrap_command))
        self.assertIn("--pre-race-summary-output", bootstrap_command)
        self.assertIn("artifacts/reports/custom_pre_race_ready_summary.json", bootstrap_command)
        self.assertIn("--primary-manifest-file", bootstrap_command)
        self.assertIn("artifacts/reports/custom_pre_race_ready_primary_materialize.json", bootstrap_command)
        self.assertIn("--benchmark-manifest-output", bootstrap_command)
        self.assertIn("artifacts/reports/custom_pre_race_ready_benchmark_gate.json", bootstrap_command)
        self.assertIn("--readiness-probe-summary", status_board_command)
        self.assertIn(expected_probe_summary, status_board_command)
        self.assertIn("--pre-race-handoff-manifest", status_board_command)
        self.assertIn("artifacts/reports/custom_pre_race_benchmark_handoff.json", status_board_command)

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
                bootstrap_handoff_script,
                "_run_command",
                return_value={
                    "label": "pre_race_benchmark_handoff",
                    "command": ["python", "handoff"],
                    "exit_code": 2,
                    "status": "failed",
                    "started_at": "2026-04-07T00:00:00Z",
                    "finished_at": "2026-04-07T00:00:01Z",
                    "log_file": "artifacts/logs/handoff.log",
                },
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
            self.assertEqual(manifest["handoff_command_result"]["log_file"], "artifacts/logs/handoff.log")
            self.assertEqual(manifest["runtime_configs"]["stack_config"], "artifacts/runtime_configs/stack.yaml")
            self.assertEqual(manifest["bootstrap_command_plan"][0]["label"], "train_win_component")
            self.assertIn("artifacts/logs/", manifest["bootstrap_command_plan"][0]["log_file"])

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
                bootstrap_handoff_script,
                "_run_command",
                return_value={
                    "label": "pre_race_benchmark_handoff",
                    "command": ["python", "handoff"],
                    "exit_code": 0,
                    "status": "completed",
                    "started_at": "2026-04-07T00:00:00Z",
                    "finished_at": "2026-04-07T00:00:01Z",
                    "log_file": "artifacts/logs/handoff.log",
                },
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
            self.assertEqual(manifest["handoff_command_result"]["log_file"], "artifacts/logs/handoff.log")
            self.assertIn("artifacts/logs/", manifest["bootstrap_command_plan"][0]["log_file"])

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
                bootstrap_handoff_script,
                "_run_command",
                side_effect=[
                    {
                        "label": "pre_race_benchmark_handoff",
                        "command": ["python", "handoff"],
                        "exit_code": 0,
                        "status": "completed",
                        "started_at": "2026-04-07T00:00:00Z",
                        "finished_at": "2026-04-07T00:00:01Z",
                        "log_file": "artifacts/logs/handoff.log",
                    },
                    {
                        "label": "train_win_component",
                        "command": ["python", "train_win"],
                        "exit_code": 0,
                        "status": "completed",
                        "started_at": "2026-04-07T00:00:02Z",
                        "finished_at": "2026-04-07T00:00:03Z",
                        "log_file": "artifacts/logs/train_win.log",
                    },
                    {
                        "label": "train_roi_component",
                        "command": ["python", "train_roi"],
                        "exit_code": 1,
                        "status": "failed",
                        "started_at": "2026-04-07T00:00:04Z",
                        "finished_at": "2026-04-07T00:00:05Z",
                        "log_file": "artifacts/logs/train_roi.log",
                    },
                ],
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
            self.assertEqual(manifest["bootstrap_runs"][1]["log_file"], "artifacts/logs/train_roi.log")

    def test_wait_then_cycle_runs_bootstrap_followup_when_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_output = tmp_path / "wait_then_cycle.json"
            log_output = tmp_path / "wait_then_cycle.log"

            def fake_read_json_dict(path: Path):
                name = path.name
                if name.endswith("readiness_cycle.json"):
                    return {"status": "partial", "current_phase": "future_only_readiness_track"}
                if name.endswith("status_board.json"):
                    return {
                        "status": "partial",
                        "current_phase": "bootstrap_pending",
                        "recommended_action": "run_bootstrap_command_plan",
                        "readiness": {"benchmark_rerun_ready": True},
                    }
                if name.endswith("pre_race_capture_loop.json"):
                    return {
                        "latest_summary": {
                            "pre_race_only_rows": 426,
                            "pre_race_only_races": 24,
                            "result_ready_races": 1,
                            "pending_result_races": 0,
                        }
                    }
                if name.endswith("bootstrap_handoff.json"):
                    return {
                        "status": "benchmark_ready",
                        "current_phase": "bootstrap_pending",
                        "recommended_action": "run_bootstrap_command_plan",
                    }
                if name.endswith("bootstrap_resume.json"):
                    return {
                        "status": "completed",
                        "current_phase": "bootstrap_completed",
                        "recommended_action": "review_bootstrap_revision_outputs",
                    }
                return {}

            with patch.object(
                wait_then_cycle_script,
                "_run_command",
                side_effect=[
                    {
                        "label": "cycle=1",
                        "command": ["python", "cycle"],
                        "exit_code": 0,
                        "status": "completed",
                        "started_at": "2026-04-07T00:00:00Z",
                        "finished_at": "2026-04-07T00:00:10Z",
                    },
                    {
                        "label": "bootstrap_resume_cycle=1",
                        "command": ["python", "resume"],
                        "exit_code": 0,
                        "status": "completed",
                        "started_at": "2026-04-07T00:00:11Z",
                        "finished_at": "2026-04-07T00:00:20Z",
                    },
                ],
            ) as run_command, patch.object(
                wait_then_cycle_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_wait_then_cycle.py",
                    "--manifest-output",
                    str(manifest_output),
                    "--log-file",
                    str(log_output),
                    "--artifact-prefix",
                    "unit_wait_then_cycle",
                    "--run-id",
                    "run01",
                    "--max-cycles",
                    "1",
                    "--run-bootstrap-on-ready",
                ],
            ):
                exit_code = wait_then_cycle_script.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_command.call_count, 2)
            manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["current_phase"], "bootstrap_completed")
            self.assertEqual(manifest["recommended_action"], "review_bootstrap_revision_outputs")
            self.assertEqual(manifest["stopped_reason"], "bootstrap_completed")
            self.assertEqual(manifest["run_id"], "run01")
            self.assertEqual(manifest["effective_artifact_prefix"], "unit_wait_then_cycle_run01")
            self.assertEqual(manifest["run_manifest_output"], str(manifest_output.with_name("wait_then_cycle_run01.json")))
            self.assertEqual(manifest["log_file"], str(log_output))
            self.assertEqual(manifest["monitor_state"], "completed")
            self.assertEqual(manifest["current_cycle_index"], 1)
            self.assertIsNone(manifest["next_cycle_index"])
            self.assertTrue(manifest["current_artifacts"]["watcher_manifest"].endswith("cycle_001_readiness_watcher.json"))
            self.assertEqual(manifest["latest_cycle"]["cycle"], 1)
            self.assertEqual(manifest["readiness_summary"]["cycle"], 1)
            self.assertEqual(manifest["readiness_summary"]["result_ready_races"], 1)
            run_manifest = json.loads((tmp_path / "wait_then_cycle_run01.json").read_text(encoding="utf-8"))
            self.assertEqual(run_manifest["run_id"], "run01")
            self.assertEqual(run_manifest["effective_artifact_prefix"], "unit_wait_then_cycle_run01")
            self.assertEqual(run_manifest["log_file"], str(log_output))
            self.assertEqual(run_manifest["latest_cycle"]["cycle"], 1)
            self.assertEqual(run_manifest["readiness_summary"]["bootstrap_status"], "benchmark_ready")
            self.assertEqual(manifest["cycles"][0]["bootstrap_status"], "benchmark_ready")
            self.assertEqual(manifest["cycles"][0]["bootstrap_followup"]["status"], "completed")
            readiness_command = run_command.call_args_list[0].kwargs["command"]
            followup_command = run_command.call_args_list[1].kwargs["command"]
            self.assertIn("--bootstrap-log-prefix", readiness_command)
            self.assertIn("unit_wait_then_cycle_run01_cycle_001_bootstrap_handoff", readiness_command)
            self.assertIn("--bootstrap-revision", readiness_command)
            self.assertIn("unit_wait_then_cycle_run01_cycle_001_bootstrap_handoff", readiness_command)
            self.assertIn("--log-prefix", followup_command)
            self.assertIn("unit_wait_then_cycle_run01_cycle_001_bootstrap_resume", followup_command)
            self.assertIn("--bootstrap-revision", followup_command)
            self.assertIn("unit_wait_then_cycle_run01_cycle_001_bootstrap_resume", followup_command)
            self.assertIn("--filtered-race-card-output", followup_command)
            self.assertIn("data/local_nankan_pre_race_ready/raw/unit_wait_then_cycle_run01_cycle_001_bootstrap_resume_race_card.csv", followup_command)
            self.assertIn("--filtered-race-result-output", followup_command)
            self.assertIn("data/local_nankan_pre_race_ready/raw/unit_wait_then_cycle_run01_cycle_001_bootstrap_resume_race_result.csv", followup_command)
            self.assertIn("--primary-output-file", followup_command)
            self.assertIn("data/local_nankan_pre_race_ready/raw/unit_wait_then_cycle_run01_cycle_001_bootstrap_resume_primary.csv", followup_command)

    def test_wait_then_cycle_updates_manifest_during_wait_interval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_output = tmp_path / "wait_then_cycle.json"
            log_output = tmp_path / "wait_then_cycle.log"
            operator_board_output = tmp_path / "operator_board.json"
            captured_payloads: list[dict[str, object]] = []
            original_write_wait_manifest = wait_then_cycle_script._write_wait_manifest

            def fake_read_json_dict(path: Path):
                name = path.name
                if name.endswith("readiness_cycle.json"):
                    return {"status": "partial", "current_phase": "future_only_readiness_track"}
                if name.endswith("status_board.json"):
                    return {
                        "status": "partial",
                        "current_phase": "future_only_readiness_track",
                        "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
                        "readiness": {"benchmark_rerun_ready": False},
                    }
                if name.endswith("pre_race_capture_loop.json"):
                    return {
                        "latest_summary": {
                            "pre_race_only_rows": 426,
                            "pre_race_only_races": 24,
                            "result_ready_races": 0,
                            "pending_result_races": 24,
                        }
                    }
                if name.endswith("bootstrap_handoff.json"):
                    return {
                        "status": "not_ready",
                        "current_phase": "await_result_arrival",
                        "recommended_action": "wait_for_result_ready_pre_race_races",
                    }
                return {}

            def capture_and_write(*, stable_manifest_path: Path, run_manifest_path: Path | None, payload: dict[str, object]) -> None:
                captured_payloads.append(json.loads(json.dumps(payload)))
                original_write_wait_manifest(
                    stable_manifest_path=stable_manifest_path,
                    run_manifest_path=run_manifest_path,
                    payload=payload,
                )

            with patch.object(
                wait_then_cycle_script,
                "_run_command",
                side_effect=[
                    {
                        "label": "cycle=1",
                        "command": ["python", "cycle1"],
                        "exit_code": 0,
                        "status": "completed",
                        "started_at": "2026-04-07T00:00:00Z",
                        "finished_at": "2026-04-07T00:00:10Z",
                    },
                    {
                        "label": "cycle=2",
                        "command": ["python", "cycle2"],
                        "exit_code": 0,
                        "status": "completed",
                        "started_at": "2026-04-07T00:00:11Z",
                        "finished_at": "2026-04-07T00:00:20Z",
                    },
                ],
            ) as run_command, patch.object(
                wait_then_cycle_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                wait_then_cycle_script,
                "_write_wait_manifest",
                side_effect=capture_and_write,
            ), patch.object(
                wait_then_cycle_script.time,
                "sleep",
                return_value=None,
            ) as sleep_mock, patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_wait_then_cycle.py",
                    "--manifest-output",
                    str(manifest_output),
                    "--log-file",
                    str(log_output),
                    "--artifact-prefix",
                    "unit_wait_then_cycle",
                    "--run-id",
                    "run02",
                    "--operator-board-output",
                    str(operator_board_output),
                    "--max-cycles",
                    "2",
                    "--wait-seconds",
                    "5",
                    "--wait-heartbeat-seconds",
                    "2",
                ],
            ):
                exit_code = wait_then_cycle_script.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_command.call_count, 2)
            self.assertEqual(sleep_mock.call_count, 3)
            wait_payloads = [payload for payload in captured_payloads if isinstance(payload.get("wait_state"), dict) and payload["wait_state"].get("active")]
            self.assertTrue(wait_payloads)
            self.assertEqual(wait_payloads[0]["monitor_state"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["monitor_phase"], "waiting_next_cycle")
            self.assertIsNone(wait_payloads[0]["finished_at"])
            self.assertIn("updated_at", wait_payloads[0])
            self.assertEqual(wait_payloads[0]["current_decision"]["monitor_state"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_decision"]["stop_reason"], "running")
            self.assertEqual(wait_payloads[0]["current_decision"]["pending_result_races"], 24)
            self.assertEqual(wait_payloads[0]["current_counts"]["pre_race_only_rows"], 426)
            self.assertEqual(wait_payloads[0]["current_counts"]["pending_result_races"], 24)
            self.assertTrue(wait_payloads[0]["current_flags"]["has_pre_race_only_rows"])
            self.assertTrue(wait_payloads[0]["current_flags"]["has_pending_result_races"])
            self.assertFalse(wait_payloads[0]["current_flags"]["has_result_ready_races"])
            self.assertTrue(wait_payloads[0]["current_flags"]["blocked_on_result_arrival"])
            self.assertFalse(wait_payloads[0]["current_flags"]["cycle_in_flight"])
            self.assertTrue(wait_payloads[0]["current_flags"]["wait_in_flight"])
            self.assertEqual(wait_payloads[0]["current_blockers"]["primary_code"], "result_arrival_pending")
            self.assertEqual(wait_payloads[0]["current_blockers"]["blocking_count"], 3)
            self.assertEqual(wait_payloads[0]["current_blockers"]["codes"], ["result_arrival_pending", "benchmark_rerun_not_ready", "bootstrap_not_ready"])
            self.assertEqual(wait_payloads[0]["current_blockers"]["details"][0]["surface"], "capture_loop")
            self.assertEqual(wait_payloads[0]["current_outcome"]["state"], "blocked")
            self.assertEqual(wait_payloads[0]["current_outcome"]["summary_code"], "result_arrival_pending")
            self.assertEqual(wait_payloads[0]["current_outcome"]["blocking_surface"], "capture_loop")
            self.assertEqual(wait_payloads[0]["current_outcome"]["monitor_state"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_refs"]["focus_surface"], "status_board")
            self.assertTrue(wait_payloads[0]["current_refs"]["focus_manifest"].endswith("cycle_001_status_board.json"))
            self.assertEqual(wait_payloads[0]["current_refs"]["blocking_surface"], "capture_loop")
            self.assertTrue(wait_payloads[0]["current_refs"]["blocking_manifest"].endswith("cycle_001_pre_race_capture_loop.json"))
            self.assertEqual(wait_payloads[0]["current_operator_card"]["headline"], "result arrival pending")
            self.assertEqual(wait_payloads[0]["current_operator_card"]["state"], "blocked")
            self.assertEqual(wait_payloads[0]["current_operator_card"]["focus_surface"], "status_board")
            self.assertEqual(wait_payloads[0]["current_operator_card"]["blocking_surface"], "capture_loop")
            self.assertEqual(wait_payloads[0]["current_surface_views"]["status_board"]["surface"], "status_board")
            self.assertEqual(wait_payloads[0]["current_surface_views"]["status_board"]["status"], "partial")
            self.assertTrue(wait_payloads[0]["current_surface_views"]["status_board"]["manifest"].endswith("cycle_001_status_board.json"))
            self.assertEqual(wait_payloads[0]["current_surface_views"]["capture_loop"]["summary"]["pending_result_races"], 24)
            self.assertEqual(wait_payloads[0]["current_statuses"]["monitor_state"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_statuses"]["readiness_cycle"], "partial")
            self.assertEqual(wait_payloads[0]["current_statuses"]["bootstrap_handoff"], "not_ready")
            self.assertEqual(wait_payloads[0]["current_phases"]["monitor_phase"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_phases"]["bootstrap_handoff"], "await_result_arrival")
            self.assertEqual(wait_payloads[0]["current_focus"]["current_surface"], "status_board")
            self.assertEqual(wait_payloads[0]["current_focus"]["current_phase"], "future_only_readiness_track")
            self.assertEqual(wait_payloads[0]["current_progress"]["completed_cycles"], 1)
            self.assertEqual(wait_payloads[0]["current_progress"]["remaining_cycles"], 1)
            self.assertEqual(wait_payloads[0]["current_progress"]["completion_percent"], 50)
            self.assertEqual(wait_payloads[0]["current_timing"]["mode"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_timing"]["elapsed_seconds"], 0)
            self.assertEqual(wait_payloads[0]["current_timing"]["seconds_remaining"], 5)
            self.assertEqual(wait_payloads[0]["current_runtime"]["monitor_state"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_runtime"]["mode"], "waiting_next_cycle")
            self.assertEqual(wait_payloads[0]["current_runtime"]["next_cycle"], 2)
            self.assertEqual(wait_payloads[0]["current_runtime"]["completion_percent"], 50)
            self.assertEqual(wait_payloads[0]["current_runtime"]["seconds_remaining"], 5)
            self.assertEqual(wait_payloads[0]["execution_role"], "readiness_supervisor")
            self.assertEqual(wait_payloads[0]["data_update_mode"], "readiness_recheck_only")
            self.assertEqual(wait_payloads[0]["execution_mode"], "bounded_wait_cycle")
            self.assertEqual(wait_payloads[0]["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(wait_payloads[0]["current_snapshot_meta"]["schema_version"], 1)
            self.assertEqual(wait_payloads[0]["current_snapshot_meta"]["execution_role"], "readiness_supervisor")
            self.assertEqual(wait_payloads[0]["current_snapshot_meta"]["data_update_mode"], "readiness_recheck_only")
            self.assertEqual(wait_payloads[0]["current_snapshot_meta"]["execution_mode"], "bounded_wait_cycle")
            self.assertEqual(wait_payloads[0]["current_snapshot_meta"]["trigger_contract"], "external_refresh_completed_only")
            self.assertIn("current_runtime", wait_payloads[0]["current_snapshot_meta"]["preferred_entrypoints"])
            self.assertEqual(wait_payloads[0]["current_snapshot"]["current_outcome"]["summary_code"], "result_arrival_pending")
            self.assertEqual(wait_payloads[0]["current_snapshot"]["current_runtime"]["seconds_remaining"], 5)
            self.assertEqual(wait_payloads[0]["current_snapshot"]["current_operator_card"]["headline"], "result arrival pending")
            self.assertEqual(wait_payloads[0]["current_snapshot"]["current_snapshot_meta"]["schema_version"], 1)
            self.assertEqual(wait_payloads[0]["current_cycle_index"], 1)
            self.assertEqual(wait_payloads[0]["next_cycle_index"], 2)
            self.assertTrue(wait_payloads[0]["current_artifacts"]["bootstrap_manifest"].endswith("cycle_001_bootstrap_handoff.json"))
            self.assertEqual(wait_payloads[0]["current_readiness_summary"]["cycle"], 1)
            self.assertEqual(wait_payloads[0]["current_readiness_summary"]["pending_result_races"], 24)
            self.assertEqual(wait_payloads[0]["current_surface_summaries"]["readiness_cycle"]["status"], "partial")
            self.assertEqual(wait_payloads[0]["current_surface_summaries"]["readiness_cycle"]["elapsed_seconds"], 10)
            self.assertEqual(wait_payloads[0]["current_surface_summaries"]["capture_loop"]["pending_result_races"], 24)
            self.assertEqual(wait_payloads[0]["wait_state"]["seconds_total"], 5)
            self.assertEqual(wait_payloads[0]["wait_state"]["seconds_remaining"], 5)
            self.assertEqual(wait_payloads[0]["wait_state"]["heartbeat_seconds"], 2)
            self.assertEqual(wait_payloads[0]["wait_state"]["next_cycle"], 2)
            final_manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertNotIn("wait_state", final_manifest)
            self.assertEqual(final_manifest["completed_cycles"], 2)
            self.assertEqual(final_manifest["monitor_phase"], "completed")
            self.assertEqual(final_manifest["current_decision"]["monitor_state"], "completed")
            self.assertEqual(final_manifest["current_decision"]["stop_reason"], "max_cycles_reached")
            self.assertEqual(final_manifest["current_counts"]["pre_race_only_races"], 24)
            self.assertTrue(final_manifest["current_flags"]["has_pre_race_only_races"])
            self.assertTrue(final_manifest["current_flags"]["has_pending_result_races"])
            self.assertTrue(final_manifest["current_flags"]["blocked_on_result_arrival"])
            self.assertFalse(final_manifest["current_flags"]["cycle_in_flight"])
            self.assertFalse(final_manifest["current_flags"]["wait_in_flight"])
            self.assertEqual(final_manifest["current_blockers"]["primary_code"], "result_arrival_pending")
            operator_board = json.loads(operator_board_output.read_text(encoding="utf-8"))
            self.assertEqual(operator_board["readiness_surfaces"]["readiness_supervisor"]["monitor_state"], "completed")
            self.assertEqual(operator_board["readiness_surfaces"]["readiness_supervisor"]["completed_cycles"], 2)
            self.assertEqual(operator_board["operator_runtime"]["monitor_state"], "completed")
            self.assertTrue(operator_board["artifacts"]["readiness_supervisor_manifest"].endswith("wait_then_cycle_run02.json"))
            self.assertIn("supervisor_monitor_state=completed", operator_board["highlights"])

    def test_build_operator_board_payload_surfaces_live_supervisor_state(self) -> None:
        board_payload = {
            "status": "partial",
            "current_phase": "future_only_readiness_track",
            "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
            "artifacts": {
                "status_board": "artifacts/reports/cycle_001_status_board.json",
            },
            "readiness_surfaces": {
                "readiness_probe": {
                    "status": "not_ready",
                    "pending_result_races": 24,
                },
            },
            "highlights": ["probe_status=not_ready"],
        }
        wait_payload = {
            "status": "partial",
            "monitor_state": "waiting_next_cycle",
            "monitor_phase": "waiting_next_cycle",
            "stopped_reason": "running",
            "completed_cycles": 2,
            "max_cycles": 3,
            "current_cycle_index": 2,
            "next_cycle_index": 3,
            "wait_state": {
                "seconds_remaining": 1200,
            },
            "cycle_state": None,
            "current_runtime": {
                "monitor_state": "waiting_next_cycle",
                "seconds_remaining": 1200,
            },
            "current_timing": {
                "mode": "waiting_next_cycle",
                "updated_at": "2026-04-11T07:47:41Z",
            },
            "current_progress": {
                "completed_cycles": 2,
                "completion_percent": 66,
            },
            "current_outcome": {
                "state": "blocked",
                "summary_code": "result_arrival_pending",
            },
            "current_operator_card": {
                "headline": "result arrival pending",
            },
            "current_refs": {
                "focus_manifest": "artifacts/reports/cycle_002_status_board.json",
            },
            "started_at": "2026-04-11T06:04:34Z",
            "updated_at": "2026-04-11T07:47:41Z",
            "finished_at": None,
        }

        operator_board = wait_then_cycle_script._build_operator_board_payload(
            board_payload=board_payload,
            wait_payload=wait_payload,
            status_board_manifest="artifacts/reports/cycle_002_status_board.json",
            stable_manifest_path=Path("artifacts/reports/wait_then_cycle.json"),
            run_manifest_path=Path("artifacts/reports/wait_then_cycle_live.json"),
        )

        self.assertEqual(operator_board["readiness_surfaces"]["readiness_supervisor"]["monitor_state"], "waiting_next_cycle")
        self.assertEqual(operator_board["readiness_surfaces"]["readiness_supervisor"]["current_runtime"]["seconds_remaining"], 1200)
        self.assertEqual(operator_board["operator_runtime"]["monitor_phase"], "waiting_next_cycle")
        self.assertEqual(operator_board["artifacts"]["live_status_board_source"], "artifacts/reports/cycle_002_status_board.json")
        self.assertEqual(operator_board["artifacts"]["readiness_supervisor_manifest"], "artifacts/reports/wait_then_cycle_live.json")
        self.assertIn("supervisor_monitor_state=waiting_next_cycle", operator_board["highlights"])

    def test_wait_then_cycle_oneshot_skips_idle_wait(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_output = tmp_path / "wait_then_cycle.json"
            log_output = tmp_path / "wait_then_cycle.log"

            def fake_read_json_dict(path: Path):
                name = path.name
                if name.endswith("readiness_cycle.json"):
                    return {
                        "status": "partial",
                        "current_phase": "future_only_readiness_track",
                        "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
                    }
                if name.endswith("status_board.json"):
                    return {
                        "status": "partial",
                        "current_phase": "future_only_readiness_track",
                        "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
                        "readiness": {"benchmark_rerun_ready": False},
                    }
                if name.endswith("pre_race_capture_loop.json"):
                    return {
                        "latest_summary": {
                            "pre_race_only_rows": 426,
                            "pre_race_only_races": 24,
                            "result_ready_races": 0,
                            "pending_result_races": 24,
                        }
                    }
                if name.endswith("bootstrap_handoff.json"):
                    return {
                        "status": "not_ready",
                        "current_phase": "await_result_arrival",
                        "recommended_action": "wait_for_result_ready_pre_race_races",
                    }
                return {}

            with patch.object(
                wait_then_cycle_script,
                "_run_command",
                return_value={
                    "label": "cycle=1",
                    "command": ["python", "cycle1"],
                    "exit_code": 0,
                    "status": "completed",
                    "started_at": "2026-04-07T00:00:00Z",
                    "finished_at": "2026-04-07T00:00:10Z",
                },
            ) as run_command, patch.object(
                wait_then_cycle_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                wait_then_cycle_script.time,
                "sleep",
                return_value=None,
            ) as sleep_mock, patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_wait_then_cycle.py",
                    "--manifest-output",
                    str(manifest_output),
                    "--log-file",
                    str(log_output),
                    "--artifact-prefix",
                    "unit_wait_then_cycle",
                    "--run-id",
                    "run03",
                    "--max-cycles",
                    "8",
                    "--wait-seconds",
                    "1800",
                    "--oneshot",
                ],
            ):
                exit_code = wait_then_cycle_script.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_command.call_count, 1)
            self.assertEqual(sleep_mock.call_count, 0)
            manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(manifest["completed_cycles"], 1)
            self.assertEqual(manifest["monitor_state"], "completed")
            self.assertEqual(manifest["monitor_phase"], "completed")
            self.assertNotIn("wait_state", manifest)
            self.assertIsNone(manifest["next_cycle_index"])
            self.assertEqual(manifest["execution_role"], "readiness_supervisor")
            self.assertEqual(manifest["data_update_mode"], "readiness_recheck_only")
            self.assertEqual(manifest["execution_mode"], "oneshot")
            self.assertEqual(manifest["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(manifest["current_progress"]["completed_cycles"], 1)
            self.assertEqual(manifest["current_progress"]["remaining_cycles"], 0)
            self.assertEqual(manifest["current_snapshot_meta"]["execution_mode"], "oneshot")
            self.assertEqual(manifest["current_snapshot_meta"]["trigger_contract"], "external_refresh_completed_only")

    def test_wait_then_cycle_updates_manifest_during_active_cycle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_output = tmp_path / "wait_then_cycle.json"
            log_output = tmp_path / "wait_then_cycle.log"
            captured_payloads: list[dict[str, object]] = []
            original_write_wait_manifest = wait_then_cycle_script._write_wait_manifest

            def fake_read_json_dict(path: Path):
                name = path.name
                if name.endswith("readiness_cycle.json"):
                    return {"status": "partial", "current_phase": "future_only_readiness_track"}
                if name.endswith("status_board.json"):
                    return {
                        "status": "partial",
                        "current_phase": "future_only_readiness_track",
                        "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
                        "readiness": {"benchmark_rerun_ready": False},
                    }
                if name.endswith("pre_race_capture_loop.json"):
                    return {
                        "latest_summary": {
                            "pre_race_only_rows": 426,
                            "pre_race_only_races": 24,
                            "result_ready_races": 0,
                            "pending_result_races": 24,
                        }
                    }
                if name.endswith("bootstrap_handoff.json"):
                    return {
                        "status": "not_ready",
                        "current_phase": "await_result_arrival",
                        "recommended_action": "wait_for_result_ready_pre_race_races",
                    }
                return {}

            def capture_and_write(*, stable_manifest_path: Path, run_manifest_path: Path | None, payload: dict[str, object]) -> None:
                captured_payloads.append(json.loads(json.dumps(payload)))
                original_write_wait_manifest(
                    stable_manifest_path=stable_manifest_path,
                    run_manifest_path=run_manifest_path,
                    payload=payload,
                )

            def fake_run_command(*, label: str, command: list[str], heartbeat_seconds: int = 10, on_start=None, on_tick=None):
                if on_start is not None:
                    on_start("2026-04-07T00:00:00Z")
                if on_tick is not None:
                    on_tick("2026-04-07T00:00:00Z", 10)
                return {
                    "label": label,
                    "command": command,
                    "exit_code": 0,
                    "status": "completed",
                    "started_at": "2026-04-07T00:00:00Z",
                    "finished_at": "2026-04-07T00:00:20Z",
                }

            with patch.object(
                wait_then_cycle_script,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                wait_then_cycle_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                wait_then_cycle_script,
                "_write_wait_manifest",
                side_effect=capture_and_write,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_wait_then_cycle.py",
                    "--manifest-output",
                    str(manifest_output),
                    "--log-file",
                    str(log_output),
                    "--artifact-prefix",
                    "unit_wait_then_cycle",
                    "--run-id",
                    "run03",
                    "--max-cycles",
                    "1",
                ],
            ):
                exit_code = wait_then_cycle_script.main()

            self.assertEqual(exit_code, 0)
            cycle_payloads = [payload for payload in captured_payloads if isinstance(payload.get("cycle_state"), dict) and payload["cycle_state"].get("active")]
            self.assertTrue(cycle_payloads)
            self.assertEqual(cycle_payloads[0]["monitor_state"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["monitor_phase"], "readiness_cycle")
            self.assertEqual(cycle_payloads[0]["current_decision"]["monitor_state"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["current_decision"]["stop_reason"], "running")
            self.assertEqual(cycle_payloads[0]["current_decision"]["pending_result_races"], 24)
            self.assertEqual(cycle_payloads[0]["current_counts"]["pre_race_only_races"], 24)
            self.assertEqual(cycle_payloads[0]["current_counts"]["result_ready_races"], 0)
            self.assertTrue(cycle_payloads[0]["current_flags"]["has_pre_race_only_races"])
            self.assertTrue(cycle_payloads[0]["current_flags"]["has_pending_result_races"])
            self.assertFalse(cycle_payloads[0]["current_flags"]["has_result_ready_races"])
            self.assertTrue(cycle_payloads[0]["current_flags"]["blocked_on_result_arrival"])
            self.assertTrue(cycle_payloads[0]["current_flags"]["cycle_in_flight"])
            self.assertFalse(cycle_payloads[0]["current_flags"]["wait_in_flight"])
            self.assertEqual(cycle_payloads[0]["current_blockers"]["primary_code"], "result_arrival_pending")
            self.assertEqual(cycle_payloads[0]["current_blockers"]["blocking_count"], 3)
            self.assertEqual(cycle_payloads[0]["current_blockers"]["details"][2]["surface"], "bootstrap_handoff")
            self.assertEqual(cycle_payloads[0]["current_outcome"]["state"], "blocked")
            self.assertEqual(cycle_payloads[0]["current_outcome"]["summary_code"], "result_arrival_pending")
            self.assertEqual(cycle_payloads[0]["current_outcome"]["current_surface"], "status_board")
            self.assertEqual(cycle_payloads[0]["current_refs"]["focus_surface"], "status_board")
            self.assertTrue(cycle_payloads[0]["current_refs"]["focus_manifest"].endswith("cycle_001_status_board.json"))
            self.assertEqual(cycle_payloads[0]["current_refs"]["blocking_surface"], "capture_loop")
            self.assertTrue(cycle_payloads[0]["current_refs"]["blocking_manifest"].endswith("cycle_001_pre_race_capture_loop.json"))
            self.assertEqual(cycle_payloads[0]["current_operator_card"]["headline"], "result arrival pending")
            self.assertEqual(cycle_payloads[0]["current_operator_card"]["recommended_action"], "capture_future_pre_race_rows_and_wait_for_results")
            self.assertTrue(cycle_payloads[0]["current_operator_card"]["focus_manifest"].endswith("cycle_001_status_board.json"))
            self.assertEqual(cycle_payloads[0]["current_surface_views"]["status_board"]["current_phase"], "future_only_readiness_track")
            self.assertTrue(cycle_payloads[0]["current_surface_views"]["readiness_cycle"]["manifest"].endswith("cycle_001_readiness_cycle.json"))
            self.assertEqual(cycle_payloads[0]["current_statuses"]["monitor_state"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["current_statuses"]["capture_loop"], None)
            self.assertEqual(cycle_payloads[0]["current_phases"]["monitor_phase"], "readiness_cycle")
            self.assertEqual(cycle_payloads[0]["current_phases"]["status_board"], "future_only_readiness_track")
            self.assertEqual(cycle_payloads[0]["current_focus"]["current_surface"], "status_board")
            self.assertEqual(cycle_payloads[0]["current_focus"]["recommended_action"], "capture_future_pre_race_rows_and_wait_for_results")
            self.assertEqual(cycle_payloads[0]["current_progress"]["completed_cycles"], 0)
            self.assertEqual(cycle_payloads[0]["current_progress"]["in_flight_cycle"], 1)
            self.assertEqual(cycle_payloads[0]["current_progress"]["completion_percent"], 0)
            self.assertEqual(cycle_payloads[0]["current_timing"]["mode"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["current_timing"]["elapsed_seconds"], 0)
            self.assertEqual(cycle_payloads[0]["current_runtime"]["monitor_state"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["current_runtime"]["in_flight_cycle"], 1)
            self.assertEqual(cycle_payloads[0]["current_runtime"]["completion_percent"], 0)
            self.assertEqual(cycle_payloads[0]["execution_role"], "readiness_supervisor")
            self.assertEqual(cycle_payloads[0]["data_update_mode"], "readiness_recheck_only")
            self.assertEqual(cycle_payloads[0]["execution_mode"], "bounded_wait_cycle")
            self.assertEqual(cycle_payloads[0]["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(cycle_payloads[0]["current_snapshot_meta"]["schema_version"], 1)
            self.assertEqual(cycle_payloads[0]["current_snapshot_meta"]["execution_mode"], "bounded_wait_cycle")
            self.assertEqual(cycle_payloads[0]["current_snapshot_meta"]["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(cycle_payloads[0]["current_snapshot"]["current_surface_views"]["status_board"]["surface"], "status_board")
            self.assertEqual(cycle_payloads[0]["current_snapshot"]["current_runtime"]["in_flight_cycle"], 1)
            self.assertEqual(cycle_payloads[0]["current_cycle_index"], 1)
            self.assertEqual(cycle_payloads[0]["next_cycle_index"], 2)
            self.assertTrue(cycle_payloads[0]["current_artifacts"]["wrapper_manifest"].endswith("cycle_001_readiness_cycle.json"))
            self.assertEqual(cycle_payloads[0]["cycle_state"]["current_cycle"], 1)
            self.assertEqual(cycle_payloads[0]["cycle_state"]["stage"], "readiness_cycle")
            self.assertEqual(cycle_payloads[0]["cycle_state"]["elapsed_seconds"], 0)
            self.assertIn("artifacts", cycle_payloads[0]["cycle_state"])
            self.assertIn("surface_summaries", cycle_payloads[0]["cycle_state"])
            self.assertEqual(cycle_payloads[0]["cycle_state"]["surface_summaries"]["readiness_cycle"]["status"], "partial")
            self.assertEqual(cycle_payloads[0]["cycle_state"]["surface_summaries"]["status_board"]["status"], "partial")
            self.assertEqual(cycle_payloads[0]["cycle_state"]["surface_summaries"]["capture_loop"]["pre_race_only_races"], 24)
            self.assertEqual(cycle_payloads[0]["cycle_state"]["surface_summaries"]["capture_loop"]["pending_result_races"], 24)
            self.assertEqual(cycle_payloads[1]["current_statuses"]["capture_loop"], None)
            self.assertEqual(cycle_payloads[0]["active_readiness_summary"]["cycle"], 1)
            self.assertEqual(cycle_payloads[0]["active_readiness_summary"]["current_surface"], "status_board")
            self.assertEqual(cycle_payloads[0]["active_readiness_summary"]["pre_race_only_races"], 24)
            self.assertEqual(cycle_payloads[0]["active_readiness_summary"]["pending_result_races"], 24)
            self.assertEqual(cycle_payloads[0]["current_readiness_summary"]["cycle"], 1)
            self.assertEqual(cycle_payloads[0]["current_readiness_summary"]["current_surface"], "status_board")
            self.assertEqual(cycle_payloads[0]["current_surface_summaries"]["status_board"]["status"], "partial")
            self.assertTrue(cycle_payloads[0]["cycle_state"]["artifacts"]["watcher_manifest"].endswith("cycle_001_readiness_watcher.json"))
            self.assertTrue(cycle_payloads[0]["cycle_state"]["artifacts"]["bootstrap_manifest"].endswith("cycle_001_bootstrap_handoff.json"))
            self.assertEqual(cycle_payloads[1]["cycle_state"]["elapsed_seconds"], 10)
            final_manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertNotIn("cycle_state", final_manifest)
            self.assertNotIn("active_readiness_summary", final_manifest)
            self.assertEqual(final_manifest["monitor_phase"], "completed")
            self.assertEqual(final_manifest["current_decision"]["monitor_state"], "completed")
            self.assertEqual(final_manifest["current_decision"]["stop_reason"], "max_cycles_reached")
            self.assertEqual(final_manifest["current_counts"]["pending_result_races"], 24)
            self.assertTrue(final_manifest["current_flags"]["has_pending_result_races"])
            self.assertTrue(final_manifest["current_flags"]["blocked_on_result_arrival"])
            self.assertFalse(final_manifest["current_flags"]["cycle_in_flight"])
            self.assertFalse(final_manifest["current_flags"]["wait_in_flight"])
            self.assertEqual(final_manifest["current_blockers"]["primary_code"], "result_arrival_pending")
            self.assertEqual(final_manifest["current_blockers"]["codes"], ["result_arrival_pending", "benchmark_rerun_not_ready", "bootstrap_not_ready"])
            self.assertEqual(final_manifest["current_outcome"]["state"], "blocked")
            self.assertEqual(final_manifest["current_outcome"]["blocking_surface"], "capture_loop")
            self.assertEqual(final_manifest["current_refs"]["focus_surface"], "status_board")
            self.assertTrue(final_manifest["current_refs"]["focus_manifest"].endswith("cycle_001_status_board.json"))
            self.assertTrue(final_manifest["current_refs"]["blocking_manifest"].endswith("cycle_001_pre_race_capture_loop.json"))
            self.assertEqual(final_manifest["current_operator_card"]["headline"], "result arrival pending")
            self.assertEqual(final_manifest["current_operator_card"]["state"], "blocked")
            self.assertEqual(final_manifest["current_operator_card"]["blocking_surface"], "capture_loop")
            self.assertEqual(final_manifest["current_surface_views"]["bootstrap_handoff"]["status"], "not_ready")
            self.assertTrue(final_manifest["current_surface_views"]["status_board"]["manifest"].endswith("cycle_001_status_board.json"))
            self.assertEqual(final_manifest["current_statuses"]["bootstrap_handoff"], "not_ready")
            self.assertEqual(final_manifest["current_phases"]["bootstrap_handoff"], "await_result_arrival")
            self.assertEqual(final_manifest["current_focus"]["current_surface"], "status_board")
            self.assertEqual(final_manifest["current_progress"]["completed_cycles"], 1)
            self.assertEqual(final_manifest["current_progress"]["remaining_cycles"], 0)
            self.assertEqual(final_manifest["current_timing"]["mode"], "completed")
            self.assertEqual(final_manifest["current_timing"]["elapsed_seconds"], 20)
            self.assertEqual(final_manifest["current_runtime"]["monitor_state"], "completed")
            self.assertEqual(final_manifest["current_runtime"]["completed_cycles"], 1)
            self.assertEqual(final_manifest["current_runtime"]["remaining_cycles"], 0)
            self.assertEqual(final_manifest["execution_role"], "readiness_supervisor")
            self.assertEqual(final_manifest["data_update_mode"], "readiness_recheck_only")
            self.assertEqual(final_manifest["execution_mode"], "bounded_wait_cycle")
            self.assertEqual(final_manifest["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(final_manifest["current_snapshot_meta"]["schema_version"], 1)
            self.assertEqual(final_manifest["current_snapshot_meta"]["execution_mode"], "bounded_wait_cycle")
            self.assertEqual(final_manifest["current_snapshot_meta"]["trigger_contract"], "external_refresh_completed_only")
            self.assertEqual(final_manifest["current_snapshot"]["current_operator_card"]["state"], "blocked")
            self.assertEqual(final_manifest["current_snapshot"]["current_surface_views"]["bootstrap_handoff"]["status"], "not_ready")
            self.assertEqual(final_manifest["latest_cycle"]["cycle"], 1)
            self.assertEqual(final_manifest["readiness_summary"]["cycle"], 1)
            self.assertEqual(final_manifest["current_readiness_summary"]["cycle"], 1)
            self.assertEqual(final_manifest["current_surface_summaries"]["capture_loop"]["pending_result_races"], 24)
            self.assertEqual(final_manifest["current_surface_summaries"]["readiness_cycle"]["elapsed_seconds"], 20)
            self.assertEqual(final_manifest["readiness_summary"]["artifacts"]["watcher_manifest"], final_manifest["cycles"][0]["watcher_manifest"])

    def test_wait_then_cycle_keeps_active_blockers_provisional_until_surfaces_observed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_output = tmp_path / "wait_then_cycle.json"
            log_output = tmp_path / "wait_then_cycle.log"
            captured_payloads: list[dict[str, object]] = []
            original_write_wait_manifest = wait_then_cycle_script._write_wait_manifest

            def fake_read_json_dict(path: Path):
                if path.name.endswith("readiness_cycle.json"):
                    return {"status": "running", "current_phase": "readiness_cycle"}
                return {}

            def capture_and_write(*, stable_manifest_path: Path, run_manifest_path: Path | None, payload: dict[str, object]) -> None:
                captured_payloads.append(json.loads(json.dumps(payload)))
                original_write_wait_manifest(
                    stable_manifest_path=stable_manifest_path,
                    run_manifest_path=run_manifest_path,
                    payload=payload,
                )

            def fake_run_command(*, label: str, command: list[str], heartbeat_seconds: int = 10, on_start=None, on_tick=None):
                if on_start is not None:
                    on_start("2026-04-07T00:00:00Z")
                if on_tick is not None:
                    on_tick("2026-04-07T00:00:00Z", 10)
                    on_tick("2026-04-07T00:00:00Z", 30)
                return {
                    "label": label,
                    "command": command,
                    "exit_code": 0,
                    "status": "completed",
                    "started_at": "2026-04-07T00:00:00Z",
                    "finished_at": "2026-04-07T00:00:40Z",
                }

            with patch.object(
                wait_then_cycle_script,
                "_run_command",
                side_effect=fake_run_command,
            ), patch.object(
                wait_then_cycle_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                wait_then_cycle_script,
                "_write_wait_manifest",
                side_effect=capture_and_write,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_wait_then_cycle.py",
                    "--manifest-output",
                    str(manifest_output),
                    "--log-file",
                    str(log_output),
                    "--artifact-prefix",
                    "unit_wait_then_cycle_sparse_active",
                    "--run-id",
                    "run04",
                    "--max-cycles",
                    "1",
                ],
            ):
                exit_code = wait_then_cycle_script.main()

            self.assertEqual(exit_code, 0)
            cycle_payloads = [payload for payload in captured_payloads if isinstance(payload.get("cycle_state"), dict) and payload["cycle_state"].get("active")]
            self.assertTrue(cycle_payloads)
            self.assertEqual(cycle_payloads[0]["current_focus"]["current_surface"], "readiness_cycle")
            self.assertIsNone(cycle_payloads[0]["finished_at"])
            self.assertIn("updated_at", cycle_payloads[0])
            self.assertEqual(cycle_payloads[0]["current_blockers"]["primary_code"], None)
            self.assertEqual(cycle_payloads[0]["current_blockers"]["blocking_count"], 0)
            self.assertFalse(cycle_payloads[0]["current_blockers"]["observed_surfaces"]["status_board"])
            self.assertFalse(cycle_payloads[0]["current_blockers"]["observed_surfaces"]["bootstrap_handoff"])
            self.assertEqual(cycle_payloads[0]["current_outcome"]["state"], "monitoring")
            self.assertEqual(cycle_payloads[0]["current_outcome"]["summary_code"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["current_outcome"]["blocking_surface"], None)
            self.assertEqual(cycle_payloads[0]["current_refs"]["focus_surface"], "readiness_cycle")
            self.assertTrue(cycle_payloads[0]["current_refs"]["focus_manifest"].endswith("cycle_001_readiness_cycle.json"))
            self.assertEqual(cycle_payloads[0]["current_refs"]["blocking_manifest"], None)
            self.assertEqual(cycle_payloads[0]["current_operator_card"]["headline"], "running_cycle")
            self.assertEqual(cycle_payloads[0]["current_surface_views"]["status_board"]["status"], None)
            self.assertEqual(cycle_payloads[0]["current_snapshot"]["current_outcome"]["state"], "monitoring")
            final_manifest = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(final_manifest["current_readiness_summary"]["artifacts"]["watcher_manifest"], final_manifest["cycles"][0]["watcher_manifest"])
            self.assertTrue(final_manifest["cycles"][0]["watcher_manifest"].endswith("cycle_001_readiness_watcher.json"))
            self.assertTrue(final_manifest["cycles"][0]["bootstrap_manifest"].endswith("cycle_001_bootstrap_handoff.json"))


if __name__ == "__main__":
    unittest.main()
