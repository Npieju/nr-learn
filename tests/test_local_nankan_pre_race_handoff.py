from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


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


pre_race_handoff_script = _load_script_module(
    "test_run_local_nankan_pre_race_benchmark_handoff",
    "scripts/run_local_nankan_pre_race_benchmark_handoff.py",
)


class LocalNankanPreRaceHandoffTest(unittest.TestCase):
    def test_pre_race_handoff_defaults_use_pre_race_ready_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            captured_commands: list[list[str]] = []

            def fake_run_command(*, label: str, command: list[str]) -> int:
                captured_commands.append(list(command))
                return 2

            with patch.object(pre_race_handoff_script, "_run_command", side_effect=fake_run_command), patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                return_value={
                    "status": "not_ready",
                    "current_phase": "await_result_arrival",
                    "recommended_action": "wait_for_result_ready_pre_race_races",
                    "materialization_summary": {"pending_result_races": 24},
                },
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 2)
            self.assertTrue(captured_commands)
            pre_race_command = captured_commands[0]
            self.assertIn("configs/data_local_nankan_pre_race_ready.yaml", pre_race_command)
            self.assertIn("data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv", pre_race_command)
            self.assertIn("data/local_nankan_pre_race_ready/raw/local_nankan_primary_pre_race_ready.csv", pre_race_command)

    def test_pre_race_handoff_returns_not_ready_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            with patch.object(pre_race_handoff_script, "_run_command", return_value=2), patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                return_value={
                    "status": "not_ready",
                    "current_phase": "await_result_arrival",
                    "recommended_action": "wait_for_result_ready_pre_race_races",
                    "materialization_summary": {"pending_result_races": 24},
                },
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 2)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "not_ready")
            self.assertEqual(manifest["current_phase"], "await_result_arrival")
            self.assertEqual(manifest["recommended_action"], "wait_for_result_ready_pre_race_races")
            self.assertEqual(manifest["read_order"][0], "status")
            self.assertEqual(manifest["read_order"][3], "pre_race_summary.status")
            self.assertEqual(manifest["read_order"][5], "pre_race_summary.recommended_action")
            self.assertEqual(manifest["attempts"], 1)
            self.assertTrue(manifest["timed_out"])
            self.assertEqual(manifest["benchmark_manifest_output"], str(benchmark_manifest))

    def test_pre_race_handoff_not_ready_manifest_normalizes_workspace_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            with patch.object(pre_race_handoff_script, "_run_command", return_value=2), patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                return_value={
                    "status": "not_ready",
                    "current_phase": "await_result_arrival",
                    "recommended_action": "wait_for_result_ready_pre_race_races",
                    "filtered_race_card_output": str(ROOT / "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_card.csv"),
                    "filtered_race_result_output": str(ROOT / "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_result.csv"),
                    "primary_output_file": str(ROOT / "data/local_nankan_pre_race_ready/raw/unit_not_ready_primary.csv"),
                    "manifest_file": str(tmp_path / "unit_not_ready_manifest.json"),
                    "materialization_summary": {"pending_result_races": 24},
                },
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 2)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["pre_race_summary"]["filtered_race_card_output"], "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_card.csv")
            self.assertEqual(manifest["pre_race_summary"]["filtered_race_result_output"], "data/local_nankan_pre_race_ready/raw/unit_not_ready_race_result.csv")
            self.assertEqual(manifest["pre_race_summary"]["primary_output_file"], "data/local_nankan_pre_race_ready/raw/unit_not_ready_primary.csv")
            self.assertEqual(manifest["pre_race_summary"]["manifest_file"], str((tmp_path / "unit_not_ready_manifest.json").relative_to(ROOT)))
            self.assertEqual(manifest["benchmark_manifest_output"], str(benchmark_manifest.relative_to(ROOT)))

    def test_pre_race_handoff_completes_when_pre_race_and_benchmark_succeed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            with patch.object(pre_race_handoff_script, "_run_command", side_effect=[0, 0]) as run_command, patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                side_effect=[
                    {
                        "status": "ready",
                        "current_phase": "ready_for_benchmark_handoff",
                        "recommended_action": "run_pre_race_benchmark_handoff",
                    },
                    {"status": "completed", "revision": "r_test"},
                ],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_command.call_count, 2)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["current_phase"], "benchmark_gate")
            self.assertEqual(manifest["recommended_action"], "review_benchmark_manifest")
            self.assertEqual(manifest["read_order"][0], "status")
            self.assertEqual(manifest["read_order"][3], "pre_race_summary.status")
            self.assertEqual(manifest["read_order"][5], "benchmark_manifest.status")
            self.assertEqual(manifest["pre_race_summary"]["status"], "ready")
            self.assertEqual(manifest["benchmark_manifest"]["status"], "completed")

    def test_pre_race_handoff_reports_failed_benchmark_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            with patch.object(pre_race_handoff_script, "_run_command", side_effect=[0, 1]), patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                side_effect=[
                    {
                        "status": "ready",
                        "current_phase": "ready_for_benchmark_handoff",
                        "recommended_action": "run_pre_race_benchmark_handoff",
                    },
                    {"status": "failed", "current_phase": "evaluate"},
                ],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 1)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_phase"], "benchmark_gate")
            self.assertEqual(manifest["recommended_action"], "inspect_benchmark_manifest")
            self.assertEqual(manifest["read_order"][0], "status")
            self.assertEqual(manifest["read_order"][3], "pre_race_summary.status")
            self.assertEqual(manifest["read_order"][5], "benchmark_manifest.status")
            self.assertEqual(manifest["benchmark_manifest"]["status"], "failed")

    def test_pre_race_handoff_reports_failed_pre_race_primary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            with patch.object(pre_race_handoff_script, "_run_command", return_value=1), patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                return_value={
                    "status": "failed",
                    "current_phase": "materialize_pre_race_subset",
                    "recommended_action": "inspect_pre_race_primary_summary",
                },
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 1)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_phase"], "pre_race_primary")
            self.assertEqual(manifest["recommended_action"], "inspect_pre_race_primary_summary")
            self.assertEqual(manifest["read_order"][0], "status")
            self.assertEqual(manifest["read_order"][3], "pre_race_summary.status")
            self.assertEqual(manifest["read_order"][5], "pre_race_summary.recommended_action")
            self.assertEqual(manifest["pre_race_summary"]["status"], "failed")

    def test_pre_race_handoff_normalizes_nested_workspace_paths_in_manifest(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_nankan_pre_race_benchmark_handoff_manifest.json"
            pre_race_summary = tmp_path / "local_nankan_pre_race_ready_summary.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_pre_race_ready.json"
            source_timing_summary = tmp_path / "local_nankan_source_timing_audit.json"

            with patch.object(pre_race_handoff_script, "_run_command", side_effect=[0, 0]), patch.object(
                pre_race_handoff_script,
                "_read_json_dict",
                side_effect=[
                    {
                        "status": "ready",
                        "current_phase": "ready_for_benchmark_handoff",
                        "filtered_race_card_output": str(ROOT / "data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv"),
                        "materialization_summary": {
                            "manifest_file": str(ROOT / "artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json"),
                        },
                    },
                    {
                        "status": "completed",
                        "command": [
                            str(ROOT / ".venv/bin/python"),
                            str(ROOT / "scripts/run_train.py"),
                        ],
                        "artifacts": {
                            "model_file": str(ROOT / "artifacts/models/unit_model.pkl"),
                        },
                    },
                ],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_benchmark_handoff.py",
                    "--pre-race-summary-output",
                    str(pre_race_summary),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--source-timing-summary-input",
                    str(source_timing_summary),
                ],
            ):
                exit_code = pre_race_handoff_script.main()

            self.assertEqual(exit_code, 0)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(
                manifest["pre_race_summary"]["filtered_race_card_output"],
                "data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv",
            )
            self.assertEqual(
                manifest["pre_race_summary"]["materialization_summary"]["manifest_file"],
                "artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json",
            )
            self.assertEqual(manifest["benchmark_manifest"]["command"][0], ".venv/bin/python")
            self.assertEqual(manifest["benchmark_manifest"]["command"][1], "scripts/run_train.py")
            self.assertEqual(manifest["benchmark_manifest"]["artifacts"]["model_file"], "artifacts/models/unit_model.pkl")


if __name__ == "__main__":
    unittest.main()