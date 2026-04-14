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


capture_loop_script = _load_script_module(
    "test_run_local_nankan_pre_race_capture_loop",
    "scripts/run_local_nankan_pre_race_capture_loop.py",
)


class LocalNankanCaptureLoopTest(unittest.TestCase):
    def test_capture_loop_writes_completed_manifest_from_latest_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            snapshot_dir = tmp_path / "snapshots"
            wrapper_manifest = tmp_path / "local_nankan_pre_race_capture_loop_manifest.json"
            pass_summary = snapshot_dir / "pass_001_coverage_summary.json"
            baseline_summary = tmp_path / "capture_coverage_summary.json"

            latest_summary = {
                "status": "completed",
                "current_phase": "capture_completed",
                "recommended_action": "review_capture_coverage_summary",
                "pre_race_only_rows": 562,
                "pre_race_only_races": 24,
                "result_ready_races": 0,
                "pending_result_races": 24,
                "baseline_comparison": {
                    "delta_pre_race_only_rows": 32,
                    "delta_pre_race_only_races": 2,
                },
            }

            baseline_summary.write_text("{}", encoding="utf-8")

            with patch.object(capture_loop_script, "_run_command", side_effect=[0, 0, 0]), patch.object(
                capture_loop_script,
                "_read_json_dict",
                side_effect=lambda path: latest_summary if path == pass_summary else {},
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_capture_loop.py",
                    "--max-passes",
                    "1",
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--baseline-summary-input",
                    str(baseline_summary),
                ],
            ):
                exit_code = capture_loop_script.main()

            self.assertEqual(exit_code, 0)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["current_phase"], "capture_completed")
            self.assertEqual(manifest["recommended_action"], "review_capture_coverage_summary")
            self.assertEqual(manifest["execution_role"], "pre_race_capture_refresh_loop")
            self.assertEqual(manifest["data_update_mode"], "capture_refresh_only")
            self.assertEqual(manifest["execution_mode"], "bounded_pass_loop")
            self.assertEqual(manifest["trigger_contract"], "direct_capture_refresh")
            self.assertEqual(manifest["completed_passes"], 1)
            self.assertEqual(manifest["initial_baseline_summary_input"], str(baseline_summary))
            self.assertEqual(manifest["pass_snapshots"][0]["status"], "completed")
            self.assertEqual(manifest["pass_snapshots"][0]["baseline_summary_input"], str(baseline_summary))
            self.assertEqual(manifest["pass_snapshots"][0]["pre_race_only_rows"], 562)

    def test_capture_loop_reports_failed_prepare_top_level_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            snapshot_dir = tmp_path / "snapshots"
            wrapper_manifest = tmp_path / "local_nankan_pre_race_capture_loop_manifest.json"

            with patch.object(capture_loop_script, "_run_command", return_value=1), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_capture_loop.py",
                    "--max-passes",
                    "1",
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                ],
            ):
                exit_code = capture_loop_script.main()

            self.assertEqual(exit_code, 1)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_phase"], "capture_loop")
            self.assertEqual(manifest["recommended_action"], "inspect_capture_loop_manifest")
            self.assertEqual(manifest["execution_role"], "pre_race_capture_refresh_loop")
            self.assertEqual(manifest["data_update_mode"], "capture_refresh_only")
            self.assertEqual(manifest["execution_mode"], "bounded_pass_loop")
            self.assertEqual(manifest["trigger_contract"], "direct_capture_refresh")
            self.assertEqual(manifest["completed_passes"], 1)
            self.assertEqual(manifest["pass_snapshots"][0]["status"], "failed_prepare")
            self.assertEqual(manifest["pass_snapshots"][0]["prepare_exit_code"], 1)

    def test_capture_loop_uses_default_baseline_then_previous_pass_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            snapshot_dir = tmp_path / "snapshots"
            wrapper_manifest = tmp_path / "local_nankan_pre_race_capture_loop_manifest.json"
            initial_baseline = tmp_path / "capture_coverage_summary.json"
            initial_baseline.write_text("{}", encoding="utf-8")

            commands: list[list[str]] = []

            def run_command(*, label: str, command: list[str]) -> int:
                del label
                commands.append(command)
                return 0

            def read_json_dict(path: Path | None) -> dict[str, object]:
                if path is None:
                    return {}
                name = path.name
                if name == "pass_001_coverage_summary.json":
                    return {
                        "status": "capturing",
                        "current_phase": "capturing_pre_race_pool",
                        "recommended_action": "continue_recrawl_cadence_and_wait_for_results",
                        "pre_race_only_rows": 281,
                        "pre_race_only_races": 24,
                        "result_ready_races": 0,
                        "pending_result_races": 24,
                        "baseline_comparison": {},
                    }
                if name == "pass_002_coverage_summary.json":
                    return {
                        "status": "completed",
                        "current_phase": "capture_completed",
                        "recommended_action": "review_capture_coverage_summary",
                        "pre_race_only_rows": 300,
                        "pre_race_only_races": 24,
                        "result_ready_races": 1,
                        "pending_result_races": 23,
                        "baseline_comparison": {},
                    }
                return {}

            with patch.object(capture_loop_script, "_run_command", side_effect=run_command), patch.object(
                capture_loop_script,
                "_read_json_dict",
                side_effect=read_json_dict,
            ), patch.object(
                capture_loop_script.time,
                "sleep",
                return_value=None,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_capture_loop.py",
                    "--max-passes",
                    "2",
                    "--snapshot-dir",
                    str(snapshot_dir),
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--baseline-summary-input",
                    str(initial_baseline),
                ],
            ):
                exit_code = capture_loop_script.main()

            self.assertEqual(exit_code, 0)
            coverage_commands = [command for command in commands if "run_local_nankan_pre_race_capture_coverage.py" in command[1]]
            self.assertEqual(len(coverage_commands), 2)
            first_baseline_index = coverage_commands[0].index("--baseline-summary-input") + 1
            second_baseline_index = coverage_commands[1].index("--baseline-summary-input") + 1
            self.assertEqual(coverage_commands[0][first_baseline_index], str(initial_baseline))
            self.assertEqual(
                coverage_commands[1][second_baseline_index],
                str(snapshot_dir / "pass_001_coverage_summary.json"),
            )

            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["initial_baseline_summary_input"], str(initial_baseline))
            self.assertEqual(manifest["pass_snapshots"][0]["baseline_summary_input"], str(initial_baseline))
            self.assertEqual(
                manifest["pass_snapshots"][1]["baseline_summary_input"],
                str(snapshot_dir / "pass_001_coverage_summary.json"),
            )


if __name__ == "__main__":
    unittest.main()