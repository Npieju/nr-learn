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


next_window_script = _load_script_module("test_run_local_nankan_next_window", "scripts/run_local_nankan_next_window.py")
window_queue_script = _load_script_module("test_run_local_nankan_window_queue", "scripts/run_local_nankan_window_queue.py")


class LocalNankanWindowOpsTest(unittest.TestCase):
    def test_next_window_runs_backfill_coverage_then_status_board(self) -> None:
        with patch.object(next_window_script, "_run", side_effect=[0, 0, 0]) as run_mock, patch.object(
            sys,
            "argv",
            [
                "run_local_nankan_next_window.py",
                "--manifest-file",
                "artifacts/reports/test_manifest.json",
                "--coverage-output",
                "artifacts/reports/test_coverage.json",
                "--status-board-output",
                "artifacts/reports/test_status.json",
                "--backfill-aggregate",
                "artifacts/reports/test_manifest.json",
            ],
        ):
            exit_code = next_window_script.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_mock.call_count, 3)
        backfill_command = run_mock.call_args_list[0].args[0]
        coverage_command = run_mock.call_args_list[1].args[0]
        status_command = run_mock.call_args_list[2].args[0]
        self.assertIn("scripts/run_backfill_local_nankan.py", backfill_command[1])
        self.assertIn("scripts/run_local_coverage_snapshot.py", coverage_command[1])
        self.assertIn("scripts/run_local_nankan_status_board.py", status_command[1])
        self.assertIn("artifacts/reports/test_manifest.json", status_command)

    def test_next_window_stops_before_status_board_when_coverage_fails(self) -> None:
        with patch.object(next_window_script, "_run", side_effect=[0, 1]) as run_mock, patch.object(
            sys,
            "argv",
            ["run_local_nankan_next_window.py"],
        ):
            exit_code = next_window_script.main()

        self.assertEqual(exit_code, 1)
        self.assertEqual(run_mock.call_count, 2)

    def test_window_queue_returns_completed_when_requested_windows_are_done(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "local_nankan_backfill.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "requested_window_count": 3,
                        "completed_window_count": 3,
                        "remaining_window_count": 0,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(window_queue_script, "_run", return_value=0) as run_mock, patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_window_queue.py",
                    "--manifest-file",
                    str(manifest_path),
                    "--poll-sec",
                    "0.01",
                ],
            ):
                exit_code = window_queue_script.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_mock.call_count, 1)
        self.assertIn("scripts/run_local_nankan_status_board.py", run_mock.call_args_list[0].args[0][1])

    def test_window_queue_launches_next_window_and_exits_on_max_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "local_nankan_backfill.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "requested_window_count": 3,
                        "completed_window_count": 1,
                        "remaining_window_count": 2,
                    }
                ),
                encoding="utf-8",
            )

            def run_side_effect(command: list[str]) -> int:
                script_name = Path(command[1]).name
                if script_name == "run_local_nankan_next_window.py":
                    manifest_path.write_text(
                        json.dumps(
                            {
                                "status": "completed",
                                "requested_window_count": 3,
                                "completed_window_count": 2,
                                "remaining_window_count": 1,
                            }
                        ),
                        encoding="utf-8",
                    )
                return 0

            with patch.object(window_queue_script, "_run", side_effect=run_side_effect) as run_mock, patch.object(
                window_queue_script.time, "sleep", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_window_queue.py",
                    "--manifest-file",
                    str(manifest_path),
                    "--poll-sec",
                    "0.01",
                    "--max-windows",
                    "1",
                ],
            ):
                exit_code = window_queue_script.main()

        self.assertEqual(exit_code, 0)
        called_scripts = [Path(call.args[0][1]).name for call in run_mock.call_args_list]
        self.assertIn("run_local_nankan_next_window.py", called_scripts)
        self.assertGreaterEqual(called_scripts.count("run_local_nankan_status_board.py"), 2)

    def test_window_queue_stops_on_failed_manifest_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "local_nankan_backfill.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "failed",
                        "requested_window_count": 3,
                        "completed_window_count": 1,
                        "remaining_window_count": 2,
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(window_queue_script, "_run", return_value=0) as run_mock, patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_window_queue.py",
                    "--manifest-file",
                    str(manifest_path),
                    "--poll-sec",
                    "0.01",
                ],
            ):
                exit_code = window_queue_script.main()

        self.assertEqual(exit_code, 2)
        self.assertEqual(run_mock.call_count, 1)


if __name__ == "__main__":
    unittest.main()