from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from racing_ml.data.local_nankan_watch import (
    build_readiness_watcher_manifest,
    should_trigger_handoff,
)


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


watcher_script = _load_script_module("test_run_local_nankan_readiness_watcher", "scripts/run_local_nankan_readiness_watcher.py")


class LocalNankanWatchTest(unittest.TestCase):
    def test_should_trigger_handoff_only_when_ready(self) -> None:
        self.assertTrue(should_trigger_handoff({"status": "ready"}))
        self.assertFalse(should_trigger_handoff({"status": "not_ready"}))
        self.assertFalse(should_trigger_handoff(None))

    def test_build_readiness_watcher_manifest_keeps_probe_and_handoff(self) -> None:
        manifest = build_readiness_watcher_manifest(
            status="completed",
            current_phase="handoff_completed",
            recommended_action="review_handoff_outputs",
            attempts=2,
            waited_seconds=60,
            timed_out=False,
            probe_summary={"status": "ready"},
            handoff_manifest={"status": "completed"},
        )

        self.assertEqual(manifest["status"], "completed")
        self.assertEqual(manifest["attempts"], 2)
        self.assertEqual(manifest["probe_summary"]["status"], "ready")
        self.assertEqual(manifest["handoff_manifest"]["status"], "completed")

    def test_readiness_watcher_returns_not_ready_manifest_when_probe_does_not_flip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            watcher_manifest = Path(tmp_dir) / "local_nankan_readiness_watcher_manifest.json"
            probe_summary = Path(tmp_dir) / "local_nankan_pre_race_readiness_probe_summary.json"

            with patch.object(watcher_script, "resolve_python_executable", return_value="/usr/bin/python3"), patch.object(
                watcher_script, "_run_command", return_value=2
            ), patch.object(
                watcher_script,
                "_read_json_dict",
                return_value={"status": "not_ready", "result_ready_races": 0, "pending_result_races": 24},
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_readiness_watcher.py",
                    "--probe-summary-output",
                    str(probe_summary),
                    "--watcher-manifest-output",
                    str(watcher_manifest),
                ],
            ):
                exit_code = watcher_script.main()

            self.assertEqual(exit_code, 2)
            manifest = json.loads(watcher_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "not_ready")
            self.assertEqual(manifest["current_phase"], "await_result_arrival")
            self.assertEqual(manifest["recommended_action"], "wait_for_result_ready_pre_race_races")
            self.assertTrue(manifest["timed_out"])
            self.assertEqual(manifest["probe_summary"]["status"], "not_ready")

    def test_readiness_watcher_triggers_handoff_when_probe_is_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            watcher_manifest = Path(tmp_dir) / "local_nankan_readiness_watcher_manifest.json"
            probe_summary = Path(tmp_dir) / "local_nankan_pre_race_readiness_probe_summary.json"
            handoff_manifest = Path(tmp_dir) / "local_nankan_result_ready_bootstrap_handoff_manifest.json"

            with patch.object(watcher_script, "resolve_python_executable", return_value="/usr/bin/python3"), patch.object(
                watcher_script, "_run_command", side_effect=[0, 0]
            ) as run_command, patch.object(
                watcher_script,
                "_read_json_dict",
                side_effect=[
                    {"status": "ready", "result_ready_races": 3},
                    {"status": "completed", "current_phase": "bootstrap_completed"},
                ],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_readiness_watcher.py",
                    "--probe-summary-output",
                    str(probe_summary),
                    "--handoff-manifest-output",
                    str(handoff_manifest),
                    "--watcher-manifest-output",
                    str(watcher_manifest),
                    "--run-bootstrap",
                ],
            ):
                exit_code = watcher_script.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual(run_command.call_count, 2)
            handoff_call = run_command.call_args_list[1]
            self.assertEqual(handoff_call.kwargs["label"], "result_ready_handoff")
            self.assertIn("--run-bootstrap", handoff_call.kwargs["command"])

            manifest = json.loads(watcher_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["current_phase"], "handoff_completed")
            self.assertEqual(manifest["recommended_action"], "review_handoff_outputs")
            self.assertEqual(manifest["probe_summary"]["status"], "ready")
            self.assertEqual(manifest["handoff_manifest"]["status"], "completed")

    def test_readiness_watcher_reports_failed_handoff_when_triggered_handoff_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            watcher_manifest = Path(tmp_dir) / "local_nankan_readiness_watcher_manifest.json"
            probe_summary = Path(tmp_dir) / "local_nankan_pre_race_readiness_probe_summary.json"
            handoff_manifest = Path(tmp_dir) / "local_nankan_result_ready_bootstrap_handoff_manifest.json"

            with patch.object(watcher_script, "resolve_python_executable", return_value="/usr/bin/python3"), patch.object(
                watcher_script, "_run_command", side_effect=[0, 1]
            ), patch.object(
                watcher_script,
                "_read_json_dict",
                side_effect=[
                    {"status": "ready", "result_ready_races": 3},
                    {"status": "failed", "current_phase": "benchmark_gate", "recommended_action": "inspect_handoff_manifest"},
                ],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_readiness_watcher.py",
                    "--probe-summary-output",
                    str(probe_summary),
                    "--handoff-manifest-output",
                    str(handoff_manifest),
                    "--watcher-manifest-output",
                    str(watcher_manifest),
                ],
            ):
                exit_code = watcher_script.main()

            self.assertEqual(exit_code, 1)
            manifest = json.loads(watcher_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "failed")
            self.assertEqual(manifest["current_phase"], "handoff_failed")
            self.assertEqual(manifest["recommended_action"], "inspect_handoff_manifest")
            self.assertEqual(manifest["probe_summary"]["status"], "ready")
            self.assertEqual(manifest["handoff_manifest"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
