from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import scripts.run_netkeiba_latest_revision_gate as latest_gate_script


class NetkeibaLatestRevisionGateTest(unittest.TestCase):
    def test_dry_run_passes_training_and_evaluate_artifact_args_to_revision_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_output = tmp / "netkeiba_latest_gate.json"
            snapshot_output = tmp / "snapshot.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_latest_revision_gate.py",
                    "--profile",
                    "current_best_eval_2025_latest",
                    "--revision",
                    "r_test_latest",
                    "--skip-train",
                    "--train-artifact-suffix",
                    "r_train_src",
                    "--evaluate-model-artifact-suffix",
                    "r_eval_src",
                    "--manifest-output",
                    str(manifest_output),
                    "--snapshot-output",
                    str(snapshot_output),
                    "--dry-run",
                ],
            ):
                exit_code = latest_gate_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(manifest_output.read_text(encoding="utf-8"))
            revision_gate_command = payload["revision_gate_command"]
            self.assertIn("--skip-train", revision_gate_command)
            self.assertIn("--train-artifact-suffix", revision_gate_command)
            self.assertIn("r_train_src", revision_gate_command)
            self.assertIn("--evaluate-model-artifact-suffix", revision_gate_command)
            self.assertIn("r_eval_src", revision_gate_command)
            self.assertEqual(payload["training"]["skip_train"], True)
            self.assertEqual(payload["training"]["train_artifact_suffix"], "r_train_src")
            self.assertEqual(payload["evaluation"]["model_artifact_suffix"], "r_eval_src")

    def test_dry_run_passes_no_model_artifact_suffix_to_revision_gate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_output = tmp / "netkeiba_latest_gate.json"
            snapshot_output = tmp / "snapshot.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_latest_revision_gate.py",
                    "--profile",
                    "current_best_eval_2025_latest",
                    "--revision",
                    "r_test_latest",
                    "--evaluate-no-model-artifact-suffix",
                    "--manifest-output",
                    str(manifest_output),
                    "--snapshot-output",
                    str(snapshot_output),
                    "--dry-run",
                ],
            ):
                exit_code = latest_gate_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(manifest_output.read_text(encoding="utf-8"))
            revision_gate_command = payload["revision_gate_command"]
            self.assertIn("--evaluate-no-model-artifact-suffix", revision_gate_command)
            self.assertEqual(payload["evaluation"]["no_model_artifact_suffix"], True)


if __name__ == "__main__":
    unittest.main()