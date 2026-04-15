from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import scripts.run_local_evaluate as local_evaluate_script


class RunLocalEvaluateTrustBlockTest(unittest.TestCase):
    def test_trust_block_writes_pointer_payload_without_launching_child(self) -> None:
        writes: list[tuple[Path, dict[str, object]]] = []

        def fake_write_json(path: Path, payload: dict[str, object]) -> None:
            writes.append((Path(path), json.loads(json.dumps(payload))))

        with patch.object(
            sys,
            "argv",
            [
                "run_local_evaluate.py",
                "--config",
                "configs/model.yaml",
                "--data-config",
                "configs/data_local_nankan.yaml",
                "--feature-config",
                "configs/features.yaml",
                "--output",
                "artifacts/reports/evaluation_local_nankan_pointer.json",
            ],
        ), patch.object(
            local_evaluate_script,
            "load_yaml",
            return_value={"dataset": {"source_dataset": "local_nankan"}},
        ), patch.object(
            local_evaluate_script,
            "resolve_local_nankan_trust_block",
            return_value={
                "error_code": "historical_local_nankan_trust_not_ready",
                "error_message": "evaluate blocked for historical local Nankan",
                "recommended_action": "rerun_local_backfill_with_pre_race_market_capture",
                "provenance_manifest_path": "/workspaces/nr-learn/artifacts/reports/local_nankan_provenance_audit_issue120_repaired.json",
                "strict_trust_ready": False,
                "pre_race": 0,
                "post_race": 728850,
                "unknown": 257,
            },
        ), patch.object(
            local_evaluate_script,
            "artifact_ensure_output_file_path",
            return_value=None,
        ), patch.object(
            local_evaluate_script,
            "write_json",
            side_effect=fake_write_json,
        ), patch.object(
            local_evaluate_script.subprocess,
            "run",
        ) as subprocess_run:
            exit_code = local_evaluate_script.main()

        self.assertEqual(exit_code, 1)
        subprocess_run.assert_not_called()
        self.assertTrue(writes)
        payload = writes[-1][1]
        self.assertEqual(payload["status"], "blocked_by_trust")
        self.assertEqual(payload["current_phase"], "trust_preflight_blocked")
        self.assertEqual(payload["error_code"], "historical_local_nankan_trust_not_ready")
        self.assertEqual(payload["recommended_action"], "rerun_local_backfill_with_pre_race_market_capture")
        self.assertEqual(
            payload["read_order"],
            [
                "status",
                "current_phase",
                "recommended_action",
                "error_code",
                "trust_context.strict_trust_ready",
                "trust_context.provenance_manifest",
            ],
        )
        self.assertEqual(payload["trust_context"]["pre_race"], 0)
        self.assertEqual(payload["trust_context"]["post_race"], 728850)
        self.assertEqual(payload["trust_context"]["unknown"], 257)

    def test_completed_pointer_exposes_top_level_read_order(self) -> None:
        writes: list[tuple[Path, dict[str, object]]] = []

        def fake_write_json(path: Path, payload: dict[str, object]) -> None:
            writes.append((Path(path), json.loads(json.dumps(payload))))

        with patch.object(
            sys,
            "argv",
            [
                "run_local_evaluate.py",
                "--config",
                "configs/model.yaml",
                "--data-config",
                "configs/data_local_nankan_pre_race_ready.yaml",
                "--feature-config",
                "configs/features.yaml",
                "--output",
                "artifacts/reports/evaluation_local_nankan_pointer.json",
            ],
        ), patch.object(
            local_evaluate_script,
            "load_yaml",
            return_value={"dataset": {"source_dataset": "local_nankan_pre_race_ready"}},
        ), patch.object(
            local_evaluate_script,
            "resolve_local_nankan_trust_block",
            return_value=None,
        ), patch.object(
            local_evaluate_script,
            "artifact_ensure_output_file_path",
            return_value=None,
        ), patch.object(
            local_evaluate_script,
            "write_json",
            side_effect=fake_write_json,
        ), patch.object(
            local_evaluate_script.subprocess,
            "run",
            return_value=type("_Result", (), {"returncode": 0})(),
        ), patch.object(
            local_evaluate_script,
            "read_json",
            return_value={"status": "completed", "output_files": {"score_output": "artifacts/predictions/local.csv"}},
        ), patch.object(
            Path,
            "exists",
            autospec=True,
            side_effect=lambda self: str(self).endswith("evaluation_manifest.json") or str(self).endswith("evaluation_summary.json"),
        ):
            exit_code = local_evaluate_script.main()

        self.assertEqual(exit_code, 0)
        payload = writes[-1][1]
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["current_phase"], "evaluate_completed")
        self.assertEqual(payload["recommended_action"], "review_local_evaluation_pointer")
        self.assertEqual(
            payload["read_order"],
            [
                "status",
                "current_phase",
                "recommended_action",
                "exit_code",
                "latest_manifest_payload.status",
                "latest_summary",
            ],
        )


if __name__ == "__main__":
    unittest.main()