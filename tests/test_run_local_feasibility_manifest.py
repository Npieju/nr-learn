from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import scripts.run_local_feasibility_manifest as feasibility_script


class RunLocalFeasibilityManifestTrustBlockTest(unittest.TestCase):
    def test_dry_run_manifest_normalizes_command_paths(self) -> None:
        artifacts_tmp = Path("/workspaces/nr-learn/artifacts/tmp")
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_output = tmp_path / "local_feasibility_manifest_local_nankan.json"

            with patch.object(
                sys,
                "argv",
                [
                    "run_local_feasibility_manifest.py",
                    "--manifest-output",
                    str(manifest_output),
                    "--dry-run",
                ],
            ), patch.object(
                feasibility_script,
                "artifact_ensure_output_file_path",
                return_value=None,
            ):
                exit_code = feasibility_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(payload["snapshot"]["command"][1], "scripts/run_local_coverage_snapshot.py")
            self.assertEqual(payload["validation"]["command"][1], "scripts/run_local_data_source_validation.py")
            self.assertEqual(payload["feature_gap"]["command"][1], "scripts/run_local_feature_gap_report.py")
            self.assertEqual(payload["evaluation"]["command"][1], "scripts/run_local_evaluate.py")

    def test_evaluation_trust_block_is_recorded_as_blocked_state(self) -> None:
        writes: list[tuple[Path, dict[str, object]]] = []

        def fake_write_json(path: Path, payload: dict[str, object]) -> None:
            writes.append((Path(path), json.loads(json.dumps(payload))))

        def fake_run_command(command: list[str], *, label: str) -> dict[str, object]:
            if label == "snapshot":
                return {
                    "label": label,
                    "command": command,
                    "status": "completed",
                    "exit_code": 0,
                    "started_at": "2026-04-13T00:00:00Z",
                    "finished_at": "2026-04-13T00:00:01Z",
                }
            if label == "evaluation":
                return {
                    "label": label,
                    "command": command,
                    "status": "failed",
                    "exit_code": 1,
                    "started_at": "2026-04-13T00:00:02Z",
                    "finished_at": "2026-04-13T00:00:03Z",
                }
            raise AssertionError(f"unexpected label: {label}")

        def fake_read_optional_json(path: Path) -> dict[str, object] | None:
            name = Path(path).name
            if name == "coverage_snapshot_local_nankan.json":
                return {
                    "readiness": {
                        "benchmark_rerun_ready": False,
                        "recommended_action": "inspect_local_nankan_provenance_audit",
                    }
                }
            if name == "evaluation_local_nankan_pointer.json":
                return {
                    "status": "blocked_by_trust",
                    "error_code": "historical_local_nankan_trust_not_ready",
                    "error_message": "historical local Nankan trust is not ready",
                    "recommended_action": "rerun_local_backfill_with_pre_race_market_capture",
                }
            return None

        with patch.object(
            sys,
            "argv",
            [
                "run_local_feasibility_manifest.py",
                "--skip-validation",
                "--skip-feature-gap",
                "--data-config",
                "configs/data_local_nankan.yaml",
                "--manifest-output",
                "artifacts/reports/local_feasibility_manifest_local_nankan.json",
            ],
        ), patch.object(
            feasibility_script,
            "artifact_ensure_output_file_path",
            return_value=None,
        ), patch.object(
            feasibility_script,
            "write_json",
            side_effect=fake_write_json,
        ), patch.object(
            feasibility_script,
            "_run_command",
            side_effect=fake_run_command,
        ), patch.object(
            feasibility_script,
            "_read_optional_json",
            side_effect=fake_read_optional_json,
        ):
            exit_code = feasibility_script.main()

        self.assertEqual(exit_code, 1)
        self.assertTrue(writes)
        final_manifest = writes[-1][1]
        self.assertEqual(final_manifest["status"], "evaluation_blocked_by_trust")
        self.assertEqual(final_manifest["error_code"], "historical_local_nankan_trust_not_ready")
        self.assertEqual(final_manifest["recommended_action"], "rerun_local_backfill_with_pre_race_market_capture")


if __name__ == "__main__":
    unittest.main()