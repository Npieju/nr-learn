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


backfill_then_benchmark_script = _load_script_module(
    "test_run_local_backfill_then_benchmark",
    "scripts/run_local_backfill_then_benchmark.py",
)
local_public_snapshot_script = _load_script_module(
    "test_run_local_public_snapshot",
    "scripts/run_local_public_snapshot.py",
)


class LocalWrapperOpsTest(unittest.TestCase):
    def test_local_backfill_then_benchmark_dry_run_writes_planned_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            wrapper_manifest = tmp_path / "local_backfill_then_benchmark_manifest.json"
            backfill_manifest = tmp_path / "local_nankan_backfill_handoff_manifest.json"
            materialize_manifest = tmp_path / "local_nankan_primary_handoff_manifest.json"
            snapshot_output = tmp_path / "coverage_snapshot_local_nankan_handoff.json"
            preflight_output = tmp_path / "data_preflight_local_nankan_handoff.json"
            benchmark_manifest = tmp_path / "benchmark_gate_local_nankan_handoff.json"

            with patch.object(backfill_then_benchmark_script, "artifact_ensure_output_file_path", return_value=None), patch.object(
                sys,
                "argv",
                [
                    "run_local_backfill_then_benchmark.py",
                    "--wrapper-manifest-output",
                    str(wrapper_manifest),
                    "--backfill-manifest-output",
                    str(backfill_manifest),
                    "--materialize-manifest-output",
                    str(materialize_manifest),
                    "--snapshot-output",
                    str(snapshot_output),
                    "--preflight-output",
                    str(preflight_output),
                    "--benchmark-manifest-output",
                    str(benchmark_manifest),
                    "--dry-run",
                ],
            ):
                exit_code = backfill_then_benchmark_script.main()

            self.assertEqual(exit_code, 0)
            manifest = json.loads(wrapper_manifest.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["current_phase"], "planned")
            self.assertEqual(manifest["recommended_action"], "run_local_backfill_then_benchmark")
            self.assertEqual(manifest["backfill"]["status"], "planned")
            self.assertIn("--dry-run", manifest["backfill"]["command"])
            self.assertEqual(manifest["benchmark_gate"]["status"], "planned")
            self.assertEqual(manifest["materialize"]["status"], "planned")

    def test_local_public_snapshot_dry_run_without_lineage_writes_planned_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "local_public_snapshot_test_revision.json"
            lineage_path = tmp_path / "local_revision_gate_test_revision.json"

            with patch.object(local_public_snapshot_script, "artifact_ensure_output_file_path", return_value=None), patch.object(
                sys,
                "argv",
                [
                    "run_local_public_snapshot.py",
                    "--revision",
                    "test_revision",
                    "--lineage-manifest",
                    str(lineage_path),
                    "--output",
                    str(output_path),
                    "--dry-run",
                ],
            ):
                exit_code = local_public_snapshot_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "planned")
            self.assertEqual(payload["current_phase"], "planned")
            self.assertEqual(payload["recommended_action"], "run_local_revision_gate")
            self.assertEqual(payload["lineage_status"], "planned")
            self.assertEqual(payload["readiness"]["benchmark_rerun_ready"], False)

    def test_local_public_snapshot_missing_lineage_writes_failure_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "local_public_snapshot_test_revision.json"
            lineage_path = tmp_path / "local_revision_gate_test_revision.json"

            with patch.object(local_public_snapshot_script, "artifact_ensure_output_file_path", return_value=None), patch.object(
                sys,
                "argv",
                [
                    "run_local_public_snapshot.py",
                    "--revision",
                    "test_revision",
                    "--lineage-manifest",
                    str(lineage_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = local_public_snapshot_script.main()

            self.assertEqual(exit_code, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["lineage_status"], "missing")
            self.assertEqual(payload["lineage_completed_step"], "missing")
            self.assertEqual(payload["recommended_action"], "run_local_revision_gate")
            self.assertIn("not found", payload["error_message"])


if __name__ == "__main__":
    unittest.main()