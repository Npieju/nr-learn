from __future__ import annotations

import argparse
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
local_revision_gate_script = _load_script_module(
    "test_run_local_revision_gate",
    "scripts/run_local_revision_gate.py",
)
local_benchmark_gate_script = _load_script_module(
    "test_run_local_benchmark_gate",
    "scripts/run_local_benchmark_gate.py",
)
local_evaluate_script = _load_script_module(
    "test_run_local_evaluate",
    "scripts/run_local_evaluate.py",
)
local_public_snapshot_script = _load_script_module(
    "test_run_local_public_snapshot",
    "scripts/run_local_public_snapshot.py",
)
future_tuning_probe_script = _load_script_module(
    "test_run_local_nankan_future_only_tuning_probe",
    "scripts/run_local_nankan_future_only_tuning_probe.py",
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
            self.assertEqual(manifest["backfill"]["command"][1], "scripts/run_backfill_local_nankan.py")
            self.assertIn("--race-id-source", manifest["backfill"]["command"])
            self.assertEqual(
                manifest["backfill"]["command"][manifest["backfill"]["command"].index("--race-id-source") + 1],
                "race_list",
            )
            self.assertEqual(manifest["benchmark_gate"]["command"][1], "scripts/run_local_benchmark_gate.py")
            self.assertEqual(manifest["benchmark_gate"]["status"], "planned")
            self.assertEqual(manifest["materialize"]["status"], "planned")

    def test_local_revision_gate_dry_run_normalizes_command_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            lineage_path = tmp_path / "local_revision_gate_test_revision.json"
            evaluation_pointer_path = tmp_path / "evaluation_test_revision_pointer.json"

            with patch.object(local_revision_gate_script, "artifact_ensure_output_file_path", return_value=None), patch.object(
                local_revision_gate_script,
                "_configure_live_log",
                return_value=None,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_revision_gate.py",
                    "--revision",
                    "test_revision",
                    "--backfill-before-benchmark",
                    "--lineage-output",
                    str(lineage_path),
                    "--evaluation-pointer-output",
                    str(evaluation_pointer_path),
                    "--dry-run",
                ],
            ):
                exit_code = local_revision_gate_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(lineage_path.read_text(encoding="utf-8"))
            self.assertIn("--race-id-source", payload["benchmark_gate"]["command"])
            self.assertEqual(
                payload["benchmark_gate"]["command"][payload["benchmark_gate"]["command"].index("--race-id-source") + 1],
                "race_list",
            )
            self.assertEqual(payload["benchmark_gate"]["command"][1], "scripts/run_local_backfill_then_benchmark.py")
            self.assertEqual(payload["revision_gate"]["command"][1], "scripts/run_revision_gate.py")
            self.assertEqual(payload["evaluation_pointer"]["command"][1], "scripts/run_local_evaluate.py")
            self.assertEqual(payload["evaluation_pointer_payload"]["evaluate_command"][1], "scripts/run_local_evaluate.py")

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

    def test_local_public_snapshot_normalizes_copied_workspace_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "local_public_snapshot_test_revision.json"
            lineage_path = tmp_path / "local_revision_gate_test_revision.json"
            absolute_eval_manifest = ROOT / "artifacts" / "reports" / "evaluation_manifest.json"
            absolute_eval_summary = ROOT / "artifacts" / "reports" / "evaluation_summary.json"
            absolute_score = ROOT / "artifacts" / "predictions" / "local_nankan_scores.csv"
            lineage_payload = {
                "status": "completed",
                "completed_step": "completed",
                "revision": "test_revision",
                "universe": "local_nankan",
                "source_scope": "local_only",
                "baseline_reference": "baseline_ref",
                "artifacts": {
                    "lineage_manifest": str(lineage_path),
                    "evaluation_pointer": str(ROOT / "artifacts" / "reports" / "evaluation_pointer.json"),
                    "score_output": str(absolute_score),
                },
                "benchmark_gate_payload": {
                    "status": "completed",
                    "completed_step": "completed",
                    "recommended_action": "review_public_snapshot",
                },
                "evaluation_pointer_payload": {
                    "status": "completed",
                    "latest_manifest": str(absolute_eval_manifest),
                    "latest_summary": str(absolute_eval_summary),
                    "output_files": {
                        "score_output": str(absolute_score),
                    },
                },
                "promotion_payload": {
                    "status": "completed",
                    "decision": "promote",
                    "recommended_action": "review_public_snapshot",
                },
            }
            lineage_path.write_text(json.dumps(lineage_payload), encoding="utf-8")

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

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["artifacts"]["score_output"], "artifacts/predictions/local_nankan_scores.csv")
            self.assertEqual(payload["evaluation_summary"]["latest_manifest"], "artifacts/reports/evaluation_manifest.json")
            self.assertEqual(payload["evaluation_summary"]["latest_summary"], "artifacts/reports/evaluation_summary.json")
            self.assertEqual(
                payload["evaluation_summary"]["output_files"]["score_output"],
                "artifacts/predictions/local_nankan_scores.csv",
            )

    def test_local_benchmark_gate_provenance_block_normalizes_workspace_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            provenance_manifest_path = tmp_path / "local_nankan_provenance_audit.json"
            provenance_summary_path = tmp_path / "local_nankan_provenance_summary.json"
            manifest_output = tmp_path / "benchmark_gate_local_nankan.json"
            preflight_output = tmp_path / "data_preflight_local_nankan.json"
            provenance_manifest_path.write_text(
                json.dumps(
                    {
                        "recommended_action": "inspect_local_nankan_provenance_audit",
                        "error_message": "blocked",
                        "readiness": {
                            "recommended_action": "inspect_local_nankan_provenance_audit",
                            "blocking_reasons": [str(ROOT / "artifacts" / "reports" / "missing_reason.txt")],
                        },
                    }
                ),
                encoding="utf-8",
            )
            provenance_summary_path.write_text(
                json.dumps({"summary_path": str(ROOT / "artifacts" / "reports" / "summary.csv")}),
                encoding="utf-8",
            )
            args = argparse.Namespace(
                schema_version="local.benchmark_gate.v1",
                universe="local_nankan",
                source_scope="local_only",
                baseline_reference="baseline_ref",
                data_config=str(ROOT / "configs" / "data_local_nankan.yaml"),
                model_config=str(ROOT / "configs" / "model_local_baseline.yaml"),
                feature_config=str(ROOT / "configs" / "features_local_baseline.yaml"),
                tail_rows=10,
                max_rows=20,
                pre_feature_max_rows=None,
                wf_mode="off",
                wf_scheme="nested",
                skip_train=False,
                skip_evaluate=False,
                race_result_path=str(ROOT / "data" / "external" / "local_nankan" / "results" / "local_race_result.csv"),
                race_card_path=str(ROOT / "data" / "external" / "local_nankan" / "racecard" / "local_race_card.csv"),
                pedigree_path=str(ROOT / "data" / "external" / "local_nankan" / "pedigree" / "local_pedigree.csv"),
                manifest_output=str(manifest_output),
                preflight_output=str(preflight_output),
                provenance_summary_output=str(provenance_summary_path),
                provenance_manifest_output=str(provenance_manifest_path),
            )

            local_benchmark_gate_script._write_provenance_block_outputs(
                args=args,
                provenance_manifest_path=provenance_manifest_path,
                provenance_summary_path=provenance_summary_path,
            )

            manifest_payload = json.loads(manifest_output.read_text(encoding="utf-8"))
            preflight_payload = json.loads(preflight_output.read_text(encoding="utf-8"))
            self.assertEqual(manifest_payload["configs"]["data_config"], "configs/data_local_nankan.yaml")
            self.assertEqual(
                manifest_payload["artifacts"]["provenance_manifest"],
                str(provenance_manifest_path.relative_to(ROOT)),
            )
            self.assertEqual(
                manifest_payload["readiness"]["reasons"][0],
                "artifacts/reports/missing_reason.txt",
            )
            self.assertEqual(
                preflight_payload["source_report"]["local_market_provenance_summary"]["summary_path"],
                "artifacts/reports/summary.csv",
            )

    def test_local_evaluate_trust_block_normalizes_workspace_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            pointer_path = tmp_path / "evaluation_local_nankan_pointer.json"

            with patch.object(local_evaluate_script, "artifact_ensure_output_file_path", return_value=None), patch.object(
                local_evaluate_script,
                "load_yaml",
                return_value={},
            ), patch.object(
                local_evaluate_script,
                "resolve_local_nankan_trust_block",
                return_value={
                    "error_code": "historical_local_nankan_trust_not_ready",
                    "error_message": "blocked",
                    "recommended_action": "inspect_local_nankan_provenance_audit",
                    "strict_trust_ready": False,
                    "pre_race": 0,
                    "post_race": 1,
                    "unknown": 0,
                    "provenance_manifest_path": str(ROOT / "artifacts" / "reports" / "local_nankan_provenance_audit.json"),
                },
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_evaluate.py",
                    "--config",
                    str(ROOT / "configs" / "model_local_baseline.yaml"),
                    "--data-config",
                    str(ROOT / "configs" / "data_local_nankan.yaml"),
                    "--feature-config",
                    str(ROOT / "configs" / "features_local_baseline.yaml"),
                    "--output",
                    str(pointer_path),
                ],
            ):
                exit_code = local_evaluate_script.main()

            self.assertEqual(exit_code, 1)
            payload = json.loads(pointer_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["run_context"]["config"], "configs/model_local_baseline.yaml")
            self.assertEqual(payload["run_context"]["data_config"], "configs/data_local_nankan.yaml")
            self.assertEqual(payload["run_context"]["feature_config"], "configs/features_local_baseline.yaml")
            self.assertEqual(
                payload["trust_context"]["provenance_manifest"],
                "artifacts/reports/local_nankan_provenance_audit.json",
            )
            self.assertNotIn("/workspaces/nr-learn", payload["error_message"])

    def test_future_tuning_probe_normalizes_copied_workspace_paths(self) -> None:
        artifacts_tmp = ROOT / "artifacts" / "tmp"
        artifacts_tmp.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=artifacts_tmp) as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "local_nankan_future_only_tuning_probe.json"
            payload_queue = [
                {
                    "status": "completed",
                    "current_phase": "completed",
                    "recommended_action": "review_status_board",
                    "run_context": {
                        "status_board_output": str(ROOT / "artifacts" / "reports" / "status_board_probe.json"),
                    },
                },
                {
                    "status": "completed",
                    "latest_summary": {
                        "baseline_comparison": {
                            "reference": str(ROOT / "artifacts" / "reports" / "baseline_summary.json"),
                        }
                    },
                },
                {
                    "current_phase": "completed",
                    "recommended_action": "review_status_board",
                    "readiness": {
                        "benchmark_rerun_ready": False,
                        "reasons": [str(ROOT / "artifacts" / "reports" / "not_ready_reason.txt")],
                    },
                },
            ]

            def fake_read_json_dict(_path):
                return payload_queue.pop(0)

            with patch.object(future_tuning_probe_script, "_run_command", return_value=0), patch.object(
                future_tuning_probe_script,
                "_read_json_dict",
                side_effect=fake_read_json_dict,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_future_only_tuning_probe.py",
                    "--output",
                    str(output_path),
                    "--scenario",
                    "unit:1:1",
                ],
            ):
                exit_code = future_tuning_probe_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            scenario = payload["scenarios"][0]
            self.assertEqual(
                scenario["run_context"]["status_board_output"],
                "artifacts/reports/status_board_probe.json",
            )
            self.assertEqual(
                scenario["readiness_reasons"][0],
                "artifacts/reports/not_ready_reason.txt",
            )


if __name__ == "__main__":
    unittest.main()