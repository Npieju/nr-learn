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


status_board_script = _load_script_module("test_run_local_nankan_status_board", "scripts/run_local_nankan_status_board.py")


class LocalNankanStatusBoardTest(unittest.TestCase):
    def test_status_board_falls_back_to_coverage_ready_when_readiness_surfaces_do_not_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coverage_path = tmp_path / "coverage.json"
            backfill_path = tmp_path / "backfill.json"
            archive_path = tmp_path / "archive.json"
            capture_path = tmp_path / "capture_loop.json"
            probe_path = tmp_path / "probe.json"
            pre_race_handoff_path = tmp_path / "pre_race_handoff.json"
            bootstrap_handoff_path = tmp_path / "bootstrap_handoff.json"
            watcher_path = tmp_path / "watcher.json"
            output_path = tmp_path / "status_board.json"

            coverage_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": True},
                        "progress": {"current_stage": "ready_for_benchmark", "completed_targets": ["race_card", "race_result", "pedigree"]},
                        "target_states": {},
                        "external_outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            backfill_path.write_text(json.dumps({"status": "completed", "remaining_window_count": 0}), encoding="utf-8")
            archive_path.write_text(json.dumps({"status": "completed", "archive_reports": []}), encoding="utf-8")
            capture_path.write_text(json.dumps({}), encoding="utf-8")
            probe_path.write_text(json.dumps({"status": "ready", "result_ready_races": 3, "pending_result_races": 0}), encoding="utf-8")
            pre_race_handoff_path.write_text(json.dumps({"status": "completed", "current_phase": "benchmark_handoff_completed"}), encoding="utf-8")
            bootstrap_handoff_path.write_text(json.dumps({}), encoding="utf-8")
            watcher_path.write_text(json.dumps({"status": "completed", "attempts": 1, "timed_out": False}), encoding="utf-8")

            with patch.object(status_board_script, "artifact_ensure_output_file_path"), patch.object(
                status_board_script, "_build_live_collect_progress", return_value={}
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_status_board.py",
                    "--coverage-snapshot",
                    str(coverage_path),
                    "--backfill-aggregate",
                    str(backfill_path),
                    "--archive-run",
                    str(archive_path),
                    "--capture-loop-manifest",
                    str(capture_path),
                    "--readiness-probe-summary",
                    str(probe_path),
                    "--pre-race-handoff-manifest",
                    str(pre_race_handoff_path),
                    "--bootstrap-handoff-manifest",
                    str(bootstrap_handoff_path),
                    "--readiness-watcher-manifest",
                    str(watcher_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["current_phase"], "ready_for_benchmark")
            self.assertIsNone(payload["recommended_action"])

    def test_status_board_includes_readiness_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coverage_path = tmp_path / "coverage.json"
            backfill_path = tmp_path / "backfill.json"
            archive_path = tmp_path / "archive.json"
            capture_path = tmp_path / "capture_loop.json"
            probe_path = tmp_path / "probe.json"
            pre_race_handoff_path = tmp_path / "pre_race_handoff.json"
            bootstrap_handoff_path = tmp_path / "bootstrap_handoff.json"
            watcher_path = tmp_path / "watcher.json"
            output_path = tmp_path / "status_board.json"

            coverage_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": False, "recommended_action": "wait_for_result_ready_pre_race_races"},
                        "progress": {"current_stage": "capturing_pre_race_pool", "completed_targets": ["race_card"]},
                        "target_states": {"race_card": {"status": "completed"}},
                        "external_outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            backfill_path.write_text(json.dumps({"status": "completed", "remaining_window_count": 0}), encoding="utf-8")
            archive_path.write_text(json.dumps({"status": "completed", "archive_reports": []}), encoding="utf-8")
            capture_path.write_text(json.dumps({}), encoding="utf-8")
            probe_path.write_text(json.dumps({"status": "not_ready", "result_ready_races": 0, "pending_result_races": 24}), encoding="utf-8")
            pre_race_handoff_path.write_text(json.dumps({"status": "not_ready", "current_phase": "await_result_arrival"}), encoding="utf-8")
            bootstrap_handoff_path.write_text(json.dumps({"status": "not_ready", "current_phase": "await_result_arrival"}), encoding="utf-8")
            watcher_path.write_text(json.dumps({"status": "not_ready", "attempts": 2, "timed_out": True}), encoding="utf-8")

            with patch.object(status_board_script, "artifact_ensure_output_file_path"), patch.object(
                status_board_script, "_build_live_collect_progress", return_value={}
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_status_board.py",
                    "--coverage-snapshot",
                    str(coverage_path),
                    "--backfill-aggregate",
                    str(backfill_path),
                    "--archive-run",
                    str(archive_path),
                    "--capture-loop-manifest",
                    str(capture_path),
                    "--readiness-probe-summary",
                    str(probe_path),
                    "--pre-race-handoff-manifest",
                    str(pre_race_handoff_path),
                    "--bootstrap-handoff-manifest",
                    str(bootstrap_handoff_path),
                    "--readiness-watcher-manifest",
                    str(watcher_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["readiness_surfaces"]["readiness_probe"]["status"], "not_ready")
            self.assertEqual(payload["readiness_surfaces"]["readiness_probe"]["pending_result_races"], 24)
            self.assertEqual(payload["readiness_surfaces"]["pre_race_handoff"]["status"], "not_ready")
            self.assertEqual(payload["readiness_surfaces"]["bootstrap_handoff"]["status"], "not_ready")
            self.assertEqual(payload["readiness_surfaces"]["readiness_watcher"]["attempts"], 2)
            self.assertEqual(payload["readiness_surfaces"]["followup_entrypoint"]["script"], "scripts/run_local_nankan_future_only_followup_oneshot.py")
            self.assertEqual(payload["readiness_surfaces"]["followup_entrypoint"]["upstream_manifest"], str(capture_path))
            self.assertFalse(payload["readiness_surfaces"]["followup_entrypoint"]["upstream_contract_ready"])
            self.assertIn("--dry-run", payload["readiness_surfaces"]["followup_entrypoint"]["dry_run_command_preview"])
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["current_phase"], "await_result_arrival")
            self.assertIn("probe_status=not_ready", payload["highlights"])
            self.assertIn("watcher_status=not_ready", payload["highlights"])
            self.assertIn("capture_baseline_chain=None", payload["highlights"])
            self.assertIn("capture_upcoming_only=None", payload["highlights"])
            self.assertIn("capture_pre_filter_rows=None", payload["highlights"])

    def test_status_board_overrides_stale_coverage_ready_with_not_ready_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coverage_path = tmp_path / "coverage.json"
            backfill_path = tmp_path / "backfill.json"
            archive_path = tmp_path / "archive.json"
            capture_path = tmp_path / "capture_loop.json"
            probe_path = tmp_path / "probe.json"
            pre_race_handoff_path = tmp_path / "pre_race_handoff.json"
            bootstrap_handoff_path = tmp_path / "bootstrap_handoff.json"
            watcher_path = tmp_path / "watcher.json"
            output_path = tmp_path / "status_board.json"

            coverage_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": True},
                        "progress": {"current_stage": "ready_for_benchmark", "completed_targets": ["race_card", "race_result", "pedigree"]},
                        "target_states": {},
                        "external_outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            backfill_path.write_text(json.dumps({"status": "completed", "remaining_window_count": 0}), encoding="utf-8")
            archive_path.write_text(json.dumps({"status": "completed", "archive_reports": []}), encoding="utf-8")
            capture_path.write_text(json.dumps({}), encoding="utf-8")
            probe_path.write_text(
                json.dumps(
                    {
                        "status": "not_ready",
                        "recommended_action": "wait_for_result_ready_pre_race_races",
                        "materialization_summary": {"result_ready_races": 0, "pending_result_races": 24, "pre_race_only_rows": 562},
                    }
                ),
                encoding="utf-8",
            )
            pre_race_handoff_path.write_text(
                json.dumps(
                    {
                        "status": "not_ready",
                        "current_phase": "await_result_arrival",
                        "recommended_action": "wait_for_result_ready_pre_race_races",
                    }
                ),
                encoding="utf-8",
            )
            bootstrap_handoff_path.write_text(
                json.dumps(
                    {
                        "status": "not_ready",
                        "current_phase": "await_result_arrival",
                        "recommended_action": "wait_for_result_ready_pre_race_races",
                    }
                ),
                encoding="utf-8",
            )
            watcher_path.write_text(json.dumps({"status": "not_ready", "attempts": 1, "timed_out": True}), encoding="utf-8")

            with patch.object(status_board_script, "artifact_ensure_output_file_path"), patch.object(
                status_board_script, "_build_live_collect_progress", return_value={}
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_status_board.py",
                    "--coverage-snapshot",
                    str(coverage_path),
                    "--backfill-aggregate",
                    str(backfill_path),
                    "--archive-run",
                    str(archive_path),
                    "--capture-loop-manifest",
                    str(capture_path),
                    "--readiness-probe-summary",
                    str(probe_path),
                    "--pre-race-handoff-manifest",
                    str(pre_race_handoff_path),
                    "--bootstrap-handoff-manifest",
                    str(bootstrap_handoff_path),
                    "--readiness-watcher-manifest",
                    str(watcher_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["current_phase"], "await_result_arrival")
            self.assertEqual(payload["recommended_action"], "wait_for_result_ready_pre_race_races")
            self.assertEqual(payload["readiness_surfaces"]["readiness_probe"]["result_ready_races"], 0)
            self.assertEqual(payload["readiness_surfaces"]["readiness_probe"]["pending_result_races"], 24)
            self.assertEqual(payload["readiness_surfaces"]["readiness_probe"]["race_card_rows"], 562)

    def test_status_board_surfaces_followup_entrypoint_for_self_describing_capture_loop(self) -> None:
        readiness_surfaces = status_board_script._build_readiness_surfaces(
            capture_loop_manifest_path="artifacts/reports/local_nankan_pre_race_capture_loop_issue122_cycle.json",
            capture_loop_manifest={
                "status": "capturing",
                "current_phase": "capturing_pre_race_pool",
                "recommended_action": "wait",
                "execution_role": "pre_race_capture_refresh_loop",
                "data_update_mode": "capture_refresh_only",
                "trigger_contract": "direct_capture_refresh",
                "initial_baseline_summary_input": "artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json",
                "latest_race_id_source_report": {
                    "upcoming_only": True,
                    "as_of": "2026-04-14T15:00:00+09:00",
                    "filtered_out_count": 5,
                },
                "pass_snapshots": [
                    {"baseline_summary_input": "artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json"},
                    {"baseline_summary_input": "artifacts/reports/pass_001_coverage_summary.json"},
                ],
            },
            readiness_probe_summary={},
            pre_race_handoff_manifest={},
            bootstrap_handoff_manifest={},
            readiness_watcher_manifest={},
        )

        followup_entrypoint = readiness_surfaces["followup_entrypoint"]
        self.assertTrue(followup_entrypoint["upstream_contract_ready"])
        self.assertEqual(
            followup_entrypoint["upstream_initial_baseline_summary_input"],
            "artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json",
        )
        self.assertEqual(
            followup_entrypoint["upstream_latest_baseline_summary_input"],
            "artifacts/reports/pass_001_coverage_summary.json",
        )
        self.assertEqual(
            readiness_surfaces["capture_loop"]["latest_race_id_source_report"]["filtered_out_count"],
            5,
        )
        self.assertEqual(followup_entrypoint["read_order"][3], "upstream_refresh.upstream_fresh")
        self.assertEqual(followup_entrypoint["read_order"][4], "upstream_refresh.age_seconds")
        self.assertIn("artifacts/reports/local_nankan_pre_race_capture_loop_issue122_cycle.json", followup_entrypoint["dry_run_command_preview"])
        self.assertIn("--run-bootstrap-on-ready", followup_entrypoint["run_command_preview"])

    def test_status_board_surfaces_capture_baseline_chain_from_watcher_and_capture_loop(self) -> None:
        readiness_surfaces = status_board_script._build_readiness_surfaces(
            capture_loop_manifest_path="artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json",
            capture_loop_manifest={
                "status": "capturing",
                "current_phase": "capturing_pre_race_pool",
                "recommended_action": "wait",
                "initial_baseline_summary_input": "artifacts/reports/initial_capture_baseline.json",
                "snapshot_dir": "artifacts/reports/capture_snapshots",
                "latest_race_id_source_report": {
                    "upcoming_only": True,
                    "as_of": "2026-04-14T15:30:00+09:00",
                    "filtered_out_count": 6,
                },
                "pass_snapshots": [
                    {"baseline_summary_input": "artifacts/reports/initial_capture_baseline.json"},
                    {"baseline_summary_input": "artifacts/reports/pass_001_coverage_summary.json"},
                ],
            },
            readiness_probe_summary={},
            pre_race_handoff_manifest={},
            bootstrap_handoff_manifest={},
            readiness_watcher_manifest={
                "capture_loop_manifest_output": "artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json",
                "capture_loop_manifest": {
                    "initial_baseline_summary_input": "artifacts/reports/initial_capture_baseline.json",
                    "pass_snapshots": [
                        {"baseline_summary_input": "artifacts/reports/pass_001_coverage_summary.json"}
                    ],
                },
            },
        )

        self.assertEqual(
            readiness_surfaces["capture_loop"]["initial_baseline_summary_input"],
            "artifacts/reports/initial_capture_baseline.json",
        )
        self.assertEqual(
            readiness_surfaces["capture_loop"]["latest_baseline_summary_input"],
            "artifacts/reports/pass_001_coverage_summary.json",
        )
        self.assertEqual(
            readiness_surfaces["readiness_watcher"]["capture_loop_manifest_output"],
            "artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json",
        )
        self.assertEqual(
            readiness_surfaces["readiness_watcher"]["capture_initial_baseline_summary_input"],
            "artifacts/reports/initial_capture_baseline.json",
        )
        self.assertEqual(
            readiness_surfaces["readiness_watcher"]["capture_latest_baseline_summary_input"],
            "artifacts/reports/pass_001_coverage_summary.json",
        )
        self.assertEqual(
            readiness_surfaces["capture_loop"]["latest_race_id_source_report"]["as_of"],
            "2026-04-14T15:30:00+09:00",
        )

    def test_status_board_normalizes_nested_readiness_payload_paths(self) -> None:
        readiness_surfaces = status_board_script._build_readiness_surfaces(
            capture_loop_manifest_path="artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json",
            capture_loop_manifest={
                "status": "capturing",
                "current_phase": "capturing_pre_race_pool",
                "recommended_action": "wait",
                "initial_baseline_summary_input": str(ROOT / "artifacts/reports/initial_capture_baseline.json"),
                "snapshot_dir": str(ROOT / "artifacts/reports/capture_snapshots"),
                "latest_summary": {
                    "baseline_summary_input": str(ROOT / "artifacts/reports/pass_001_coverage_summary.json"),
                    "source_timing_summary_input": str(ROOT / "artifacts/reports/local_nankan_source_timing_audit.json"),
                },
                "pass_snapshots": [
                    {"baseline_summary_input": str(ROOT / "artifacts/reports/initial_capture_baseline.json")},
                    {"baseline_summary_input": str(ROOT / "artifacts/reports/pass_001_coverage_summary.json")},
                ],
            },
            readiness_probe_summary={
                "historical_source_timing": {
                    "source_timing_summary_input": str(ROOT / "artifacts/reports/local_nankan_source_timing_audit.json")
                }
            },
            pre_race_handoff_manifest={},
            bootstrap_handoff_manifest={},
            readiness_watcher_manifest={
                "probe_summary": {
                    "source_timing_summary_input": str(ROOT / "artifacts/reports/local_nankan_source_timing_audit.json")
                }
            },
        )

        self.assertEqual(
            readiness_surfaces["capture_loop"]["latest_summary"]["baseline_summary_input"],
            "artifacts/reports/pass_001_coverage_summary.json",
        )
        self.assertEqual(
            readiness_surfaces["capture_loop"]["latest_summary"]["source_timing_summary_input"],
            "artifacts/reports/local_nankan_source_timing_audit.json",
        )
        self.assertEqual(
            readiness_surfaces["readiness_probe"]["historical_source_timing"]["source_timing_summary_input"],
            "artifacts/reports/local_nankan_source_timing_audit.json",
        )
        self.assertEqual(
            readiness_surfaces["readiness_watcher"]["probe_summary"]["source_timing_summary_input"],
            "artifacts/reports/local_nankan_source_timing_audit.json",
        )

    def test_status_board_surfaces_benchmark_ready_from_bootstrap_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coverage_path = tmp_path / "coverage.json"
            backfill_path = tmp_path / "backfill.json"
            archive_path = tmp_path / "archive.json"
            capture_path = tmp_path / "capture_loop.json"
            probe_path = tmp_path / "probe.json"
            pre_race_handoff_path = tmp_path / "pre_race_handoff.json"
            bootstrap_handoff_path = tmp_path / "bootstrap_handoff.json"
            watcher_path = tmp_path / "watcher.json"
            output_path = tmp_path / "status_board.json"

            coverage_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": True},
                        "progress": {"current_stage": "ready_for_benchmark", "completed_targets": ["race_card", "race_result", "pedigree"]},
                        "target_states": {},
                        "external_outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            backfill_path.write_text(json.dumps({"status": "completed", "remaining_window_count": 0}), encoding="utf-8")
            archive_path.write_text(json.dumps({"status": "completed", "archive_reports": []}), encoding="utf-8")
            capture_path.write_text(json.dumps({}), encoding="utf-8")
            probe_path.write_text(json.dumps({"status": "ready", "result_ready_races": 3, "pending_result_races": 0}), encoding="utf-8")
            pre_race_handoff_path.write_text(json.dumps({"status": "completed", "current_phase": "benchmark_handoff_completed"}), encoding="utf-8")
            bootstrap_handoff_path.write_text(
                json.dumps(
                    {
                        "status": "benchmark_ready",
                        "current_phase": "bootstrap_pending",
                        "recommended_action": "run_bootstrap_command_plan",
                    }
                ),
                encoding="utf-8",
            )
            watcher_path.write_text(json.dumps({"status": "completed", "attempts": 1, "timed_out": False}), encoding="utf-8")

            with patch.object(status_board_script, "artifact_ensure_output_file_path"), patch.object(
                status_board_script, "_build_live_collect_progress", return_value={}
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_status_board.py",
                    "--coverage-snapshot",
                    str(coverage_path),
                    "--backfill-aggregate",
                    str(backfill_path),
                    "--archive-run",
                    str(archive_path),
                    "--capture-loop-manifest",
                    str(capture_path),
                    "--readiness-probe-summary",
                    str(probe_path),
                    "--pre-race-handoff-manifest",
                    str(pre_race_handoff_path),
                    "--bootstrap-handoff-manifest",
                    str(bootstrap_handoff_path),
                    "--readiness-watcher-manifest",
                    str(watcher_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["current_phase"], "bootstrap_pending")
            self.assertEqual(payload["recommended_action"], "run_bootstrap_command_plan")
            self.assertEqual(payload["readiness_surfaces"]["bootstrap_handoff"]["status"], "benchmark_ready")

    def test_status_board_surfaces_completed_bootstrap_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            coverage_path = tmp_path / "coverage.json"
            backfill_path = tmp_path / "backfill.json"
            archive_path = tmp_path / "archive.json"
            capture_path = tmp_path / "capture_loop.json"
            probe_path = tmp_path / "probe.json"
            pre_race_handoff_path = tmp_path / "pre_race_handoff.json"
            bootstrap_handoff_path = tmp_path / "bootstrap_handoff.json"
            watcher_path = tmp_path / "watcher.json"
            output_path = tmp_path / "status_board.json"

            coverage_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": True},
                        "progress": {"current_stage": "ready_for_benchmark", "completed_targets": ["race_card", "race_result", "pedigree"]},
                        "target_states": {},
                        "external_outputs": {},
                    }
                ),
                encoding="utf-8",
            )
            backfill_path.write_text(json.dumps({"status": "completed", "remaining_window_count": 0}), encoding="utf-8")
            archive_path.write_text(json.dumps({"status": "completed", "archive_reports": []}), encoding="utf-8")
            capture_path.write_text(json.dumps({}), encoding="utf-8")
            probe_path.write_text(json.dumps({"status": "ready", "result_ready_races": 3, "pending_result_races": 0}), encoding="utf-8")
            pre_race_handoff_path.write_text(json.dumps({"status": "completed", "current_phase": "benchmark_handoff_completed"}), encoding="utf-8")
            bootstrap_handoff_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "current_phase": "bootstrap_completed",
                        "recommended_action": "review_bootstrap_revision_outputs",
                    }
                ),
                encoding="utf-8",
            )
            watcher_path.write_text(json.dumps({"status": "completed", "attempts": 1, "timed_out": False}), encoding="utf-8")

            with patch.object(status_board_script, "artifact_ensure_output_file_path"), patch.object(
                status_board_script, "_build_live_collect_progress", return_value={}
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_status_board.py",
                    "--coverage-snapshot",
                    str(coverage_path),
                    "--backfill-aggregate",
                    str(backfill_path),
                    "--archive-run",
                    str(archive_path),
                    "--capture-loop-manifest",
                    str(capture_path),
                    "--readiness-probe-summary",
                    str(probe_path),
                    "--pre-race-handoff-manifest",
                    str(pre_race_handoff_path),
                    "--bootstrap-handoff-manifest",
                    str(bootstrap_handoff_path),
                    "--readiness-watcher-manifest",
                    str(watcher_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["current_phase"], "bootstrap_completed")
            self.assertEqual(payload["recommended_action"], "review_bootstrap_revision_outputs")
            self.assertEqual(payload["readiness_surfaces"]["bootstrap_handoff"]["status"], "completed")
