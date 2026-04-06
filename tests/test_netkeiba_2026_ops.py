from __future__ import annotations

from datetime import datetime
import importlib.util
import json
from pathlib import Path
import subprocess
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


backfill_script = _load_script_module("test_run_netkeiba_2026_ytd_backfill", "scripts/run_netkeiba_2026_ytd_backfill.py")
handoff_script = _load_script_module("test_run_netkeiba_2026_live_handoff", "scripts/run_netkeiba_2026_live_handoff.py")
rollover_script = _load_script_module("test_run_netkeiba_2026_backfill_rollover", "scripts/run_netkeiba_2026_backfill_rollover.py")
same_day_ops_script = _load_script_module("test_run_netkeiba_2026_same_day_ops", "scripts/run_netkeiba_2026_same_day_ops.py")
status_board_script = _load_script_module("test_run_netkeiba_2026_status_board", "scripts/run_netkeiba_2026_status_board.py")
snapshot_script = _load_script_module("test_run_netkeiba_2026_ytd_snapshot", "scripts/run_netkeiba_2026_ytd_snapshot.py")
benchmark_gate_script = _load_script_module("test_run_netkeiba_2026_benchmark_gate", "scripts/run_netkeiba_2026_benchmark_gate.py")


class Netkeiba2026OpsTest(unittest.TestCase):
    def test_live_handoff_main_skips_rerun_when_completed_outputs_already_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            manifest_path = tmp_path / "netkeiba_2026_live_handoff_manifest.json"
            prediction_path = tmp_path / "predictions_20260405_jra_live.csv"
            report_path = tmp_path / "predictions_20260405_jra_live.report.md"

            prediction_path.write_text("race_id\n202604050811\n", encoding="utf-8")
            report_path.write_text("# report\n", encoding="utf-8")
            manifest_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "race_date": "2026-04-05",
                        "live_prediction_file": str(prediction_path),
                        "live_report_file": str(report_path),
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(handoff_script, "_refresh_status_board", return_value=0) as refresh_mock, patch.object(
                handoff_script, "_run_command", side_effect=AssertionError("_run_command should not be called")
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_live_handoff.py",
                    "--race-date",
                    "2026-04-05",
                    "--wrapper-manifest-output",
                    str(manifest_path),
                ],
            ):
                exit_code = handoff_script.main()

            self.assertEqual(exit_code, 0)
            refresh_mock.assert_called_once()

    def test_completed_handoff_reusable_requires_matching_date_and_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prediction_path = tmp_path / "predictions_20260405_jra_live.csv"
            report_path = tmp_path / "predictions_20260405_jra_live.report.md"
            prediction_path.write_text("race_id\n202604050811\n", encoding="utf-8")
            report_path.write_text("# report\n", encoding="utf-8")

            manifest = {
                "status": "completed",
                "race_date": "2026-04-05",
                "live_prediction_file": str(prediction_path),
                "live_report_file": str(report_path),
            }

            self.assertTrue(handoff_script._completed_handoff_reusable(manifest=manifest, race_date="2026-04-05"))
            self.assertFalse(handoff_script._completed_handoff_reusable(manifest=manifest, race_date="2026-04-06"))

            report_path.unlink()
            self.assertFalse(handoff_script._completed_handoff_reusable(manifest=manifest, race_date="2026-04-05"))

    def test_status_board_main_writes_waiting_top_level_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"

            backfill_path.write_text(json.dumps({"status": "running", "completed_cycles": 2}), encoding="utf-8")
            snapshot_path.write_text(json.dumps({"readiness": {}, "progress": {}}), encoding="utf-8")
            handoff_path.write_text(
                json.dumps(
                    {
                        "status": "waiting",
                        "current_phase": "await_history_ready",
                        "recommended_action": "wait_for_2026_history_frontier",
                        "race_date": "2026-04-05",
                        "history_ready_date": "2026-04-04",
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "waiting")
            self.assertEqual(payload["current_phase"], "await_history_ready")
            self.assertEqual(payload["recommended_action"], "wait_for_2026_history_frontier")
            self.assertEqual(payload["handoff"]["status"], "waiting")
            self.assertIn("status=waiting", payload["highlights"])

    def test_status_board_main_writes_handed_off_status_and_live_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"
            prediction_path = tmp_path / "predictions_20260405_jra_live.csv"
            summary_path = prediction_path.with_suffix(".summary.json")
            live_path = prediction_path.with_suffix(".live.json")

            backfill_path.write_text(json.dumps({"status": "completed", "completed_cycles": 4}), encoding="utf-8")
            snapshot_path.write_text(json.dumps({"readiness": {}, "progress": {}}), encoding="utf-8")
            handoff_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "current_phase": "live_predict_completed",
                        "recommended_action": "review_live_prediction_outputs",
                        "race_date": "2026-04-05",
                        "history_ready_date": "2026-04-04",
                        "live_prediction_file": str(prediction_path),
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(json.dumps({}), encoding="utf-8")
            summary_path.write_text(
                json.dumps(
                    {
                        "summary_file": "artifacts/predictions/predictions_20260405_jra_live.summary.json",
                        "policy_name": "runtime_portfolio_probe",
                        "policy_selected_rows": 2,
                        "records": 12,
                        "num_races": 3,
                    }
                ),
                encoding="utf-8",
            )
            live_path.write_text(json.dumps({"record_count": 12, "race_count": 3}), encoding="utf-8")

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "handed_off")
            self.assertEqual(payload["current_phase"], "live_predict_completed")
            self.assertEqual(payload["recommended_action"], "review_live_prediction_outputs")
            self.assertEqual(payload["live_outputs"]["policy_name"], "runtime_portfolio_probe")
            self.assertEqual(payload["live_outputs"]["policy_selected_rows"], 2)
            self.assertEqual(payload["live_outputs"]["num_races"], 3)
            self.assertIn("status=handed_off", payload["highlights"])

    def test_status_board_main_prioritizes_completed_benchmark_gate_over_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"

            backfill_path.write_text(json.dumps({"status": "completed", "completed_cycles": 4}), encoding="utf-8")
            snapshot_path.write_text(json.dumps({"readiness": {}, "progress": {}}), encoding="utf-8")
            handoff_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "current_phase": "live_predict_completed",
                        "recommended_action": "review_live_prediction_outputs",
                        "race_date": "2026-04-05",
                        "history_ready_date": "2026-04-04",
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "current_phase": "benchmark_gate_completed",
                        "recommended_action": "review_benchmark_outputs",
                    }
                ),
                encoding="utf-8",
            )

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["current_phase"], "benchmark_gate_completed")
            self.assertEqual(payload["recommended_action"], "review_benchmark_outputs")
            self.assertEqual(payload["benchmark_gate"]["status"], "completed")
            self.assertIn("status=completed", payload["highlights"])

    def test_status_board_main_writes_ready_before_live_handoff(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"

            backfill_path.write_text(json.dumps({"status": "completed", "completed_cycles": 4}), encoding="utf-8")
            snapshot_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": True},
                        "progress": {},
                    }
                ),
                encoding="utf-8",
            )
            handoff_path.write_text(json.dumps({}), encoding="utf-8")
            benchmark_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "ready")
            self.assertEqual(payload["current_phase"], "history_ready_for_live_handoff")
            self.assertEqual(payload["recommended_action"], "run_netkeiba_2026_live_handoff")
            self.assertIn("status=ready", payload["highlights"])

    def test_status_board_main_surfaces_timeout_handoff_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"

            backfill_path.write_text(json.dumps({"status": "completed", "completed_cycles": 4}), encoding="utf-8")
            snapshot_path.write_text(
                json.dumps(
                    {
                        "readiness": {"benchmark_rerun_ready": True},
                        "progress": {},
                    }
                ),
                encoding="utf-8",
            )
            handoff_path.write_text(
                json.dumps(
                    {
                        "status": "timeout",
                        "current_phase": "await_history_ready_timeout",
                        "recommended_action": "rerun_when_2026_history_frontier_ready",
                        "race_date": "2026-04-05",
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["current_phase"], "await_history_ready_timeout")
            self.assertEqual(payload["recommended_action"], "rerun_when_2026_history_frontier_ready")
            self.assertEqual(payload["handoff"]["status"], "timeout")
            self.assertIn("status=partial", payload["highlights"])

    def test_status_board_main_surfaces_handoff_failed_as_failed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"

            backfill_path.write_text(json.dumps({"status": "completed", "completed_cycles": 4}), encoding="utf-8")
            snapshot_path.write_text(json.dumps({"readiness": {}, "progress": {}}), encoding="utf-8")
            handoff_path.write_text(
                json.dumps(
                    {
                        "status": "handoff_failed",
                        "current_phase": "live_predict_failed",
                        "recommended_action": "inspect_live_predict_failure",
                        "race_date": "2026-04-05",
                    }
                ),
                encoding="utf-8",
            )
            benchmark_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "failed")
            self.assertEqual(payload["current_phase"], "live_predict_failed")
            self.assertEqual(payload["recommended_action"], "inspect_live_predict_failure")
            self.assertEqual(payload["handoff"]["status"], "handoff_failed")
            self.assertIn("status=failed", payload["highlights"])

    def test_status_board_main_surfaces_unknown_state_as_partial(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            backfill_path = tmp_path / "backfill.json"
            snapshot_path = tmp_path / "snapshot.json"
            handoff_path = tmp_path / "handoff.json"
            benchmark_path = tmp_path / "benchmark.json"
            output_path = tmp_path / "status_board.json"

            backfill_path.write_text(json.dumps({}), encoding="utf-8")
            snapshot_path.write_text(json.dumps({"readiness": {}, "progress": {}}), encoding="utf-8")
            handoff_path.write_text(json.dumps({}), encoding="utf-8")
            benchmark_path.write_text(json.dumps({}), encoding="utf-8")

            with patch.object(status_board_script, "ROOT", tmp_path), patch.object(
                status_board_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_status_board.py",
                    "--backfill-manifest",
                    str(backfill_path),
                    "--snapshot",
                    str(snapshot_path),
                    "--handoff-manifest",
                    str(handoff_path),
                    "--benchmark-gate-manifest",
                    str(benchmark_path),
                    "--output",
                    str(output_path),
                ],
            ):
                exit_code = status_board_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "partial")
            self.assertEqual(payload["current_phase"], "status_unknown")
            self.assertEqual(payload["recommended_action"], "inspect_2026_manifests")
            self.assertIn("status=partial", payload["highlights"])

    def test_write_readiness_manifest_preserves_status_phase_action_and_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "netkeiba_2026_live_handoff_manifest.json"

            handoff_script._write_readiness_manifest(
                wrapper_manifest_path=manifest_path,
                status="timeout",
                current_phase="await_history_ready_timeout",
                recommended_action="rerun_when_2026_history_frontier_ready",
                attempts=3,
                waited_seconds=600,
                race_date="2026-04-05",
                history_ready_date=datetime(2026, 4, 4),
                snapshot_payload={"readiness": {"snapshot_consistent": False}},
                race_result_max_date=None,
                race_card_max_date=None,
                live_command=["python", "scripts/run_jra_live_predict.py"],
                snapshot_exit=0,
                race_targets_completed=False,
                snapshot_consistent=False,
                history_dates_ready=False,
                reason="max_wait_seconds_exceeded",
            )

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        self.assertEqual(manifest["status"], "timeout")
        self.assertEqual(manifest["current_phase"], "await_history_ready_timeout")
        self.assertEqual(manifest["recommended_action"], "rerun_when_2026_history_frontier_ready")
        self.assertEqual(manifest["attempts"], 3)
        self.assertEqual(manifest["waited_seconds"], 600)
        self.assertEqual(manifest["history_ready_date"], "2026-04-04")
        self.assertIn("reason=max_wait_seconds_exceeded", manifest["error"])
        self.assertEqual(manifest["live_command"], ["python", "scripts/run_jra_live_predict.py"])

    def test_backfill_wrapper_runs_single_cycle_with_2026_manifest_and_snapshot_command(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "netkeiba_backfill_manifest_2026_ytd.json"

            with patch.object(backfill_script, "load_yaml", side_effect=[{"dataset": {}}, {"crawl": {}}]), patch.object(
                backfill_script, "artifact_ensure_output_file_path"
            ), patch.object(backfill_script, "run_netkeiba_backfill_from_config") as run_backfill, patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_ytd_backfill.py",
                    "--max-cycles",
                    "1",
                    "--manifest-file",
                    str(manifest_path),
                ],
            ):
                run_backfill.return_value = {
                    "completed_cycles": 1,
                    "stopped_reason": "max_cycles_reached",
                    "date_order": "asc",
                    "race_id_source": "race_list",
                }

                exit_code = backfill_script.main()

            self.assertEqual(exit_code, 0)
            kwargs = run_backfill.call_args.kwargs
            self.assertEqual(kwargs["start_date"], "2026-01-01")
            self.assertEqual(kwargs["max_cycles"], 1)
            self.assertEqual(kwargs["race_id_source"], "race_list")
            self.assertEqual(kwargs["post_cycle_command"], backfill_script.DEFAULT_POST_CYCLE_SNAPSHOT_COMMAND)
            self.assertEqual(kwargs["manifest_file"], str(manifest_path))

    def test_backfill_rollover_dry_run_writes_ready_manifest_at_safe_boundary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "netkeiba_2026_backfill_rollover_manifest.json"

            with patch.object(rollover_script, "ROOT", tmp_path), patch.object(
                rollover_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                rollover_script,
                "_read_lock_pid",
                return_value=(4321, {"pid": 4321, "started_at": "2026-04-06T10:00:00Z"}),
            ), patch.object(
                rollover_script, "_pid_running", return_value=True
            ), patch.object(
                rollover_script,
                "_target_summary",
                side_effect=[
                    {"status": "completed", "requested_ids": 10, "processed_ids": 10},
                    {"status": "completed", "requested_ids": 8, "processed_ids": 8},
                    {"status": "completed", "requested_ids": 5, "processed_ids": 5},
                ],
            ), patch.object(
                rollover_script, "_any_running", return_value=False
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_backfill_rollover.py",
                    "--output",
                    str(output_path),
                    "--dry-run",
                ],
            ):
                exit_code = rollover_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "dry_run_ready")
            self.assertEqual(payload["current_phase"], "ready_to_restart")
            self.assertEqual(payload["recommended_action"], "launch_rollover_without_dry_run")
            self.assertTrue(payload["safe_to_restart"])
            self.assertEqual(payload["active_pid"], 4321)

    def test_backfill_rollover_times_out_when_safe_boundary_never_arrives(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            output_path = tmp_path / "netkeiba_2026_backfill_rollover_manifest.json"

            with patch.object(rollover_script, "ROOT", tmp_path), patch.object(
                rollover_script, "artifact_ensure_output_file_path", return_value=None
            ), patch.object(
                rollover_script,
                "_read_lock_pid",
                return_value=(4321, {"pid": 4321, "started_at": "2026-04-06T10:00:00Z"}),
            ), patch.object(
                rollover_script, "_pid_running", return_value=True
            ), patch.object(
                rollover_script,
                "_target_summary",
                side_effect=[
                    {"status": "running", "requested_ids": 10, "processed_ids": 7},
                    {"status": "completed", "requested_ids": 8, "processed_ids": 8},
                    {"status": "completed", "requested_ids": 5, "processed_ids": 5},
                ],
            ), patch.object(
                rollover_script, "_any_running", return_value=True
            ), patch.object(
                rollover_script.time,
                "monotonic",
                side_effect=[0.0, 5.0],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_backfill_rollover.py",
                    "--output",
                    str(output_path),
                    "--max-wait-seconds",
                    "1",
                ],
            ):
                exit_code = rollover_script.main()

            self.assertEqual(exit_code, 1)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "timeout")
            self.assertEqual(payload["current_phase"], "wait_cycle_boundary")
            self.assertEqual(payload["recommended_action"], "re-run_rollover_or_restart_manually")
            self.assertFalse(payload["safe_to_restart"])
            self.assertIn("max wait exceeded before safe cycle boundary", payload["error"])

    def test_ytd_snapshot_runs_snapshot_then_status_board(self) -> None:
        with patch.object(
            snapshot_script.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(args=["snapshot"], returncode=0),
                subprocess.CompletedProcess(args=["board"], returncode=0),
            ],
        ) as run_mock, patch.object(
            sys,
            "argv",
            [
                "run_netkeiba_2026_ytd_snapshot.py",
                "--config",
                "configs/data_2025_latest.yaml",
                "--output",
                "artifacts/reports/test_snapshot.json",
            ],
        ):
            exit_code = snapshot_script.main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(run_mock.call_count, 2)
        snapshot_call = run_mock.call_args_list[0]
        board_call = run_mock.call_args_list[1]
        self.assertIn("scripts/run_netkeiba_coverage_snapshot.py", snapshot_call.args[0][1])
        self.assertIn("--output", snapshot_call.args[0])
        self.assertIn("artifacts/reports/test_snapshot.json", snapshot_call.args[0])
        self.assertEqual(board_call.kwargs["shell"], True)

    def test_ytd_snapshot_returns_status_board_exit_code_when_board_update_fails(self) -> None:
        with patch.object(
            snapshot_script.subprocess,
            "run",
            side_effect=[
                subprocess.CompletedProcess(args=["snapshot"], returncode=0),
                subprocess.CompletedProcess(args=["board"], returncode=7),
            ],
        ), patch.object(
            sys,
            "argv",
            ["run_netkeiba_2026_ytd_snapshot.py"],
        ):
            exit_code = snapshot_script.main()

        self.assertEqual(exit_code, 7)

    def test_benchmark_gate_forwards_optional_flags_and_status_board_failure(self) -> None:
        with patch.object(
            benchmark_gate_script,
            "_run_command",
            return_value=subprocess.CompletedProcess(args=["benchmark"], returncode=0),
        ) as run_command, patch.object(
            benchmark_gate_script.subprocess,
            "run",
            return_value=subprocess.CompletedProcess(args=["board"], returncode=5),
        ), patch.object(
            sys,
            "argv",
            [
                "run_netkeiba_2026_benchmark_gate.py",
                "--pre-feature-max-rows",
                "12345",
                "--skip-train",
                "--skip-evaluate",
            ],
        ):
            exit_code = benchmark_gate_script.main()

        self.assertEqual(exit_code, 5)
        command = run_command.call_args.kwargs["command"]
        self.assertIn("scripts/run_netkeiba_benchmark_gate.py", command[1])
        self.assertIn("--pre-feature-max-rows", command)
        self.assertIn("12345", command)
        self.assertIn("--skip-train", command)
        self.assertIn("--skip-evaluate", command)

    def test_live_handoff_timeout_writes_timeout_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            manifest_path = Path(tmp_dir) / "netkeiba_2026_live_handoff_manifest.json"
            snapshot_path = Path(tmp_dir) / "netkeiba_coverage_snapshot_2026_ytd.json"
            snapshot_path.write_text("{}", encoding="utf-8")

            with patch.object(handoff_script, "_run_command", return_value=0), patch.object(
                handoff_script,
                "_read_json_dict",
                return_value={
                    "readiness": {"snapshot_consistent": False},
                    "target_states": {
                        "race_result": {"status": "running"},
                        "race_card": {"status": "running"},
                    },
                    "progress": {},
                },
            ), patch.object(handoff_script, "_extract_external_max_date", return_value=None), patch.object(
                handoff_script, "_refresh_status_board", return_value=0
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_live_handoff.py",
                    "--race-date",
                    "2026-04-05",
                    "--snapshot-output",
                    str(snapshot_path),
                    "--wrapper-manifest-output",
                    str(manifest_path),
                ],
            ):
                exit_code = handoff_script.main()

            self.assertEqual(exit_code, 2)
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["status"], "timeout")
            self.assertEqual(manifest["current_phase"], "await_history_ready_timeout")
            self.assertEqual(manifest["recommended_action"], "rerun_when_2026_history_frontier_ready")
            self.assertIn("reason=wait_for_ready_disabled", manifest["error"])

    def test_status_board_derives_waiting_and_handed_off_states(self) -> None:
        waiting_status = status_board_script._derive_status(
            backfill={"status": "running"},
            snapshot={"readiness": {}, "progress": {}},
            handoff={"status": "waiting", "current_phase": "await_history_ready", "recommended_action": "wait_for_2026_history_frontier"},
            benchmark_gate={},
        )
        timeout_status = status_board_script._derive_status(
            backfill={"status": "completed"},
            snapshot={"readiness": {"benchmark_rerun_ready": True}, "progress": {}},
            handoff={"status": "timeout", "current_phase": "await_history_ready_timeout", "recommended_action": "rerun_when_2026_history_frontier_ready"},
            benchmark_gate={},
        )
        handed_off_status = status_board_script._derive_status(
            backfill={"status": "completed"},
            snapshot={"readiness": {}, "progress": {}},
            handoff={"status": "completed", "current_phase": "live_predict_completed", "recommended_action": "review_live_prediction_outputs"},
            benchmark_gate={},
        )
        completed_status = status_board_script._derive_status(
            backfill={"status": "completed"},
            snapshot={"readiness": {}, "progress": {}},
            handoff={"status": "completed", "current_phase": "live_predict_completed", "recommended_action": "review_live_prediction_outputs"},
            benchmark_gate={"status": "completed", "current_phase": "benchmark_gate_completed", "recommended_action": "review_benchmark_outputs"},
        )

        self.assertEqual(waiting_status, ("waiting", "await_history_ready", "wait_for_2026_history_frontier"))
        self.assertEqual(timeout_status, ("partial", "await_history_ready_timeout", "rerun_when_2026_history_frontier_ready"))
        self.assertEqual(handed_off_status, ("handed_off", "live_predict_completed", "review_live_prediction_outputs"))
        self.assertEqual(completed_status, ("completed", "benchmark_gate_completed", "review_benchmark_outputs"))

    def test_same_day_ops_treats_handed_off_board_as_already_completed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "netkeiba_2026_same_day_ops_manifest.json"
            board_payload = {
                "status": "handed_off",
                "live_outputs": {"prediction_file": "artifacts/predictions/predictions_20260405_jra_live.csv"},
            }
            handoff_payload = {
                "status": "completed",
                "live_prediction_file": "artifacts/predictions/predictions_20260405_jra_live.csv",
            }

            with patch.object(same_day_ops_script, "_refresh_status_board", return_value=0), patch.object(
                same_day_ops_script,
                "_read_json_dict",
                side_effect=[board_payload, handoff_payload, handoff_payload],
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_same_day_ops.py",
                    "--race-date",
                    "2026-04-05",
                    "--output",
                    str(output_path),
                    "--dry-run",
                ],
            ):
                exit_code = same_day_ops_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "completed")
            self.assertEqual(payload["current_phase"], "already_completed")

    def test_same_day_ops_dry_run_plans_handoff_and_rollover_when_backfill_is_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "netkeiba_2026_same_day_ops_manifest.json"
            board_payload_initial = {
                "status": "waiting",
                "current_phase": "await_history_ready",
            }
            board_payload_final = {
                "status": "waiting",
                "current_phase": "await_history_ready",
            }
            handoff_payload = {
                "status": "waiting",
                "race_date": "2026-04-05",
            }

            with patch.object(same_day_ops_script, "_refresh_status_board", return_value=0), patch.object(
                same_day_ops_script,
                "_read_json_dict",
                side_effect=[board_payload_initial, handoff_payload, handoff_payload, board_payload_final],
            ), patch.object(
                same_day_ops_script,
                "_find_running_processes",
                side_effect=[[{"pid": 1234, "command": "python scripts/run_netkeiba_2026_ytd_backfill.py"}], [], []],
            ), patch.object(
                same_day_ops_script,
                "_launch_background",
                side_effect=AssertionError("_launch_background should not run during dry-run"),
            ), patch.object(
                sys,
                "argv",
                [
                    "run_netkeiba_2026_same_day_ops.py",
                    "--race-date",
                    "2026-04-05",
                    "--output",
                    str(output_path),
                    "--dry-run",
                ],
            ):
                exit_code = same_day_ops_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["status"], "dry_run")
            self.assertEqual(payload["current_phase"], "ops_plan_ready")
            self.assertEqual(payload["actions"]["backfill"]["status"], "already_running")
            self.assertEqual(payload["actions"]["handoff"]["status"], "would_start")
            self.assertEqual(payload["actions"]["rollover"]["status"], "would_start")
            self.assertIn("backfill=already_running", payload["highlights"])
            self.assertIn("handoff=would_start", payload["highlights"])
            self.assertIn("rollover=would_start", payload["highlights"])


if __name__ == "__main__":
    unittest.main()