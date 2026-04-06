from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import scripts.run_revision_gate as revision_gate_script
from scripts.run_revision_gate import (
    _acquire_run_lock,
    _build_evaluation_promotion_alignment_report,
    _build_running_manifest_payload,
    _build_challenger_equivalence_report,
    _build_lock_path,
    _write_manifest,
    _classify_step_failure,
)


class RevisionGateFailureClassificationTest(unittest.TestCase):
    def test_wf_window_coverage_failure_is_classified_for_soft_block(self) -> None:
        result = subprocess.CompletedProcess(
            args=["wf"],
            returncode=1,
            stdout="No nested walk-forward slices available for the requested window\n",
            stderr="",
        )

        failure = _classify_step_failure(step_name="wf_feasibility", result=result)

        self.assertEqual(failure["error_code"], "insufficient_wf_window_coverage")
        self.assertEqual(failure["current_phase"], "wf_feasibility")

    def test_missing_odds_failure_is_classified_for_soft_block(self) -> None:
        result = subprocess.CompletedProcess(
            args=["wf"],
            returncode=1,
            stdout="",
            stderr="RuntimeError: Odds column is required for feasibility diagnostic\n",
        )

        failure = _classify_step_failure(step_name="wf_feasibility", result=result)

        self.assertEqual(failure["error_code"], "missing_market_odds")
        self.assertEqual(failure["recommended_action"], "populate_historical_odds_or_accept_formal_block")


class RevisionGateDuplicateRunGuardTest(unittest.TestCase):
    def test_build_lock_path_uses_revision_sidecar_for_default_manifest(self) -> None:
        manifest_output = Path("/tmp/revision_gate_r20260329_sample.json")
        lock_path = _build_lock_path(manifest_output, revision_slug="r20260329_sample")
        self.assertEqual(lock_path, Path("/tmp/revision_gate_r20260329_sample.json.lock"))

    def test_acquire_run_lock_returns_existing_live_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            lock_path = Path(tmpdir) / "revision_gate_r20260329_sample.lock"
            manifest_output = Path(tmpdir) / "revision_gate_r20260329_sample.json"
            existing = {
                "pid": os.getpid(),
                "revision": "r20260329_sample",
                "manifest_output": str(manifest_output),
                "started_at": "2026-03-29T00:00:00Z",
            }
            lock_path.write_text(json.dumps(existing), encoding="utf-8")

            returned = _acquire_run_lock(
                lock_path=lock_path,
                revision_slug="r20260329_sample",
                manifest_output=manifest_output,
            )

            self.assertEqual(returned["pid"], os.getpid())
            self.assertEqual(returned["revision"], "r20260329_sample")


class RevisionGateChallengerEquivalenceTest(unittest.TestCase):
    def test_equivalence_report_marks_identical_summaries_equivalent(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor_path = Path(tmpdir) / "anchor.json"
            candidate_path = Path(tmpdir) / "candidate.json"
            payload = {
                "stability_assessment": "representative",
                "auc": 0.84,
                "top1_roi": 0.8,
                "ev_top1_roi": 0.55,
                "wf_nested_actual_folds": 5,
                "wf_nested_test_roi_weighted": 0.91,
                "wf_nested_test_roi_mean": 0.89,
                "wf_nested_test_bets_total": 410,
            }
            anchor_path.write_text(json.dumps(payload), encoding="utf-8")
            candidate_path.write_text(json.dumps(payload), encoding="utf-8")

            report = _build_challenger_equivalence_report(
                anchor_summary_path=anchor_path,
                candidate_summary_path=candidate_path,
                tolerance=0.0,
            )

            self.assertEqual(report["status"], "equivalent")
            self.assertTrue(all(item["equivalent"] for item in report["comparisons"]))

    def test_equivalence_report_marks_changed_summary_different(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            anchor_path = Path(tmpdir) / "anchor.json"
            candidate_path = Path(tmpdir) / "candidate.json"
            anchor_payload = {
                "stability_assessment": "representative",
                "auc": 0.84,
                "top1_roi": 0.8,
                "ev_top1_roi": 0.55,
                "wf_nested_actual_folds": 5,
                "wf_nested_test_roi_weighted": 0.91,
                "wf_nested_test_roi_mean": 0.89,
                "wf_nested_test_bets_total": 410,
            }
            candidate_payload = dict(anchor_payload)
            candidate_payload["wf_nested_test_roi_weighted"] = 0.93
            anchor_path.write_text(json.dumps(anchor_payload), encoding="utf-8")
            candidate_path.write_text(json.dumps(candidate_payload), encoding="utf-8")

            report = _build_challenger_equivalence_report(
                anchor_summary_path=anchor_path,
                candidate_summary_path=candidate_path,
                tolerance=0.0,
            )

            self.assertEqual(report["status"], "different")
            differing = [item["field"] for item in report["comparisons"] if not item["equivalent"]]
            self.assertEqual(differing, ["wf_nested_test_roi_weighted"])


class RevisionGatePromotionAlignmentTest(unittest.TestCase):
    def test_alignment_report_triggers_for_all_no_bet_zero_bets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "evaluation_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "wf_nested_actual_folds": 3,
                        "wf_nested_test_bets_total": 0,
                        "wf_nested_folds": [
                            {"fold": 1, "strategy_kind": "no_bet", "test_bets": 0},
                            {"fold": 2, "strategy_kind": "no_bet", "test_bets": 0},
                            {"fold": 3, "strategy_kind": "no_bet", "test_bets": 0},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = _build_evaluation_promotion_alignment_report(
                evaluation_summary_path=summary_path
            )

            self.assertEqual(report["status"], "triggered")
            self.assertEqual(report["reason"], "evaluation_nested_all_no_bet")
            self.assertEqual(report["wf_nested_test_bets_total"], 0)

    def test_alignment_report_remains_clear_when_nested_policy_places_bets(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = Path(tmpdir) / "evaluation_summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        "wf_nested_actual_folds": 3,
                        "wf_nested_test_bets_total": 2302,
                        "wf_nested_folds": [
                            {"fold": 1, "strategy_kind": "portfolio", "test_bets": 911},
                            {"fold": 2, "strategy_kind": "portfolio", "test_bets": 701},
                            {"fold": 3, "strategy_kind": "portfolio", "test_bets": 690},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            report = _build_evaluation_promotion_alignment_report(
                evaluation_summary_path=summary_path
            )

            self.assertEqual(report["status"], "clear")
            self.assertEqual(report["reason"], "evaluation_nested_has_feasible_policy")
            self.assertEqual(report["wf_nested_test_bets_total"], 2302)


class RevisionGateRunningManifestTest(unittest.TestCase):
    def test_running_manifest_overwrites_stale_failed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_output = tmp / "revision_gate_r20260331_sample.json"
            wf_summary_output = tmp / "wf_summary.json"
            promotion_output = tmp / "promotion_gate.json"
            stale_payload = {
                "revision": "r20260331_sample",
                "status": "failed",
                "decision": "error",
                "current_phase": "train",
            }
            manifest_output.write_text(json.dumps(stale_payload), encoding="utf-8")

            running_payload = _build_running_manifest_payload(
                revision_slug="r20260331_sample",
                started_at="2026-03-31T00:00:00Z",
                resolved_profile=None,
                config_path="configs/model.yaml",
                data_config_path="configs/data.yaml",
                feature_config_path="configs/features.yaml",
                train_artifact_suffix="r20260331_sample",
                skip_train=False,
                train_max_train_rows=None,
                train_max_valid_rows=None,
                evaluate_model_artifact_suffix=None,
                evaluate_max_rows=120000,
                evaluate_pre_feature_max_rows=None,
                evaluate_start_date=None,
                evaluate_end_date=None,
                evaluate_wf_mode="fast",
                evaluate_wf_scheme="nested",
                wf_summary_output=wf_summary_output,
                promotion_min_feasible_folds=1,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=[],
                challenger_equivalence=None,
            )
            _write_manifest(manifest_output, running_payload, label="test running manifest overwrite")

            overwritten = json.loads(manifest_output.read_text(encoding="utf-8"))
            self.assertEqual(overwritten["status"], "running")
            self.assertEqual(overwritten["decision"], "in_progress")
            self.assertEqual(overwritten["current_phase"], "train")
            self.assertEqual(overwritten["recommended_action"], "wait_for_revision_gate_completion")

    def test_dry_run_propagates_no_model_artifact_suffix_to_evaluate_and_wf(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            manifest_output = tmp / "revision_gate_test.json"
            wf_summary_output = tmp / "wf_summary.json"
            promotion_output = tmp / "promotion_gate.json"
            log_file = tmp / "revision_gate.log"

            with patch.object(
                sys,
                "argv",
                [
                    "run_revision_gate.py",
                    "--revision",
                    "r_test",
                    "--config",
                    "configs/model_catboost_value_stack_lgbm_roi_high_coverage_diag.yaml",
                    "--data-config",
                    "configs/data.yaml",
                    "--feature-config",
                    "configs/features_catboost_rich_high_coverage_diag.yaml",
                    "--evaluate-no-model-artifact-suffix",
                    "--manifest-output",
                    str(manifest_output),
                    "--wf-summary-output",
                    str(wf_summary_output),
                    "--promotion-output",
                    str(promotion_output),
                    "--log-file",
                    str(log_file),
                    "--dry-run",
                ],
            ):
                exit_code = revision_gate_script.main()

            self.assertEqual(exit_code, 0)
            payload = json.loads(manifest_output.read_text(encoding="utf-8"))
            evaluate_step = next(step for step in payload["steps"] if step["name"] == "evaluate")
            wf_step = next(step for step in payload["steps"] if step["name"] == "wf_feasibility")
            self.assertIn("--model-artifact-suffix", evaluate_step["command"])
            self.assertIn(revision_gate_script.NO_MODEL_ARTIFACT_SUFFIX, evaluate_step["command"])
            self.assertIn("--model-artifact-suffix", wf_step["command"])
            self.assertIn(revision_gate_script.NO_MODEL_ARTIFACT_SUFFIX, wf_step["command"])


if __name__ == "__main__":
    unittest.main()
