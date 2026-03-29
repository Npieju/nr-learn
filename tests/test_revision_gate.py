from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from scripts.run_revision_gate import (
    _acquire_run_lock,
    _build_challenger_equivalence_report,
    _build_lock_path,
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


if __name__ == "__main__":
    unittest.main()
