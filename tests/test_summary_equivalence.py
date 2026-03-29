from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from racing_ml.evaluation.summary_equivalence import DEFAULT_IGNORED_PATHS, compare_summary_files


class SummaryEquivalenceTest(unittest.TestCase):
    def test_compare_summary_files_ignores_default_volatile_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left.json"
            right_path = Path(tmpdir) / "right.json"
            left_path.write_text(
                json.dumps(
                    {
                        "auc": 0.84,
                        "output_files": {"versioned_summary": "left.json"},
                        "run_context": {
                            "artifact_suffix": "left",
                            "model_artifact_suffix": "m1",
                            "artifact_manifest": "left.manifest.json",
                        },
                    }
                )
            )
            right_path.write_text(
                json.dumps(
                    {
                        "auc": 0.84,
                        "output_files": {"versioned_summary": "right.json"},
                        "run_context": {
                            "artifact_suffix": "right",
                            "model_artifact_suffix": "m2",
                            "artifact_manifest": "right.manifest.json",
                        },
                    }
                )
            )

            result = compare_summary_files(
                left_summary=left_path,
                right_summary=right_path,
                ignored_paths=set(DEFAULT_IGNORED_PATHS),
            )

            self.assertTrue(result["exact_equal"])
            self.assertEqual(result["difference_count"], 0)

    def test_compare_summary_files_reports_value_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            left_path = Path(tmpdir) / "left.json"
            right_path = Path(tmpdir) / "right.json"
            left_path.write_text(json.dumps({"auc": 0.84, "top1_roi": 1.1}))
            right_path.write_text(json.dumps({"auc": 0.84, "top1_roi": 1.2}))

            result = compare_summary_files(
                left_summary=left_path,
                right_summary=right_path,
                ignored_paths=set(DEFAULT_IGNORED_PATHS),
            )

            self.assertFalse(result["exact_equal"])
            self.assertEqual(result["difference_count"], 1)
            self.assertEqual(result["difference_samples"][0]["path"], "top1_roi")
