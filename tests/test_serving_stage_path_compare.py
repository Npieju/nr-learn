from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_serving_stage_path_compare import _build_comparison, _stage_signature


class ServingStagePathCompareTest(unittest.TestCase):
    def test_stage_signature_normalizes_empty_and_sorted_values(self) -> None:
        self.assertEqual(_stage_signature(None), "(none)")
        self.assertEqual(_stage_signature([]), "(none)")
        self.assertEqual(_stage_signature(["kelly_fallback_1", "portfolio_ev_only"]), "kelly_fallback_1|portfolio_ev_only")

    def test_build_comparison_detects_differing_stage_dates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            left = root / "left.json"
            right = root / "right.json"

            left.write_text(
                """
{
  "profile": "left",
  "artifact_suffix": "left_probe",
  "cases": [
    {
      "date": "2024-09-22",
      "status": "ok",
      "policy_name": "runtime_staged_probe",
      "policy_stage_names": ["portfolio_ev_only"],
      "policy_bets": 3,
      "policy_roi": 0.0
    },
    {
      "date": "2024-09-29",
      "status": "ok",
      "policy_name": "runtime_staged_probe",
      "policy_stage_names": ["portfolio_ev_only"],
      "policy_bets": 1,
      "policy_roi": 0.0
    }
  ]
}
""".strip()
                + "\n",
                encoding="utf-8",
            )
            right.write_text(
                """
{
  "profile": "right",
  "artifact_suffix": "right_probe",
  "cases": [
    {
      "date": "2024-09-22",
      "status": "ok",
      "policy_name": "runtime_staged_probe",
      "policy_stage_names": ["kelly_fallback_1"],
      "policy_bets": 2,
      "policy_roi": 0.0
    },
    {
      "date": "2024-09-29",
      "status": "ok",
      "policy_name": "runtime_staged_probe",
      "policy_stage_names": ["portfolio_ev_only"],
      "policy_bets": 1,
      "policy_roi": 0.0
    }
  ]
}
""".strip()
                + "\n",
                encoding="utf-8",
            )

            payload, row_df = _build_comparison([
                ("left", left),
                ("right", right),
            ])

            self.assertEqual(payload["comparison"]["shared_ok_dates_all"], ["2024-09-22", "2024-09-29"])
            self.assertEqual(payload["comparison"]["differing_policy_dates"], [])
            self.assertEqual(payload["comparison"]["differing_stage_dates"], ["2024-09-22"])
            self.assertEqual(len(row_df), 2)


if __name__ == "__main__":
    unittest.main()