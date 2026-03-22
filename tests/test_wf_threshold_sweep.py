from __future__ import annotations

import unittest

from scripts.run_wf_threshold_sweep import _strictest_threshold_matching


class WfThresholdSweepTest(unittest.TestCase):
    def test_strictest_threshold_matching_prefers_highest_passing_threshold(self) -> None:
        analyses = [
            {"policy_constraints": {"min_bets_abs": 30}, "feasible_fold_count": 5},
            {"policy_constraints": {"min_bets_abs": 40}, "feasible_fold_count": 4},
            {"policy_constraints": {"min_bets_abs": 58}, "feasible_fold_count": 1},
            {"policy_constraints": {"min_bets_abs": 60}, "feasible_fold_count": 0},
            {"policy_constraints": {"min_bets_abs": 100}, "feasible_fold_count": 0},
        ]

        result = _strictest_threshold_matching(
            analyses,
            lambda analysis: int(analysis.get("feasible_fold_count") or 0) > 0,
        )

        self.assertEqual(result, 58)

    def test_strictest_threshold_matching_returns_none_when_no_threshold_passes(self) -> None:
        analyses = [
            {"policy_constraints": {"min_bets_abs": 30}, "feasible_fold_count": 0},
            {"policy_constraints": {"min_bets_abs": 40}, "feasible_fold_count": 0},
        ]

        result = _strictest_threshold_matching(
            analyses,
            lambda analysis: int(analysis.get("feasible_fold_count") or 0) > 0,
        )

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()