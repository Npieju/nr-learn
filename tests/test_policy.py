from __future__ import annotations

import math
import unittest

import pandas as pd

from racing_ml.evaluation.policy import _is_winning_rank, simulate_ev_portfolio


class PolicyRankSafetyTest(unittest.TestCase):
    def test_is_winning_rank_rejects_nan(self) -> None:
        self.assertFalse(_is_winning_rank(float("nan")))
        self.assertFalse(_is_winning_rank(None))
        self.assertTrue(_is_winning_rank(1.0))

    def test_simulate_ev_portfolio_tolerates_nan_rank(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": [1, 1, 2, 2],
                "prob": [0.30, 0.20, 0.40, 0.10],
                "odds": [5.0, 7.0, 3.0, 10.0],
                "rank": [float("nan"), 2.0, 1.0, float("nan")],
            }
        )

        metrics = simulate_ev_portfolio(
            frame,
            prob_col="prob",
            odds_col="odds",
            min_prob=0.05,
            odds_min=1.0,
            odds_max=20.0,
            top_k=1,
            min_expected_value=1.0,
        )

        self.assertEqual(metrics["portfolio_bets"], 2)
        self.assertEqual(metrics["portfolio_hit_rate"], 0.5)
        self.assertTrue(math.isfinite(float(metrics["portfolio_roi"])))


if __name__ == "__main__":
    unittest.main()
