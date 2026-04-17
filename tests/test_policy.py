from __future__ import annotations

import math
import unittest

import pandas as pd

from racing_ml.evaluation.policy import (
    PolicyConstraints,
    StrategyCandidate,
    _is_winning_rank,
    _pick_flat_candidate,
    simulate_ev_portfolio,
)


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

    def test_pick_flat_candidate_uses_max_without_sorting(self) -> None:
        group = pd.DataFrame(
            {
                "race_id": [1, 1, 1],
                "score": [0.25, 0.40, 0.35],
                "odds": [5.0, 3.0, 7.0],
                "expected_value": [1.25, 1.20, 2.45],
                "edge": [0.05, 0.10, 0.08],
            },
            index=[10, 11, 12],
        )

        top1 = _pick_flat_candidate(group, StrategyCandidate(strategy="top1"), score_col="score", odds_col="odds")
        ev = _pick_flat_candidate(
            group,
            StrategyCandidate(strategy="ev", threshold=1.0, odds_min=1.0, odds_max=10.0),
            score_col="score",
            odds_col="odds",
        )
        edge = _pick_flat_candidate(
            group,
            StrategyCandidate(strategy="edge", threshold=0.01, odds_min=1.0, odds_max=10.0),
            score_col="score",
            odds_col="odds",
        )

        self.assertEqual(int(top1.name), 11)
        self.assertEqual(int(ev.name), 12)
        self.assertEqual(int(edge.name), 11)

    def test_policy_constraints_apply_regime_override(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-09-28", "2025-10-01"],
                "race_id": [1, 2],
            }
        )

        constraints = PolicyConstraints.from_config(
            {
                "policy": {
                    "min_bet_ratio": 0.03,
                    "min_bets_abs": 90,
                    "selection_mode": "gate_then_roi",
                    "regime_overrides": [
                        {
                            "name": "october_relaxed_support",
                            "when": {"valid_end_month_in": [10]},
                            "min_bets_abs": 45,
                            "min_bet_ratio": 0.015,
                            "selection_mode": "roi_penalized",
                        }
                    ],
                }
            },
            frame=frame,
        )

        self.assertEqual(constraints.min_bets_abs, 45)
        self.assertAlmostEqual(constraints.min_bet_ratio, 0.015)
        self.assertEqual(constraints.selection_mode, "roi_penalized")

    def test_policy_constraints_keep_base_when_override_does_not_match(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-07-01", "2025-07-15"],
                "race_id": [1, 2],
            }
        )

        constraints = PolicyConstraints.from_config(
            {
                "policy": {
                    "min_bet_ratio": 0.03,
                    "min_bets_abs": 90,
                    "regime_overrides": [
                        {
                            "name": "october_relaxed_support",
                            "when": {"valid_end_month_in": [10]},
                            "min_bets_abs": 45,
                        }
                    ],
                }
            },
            frame=frame,
        )

        self.assertEqual(constraints.min_bets_abs, 90)
        self.assertAlmostEqual(constraints.min_bet_ratio, 0.03)


if __name__ == "__main__":
    unittest.main()
