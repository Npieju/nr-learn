from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from racing_ml.evaluation.policy import PolicyConstraints
from racing_ml.evaluation.walk_forward import optimize_roi_strategy


class OptimizeRoiStrategySearchDedupTest(unittest.TestCase):
    def test_portfolio_candidates_are_not_repeated_per_min_edge(self) -> None:
        train_df = pd.DataFrame(
            {
                "race_id": [1, 1, 2, 2],
                "score": [0.2, 0.1, 0.3, 0.15],
                "odds": [3.0, 5.0, 4.0, 6.0],
                "rank": [1, 2, 1, 2],
                "is_win": [1, 0, 1, 0],
            }
        )
        valid_df = pd.DataFrame(
            {
                "race_id": [3, 3, 4, 4],
                "score": [0.25, 0.12, 0.35, 0.18],
                "odds": [3.5, 5.5, 4.5, 6.5],
                "rank": [1, 2, 1, 2],
                "is_win": [1, 0, 1, 0],
            }
        )
        constraints = PolicyConstraints(min_bet_ratio=0.0, min_bets_abs=0)
        search_config = {
            "full": {
                "blend_weights": [0.8],
                "min_edges": [0.0, 0.01],
                "min_probabilities": [0.05],
                "fractional_kelly_values": [0.25],
                "max_fraction_values": [0.02],
                "odds_mins": [1.0],
                "odds_maxs": [25.0],
                "top_ks": [1],
                "min_expected_values": [1.0],
            }
        }

        calls: list[str] = []

        def fake_run_policy_strategy(frame: pd.DataFrame, prob_col: str, odds_col: str, params: dict[str, float | str]) -> dict[str, float | int | None]:
            strategy_kind = str(params["strategy_kind"])
            calls.append(strategy_kind)
            if strategy_kind == "kelly":
                return {
                    "kelly_roi": 1.01,
                    "kelly_bets": 2,
                    "kelly_hit_rate": 0.5,
                    "kelly_final_bankroll": 1.02,
                    "kelly_max_drawdown": 0.01,
                }
            return {
                "portfolio_roi": 1.00,
                "portfolio_bets": 2,
                "portfolio_hit_rate": 0.5,
                "portfolio_final_bankroll": 1.01,
                "portfolio_max_drawdown": 0.01,
            }

        with patch("racing_ml.evaluation.walk_forward.run_policy_strategy", side_effect=fake_run_policy_strategy):
            optimize_roi_strategy(
                train_df=train_df,
                valid_df=valid_df,
                label_col="is_win",
                odds_col="odds",
                constraints=constraints,
                mode="full",
                search_config=search_config,
                progress_interval_sec=999.0,
            )

        self.assertEqual(calls.count("kelly"), 2)
        self.assertEqual(calls.count("portfolio"), 1)

    def test_isotonic_fit_runs_once_per_optimization(self) -> None:
        train_df = pd.DataFrame(
            {
                "race_id": [1, 1, 2, 2],
                "score": [0.2, 0.1, 0.3, 0.15],
                "odds": [3.0, 5.0, 4.0, 6.0],
                "rank": [1, 2, 1, 2],
                "is_win": [1, 0, 1, 0],
            }
        )
        valid_df = pd.DataFrame(
            {
                "race_id": [3, 3, 4, 4],
                "score": [0.25, 0.12, 0.35, 0.18],
                "odds": [3.5, 5.5, 4.5, 6.5],
                "rank": [1, 2, 1, 2],
                "is_win": [1, 0, 1, 0],
            }
        )
        constraints = PolicyConstraints(min_bet_ratio=0.0, min_bets_abs=0)
        search_config = {
            "fast": {
                "blend_weights": [0.8],
                "min_edges": [0.0, 0.01],
                "min_probabilities": [0.05],
                "fractional_kelly_values": [0.25],
                "max_fraction_values": [0.02],
                "odds_mins": [1.0],
                "odds_maxs": [25.0],
                "top_ks": [1],
                "min_expected_values": [1.0],
            }
        }

        fit_calls = 0

        class FakeIsotonicModel:
            def transform(self, values):
                return values

        def fake_fit_isotonic_model(train_scores, train_labels):
            nonlocal fit_calls
            fit_calls += 1
            return FakeIsotonicModel()

        def fake_run_policy_strategy(frame: pd.DataFrame, prob_col: str, odds_col: str, params: dict[str, float | str]) -> dict[str, float | int | None]:
            strategy_kind = str(params["strategy_kind"])
            if strategy_kind == "kelly":
                return {
                    "kelly_roi": 1.01,
                    "kelly_bets": 2,
                    "kelly_hit_rate": 0.5,
                    "kelly_final_bankroll": 1.02,
                    "kelly_max_drawdown": 0.01,
                }
            return {
                "portfolio_roi": 1.00,
                "portfolio_bets": 2,
                "portfolio_hit_rate": 0.5,
                "portfolio_final_bankroll": 1.01,
                "portfolio_max_drawdown": 0.01,
            }

        with patch("racing_ml.evaluation.walk_forward.fit_isotonic_model", side_effect=fake_fit_isotonic_model):
            with patch("racing_ml.evaluation.walk_forward.run_policy_strategy", side_effect=fake_run_policy_strategy):
                optimize_roi_strategy(
                    train_df=train_df,
                    valid_df=valid_df,
                    label_col="is_win",
                    odds_col="odds",
                    constraints=constraints,
                    mode="fast",
                    search_config=search_config,
                    progress_interval_sec=999.0,
                )

        self.assertEqual(fit_calls, 1)


if __name__ == "__main__":
    unittest.main()
