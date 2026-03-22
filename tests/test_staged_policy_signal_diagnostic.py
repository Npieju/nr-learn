from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_staged_policy_signal_diagnostic import _aggregate_stage_rows, _race_stage_row


class StagedPolicySignalDiagnosticTest(unittest.TestCase):
    def test_race_stage_row_records_fallback_and_returns(self) -> None:
        stage_race = pd.DataFrame(
            [
                {
                    "horse_id": "h1",
                    "horse_name": "Horse 1",
                    "rank": 1,
                    "odds": 3.5,
                    "policy_selected": True,
                    "policy_weight": 1.0,
                    "policy_expected_value": 1.04,
                    "policy_prob": 0.30,
                    "policy_edge": 0.04,
                },
                {
                    "horse_id": "h2",
                    "horse_name": "Horse 2",
                    "rank": 3,
                    "odds": 8.0,
                    "policy_selected": False,
                    "policy_weight": 0.0,
                    "policy_expected_value": 0.90,
                    "policy_prob": 0.10,
                    "policy_edge": -0.10,
                },
            ]
        )

        row = _race_stage_row(
            stage_race,
            date_value="2024-08-03",
            race_id="r1",
            stage_index=1,
            stage_name="portfolio_ev_only",
            stage_cfg={"fallback_when": {"max_expected_value_below": 1.08}},
            odds_col="odds",
            final_stage_name="kelly_fallback_1",
            final_stage_trace="portfolio_ev_only:fallback(max_expected_value_below) > kelly_fallback_1:selected",
            final_stage_fallback_reasons="portfolio_ev_only:max_expected_value_below",
        )

        self.assertEqual(row["selected_count"], 1)
        self.assertTrue(row["fallback"])
        self.assertEqual(row["fallback_reasons"], ["max_expected_value_below"])
        self.assertEqual(row["selected_return_units"], 3.5)
        self.assertEqual(row["selected_net_units"], 2.5)
        self.assertTrue(row["selected_hit"])
        self.assertFalse(row["stage_is_final"])

    def test_aggregate_stage_rows_summarizes_reason_and_final_stage_counts(self) -> None:
        summary = _aggregate_stage_rows(
            [
                {
                    "selected_count": 1,
                    "selected_stake_units": 1.0,
                    "selected_return_units": 0.0,
                    "selected_net_units": -1.0,
                    "selected_hit": False,
                    "fallback": True,
                    "fallback_reasons": ["max_expected_value_below"],
                    "final_stage_name": "kelly_fallback_1",
                    "stage_is_final": False,
                    "max_expected_value": 1.04,
                    "max_prob": 0.22,
                    "max_edge": 0.04,
                    "ev_guard": 1.08,
                },
                {
                    "selected_count": 0,
                    "selected_stake_units": 0.0,
                    "selected_return_units": 0.0,
                    "selected_net_units": 0.0,
                    "selected_hit": False,
                    "fallback": True,
                    "fallback_reasons": ["no_selection"],
                    "final_stage_name": "kelly_fallback_1",
                    "stage_is_final": False,
                    "max_expected_value": None,
                    "max_prob": None,
                    "max_edge": None,
                    "ev_guard": 1.08,
                },
                {
                    "selected_count": 1,
                    "selected_stake_units": 1.0,
                    "selected_return_units": 2.2,
                    "selected_net_units": 1.2,
                    "selected_hit": True,
                    "fallback": False,
                    "fallback_reasons": [],
                    "final_stage_name": "portfolio_ev_only",
                    "stage_is_final": True,
                    "max_expected_value": 1.11,
                    "max_prob": 0.31,
                    "max_edge": 0.11,
                    "ev_guard": 1.08,
                },
            ]
        )

        self.assertEqual(summary["num_races"], 3)
        self.assertEqual(summary["selected_race_count"], 2)
        self.assertEqual(summary["fallback_race_count"], 2)
        self.assertEqual(summary["no_selection_race_count"], 1)
        self.assertEqual(summary["hit_race_count"], 1)
        self.assertEqual(summary["stage_is_final_count"], 1)
        self.assertEqual(summary["total_selected_net_units"], 0.19999999999999996)
        self.assertEqual(summary["ev_guard_pass_race_count"], 1)
        self.assertEqual(summary["fallback_reason_counts"], {"max_expected_value_below": 1, "no_selection": 1})
        self.assertEqual(summary["final_stage_counts"], {"kelly_fallback_1": 2, "portfolio_ev_only": 1})


if __name__ == "__main__":
    unittest.main()