from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_staged_trace_date_report import _aggregate_date_rows, _trace_depth


class StagedTraceDateReportTest(unittest.TestCase):
    def test_trace_depth_counts_steps(self) -> None:
        self.assertEqual(_trace_depth(None), 0)
        self.assertEqual(_trace_depth("portfolio:selected"), 1)
        self.assertEqual(_trace_depth("portfolio:fallback(x) > kelly:selected"), 2)
        self.assertEqual(_trace_depth("a > b > c"), 3)

    def test_aggregate_date_rows_summarizes_final_stages_and_depth(self) -> None:
        raw_rows = pd.DataFrame(
            [
                {
                    "date": "2024-09-28",
                    "race_id": "r1",
                    "stage_index": 1,
                    "stage_name": "portfolio_aug_baseline",
                    "selected_count": 1,
                    "final_stage_name": "kelly_fallback_1",
                    "final_stage_trace": "portfolio_aug_baseline:fallback(selected_rows_at_most) > kelly_fallback_1:selected",
                    "final_stage_fallback_reasons": "portfolio_aug_baseline:selected_rows_at_most",
                    "stage_is_final": False,
                    "selected_hit": False,
                    "selected_net_units": -1.0,
                },
                {
                    "date": "2024-09-28",
                    "race_id": "r1",
                    "stage_index": 2,
                    "stage_name": "kelly_fallback_1",
                    "selected_count": 1,
                    "final_stage_name": "kelly_fallback_1",
                    "final_stage_trace": "portfolio_aug_baseline:fallback(selected_rows_at_most) > kelly_fallback_1:selected",
                    "final_stage_fallback_reasons": "portfolio_aug_baseline:selected_rows_at_most",
                    "stage_is_final": True,
                    "selected_hit": False,
                    "selected_net_units": -0.01,
                },
                {
                    "date": "2024-09-28",
                    "race_id": "r2",
                    "stage_index": 1,
                    "stage_name": "portfolio_aug_baseline",
                    "selected_count": 0,
                    "final_stage_name": None,
                    "final_stage_trace": "portfolio_aug_baseline:no_selection > kelly_fallback_1:no_selection > kelly_fallback_2:no_selection",
                    "final_stage_fallback_reasons": "portfolio_aug_baseline:no_selection|kelly_fallback_1:no_selection|kelly_fallback_2:no_selection",
                    "stage_is_final": False,
                    "selected_hit": False,
                    "selected_net_units": 0.0,
                },
                {
                    "date": "2024-09-28",
                    "race_id": "r3",
                    "stage_index": 1,
                    "stage_name": "portfolio_aug_baseline",
                    "selected_count": 1,
                    "final_stage_name": "kelly_fallback_2",
                    "final_stage_trace": "portfolio_aug_baseline:fallback(selected_rows_at_most) > kelly_fallback_1:no_selection > kelly_fallback_2:selected",
                    "final_stage_fallback_reasons": "portfolio_aug_baseline:selected_rows_at_most|kelly_fallback_1:no_selection",
                    "stage_is_final": False,
                    "selected_hit": False,
                    "selected_net_units": -1.0,
                },
                {
                    "date": "2024-09-28",
                    "race_id": "r3",
                    "stage_index": 3,
                    "stage_name": "kelly_fallback_2",
                    "selected_count": 1,
                    "final_stage_name": "kelly_fallback_2",
                    "final_stage_trace": "portfolio_aug_baseline:fallback(selected_rows_at_most) > kelly_fallback_1:no_selection > kelly_fallback_2:selected",
                    "final_stage_fallback_reasons": "portfolio_aug_baseline:selected_rows_at_most|kelly_fallback_1:no_selection",
                    "stage_is_final": True,
                    "selected_hit": True,
                    "selected_net_units": 0.5,
                },
            ]
        )

        summary = _aggregate_date_rows(raw_rows)

        self.assertEqual(summary["num_races"], 3)
        self.assertEqual(summary["races_with_final_selection"], 2)
        self.assertEqual(summary["no_final_selection_race_count"], 1)
        self.assertEqual(summary["final_selected_hit_count"], 1)
        self.assertAlmostEqual(summary["final_selected_net_units"], 0.49)
        self.assertEqual(summary["max_trace_depth"], 3)
        self.assertEqual(summary["stage2_plus_trace_race_count"], 3)
        self.assertEqual(summary["stage3_plus_trace_race_count"], 2)
        self.assertEqual(summary["max_selected_stage_index"], 3)
        self.assertEqual(summary["stage2_plus_selected_race_count"], 2)
        self.assertEqual(summary["stage3_plus_selected_race_count"], 1)
        self.assertEqual(summary["final_stage_counts"], {"kelly_fallback_1": 1, "kelly_fallback_2": 1})
        self.assertEqual(summary["deepest_selected_stage_counts"], {"kelly_fallback_1": 1, "kelly_fallback_2": 1})
        self.assertTrue(summary["reaches_stage3"])


if __name__ == "__main__":
    unittest.main()