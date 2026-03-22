from __future__ import annotations

import unittest

from scripts.run_staged_date_selected_rows_sweep import _aggregate_variant, _variant_config, _variant_label


class StagedDateSelectedRowsSweepTest(unittest.TestCase):
    def test_variant_config_updates_stage1_date_selected_rows_threshold(self) -> None:
        base_config = {
            "serving": {
                "policy": {
                    "strategy_kind": "staged",
                    "stages": [
                        {"name": "stage1", "fallback_when": {"date_selected_rows_at_most": 5}},
                        {"name": "stage2", "policy": {"strategy_kind": "kelly"}},
                    ],
                }
            }
        }

        variant = _variant_config(base_config, 3)
        stages = variant["serving"]["policy"]["stages"]

        self.assertEqual(stages[0]["fallback_when"]["date_selected_rows_at_most"], 3)
        self.assertEqual(base_config["serving"]["policy"]["stages"][0]["fallback_when"]["date_selected_rows_at_most"], 5)

    def test_aggregate_variant_counts_kelly_and_reasons(self) -> None:
        summary = _aggregate_variant(
            [
                {
                    "date": "2024-09-21",
                    "policy_bets": 1,
                    "policy_net": -1.0,
                    "policy_stage_names": ["kelly_fallback_1"],
                    "policy_stage_traces": ["portfolio_sep_baseline:fallback(date_selected_rows_at_most) > kelly_fallback_1:selected"],
                    "policy_stage_fallback_reasons": ["portfolio_sep_baseline:date_selected_rows_at_most"],
                },
                {
                    "date": "2024-09-22",
                    "policy_bets": 10,
                    "policy_net": -10.0,
                    "policy_stage_names": [],
                    "policy_stage_traces": [],
                    "policy_stage_fallback_reasons": [],
                },
            ]
        )

        self.assertEqual(summary["num_dates"], 2)
        self.assertEqual(summary["total_policy_bets"], 11)
        self.assertEqual(summary["total_policy_net"], -11.0)
        self.assertEqual(summary["kelly_fallback_dates"], ["2024-09-21"])
        self.assertEqual(summary["fallback_reason_dates"], ["2024-09-21"])
        self.assertEqual(summary["stage_name_counts"], {"kelly_fallback_1": 1})
        self.assertEqual(summary["stage_fallback_reason_counts"], {"portfolio_sep_baseline:date_selected_rows_at_most": 1})

    def test_variant_label_is_stable(self) -> None:
        self.assertEqual(_variant_label(5), "date_rows_at_most_5")


if __name__ == "__main__":
    unittest.main()