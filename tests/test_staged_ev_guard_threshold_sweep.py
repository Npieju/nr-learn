from __future__ import annotations

import unittest

from scripts.run_staged_ev_guard_threshold_sweep import _aggregate_variant, _variant_config, _variant_label


class StagedEvGuardThresholdSweepTest(unittest.TestCase):
    def test_variant_config_updates_first_two_stage_thresholds(self) -> None:
        base_config = {
            "serving": {
                "policy": {
                    "strategy_kind": "staged",
                    "stages": [
                        {"name": "stage1", "fallback_when": {"max_expected_value_below": 1.08}},
                        {"name": "stage2", "fallback_when": {"max_expected_value_below": 1.03}},
                        {"name": "stage3", "policy": {"strategy_kind": "kelly"}},
                    ],
                }
            }
        }

        variant = _variant_config(base_config, 1.02, 0.99)
        stages = variant["serving"]["policy"]["stages"]

        self.assertEqual(stages[0]["fallback_when"]["max_expected_value_below"], 1.02)
        self.assertEqual(stages[1]["fallback_when"]["max_expected_value_below"], 0.99)
        self.assertEqual(base_config["serving"]["policy"]["stages"][0]["fallback_when"]["max_expected_value_below"], 1.08)

    def test_aggregate_variant_counts_kelly_and_reasons(self) -> None:
        summary = _aggregate_variant(
            [
                {
                    "date": "2024-08-03",
                    "policy_bets": 1,
                    "policy_net": -1.0,
                    "policy_stage_names": ["kelly_fallback_2"],
                    "policy_stage_traces": ["stage1:fallback(x) > stage2:selected"],
                    "policy_stage_fallback_reasons": ["stage1:max_expected_value_below"],
                },
                {
                    "date": "2024-08-04",
                    "policy_bets": 0,
                    "policy_net": 0.0,
                    "policy_stage_names": [],
                    "policy_stage_traces": [],
                    "policy_stage_fallback_reasons": [],
                },
            ]
        )

        self.assertEqual(summary["num_dates"], 2)
        self.assertEqual(summary["total_policy_bets"], 1)
        self.assertEqual(summary["total_policy_net"], -1.0)
        self.assertEqual(summary["kelly_fallback_dates"], ["2024-08-03"])
        self.assertEqual(summary["fallback_reason_dates"], ["2024-08-03"])
        self.assertEqual(summary["stage_name_counts"], {"kelly_fallback_2": 1})
        self.assertEqual(summary["stage_fallback_reason_counts"], {"stage1:max_expected_value_below": 1})

    def test_variant_label_is_stable(self) -> None:
        self.assertEqual(_variant_label(1.08, 1.03), "s1_1.08_s2_1.03")


if __name__ == "__main__":
    unittest.main()