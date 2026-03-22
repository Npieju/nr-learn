from __future__ import annotations

import unittest

from scripts.run_portfolio_policy_family_sweep import _aggregate_variant, _variant_config, _variant_label


class PortfolioPolicyFamilySweepTest(unittest.TestCase):
    def test_variant_config_replaces_serving_policy_and_drops_overrides(self) -> None:
        base_config = {
            "serving": {
                "policy": {"name": "base", "strategy_kind": "portfolio", "blend_weight": 0.8, "min_prob": 0.03, "min_expected_value": 0.95},
                "policy_regime_overrides": [{"name": "aug", "policy": {"strategy_kind": "portfolio"}}],
            }
        }

        variant = _variant_config(
            base_config,
            policy_name="aug_probe",
            base_policy={"strategy_kind": "portfolio", "blend_weight": 0.8, "min_prob": 0.03, "min_expected_value": 0.95},
            blend_weight=0.6,
            min_prob=0.05,
            min_expected_value=1.0,
        )

        policy = variant["serving"]["policy"]
        self.assertEqual(policy["name"], "aug_probe")
        self.assertEqual(policy["blend_weight"], 0.6)
        self.assertEqual(policy["min_prob"], 0.05)
        self.assertEqual(policy["min_expected_value"], 1.0)
        self.assertNotIn("policy_regime_overrides", variant["serving"])
        self.assertIn("policy_regime_overrides", base_config["serving"])

    def test_aggregate_variant_tracks_profitable_and_zero_bet_dates(self) -> None:
        summary = _aggregate_variant(
            [
                {"date": "2024-08-03", "policy_bets": 2, "policy_net": -2.0},
                {"date": "2024-08-11", "policy_bets": 5, "policy_net": 3.8},
                {"date": "2024-08-17", "policy_bets": 0, "policy_net": 0.0},
            ]
        )

        self.assertEqual(summary["num_dates"], 3)
        self.assertEqual(summary["total_policy_bets"], 7)
        self.assertEqual(summary["total_policy_net"], 1.7999999999999998)
        self.assertEqual(summary["profitable_dates"], ["2024-08-11"])
        self.assertEqual(summary["losing_dates"], ["2024-08-03"])
        self.assertEqual(summary["zero_bet_dates"], ["2024-08-17"])

    def test_variant_label_is_stable(self) -> None:
        self.assertEqual(_variant_label(0.8, 0.03, 0.95), "blend_0.80_prob_0.03_ev_0.95")


if __name__ == "__main__":
    unittest.main()