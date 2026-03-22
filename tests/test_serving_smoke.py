from __future__ import annotations

import unittest

from scripts.run_serving_smoke import _select_cases


class ServingSmokeCaseSelectionTest(unittest.TestCase):
    def test_explicit_dates_use_config_resolved_policy_instead_of_profile_preset(self) -> None:
        model_config = {
            "serving": {
                "policy": {
                    "name": "runtime_portfolio_lower_blend_probe",
                    "strategy_kind": "portfolio",
                }
            }
        }

        cases = _select_cases(
            "fallback_hybrid_june_strict",
            ["2024-08-10", "2024-08-11"],
            model_config,
        )

        self.assertEqual(
            cases,
            [
                {
                    "date": "2024-08-10",
                    "score_source": "default",
                    "policy_name": "runtime_portfolio_lower_blend_probe",
                },
                {
                    "date": "2024-08-11",
                    "score_source": "default",
                    "policy_name": "runtime_portfolio_lower_blend_probe",
                },
            ],
        )

    def test_profile_without_explicit_dates_still_uses_preset_cases(self) -> None:
        model_config = {
            "serving": {
                "policy": {
                    "name": "runtime_portfolio_lower_blend_probe",
                    "strategy_kind": "portfolio",
                }
            }
        }

        cases = _select_cases("fallback_hybrid_june_strict", None, model_config)

        self.assertTrue(cases)
        self.assertEqual(cases[0]["date"], "2024-05-25")
        self.assertEqual(cases[0]["policy_name"], "may_runtime_kelly")


if __name__ == "__main__":
    unittest.main()