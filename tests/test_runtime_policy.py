from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.serving.runtime_policy import annotate_runtime_policy, summarize_policy_diagnostics


class RuntimePolicyStageFallbackTest(unittest.TestCase):
    def test_portfolio_policy_records_primary_reject_reasons_and_summary(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "horse_name": "A", "odds": 2.0, "score": 0.20, "pred_rank": 1, "expected_value": 0.40},
                {"race_id": "r1", "horse_name": "B", "odds": 30.0, "score": 0.10, "pred_rank": 2, "expected_value": 3.00},
                {"race_id": "r1", "horse_name": "C", "odds": 5.0, "score": 0.01, "pred_rank": 3, "expected_value": 0.05},
            ]
        )
        policy = {
            "strategy_kind": "portfolio",
            "blend_weight": 0.8,
            "min_prob": 0.05,
            "odds_min": 1.0,
            "odds_max": 25.0,
            "top_k": 1,
            "min_expected_value": 0.95,
        }

        annotated = annotate_runtime_policy(
            frame,
            odds_col="odds",
            policy_name="runtime_portfolio_probe",
            policy_config=policy,
            score_col="score",
        )
        summary = summarize_policy_diagnostics(annotated)

        self.assertEqual(annotated.loc[0, "policy_reject_reason_primary"], "expected_value_below_min_expected_value")
        self.assertEqual(annotated.loc[1, "policy_reject_reason_primary"], "odds_above_max")
        self.assertTrue(pd.notna(annotated.loc[2, "policy_reject_reason_primary"]))
        self.assertEqual(summary["policy_selected_rows"], 0)
        self.assertEqual(summary["likely_blocker_reason"], "expected_value_below_min_expected_value")
        self.assertEqual(summary["primary_reject_reason_counts"]["expected_value_below_min_expected_value"], 2)

    def test_staged_policy_records_fallback_reason_and_trace(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "odds": 2.0, "score": 0.55},
                {"race_id": "r1", "odds": 5.0, "score": 0.10},
                {"race_id": "r2", "odds": 2.0, "score": 0.80},
                {"race_id": "r2", "odds": 5.0, "score": 0.05},
                {"race_id": "r3", "odds": 2.0, "score": 0.20},
                {"race_id": "r3", "odds": 5.0, "score": 0.10},
            ]
        )
        policy = {
            "strategy_kind": "staged",
            "stages": [
                {
                    "name": "stage1",
                    "fallback_when": {"max_expected_value_below": 1.2},
                    "policy": {
                        "strategy_kind": "portfolio",
                        "blend_weight": 0.8,
                        "min_prob": 0.03,
                        "odds_min": 1.0,
                        "odds_max": 25.0,
                        "top_k": 1,
                        "min_expected_value": 1.0,
                    },
                },
                {
                    "name": "stage2",
                    "policy": {
                        "strategy_kind": "kelly",
                        "blend_weight": 0.6,
                        "min_prob": 0.03,
                        "odds_min": 1.0,
                        "odds_max": 25.0,
                        "min_edge": 0.0,
                        "fractional_kelly": 0.25,
                        "max_fraction": 0.02,
                    },
                },
            ],
        }

        annotated = annotate_runtime_policy(
            frame,
            odds_col="odds",
            policy_name="runtime_staged_probe",
            policy_config=policy,
            score_col="score",
        )

        race1 = annotated[annotated["race_id"] == "r1"].reset_index(drop=True)
        race2 = annotated[annotated["race_id"] == "r2"].reset_index(drop=True)
        race3 = annotated[annotated["race_id"] == "r3"].reset_index(drop=True)

        self.assertEqual(race1.loc[0, "policy_stage_name"], "stage2")
        self.assertEqual(race1.loc[0, "policy_stage_fallback_reasons"], "stage1:max_expected_value_below")
        self.assertEqual(race1.loc[0, "policy_stage_trace"], "stage1:fallback(max_expected_value_below) > stage2:selected")

        self.assertEqual(race2.loc[0, "policy_stage_name"], "stage1")
        self.assertTrue(pd.isna(race2.loc[0, "policy_stage_fallback_reasons"]))
        self.assertEqual(race2.loc[0, "policy_stage_trace"], "stage1:selected")

        self.assertTrue(pd.isna(race3.loc[0, "policy_stage_name"]))
        self.assertEqual(race3.loc[0, "policy_stage_fallback_reasons"], "stage1:no_selection|stage2:no_selection")
        self.assertEqual(race3.loc[0, "policy_stage_trace"], "stage1:no_selection > stage2:no_selection")

    def test_staged_policy_supports_date_selected_rows_guard(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "odds": 2.0, "score": 0.80},
                {"race_id": "r1", "odds": 5.0, "score": 0.10},
                {"race_id": "r2", "odds": 2.5, "score": 0.82},
                {"race_id": "r2", "odds": 6.0, "score": 0.05},
            ]
        )
        policy = {
            "strategy_kind": "staged",
            "stages": [
                {
                    "name": "stage1",
                    "fallback_when": {"date_selected_rows_at_most": 2},
                    "policy": {
                        "strategy_kind": "portfolio",
                        "blend_weight": 0.8,
                        "min_prob": 0.03,
                        "odds_min": 1.0,
                        "odds_max": 25.0,
                        "top_k": 1,
                        "min_expected_value": 1.0,
                    },
                },
                {
                    "name": "stage2",
                    "policy": {
                        "strategy_kind": "kelly",
                        "blend_weight": 0.6,
                        "min_prob": 0.03,
                        "odds_min": 1.0,
                        "odds_max": 25.0,
                        "min_edge": 0.0,
                        "fractional_kelly": 0.25,
                        "max_fraction": 0.02,
                    },
                },
            ],
        }

        annotated = annotate_runtime_policy(
            frame,
            odds_col="odds",
            policy_name="runtime_staged_probe",
            policy_config=policy,
            score_col="score",
        )

        race1 = annotated[annotated["race_id"] == "r1"].reset_index(drop=True)
        race2 = annotated[annotated["race_id"] == "r2"].reset_index(drop=True)

        self.assertEqual(race1.loc[0, "policy_stage_name"], "stage2")
        self.assertEqual(race2.loc[0, "policy_stage_name"], "stage2")
        self.assertEqual(race1.loc[0, "policy_stage_fallback_reasons"], "stage1:date_selected_rows_at_most")
        self.assertEqual(race2.loc[0, "policy_stage_fallback_reasons"], "stage1:date_selected_rows_at_most")
        self.assertEqual(race1.loc[0, "policy_stage_trace"], "stage1:fallback(date_selected_rows_at_most) > stage2:selected")

    def test_staged_policy_without_selection_keeps_last_stage_reject_reason(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "odds": 2.0, "score": 0.10},
                {"race_id": "r1", "odds": 5.0, "score": 0.05},
            ]
        )
        policy = {
            "strategy_kind": "staged",
            "stages": [
                {
                    "name": "stage1",
                    "fallback_when": {"selected_rows_at_most": 1},
                    "policy": {
                        "strategy_kind": "portfolio",
                        "blend_weight": 0.8,
                        "min_prob": 0.03,
                        "odds_min": 1.0,
                        "odds_max": 25.0,
                        "top_k": 1,
                        "min_expected_value": 1.1,
                    },
                },
                {
                    "name": "stage2",
                    "policy": {
                        "strategy_kind": "kelly",
                        "blend_weight": 0.8,
                        "min_prob": 0.20,
                        "odds_min": 1.0,
                        "odds_max": 25.0,
                        "min_edge": 0.01,
                        "fractional_kelly": 0.25,
                        "max_fraction": 0.02,
                    },
                },
            ],
        }

        annotated = annotate_runtime_policy(
            frame,
            odds_col="odds",
            policy_name="runtime_staged_probe",
            policy_config=policy,
            score_col="score",
        )

        self.assertTrue(pd.isna(annotated.loc[0, "policy_stage_name"]))
        self.assertTrue(pd.notna(annotated.loc[0, "policy_reject_reason_primary"]))
        self.assertEqual(annotated.loc[0, "policy_stage_fallback_reasons"], "stage1:no_selection|stage2:no_selection")


if __name__ == "__main__":
    unittest.main()