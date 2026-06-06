from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from scripts.run_serving_smoke import _prepare_replay_existing_frame, _select_cases


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

    def test_prepare_replay_existing_frame_refreshes_market_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prediction_path = tmp_path / "predictions_20251228_probe.csv"
            market_path = tmp_path / "market.csv"
            pd.DataFrame(
                [
                    {"race_id": "r1", "gate_no": 1, "score": 0.62, "odds": 9.9, "popularity": 9},
                    {"race_id": "r1", "gate_no": 2, "score": 0.24, "odds": 8.8, "popularity": 8},
                ]
            ).to_csv(prediction_path, index=False)
            pd.DataFrame(
                [
                    {"race_id": "r1", "gate_no": 1, "odds": 2.4, "popularity": 1},
                    {"race_id": "r1", "gate_no": 2, "odds": 5.1, "popularity": 3},
                ]
            ).to_csv(market_path, index=False)

            frame, refresh_summary = _prepare_replay_existing_frame(
                prediction_path,
                market_file=market_path,
                market_join_keys=None,
                market_columns=None,
            )

        self.assertIsNotNone(refresh_summary)
        self.assertEqual(refresh_summary["join_keys"], ["race_id", "gate_no"])
        self.assertListEqual(frame["score"].tolist(), [0.62, 0.24])
        self.assertListEqual(frame["odds"].tolist(), [2.4, 5.1])


if __name__ == "__main__":
    unittest.main()