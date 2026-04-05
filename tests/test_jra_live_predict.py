from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.serving.jra_live_predict import (
    _merge_live_racecard_with_pedigree,
    _parse_live_win_odds_payload,
    build_live_prediction_report,
    summarize_live_prediction_tradeoff,
)


class JraLivePredictTest(unittest.TestCase):
    def test_merge_live_racecard_with_pedigree_keeps_live_odds_and_adds_metadata(self) -> None:
        racecard = pd.DataFrame(
            [
                {
                    "date": "2026-04-05",
                    "race_id": "202609020411",
                    "horse_id": "20260902041101",
                    "horse_key": "2019100001",
                    "horse_name": "サンプルA",
                    "gate_no": 1,
                    "odds": "3.4",
                    "popularity": "1",
                }
            ]
        )
        pedigree = pd.DataFrame(
            [
                {
                    "horse_key": "2019100001",
                    "owner_name": "owner-a",
                    "breeder_name": "breeder-a",
                    "sire_name": "sire-a",
                    "dam_name": "dam-a",
                    "damsire_name": "damsire-a",
                }
            ]
        )
        race_list = pd.DataFrame(
            [
                {
                    "date": "2026-04-05",
                    "race_id": "202609020411",
                    "race_no": 11,
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                }
            ]
        )

        merged = _merge_live_racecard_with_pedigree(
            racecard,
            pedigree,
            race_list,
            target_date="2026-04-05",
        )

        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.loc[0, "headline"], "11R 大阪杯 GI 15:40 芝2000m 15頭")
        self.assertEqual(merged.loc[0, "owner_name"], "owner-a")
        self.assertAlmostEqual(float(merged.loc[0, "odds"]), 3.4)
        self.assertEqual(int(merged.loc[0, "popularity"]), 1)
        self.assertTrue(pd.isna(merged.loc[0, "rank"]))
        self.assertEqual(int(merged.loc[0, "is_win"]), 0)

    def test_build_live_prediction_report_includes_headline_and_policy_state(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "race_id": "202609020411",
                    "headline": "11R 大阪杯 GI 15:40 芝2000m 15頭",
                    "pred_rank": 1,
                    "horse_name": "サンプルA",
                    "score": 0.321,
                    "odds": 3.4,
                    "popularity": 1,
                    "expected_value": 1.091,
                    "ev_rank": 1,
                    "policy_selected": True,
                }
            ]
        )

        report = build_live_prediction_report(predictions, race_date="2026-04-05", odds_available_rows=1)

        self.assertIn("JRA live prediction report 2026-04-05", report)
        self.assertIn("11R 大阪杯 GI 15:40 芝2000m 15頭", report)
        self.assertIn("サンプルA", report)
        self.assertIn("yes", report)
        self.assertIn("Value-First", report)
        self.assertIn("Divergence", report)

    def test_parse_live_win_odds_payload_maps_gate_to_odds_and_popularity(self) -> None:
        payload = {
            "official_datetime": "2026-04-05 14:35:19",
            "odds": {
                "1": {
                    "01": ["222.0", "0.0", 15],
                    "04": ["4.3", "0.0", 2],
                    "15": ["2.8", "0.0", 1],
                }
            },
        }

        frame = _parse_live_win_odds_payload(payload, race_id="202609020411")

        self.assertEqual(len(frame), 3)
        self.assertAlmostEqual(float(frame.loc[frame["gate_no"] == 4, "odds"].iloc[0]), 4.3)
        self.assertEqual(int(frame.loc[frame["gate_no"] == 15, "popularity"].iloc[0]), 1)
        self.assertEqual(frame.loc[frame["gate_no"] == 1, "odds_official_datetime"].iloc[0], "2026-04-05 14:35:19")

    def test_summarize_live_prediction_tradeoff_reports_top_score_top_ev_and_gap(self) -> None:
        predictions = pd.DataFrame(
            [
                {
                    "horse_name": "A",
                    "score": 0.30,
                    "expected_value": 0.90,
                    "pred_rank": 1,
                    "ev_rank": 3,
                    "policy_selected": False,
                },
                {
                    "horse_name": "B",
                    "score": 0.20,
                    "expected_value": 1.20,
                    "pred_rank": 3,
                    "ev_rank": 1,
                    "policy_selected": True,
                },
                {
                    "horse_name": "C",
                    "score": 0.25,
                    "expected_value": 1.00,
                    "pred_rank": 2,
                    "ev_rank": 2,
                    "policy_selected": False,
                },
            ]
        )

        summary = summarize_live_prediction_tradeoff(predictions)

        self.assertEqual(summary["top_score_horse"], "A")
        self.assertEqual(summary["top_expected_value_horse"], "B")
        self.assertIn(summary["largest_divergence_horse"], {"A", "B"})
        self.assertEqual(summary["policy_selected_rows"], 1)


if __name__ == "__main__":
    unittest.main()