from __future__ import annotations

from pathlib import Path
import sys
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.serving.prediction_market_refresh import refresh_prediction_market_data, resolve_market_join_keys


class PredictionMarketRefreshTest(unittest.TestCase):
    def test_refresh_updates_market_columns_without_touching_scores(self) -> None:
        predictions = pd.DataFrame(
            [
                {"race_id": "r1", "gate_no": 1, "score": 0.62, "pred_rank": 1, "odds": 9.9, "popularity": 9},
                {"race_id": "r1", "gate_no": 2, "score": 0.24, "pred_rank": 2, "odds": 8.8, "popularity": 8},
            ]
        )
        market = pd.DataFrame(
            [
                {"race_id": "r1", "gate_no": 1, "odds": 2.4, "popularity": 1, "odds_updated_at": "2026-06-06T04:00:00Z"},
                {"race_id": "r1", "gate_no": 2, "odds": 5.1, "popularity": 3, "odds_updated_at": "2026-06-06T04:00:00Z"},
            ]
        )

        refreshed, summary = refresh_prediction_market_data(predictions, market)

        self.assertEqual(summary["join_keys"], ["race_id", "gate_no"])
        self.assertEqual(summary["rows_with_odds_after_refresh"], 2)
        self.assertListEqual(refreshed["score"].tolist(), [0.62, 0.24])
        self.assertListEqual(refreshed["pred_rank"].tolist(), [1, 2])
        self.assertListEqual(refreshed["odds"].tolist(), [2.4, 5.1])
        self.assertListEqual(refreshed["popularity"].tolist(), [1, 3])

    def test_refresh_deduplicates_market_rows_by_last_snapshot(self) -> None:
        predictions = pd.DataFrame(
            [{"race_id": "r1", "horse_id": "h1", "score": 0.55}]
        )
        market = pd.DataFrame(
            [
                {"race_id": "r1", "horse_id": "h1", "odds": 4.5, "popularity": 4},
                {"race_id": "r1", "horse_id": "h1", "odds": 3.8, "popularity": 2},
            ]
        )

        refreshed, summary = refresh_prediction_market_data(predictions, market, join_keys=["race_id", "horse_id"])

        self.assertEqual(summary["market_duplicate_rows_dropped"], 2)
        self.assertEqual(refreshed.loc[0, "odds"], 3.8)
        self.assertEqual(refreshed.loc[0, "popularity"], 2)

    def test_resolve_market_join_keys_prefers_race_gate_path(self) -> None:
        predictions = pd.DataFrame([{"race_id": "r1", "gate_no": 1, "horse_id": "h1"}])
        market = pd.DataFrame([{"race_id": "r1", "gate_no": 1, "horse_id": "h1"}])

        resolved = resolve_market_join_keys(predictions, market)

        self.assertEqual(resolved, ["race_id", "gate_no"])


if __name__ == "__main__":
    unittest.main()