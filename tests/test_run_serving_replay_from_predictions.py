from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scripts.run_serving_replay_from_predictions as replay_script


class RunServingReplayFromPredictionsTest(unittest.TestCase):
    def test_prepare_replay_frame_refreshes_market_columns_when_file_is_given(self) -> None:
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

            frame, refresh_summary = replay_script._prepare_replay_frame(
                prediction_path,
                market_path=market_path,
                join_keys=None,
                market_columns=None,
            )

        self.assertIsNotNone(refresh_summary)
        self.assertEqual(refresh_summary["join_keys"], ["race_id", "gate_no"])
        self.assertListEqual(frame["score"].tolist(), [0.62, 0.24])
        self.assertListEqual(frame["odds"].tolist(), [2.4, 5.1])
        self.assertListEqual(frame["popularity"].tolist(), [1, 3])

    def test_prepare_replay_frame_keeps_original_when_market_file_is_absent(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            prediction_path = tmp_path / "predictions_20251228_probe.csv"
            expected = pd.DataFrame(
                [{"race_id": "r1", "gate_no": 1, "score": 0.62, "odds": 9.9, "popularity": 9}]
            )
            expected.to_csv(prediction_path, index=False)

            frame, refresh_summary = replay_script._prepare_replay_frame(prediction_path, market_path=None)

        self.assertIsNone(refresh_summary)
        self.assertEqual(frame.to_dict(orient="records"), expected.to_dict(orient="records"))


if __name__ == "__main__":
    unittest.main()