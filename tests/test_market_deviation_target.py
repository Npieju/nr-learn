from __future__ import annotations

from pathlib import Path
import sys
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.models.trainer import _compute_market_deviation_target


class MarketDeviationTargetModeTest(unittest.TestCase):
    def setUp(self) -> None:
        self.frame = pd.DataFrame(
            [
                {"race_id": "r1", "odds": 2.0, "rank": 1, "is_win": 1},
                {"race_id": "r1", "odds": 4.0, "rank": 2, "is_win": 0},
                {"race_id": "r1", "odds": 8.0, "rank": 3, "is_win": 0},
                {"race_id": "r2", "odds": 1.6, "rank": 1, "is_win": 1},
                {"race_id": "r2", "odds": 3.5, "rank": 2, "is_win": 0},
                {"race_id": "r2", "odds": 12.0, "rank": 3, "is_win": 0},
            ]
        )

    def test_logit_residual_mode_preserves_existing_behavior(self) -> None:
        target = _compute_market_deviation_target(
            self.frame,
            label_column="is_win",
            target_mode="logit_residual",
        )

        self.assertEqual(target.shape[0], len(self.frame))
        self.assertTrue(np.isfinite(target).all())
        self.assertGreater(float(np.std(target)), 0.0)

    def test_race_normalized_residual_mode_centers_each_race(self) -> None:
        target = _compute_market_deviation_target(
            self.frame,
            label_column="is_win",
            target_mode="race_normalized_residual",
        )

        series = pd.Series(target, index=self.frame.index)
        by_race_mean = series.groupby(self.frame["race_id"]).mean()

        self.assertTrue(np.isfinite(target).all())
        self.assertTrue((by_race_mean.abs() < 1e-9).all())
        self.assertGreater(float(np.std(target)), 0.0)

    def test_unsupported_target_mode_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported market_deviation target_mode"):
            _compute_market_deviation_target(
                self.frame,
                label_column="is_win",
                target_mode="unsupported_mode",
            )


if __name__ == "__main__":
    unittest.main()