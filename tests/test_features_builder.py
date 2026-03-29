from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.features.builder import _group_shifted_rolling_mean, _group_shifted_rolling_mean_many


class FeatureBuilderRollingReuseTest(unittest.TestCase):
    def test_group_shifted_rolling_mean_many_matches_single_column_helper(self) -> None:
        frame = pd.DataFrame(
            {
                "horse_history_key": ["a", "a", "a", "b", "b", "a"],
                "rank": [3, 1, 2, 4, 1, 5],
                "is_win": [0, 1, 0, 0, 1, 0],
                "time_margin_sec": [0.4, -0.2, 0.1, 0.8, -0.5, 0.3],
            }
        )

        many = _group_shifted_rolling_mean_many(
            frame,
            "horse_history_key",
            ["rank", "is_win", "time_margin_sec"],
            window=3,
        )

        for column in ["rank", "is_win", "time_margin_sec"]:
            single = _group_shifted_rolling_mean(frame, "horse_history_key", column, window=3)
            pd.testing.assert_series_equal(many[column], single, check_names=False)


if __name__ == "__main__":
    unittest.main()
