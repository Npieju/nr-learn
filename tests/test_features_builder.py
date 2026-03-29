from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.features.builder import (
    _entity_race_shifted_rolling_mean,
    _entity_race_shifted_rolling_mean_many,
    _group_shifted_rolling_mean,
    _group_shifted_rolling_mean_many,
)


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

    def test_entity_race_shifted_rolling_mean_many_matches_single_column_helper(self) -> None:
        frame = pd.DataFrame(
            {
                "race_id": [1, 1, 2, 2, 3, 3, 4, 4],
                "jockey_id": ["j1", "j1", "j1", "j1", "j2", "j2", "j1", "j1"],
                "is_win": [1, 0, 0, 1, 0, 1, 1, 0],
                "closing_time_3f": [34.8, 35.1, 35.0, 34.7, 36.0, 35.2, 34.9, 35.4],
                "corner_gain_2_to_4": [1.0, 0.5, -0.5, 0.0, 1.5, 0.2, -1.0, 0.3],
            }
        )

        many = _entity_race_shifted_rolling_mean_many(
            frame,
            "jockey_id",
            ["is_win", "closing_time_3f", "corner_gain_2_to_4"],
            window=3,
        )

        for column in ["is_win", "closing_time_3f", "corner_gain_2_to_4"]:
            single = _entity_race_shifted_rolling_mean(frame, "jockey_id", column, window=3)
            pd.testing.assert_series_equal(many[column], single, check_names=False)


if __name__ == "__main__":
    unittest.main()
