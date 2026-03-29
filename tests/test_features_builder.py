from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.features.builder import (
    _build_early_corner_position,
    _build_surface_series,
    _entity_race_shifted_rolling_mean,
    _entity_race_shifted_rolling_mean_many,
    _group_shifted_rolling_mean,
    _group_shifted_rolling_mean_many,
)


class FeatureBuilderRollingReuseTest(unittest.TestCase):
    def test_build_surface_series_prefers_first_non_empty_surface(self) -> None:
        frame = pd.DataFrame(
            {
                "芝・ダート区分": ["芝", None, "", None],
                "芝・ダート区分2": [None, "ダート", "芝", None],
            }
        )

        actual = _build_surface_series(frame)
        expected = pd.Series(["芝", "ダート", "芝", pd.NA], dtype="string")
        pd.testing.assert_series_equal(actual, expected, check_names=False)

    def test_build_early_corner_position_prefers_corner_2_then_fallbacks(self) -> None:
        frame = pd.DataFrame(
            {
                "corner_2_position": [2, None, None, 5],
                "corner_1_position": [1, 4, None, 6],
                "corner_3_position": [3, 7, 8, 9],
            }
        )

        actual = _build_early_corner_position(frame)
        expected = pd.Series([2.0, 4.0, 8.0, 5.0], dtype=float)
        pd.testing.assert_series_equal(actual, expected, check_names=False)

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
