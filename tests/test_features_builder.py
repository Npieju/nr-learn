from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.features.builder import (
    _build_early_corner_position,
    _build_surface_series,
    build_features,
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

    def test_build_features_adds_rest_surface_and_class_interactions(self) -> None:
        frame = pd.DataFrame(
            {
                "date": ["2025-01-01", "2025-01-10", "2025-05-01"],
                "race_id": ["r1", "r2", "r3"],
                "horse_id": ["h1", "h1", "h1"],
                "horse_name": ["horse-1", "horse-1", "horse-1"],
                "track": ["tokyo", "tokyo", "tokyo"],
                "distance": [1200, 1400, 1400],
                "weight": [500, 498, 502],
                "rank": [2, 1, 3],
                "芝・ダート区分": ["芝", "ダート", "ダート"],
                "競争条件": ["1勝クラス", "2勝クラス", "1勝クラス"],
                "is_win": [0, 1, 0],
            }
        )

        actual = build_features(frame)

        self.assertTrue(pd.isna(actual.loc[0, "horse_surface_switch_short_turnaround"]))
        self.assertEqual(actual.loc[1, "horse_surface_switch_short_turnaround"], 1.0)
        self.assertEqual(actual.loc[1, "horse_class_up_short_turnaround"], 1.0)
        self.assertEqual(actual.loc[1, "horse_class_down_short_turnaround"], 0.0)
        self.assertEqual(actual.loc[2, "horse_surface_switch_long_layoff"], 0.0)
        self.assertEqual(actual.loc[2, "horse_class_down_long_layoff"], 1.0)
        self.assertEqual(actual.loc[2, "horse_class_up_long_layoff"], 0.0)


if __name__ == "__main__":
    unittest.main()
