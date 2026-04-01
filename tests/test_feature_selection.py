from __future__ import annotations

import unittest

import pandas as pd

from racing_ml.features.selection import resolve_feature_selection


class FeatureSelectionExplicitModeTest(unittest.TestCase):
    def test_explicit_mode_includes_declared_custom_feature_groups(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [3, 4],
                "horse_last_3_avg_rank": [2.0, 1.0],
                "horse_days_since_last_race": [12.0, 28.0],
                "horse_weight_change": [-2.0, 3.0],
                "is_win": [0, 1],
            }
        )
        feature_config = {
            "features": {
                "base": ["age"],
                "history": ["horse_last_3_avg_rank"],
                "class_rest_surface": [
                    "horse_days_since_last_race",
                    "horse_weight_change",
                ],
            }
        }

        selection = resolve_feature_selection(frame, feature_config, label_column="is_win")

        self.assertEqual(
            selection.feature_columns,
            [
                "age",
                "horse_last_3_avg_rank",
                "horse_days_since_last_race",
                "horse_weight_change",
            ],
        )

    def test_explicit_mode_still_ignores_missing_declared_columns(self) -> None:
        frame = pd.DataFrame(
            {
                "age": [3, 4],
                "is_win": [0, 1],
            }
        )
        feature_config = {
            "features": {
                "base": ["age"],
                "class_rest_surface": ["horse_days_since_last_race"],
            }
        }

        selection = resolve_feature_selection(frame, feature_config, label_column="is_win")

        self.assertEqual(selection.feature_columns, ["age"])


if __name__ == "__main__":
    unittest.main()
