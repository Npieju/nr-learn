from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import pandas as pd

from racing_ml.serving.score_calibration import apply_score_calibration


class ScoreCalibrationTest(unittest.TestCase):
    def test_favorite_only_guard_preserves_non_favorite_without_lift(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            train_path = workspace_root / "train_predictions.csv"
            pd.DataFrame(
                [
                    {"race_id": "t1", "horse_id": "a", "score": 0.90, "rank": 1, "popularity": 1},
                    {"race_id": "t1", "horse_id": "b", "score": 0.10, "rank": 2, "popularity": 2},
                    {"race_id": "t2", "horse_id": "c", "score": 0.80, "rank": 1, "popularity": 1},
                    {"race_id": "t2", "horse_id": "d", "score": 0.20, "rank": 2, "popularity": 2},
                ]
            ).to_csv(train_path, index=False)

            frame = pd.DataFrame(
                [
                    {"race_id": "r1", "horse_id": "h1", "score": 0.85, "rank": 1, "popularity": 1},
                    {"race_id": "r1", "horse_id": "h2", "score": 0.25, "rank": 2, "popularity": 2},
                ]
            )
            calibrated, summary = apply_score_calibration(
                frame,
                {
                    "enabled": True,
                    "method": "isotonic",
                    "train_glob": "train_predictions.csv",
                    "top_popularity_max": 1,
                    "non_top_max_lift": 0.0,
                    "shrinkage": 1.0,
                    "min_calibration_rows": 1,
                },
                workspace_root=workspace_root,
                score_col="score",
            )

        self.assertEqual(summary["calibration_rows"], 4)
        self.assertIn("score_before_calibration", calibrated.columns)
        self.assertNotEqual(float(calibrated.loc[0, "score"]), float(calibrated.loc[0, "score_before_calibration"]))
        self.assertLessEqual(float(calibrated.loc[1, "score"]), float(calibrated.loc[1, "score_before_calibration"]))

    def test_disabled_calibration_returns_frame_and_no_summary(self) -> None:
        frame = pd.DataFrame([{"race_id": "r1", "score": 0.5}])
        calibrated, summary = apply_score_calibration(frame, {"enabled": False}, workspace_root=Path.cwd())

        self.assertIs(calibrated, frame)
        self.assertIsNone(summary)

    def test_train_glob_expands_artifact_suffix_placeholder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            train_path = workspace_root / "train_rev1.csv"
            pd.DataFrame(
                [
                    {"race_id": "t1", "horse_id": "a", "score": 0.90, "rank": 1},
                    {"race_id": "t1", "horse_id": "b", "score": 0.10, "rank": 2},
                ]
            ).to_csv(train_path, index=False)

            frame = pd.DataFrame([{"race_id": "r1", "horse_id": "h1", "score": 0.85, "rank": 1}])
            _, summary = apply_score_calibration(
                frame,
                {
                    "enabled": True,
                    "method": "isotonic",
                    "train_glob": "train_{artifact_suffix}.csv",
                    "min_calibration_rows": 1,
                },
                workspace_root=workspace_root,
                score_col="score",
                format_context={"artifact_suffix": "rev1"},
            )

        self.assertEqual(summary["train_glob"], "train_rev1.csv")
        self.assertEqual(summary["train_glob_template"], "train_{artifact_suffix}.csv")

    def test_config_label_col_overrides_call_site_label(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            pd.DataFrame(
                [
                    {"race_id": "t1", "horse_id": "a", "score": 0.90, "rank": 1},
                    {"race_id": "t1", "horse_id": "b", "score": 0.10, "rank": 2},
                ]
            ).to_csv(workspace_root / "train.csv", index=False)

            frame = pd.DataFrame([{"race_id": "r1", "horse_id": "h1", "score": 0.85, "rank": 1}])
            _, summary = apply_score_calibration(
                frame,
                {
                    "enabled": True,
                    "method": "isotonic",
                    "train_glob": "train.csv",
                    "label_col": "rank",
                    "min_calibration_rows": 1,
                },
                workspace_root=workspace_root,
                score_col="score",
                label_col="is_win",
            )

        self.assertEqual(summary["label_col"], "rank")


if __name__ == "__main__":
    unittest.main()
