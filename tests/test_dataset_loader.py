from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from racing_ml.data.dataset_loader import (
    _normalize_columns,
    _read_csv_tail,
    _select_table_columns,
    load_training_table,
    materialize_supplemental_table,
)


class DatasetLoaderTailReadTest(unittest.TestCase):
    def test_normalize_columns_reuses_frame_when_already_normalized(self) -> None:
        frame = pd.DataFrame({"date": ["2025-01-01"], "race_id": [1]})

        normalized = _normalize_columns(frame)

        self.assertIs(normalized, frame)

    def test_read_csv_tail_returns_requested_tail_and_total_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            frame = pd.DataFrame(
                {
                    "date": pd.date_range("2025-01-01", periods=12, freq="D"),
                    "race_id": [f"r{i}" for i in range(12)],
                    "value": list(range(12)),
                }
            )
            frame.to_csv(csv_path, index=False)

            tail_frame, total_rows = _read_csv_tail(csv_path, tail_rows=5)

            self.assertEqual(total_rows, 12)
            self.assertEqual(len(tail_frame), 5)
            self.assertEqual(tail_frame["value"].tolist(), [7, 8, 9, 10, 11])

    def test_read_csv_tail_preserves_current_multi_overflow_behavior(self) -> None:
        chunks = [
            pd.DataFrame({"value": [0, 1, 2]}),
            pd.DataFrame({"value": [3, 4, 5, 6]}),
            pd.DataFrame({"value": [7, 8, 9]}),
        ]

        with mock.patch("racing_ml.data.dataset_loader.pd.read_csv", return_value=iter(chunks)):
            tail_frame, total_rows = _read_csv_tail(Path("dummy.csv"), tail_rows=5)

        self.assertEqual(total_rows, 10)
        self.assertEqual(tail_frame["value"].tolist(), [5, 6, 7, 8, 9])

    def test_materialize_supplemental_table_writes_corner_passing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            corner_path = raw_dir / "sample_corner_passing_order.csv"
            pd.DataFrame(
                {
                    "レースID": [101],
                    "1コーナー": ["(3,1)2"],
                    "2コーナー": ["1,2,3"],
                    "3コーナー": [None],
                    "4コーナー": ["3-2-1"],
                }
            ).to_csv(corner_path, index=False)

            dataset_config = {
                "supplemental_tables": [
                    {
                        "name": "corner_passing_order",
                        "pattern": "**/*corner_passing_order*.csv",
                        "materialized_file": "processed/corner_passing_order.csv",
                        "join_on": ["race_id", "gate_no"],
                        "required_columns": ["race_id", "gate_no"],
                        "keep_columns": ["race_id", "gate_no", "corner_1_position", "corner_2_position", "corner_4_position"],
                        "dedupe_on": ["race_id", "gate_no"],
                        "table_loader": "corner_passing_order",
                    }
                ]
            }

            summary = materialize_supplemental_table(
                raw_dir,
                table_name="corner_passing_order",
                dataset_config=dataset_config,
                base_dir=root,
            )

            self.assertEqual(summary["status"], "completed")
            output_path = root / "processed/corner_passing_order.csv"
            self.assertTrue(output_path.exists())
            materialized = pd.read_csv(output_path)
            self.assertEqual(materialized["gate_no"].tolist(), [1, 2, 3])
            self.assertEqual(materialized["corner_4_position"].tolist(), [3, 2, 1])

    def test_select_table_columns_reuses_frame_for_noop_selection(self) -> None:
        frame = pd.DataFrame({"race_id": [101], "horse_id": ["h1"], "horse_key": ["k1"]})

        selected = _select_table_columns(frame, keep_columns=["race_id", "horse_id", "horse_key"], join_on=["race_id", "horse_id"])

        self.assertIs(selected, frame)

    def test_load_training_table_prefers_materialized_supplemental_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            raw_dir = root / "raw"
            raw_dir.mkdir(parents=True, exist_ok=True)
            supplemental_dir = root / "external"
            supplemental_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "date": ["2025-01-01"],
                    "race_id": [101],
                    "horse_id": ["h1"],
                    "horse_name": ["horse-a"],
                    "track": ["tokyo"],
                    "distance": [1600],
                    "gate_no": [1],
                    "rank": [1],
                }
            ).to_csv(raw_dir / "primary.csv", index=False)
            pd.DataFrame(
                {
                    "レースID": [101],
                    "1コーナー": ["1"],
                    "2コーナー": ["1"],
                    "3コーナー": ["1"],
                    "4コーナー": ["1"],
                }
            ).to_csv(supplemental_dir / "sample_corner_passing_order.csv", index=False)

            materialized_dir = root / "processed"
            materialized_dir.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(
                {
                    "race_id": [101],
                    "gate_no": [1],
                    "corner_1_position": [7],
                    "corner_2_position": [8],
                    "corner_3_position": [9],
                    "corner_4_position": [10],
                }
            ).to_csv(materialized_dir / "corner_passing_order.csv", index=False)

            dataset_config = {
                "supplemental_tables": [
                    {
                        "name": "corner_passing_order",
                        "pattern": "**/*corner_passing_order*.csv",
                        "search_dirs": ["external"],
                        "materialized_file": "processed/corner_passing_order.csv",
                        "join_on": ["race_id", "gate_no"],
                        "required_columns": ["race_id", "gate_no"],
                        "keep_columns": [
                            "race_id",
                            "gate_no",
                            "corner_1_position",
                            "corner_2_position",
                            "corner_3_position",
                            "corner_4_position",
                        ],
                        "dedupe_on": ["race_id", "gate_no"],
                        "merge_mode": "fill_missing",
                        "table_loader": "corner_passing_order",
                    }
                ]
            }

            loaded = load_training_table(raw_dir, dataset_config=dataset_config, base_dir=root)

            self.assertEqual(int(loaded.loc[0, "corner_1_position"]), 7)
            self.assertEqual(int(loaded.loc[0, "corner_4_position"]), 10)


if __name__ == "__main__":
    unittest.main()
