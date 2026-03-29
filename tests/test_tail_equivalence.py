from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from racing_ml.data.tail_equivalence import compare_tail_readers


class TailEquivalenceTest(unittest.TestCase):
    def test_compare_tail_readers_reports_exact_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

            def reader(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                frame = pd.read_csv(csv_path).tail(tail_rows).reset_index(drop=True)
                return frame, 3

            result = compare_tail_readers(
                left_name="left",
                left_reader=reader,
                right_name="right",
                right_reader=reader,
                csv_path=csv_path,
                tail_rows=2,
            )

            self.assertTrue(result["comparison"]["raw"]["exact_equal"])
            self.assertTrue(result["comparison"]["raw"]["value_equal"])
            self.assertFalse(result["comparison"]["raw"]["dtype_only_difference"])
            self.assertFalse(result["comparison"]["raw"]["canonical_dtype_only_difference"])
            self.assertEqual(result["comparison"]["raw"]["first_diff_column"], None)
            self.assertTrue(result["comparison"]["normalized"]["exact_equal"])
            self.assertTrue(result["comparison"]["normalized"]["value_equal"])
            self.assertFalse(result["comparison"]["normalized"]["dtype_only_difference"])
            self.assertFalse(result["comparison"]["normalized"]["canonical_dtype_only_difference"])

    def test_compare_tail_readers_reports_first_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

            def left_reader(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                return pd.DataFrame({"value": [2, 3]}), 3

            def right_reader(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                return pd.DataFrame({"value": [2, 4]}), 3

            result = compare_tail_readers(
                left_name="left",
                left_reader=left_reader,
                right_name="right",
                right_reader=right_reader,
                csv_path=csv_path,
                tail_rows=2,
            )

            self.assertFalse(result["comparison"]["raw"]["exact_equal"])
            self.assertFalse(result["comparison"]["raw"]["value_equal"])
            self.assertFalse(result["comparison"]["raw"]["dtype_only_difference"])
            self.assertFalse(result["comparison"]["raw"]["canonical_dtype_only_difference"])
            self.assertEqual(result["comparison"]["raw"]["value_difference_count"], 1)
            self.assertEqual(result["comparison"]["raw"]["first_diff_column"], "value")
            self.assertEqual(result["comparison"]["raw"]["first_diff_indices"], [1])
            self.assertFalse(result["comparison"]["normalized"]["exact_equal"])
            self.assertFalse(result["comparison"]["normalized"]["value_equal"])
            self.assertFalse(result["comparison"]["normalized"]["dtype_only_difference"])
            self.assertFalse(result["comparison"]["normalized"]["canonical_dtype_only_difference"])

    def test_compare_tail_readers_reports_dtype_only_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

            def left_reader(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                return pd.DataFrame({"value": pd.Series([2, 3], dtype="int64")}), 3

            def right_reader(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                return pd.DataFrame({"value": pd.Series([2.0, 3.0], dtype="float64")}), 3

            result = compare_tail_readers(
                left_name="left",
                left_reader=left_reader,
                right_name="right",
                right_reader=right_reader,
                csv_path=csv_path,
                tail_rows=2,
            )

            self.assertFalse(result["comparison"]["raw"]["exact_equal"])
            self.assertTrue(result["comparison"]["raw"]["value_equal"])
            self.assertTrue(result["comparison"]["raw"]["dtype_only_difference"])
            self.assertTrue(result["comparison"]["raw"]["canonical_dtype_only_difference"])
            self.assertEqual(result["comparison"]["raw"]["value_difference_count"], 0)
            self.assertEqual(
                result["comparison"]["raw"]["dtype_difference_categories"][0]["classification"],
                "numeric_integral_equivalent",
            )
            self.assertFalse(result["comparison"]["normalized"]["exact_equal"])
            self.assertTrue(result["comparison"]["normalized"]["value_equal"])
            self.assertTrue(result["comparison"]["normalized"]["dtype_only_difference"])
            self.assertTrue(result["comparison"]["normalized"]["canonical_dtype_only_difference"])

    def test_compare_tail_readers_classifies_all_null_dtype_difference(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "sample.csv"
            pd.DataFrame({"value": [1, 2, 3]}).to_csv(csv_path, index=False)

            def left_reader(_: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                return pd.DataFrame({"value": pd.Series([None, None], dtype="object")}), 3

            def right_reader(_: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
                return pd.DataFrame({"value": pd.Series([float("nan"), float("nan")], dtype="float64")}), 3

            result = compare_tail_readers(
                left_name="left",
                left_reader=left_reader,
                right_name="right",
                right_reader=right_reader,
                csv_path=csv_path,
                tail_rows=2,
            )

            self.assertFalse(result["comparison"]["raw"]["exact_equal"])
            self.assertTrue(result["comparison"]["raw"]["value_equal"])
            self.assertTrue(result["comparison"]["raw"]["dtype_only_difference"])
            self.assertTrue(result["comparison"]["raw"]["canonical_dtype_only_difference"])
            self.assertEqual(
                result["comparison"]["raw"]["dtype_difference_categories"][0]["classification"],
                "all_null",
            )
