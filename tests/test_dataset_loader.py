from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from racing_ml.data.dataset_loader import _read_csv_tail


class DatasetLoaderTailReadTest(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
