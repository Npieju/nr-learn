from __future__ import annotations

import unittest
from pathlib import Path

from racing_ml.data.local_nankan_collect import _decode_html_bytes, parse_local_nankan_race_result_html


class LocalNankanCollectParserTest(unittest.TestCase):
    def test_parse_race_result_supports_legacy_nankan_layout(self) -> None:
        html_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "local_nankan"
            / "raw_html"
            / "race_result"
            / "2006032921140301.html"
        )
        html = _decode_html_bytes(html_path.read_bytes())

        frame = parse_local_nankan_race_result_html(html, "2006032921140301")

        self.assertEqual(len(frame), 10)
        self.assertEqual(frame.iloc[0]["date"], "2006-03-29")
        self.assertEqual(frame.iloc[0]["track"], "川崎")
        self.assertEqual(frame.iloc[0]["distance"], 1400)
        self.assertEqual(frame.iloc[0]["weather"], "晴")
        self.assertEqual(frame.iloc[0]["ground_condition"], "良")
        self.assertEqual(frame.iloc[0]["race_name"], "Ｃ３(一)(二)")
        self.assertEqual(frame.iloc[0]["horse_key"], "1999100679")
        self.assertEqual(frame.iloc[0]["horse_name"], "カソクソウチ")
        self.assertEqual(frame.iloc[0]["jockey_key"], "031113")
        self.assertEqual(frame.iloc[0]["trainer_key"], "010298")
        self.assertEqual(frame.iloc[0]["finish_time"], "1:30.8")
        self.assertEqual(frame.iloc[0]["closing_time_3f"], "40.8")

    def test_parse_race_result_handles_cancelled_race_page(self) -> None:
        html_path = (
            Path(__file__).resolve().parents[1]
            / "data"
            / "external"
            / "local_nankan"
            / "raw_html"
            / "race_result"
            / "2007081820090301.html"
        )
        html = _decode_html_bytes(html_path.read_bytes())

        frame = parse_local_nankan_race_result_html(html, "2007081820090301")

        self.assertTrue(frame.empty)
        self.assertEqual(frame.columns.tolist()[0:7], [
            "date",
            "race_id",
            "track",
            "distance",
            "weather",
            "ground_condition",
            "race_name",
        ])


if __name__ == "__main__":
    unittest.main()