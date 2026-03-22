from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.run_staged_trace_support_check import _build_support_summary, _load_report_rows, _selection_depth_bucket


class StagedTraceSupportCheckTest(unittest.TestCase):
    def test_selection_depth_bucket_distinguishes_deepest_and_intermediate(self) -> None:
        self.assertEqual(
            _selection_depth_bucket({"deepest_stage_selected_present": True, "intermediate_stage_selected_present": False, "races_with_final_selection": 1}),
            "deepest_stage_selected",
        )
        self.assertEqual(
            _selection_depth_bucket({"deepest_stage_selected_present": False, "intermediate_stage_selected_present": True, "races_with_final_selection": 2}),
            "intermediate_stage_selected",
        )
        self.assertEqual(
            _selection_depth_bucket({"deepest_stage_selected_present": False, "intermediate_stage_selected_present": False, "races_with_final_selection": 1}),
            "stage1_only_selected",
        )
        self.assertEqual(
            _selection_depth_bucket({"deepest_stage_selected_present": False, "intermediate_stage_selected_present": False, "races_with_final_selection": 0}),
            "no_final_selection",
        )

    def test_build_support_summary_counts_stage3_non_positive_dates(self) -> None:
        rows = [
            {
                "report_label": "aug",
                "date": "2024-08-03",
                "net_sign": "negative",
                "deepest_stage_selected_present": True,
                "intermediate_stage_selected_present": False,
            },
            {
                "report_label": "aug",
                "date": "2024-08-04",
                "net_sign": "zero",
                "deepest_stage_selected_present": False,
                "intermediate_stage_selected_present": False,
            },
            {
                "report_label": "sep",
                "date": "2024-09-28",
                "net_sign": "zero",
                "deepest_stage_selected_present": True,
                "intermediate_stage_selected_present": False,
            },
            {
                "report_label": "sep",
                "date": "2024-09-29",
                "net_sign": "negative",
                "deepest_stage_selected_present": False,
                "intermediate_stage_selected_present": True,
            },
        ]

        summary = _build_support_summary(rows)

        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["date_count"], 4)
        self.assertEqual(summary["deepest_stage_selected_date_count"], 2)
        self.assertEqual(summary["deepest_stage_selected_positive_net_date_count"], 0)
        self.assertEqual(summary["deepest_stage_selected_non_positive_net_date_count"], 2)
        self.assertEqual(summary["deepest_stage_selected_dates"], ["2024-08-03", "2024-09-28"])
        self.assertEqual(summary["deepest_stage_selected_positive_dates"], [])
        self.assertEqual(summary["deepest_stage_selected_non_positive_dates"], ["2024-08-03", "2024-09-28"])
        self.assertEqual(summary["intermediate_stage_selected_date_count"], 1)
        self.assertEqual(summary["intermediate_stage_selected_positive_net_date_count"], 0)
        self.assertEqual(summary["intermediate_stage_selected_non_positive_net_date_count"], 1)
        self.assertEqual(summary["intermediate_stage_selected_dates"], ["2024-09-29"])
        self.assertEqual(summary["intermediate_stage_selected_positive_dates"], [])
        self.assertEqual(summary["intermediate_stage_selected_non_positive_dates"], ["2024-09-29"])

    def test_load_report_rows_parses_dict_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "report.csv"
            path.write_text(
                "date,window_label,races_with_final_selection,final_selected_net_units,max_selected_stage_index,stage2_plus_selected_race_count,stage3_plus_selected_race_count,deepest_selected_stage_counts\n"
                "2024-09-28,late_sep,1,-0.00557,3,1,1,\"{'kelly_fallback_2': 1, 'portfolio_aug_baseline': 3}\"\n",
                encoding="utf-8",
            )

            rows = _load_report_rows("sep_guard", path)

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["report_label"], "sep_guard")
            self.assertTrue(rows[0]["deepest_stage_selected_present"])
            self.assertFalse(rows[0]["intermediate_stage_selected_present"])
            self.assertEqual(rows[0]["highest_stage_index"], 3)
            self.assertEqual(rows[0]["deepest_stage_selected_race_count"], 1)
            self.assertEqual(rows[0]["deepest_stage3_selected_count"], 1)
            self.assertEqual(rows[0]["net_sign"], "negative")


if __name__ == "__main__":
    unittest.main()