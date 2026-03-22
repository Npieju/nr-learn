from __future__ import annotations

import unittest

import pandas as pd

from scripts.run_policy_date_signal_report import _selected_signal_summary


class PolicyDateSignalReportTest(unittest.TestCase):
    def test_selected_signal_summary_handles_empty(self) -> None:
        summary = _selected_signal_summary(pd.DataFrame())

        self.assertEqual(summary["selected_count"], 0)
        self.assertEqual(summary["count_ev_ge_1_0"], 0)
        self.assertIsNone(summary["ev_mean"])
        self.assertIsNone(summary["share_edge_pos"])

    def test_selected_signal_summary_computes_threshold_counts(self) -> None:
        selected = pd.DataFrame(
            [
                {"policy_expected_value": 0.98, "policy_edge": -0.02, "policy_prob": 0.08},
                {"policy_expected_value": 1.00, "policy_edge": 0.00, "policy_prob": 0.09},
                {"policy_expected_value": 1.02, "policy_edge": 0.02, "policy_prob": 0.10},
            ]
        )

        summary = _selected_signal_summary(selected)

        self.assertEqual(summary["selected_count"], 3)
        self.assertEqual(summary["count_ev_ge_1_0"], 2)
        self.assertEqual(summary["count_ev_ge_1_01"], 1)
        self.assertEqual(summary["count_edge_pos"], 1)
        self.assertEqual(summary["count_edge_ge_0_01"], 1)
        self.assertAlmostEqual(summary["share_ev_ge_1_0"], 2 / 3)
        self.assertAlmostEqual(summary["share_edge_pos"], 1 / 3)


if __name__ == "__main__":
    unittest.main()