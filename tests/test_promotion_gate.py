from __future__ import annotations

import unittest

from scripts.run_promotion_gate import _summarize_formal_benchmark


class PromotionGateFormalBenchmarkTest(unittest.TestCase):
    def test_formal_benchmark_prefers_holdout_test_metrics(self) -> None:
        summary = _summarize_formal_benchmark(
            [
                {
                    "fold": 1,
                    "best_feasible": {"strategy_kind": "portfolio", "bets": 800, "roi": 3.5},
                    "best_feasible_test": {"strategy_kind": "portfolio", "bets": 200, "roi": 1.2},
                },
                {
                    "fold": 2,
                    "best_feasible": {"strategy_kind": "kelly", "bets": 600, "roi": 2.5},
                    "best_feasible_test": {"strategy_kind": "kelly", "bets": 100, "roi": 0.8},
                },
            ]
        )

        self.assertEqual(summary["feasible_fold_count"], 2)
        self.assertEqual(summary["bets_total"], 300)
        self.assertAlmostEqual(summary["weighted_roi"], (200 * 1.2 + 100 * 0.8) / 300)
        self.assertEqual(summary["metric_source_counts"], {"test": 2})
        self.assertEqual([fold["metric_source"] for fold in summary["folds"]], ["test", "test"])

    def test_formal_benchmark_falls_back_to_valid_when_test_metrics_missing(self) -> None:
        summary = _summarize_formal_benchmark(
            [
                {
                    "fold": 1,
                    "best_feasible": {"strategy_kind": "portfolio", "bets": 150, "roi": 1.1},
                }
            ]
        )

        self.assertEqual(summary["feasible_fold_count"], 1)
        self.assertEqual(summary["bets_total"], 150)
        self.assertAlmostEqual(summary["weighted_roi"], 1.1)
        self.assertEqual(summary["metric_source_counts"], {"valid_fallback": 1})
        self.assertEqual(summary["folds"][0]["metric_source"], "valid_fallback")


if __name__ == "__main__":
    unittest.main()
