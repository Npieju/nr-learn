from __future__ import annotations

import unittest

from scripts.run_wf_feasibility_diag import (
    _count_outer_search_steps,
    _count_strategy_evals_per_outer_step,
    _should_emit_checkpoint,
)


class WfFeasibilityProgressHelpersTest(unittest.TestCase):
    def test_count_outer_search_steps_counts_grid(self) -> None:
        total = _count_outer_search_steps(
            blend_candidates=[0.2, 0.4],
            edge_candidates=[0.01, 0.03, 0.05],
            min_prob_candidates=[0.03, 0.05],
            odds_min_candidates=[1.0],
            odds_max_candidates=[25.0, 40.0],
        )
        self.assertEqual(total, 24)

    def test_count_strategy_evals_per_outer_step_counts_inner_trials(self) -> None:
        total = _count_strategy_evals_per_outer_step(
            kelly_frac_candidates=[0.25, 0.5],
            max_frac_candidates=[0.02, 0.05],
            top_k_candidates=[1, 2],
            min_ev_candidates=[1.0, 1.05, 1.10],
        )
        self.assertEqual(total, 10)

    def test_should_emit_checkpoint_on_edges_and_intervals(self) -> None:
        self.assertTrue(
            _should_emit_checkpoint(
                processed=1,
                total=100,
                checkpoint_interval=10,
                now_monotonic=10.0,
                last_progress_at=0.0,
                max_silent_seconds=60.0,
            )
        )
        self.assertTrue(
            _should_emit_checkpoint(
                processed=20,
                total=100,
                checkpoint_interval=10,
                now_monotonic=10.0,
                last_progress_at=0.0,
                max_silent_seconds=60.0,
            )
        )
        self.assertTrue(
            _should_emit_checkpoint(
                processed=100,
                total=100,
                checkpoint_interval=10,
                now_monotonic=10.0,
                last_progress_at=0.0,
                max_silent_seconds=60.0,
            )
        )

    def test_should_emit_checkpoint_when_silent_interval_exceeded(self) -> None:
        self.assertTrue(
            _should_emit_checkpoint(
                processed=7,
                total=100,
                checkpoint_interval=10,
                now_monotonic=65.0,
                last_progress_at=0.0,
                max_silent_seconds=60.0,
            )
        )
        self.assertFalse(
            _should_emit_checkpoint(
                processed=7,
                total=100,
                checkpoint_interval=10,
                now_monotonic=30.0,
                last_progress_at=0.0,
                max_silent_seconds=60.0,
            )
        )


if __name__ == "__main__":
    unittest.main()
