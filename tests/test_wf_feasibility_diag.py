from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from scripts.run_wf_feasibility_diag import (
    _count_outer_search_steps,
    _count_strategy_evals_per_outer_step,
    _resolve_feasibility_output_paths,
    _should_emit_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


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

    def test_resolve_feasibility_output_paths_uses_explicit_summary_and_derives_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path, detail_path = _resolve_feasibility_output_paths(
                report_dir=Path(tmpdir),
                config_path="configs/model_local_baseline_wf_runtime_narrow.yaml",
                model_path=Path("artifacts/models/local_nankan_baseline_model.joblib"),
                wf_mode="fast",
                wf_scheme="nested",
                start_date=None,
                end_date=None,
                summary_output="artifacts/reports/custom_summary.json",
                detail_output=None,
            )
            self.assertEqual(summary_path, ROOT / "artifacts/reports/custom_summary.json")
            self.assertEqual(detail_path, summary_path.with_suffix(".csv"))

    def test_resolve_feasibility_output_paths_builds_default_versioned_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            summary_path, detail_path = _resolve_feasibility_output_paths(
                report_dir=report_dir,
                config_path="configs/model_local_baseline_wf_runtime_narrow.yaml",
                model_path=Path("artifacts/models/local_nankan_baseline_model.joblib"),
                wf_mode="fast",
                wf_scheme="nested",
                start_date=None,
                end_date=None,
                summary_output=None,
                detail_output=None,
            )
            self.assertEqual(summary_path, report_dir / "wf_feasibility_diag_local_baseline_wf_runtime_narrow_wf_fast_nested.json")
            self.assertEqual(detail_path, report_dir / "wf_feasibility_diag_local_baseline_wf_runtime_narrow_wf_fast_nested.csv")


if __name__ == "__main__":
    unittest.main()
