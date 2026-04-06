from __future__ import annotations

from contextlib import ExitStack
import json
from pathlib import Path
from types import SimpleNamespace
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.run_wf_feasibility_diag as wf_diag_script

from scripts.run_wf_feasibility_diag import (
    _build_candidate_snapshot_from_metrics,
    _count_outer_search_steps,
    _count_strategy_evals_per_outer_step,
    _resolve_feasibility_output_paths,
    _extract_strategy_params,
    _should_emit_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def _minimal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"race_id": "r1", "date": "2026-04-01", "is_win": 1, "f1": 0.9, "odds": 3.2},
            {"race_id": "r1", "date": "2026-04-01", "is_win": 0, "f1": 0.1, "odds": 6.8},
            {"race_id": "r2", "date": "2026-04-02", "is_win": 1, "f1": 0.8, "odds": 2.9},
            {"race_id": "r2", "date": "2026-04-02", "is_win": 0, "f1": 0.2, "odds": 8.4},
        ]
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

    def test_extract_strategy_params_and_snapshot_for_portfolio(self) -> None:
        row = {
            "strategy_kind": "portfolio",
            "blend_weight": 0.4,
            "min_prob": 0.05,
            "odds_min": 1.0,
            "odds_max": 40.0,
            "top_k": 1,
            "min_expected_value": 1.05,
        }
        params = _extract_strategy_params(row)
        snapshot = _build_candidate_snapshot_from_metrics(
            strategy_kind="portfolio",
            params=params,
            metrics={
                "portfolio_roi": 0.82,
                "portfolio_bets": 611,
                "portfolio_hit_rate": 0.14,
                "portfolio_final_bankroll": 0.93,
                "portfolio_max_drawdown": 0.31,
            },
        )
        self.assertEqual(snapshot["strategy_kind"], "portfolio")
        self.assertEqual(snapshot["params"]["top_k"], 1)
        self.assertEqual(snapshot["params"]["min_expected_value"], 1.05)
        self.assertEqual(snapshot["bets"], 611)
        self.assertAlmostEqual(snapshot["roi"], 0.82)


class WfFeasibilityMainContractTest(unittest.TestCase):
    def test_main_writes_resolved_profile_and_no_suffix_contract(self) -> None:
        frame = _minimal_frame()
        writes: list[tuple[str | None, str, str]] = []

        def fake_write_text_file(path: Path, content: str, *, label: str | None = None, **_: object) -> None:
            writes.append((label, str(path), content))

        def fake_prepare_scored_frame(input_frame: pd.DataFrame, scores: np.ndarray, *, odds_col: str | None, score_col: str) -> pd.DataFrame:
            output = input_frame.copy()
            output[score_col] = scores
            return output

        with ExitStack() as stack:
            stack.enter_context(
                patch.object(
                    sys,
                    "argv",
                    [
                        "run_wf_feasibility_diag.py",
                        "--profile",
                        "current_best_eval_2025_latest",
                        "--artifact-suffix",
                        "r_eval",
                        "--model-artifact-suffix",
                        wf_diag_script.NO_MODEL_ARTIFACT_SUFFIX,
                        "--wf-mode",
                        "fast",
                    ],
                )
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "resolve_model_run_profile",
                    return_value=(
                        "current_best_eval_2025_latest",
                        "configs/model.yaml",
                        "configs/data.yaml",
                        "configs/features.yaml",
                    ),
                )
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "load_yaml",
                    side_effect=[
                        {
                            "label": "is_win",
                            "output": {
                                "model_file": "artifacts/models/model.joblib",
                                "manifest_file": "artifacts/models/model.manifest.json",
                            },
                            "evaluation": {"policy_search": {}},
                        },
                        {"dataset": {"raw_dir": "data/raw"}},
                        {},
                    ],
                )
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "load_training_table_for_feature_build",
                    return_value=SimpleNamespace(
                        frame=frame.copy(),
                        loaded_rows=len(frame),
                        pre_feature_rows=len(frame),
                        data_load_strategy="full_scan",
                        primary_source_rows_total=len(frame),
                    ),
                )
            )
            stack.enter_context(patch.object(wf_diag_script, "build_features", side_effect=lambda input_frame: input_frame))
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "resolve_output_artifacts",
                    side_effect=lambda cfg: SimpleNamespace(
                        model_path=Path(str(cfg.get("model_file", "artifacts/models/model.joblib"))),
                        manifest_path=Path(str(cfg.get("manifest_file", "artifacts/models/model.manifest.json"))),
                    ),
                )
            )
            stack.enter_context(patch.object(wf_diag_script.joblib, "load", return_value=object()))
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "resolve_feature_selection",
                    return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[], mode="config"),
                )
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "resolve_model_feature_selection",
                    return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[], mode="model"),
                )
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "prepare_model_input_frame",
                    side_effect=lambda input_frame, feature_columns, categorical_columns: input_frame[list(feature_columns)].copy(),
                )
            )
            stack.enter_context(patch.object(wf_diag_script, "resolve_odds_column", return_value="odds"))
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "generate_prediction_outputs",
                    return_value=SimpleNamespace(score=np.array([0.9, 0.1, 0.8, 0.2]), top3_probs=None),
                )
            )
            stack.enter_context(patch.object(wf_diag_script, "prepare_scored_frame", side_effect=fake_prepare_scored_frame))
            stack.enter_context(
                patch.object(wf_diag_script, "add_market_signals", side_effect=lambda input_frame, score_col, odds_col: input_frame)
            )
            stack.enter_context(
                patch.object(wf_diag_script, "build_stability_guardrail", return_value={"assessment": "representative", "warnings": []})
            )
            stack.enter_context(
                patch.object(wf_diag_script, "build_nested_wf_slices", return_value=[(frame.copy(), frame.copy(), frame.copy())])
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "_summarize_fold_candidates",
                    return_value=({"fold": 1, "feasible_candidates": 1}, [{"fold": 1, "strategy_kind": "portfolio", "gate_failures": []}]),
                )
            )
            stack.enter_context(
                patch.object(
                    wf_diag_script,
                    "_resolve_feasibility_output_paths",
                    return_value=(ROOT / "artifacts/reports/wf_summary_test.json", ROOT / "artifacts/reports/wf_detail_test.csv"),
                )
            )
            stack.enter_context(patch.object(wf_diag_script, "artifact_ensure_output_file_path", return_value=None))
            stack.enter_context(patch.object(wf_diag_script, "write_text_file", side_effect=fake_write_text_file))
            stack.enter_context(patch.object(wf_diag_script, "write_csv_file", return_value=None))
            exit_code = wf_diag_script.main()

        self.assertEqual(exit_code, 0)
        summary_write = next(item for item in writes if item[0] == "summary output")
        summary = json.loads(summary_write[2])
        self.assertEqual(summary["run_context"]["profile"], "current_best_eval_2025_latest")
        self.assertEqual(summary["run_context"]["config"], "configs/model.yaml")
        self.assertEqual(summary["run_context"]["data_config"], "configs/data.yaml")
        self.assertEqual(summary["run_context"]["feature_config"], "configs/features.yaml")
        self.assertEqual(summary["run_context"]["artifact_suffix"], "r_eval")
        self.assertEqual(summary["run_context"]["model_artifact_suffix"], "")
        self.assertEqual(summary["run_context"]["model_path"], "artifacts/models/model.joblib")


if __name__ == "__main__":
    unittest.main()
