from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import scripts.run_evaluate as evaluate_script


def _minimal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"race_id": "r1", "date": "2026-04-01", "is_win": 1, "f1": 0.9},
            {"race_id": "r1", "date": "2026-04-01", "is_win": 0, "f1": 0.1},
            {"race_id": "r2", "date": "2026-04-02", "is_win": 1, "f1": 0.8},
            {"race_id": "r2", "date": "2026-04-02", "is_win": 0, "f1": 0.2},
        ]
    )


class RunEvaluateSuffixContractTest(unittest.TestCase):
    def _run_main_and_capture_summary(self, extra_argv: list[str]) -> dict[str, object]:
        frame = _minimal_frame()
        writes: list[tuple[str | None, str, str]] = []

        def fake_write_text_file(path: Path, content: str, *, label: str | None = None, **_: object) -> None:
            writes.append((label, str(path), content))

        def fake_prepare_scored_frame(input_frame: pd.DataFrame, scores: np.ndarray, *, odds_col: str | None, score_col: str) -> pd.DataFrame:
            output = input_frame.copy()
            output[score_col] = scores
            return output

        with patch.object(
            sys,
            "argv",
            [
                "run_evaluate.py",
                "--config",
                "configs/model.yaml",
                "--data-config",
                "configs/data.yaml",
                "--feature-config",
                "configs/features.yaml",
                "--max-rows",
                "10",
                "--wf-mode",
                "off",
                *extra_argv,
            ],
        ), patch.object(
            evaluate_script,
            "load_yaml",
            side_effect=[
                {
                    "task": "classification",
                    "label": "is_win",
                    "output": {
                        "model_file": "artifacts/models/model.joblib",
                        "report_file": "artifacts/reports/train_metrics.json",
                        "manifest_file": "artifacts/models/model.manifest.json",
                    },
                    "evaluation": {"leakage_audit": {"enabled": False}},
                },
                {"dataset": {"raw_dir": "data/raw"}},
                {},
            ],
        ), patch.object(
            evaluate_script,
            "load_training_table_for_feature_build",
            return_value=SimpleNamespace(
                frame=frame.copy(),
                loaded_rows=len(frame),
                pre_feature_rows=len(frame),
                data_load_strategy="full_scan",
                primary_source_rows_total=len(frame),
            ),
        ), patch.object(
            evaluate_script,
            "build_features",
            side_effect=lambda input_frame: input_frame,
        ), patch.object(
            evaluate_script.joblib,
            "load",
            return_value=object(),
        ), patch.object(
            evaluate_script,
            "resolve_feature_selection",
            return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[], mode="config"),
        ), patch.object(
            evaluate_script,
            "resolve_model_feature_selection",
            return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[], mode="model"),
        ), patch.object(
            evaluate_script,
            "summarize_feature_coverage",
            return_value={
                "missing_force_include_features": [],
                "low_coverage_force_include_features": [],
            },
        ), patch.object(
            evaluate_script,
            "prepare_model_input_frame",
            side_effect=lambda input_frame, feature_columns, categorical_columns: input_frame[list(feature_columns)].copy(),
        ), patch.object(
            evaluate_script,
            "generate_prediction_outputs",
            return_value=SimpleNamespace(score=np.array([0.9, 0.1, 0.8, 0.2]), top3_probs=None),
        ), patch.object(
            evaluate_script,
            "prepare_scored_frame",
            side_effect=fake_prepare_scored_frame,
        ), patch.object(
            evaluate_script,
            "resolve_odds_column",
            return_value=None,
        ), patch.object(
            evaluate_script,
            "build_stability_guardrail",
            return_value={"assessment": "representative", "warnings": []},
        ), patch.object(
            evaluate_script,
            "resolve_output_artifacts",
            side_effect=lambda cfg: SimpleNamespace(
                model_path=Path(str(cfg.get("model_file", "artifacts/models/model.joblib"))),
                manifest_path=Path(str(cfg.get("manifest_file", "artifacts/models/model.manifest.json"))),
            ),
        ), patch.object(
            evaluate_script,
            "artifact_ensure_output_file_path",
            return_value=None,
        ), patch.object(
            evaluate_script,
            "write_text_file",
            side_effect=fake_write_text_file,
        ), patch.object(
            evaluate_script,
            "write_json",
            return_value=None,
        ):
            exit_code = evaluate_script.main()

        self.assertEqual(exit_code, 0)
        latest_summary_write = next(item for item in writes if item[0] == "latest summary output")
        return json.loads(latest_summary_write[2])

    def test_no_model_artifact_suffix_is_reported_as_none(self) -> None:
        summary = self._run_main_and_capture_summary(
            [
                "--artifact-suffix",
                "r_eval",
                "--model-artifact-suffix",
                evaluate_script.NO_MODEL_ARTIFACT_SUFFIX,
            ]
        )

        self.assertIsNone(summary["run_context"]["model_artifact_suffix"])
        self.assertEqual(summary["run_context"]["artifact_suffix"], "r_eval")

    def test_explicit_model_artifact_suffix_is_preserved_in_summary(self) -> None:
        summary = self._run_main_and_capture_summary(
            [
                "--artifact-suffix",
                "r_eval",
                "--model-artifact-suffix",
                "r_source_model",
            ]
        )

        self.assertEqual(summary["run_context"]["model_artifact_suffix"], "r_source_model")
        self.assertEqual(summary["run_context"]["artifact_suffix"], "r_eval")


class RunEvaluateLocalNankanTrustGuardTest(unittest.TestCase):
    def test_historical_local_nankan_is_blocked_without_override(self) -> None:
        with patch.object(
            sys,
            "argv",
            [
                "run_evaluate.py",
                "--config",
                "configs/model.yaml",
                "--data-config",
                "configs/data_local_nankan.yaml",
                "--feature-config",
                "configs/features.yaml",
            ],
        ), patch.object(
            evaluate_script,
            "load_yaml",
            side_effect=[
                {"task": "classification", "label": "is_win", "output": {}},
                {"dataset": {"source_dataset": "local_nankan", "raw_dir": "data/local_nankan/raw"}},
                {},
            ],
        ), patch.object(
            evaluate_script,
            "require_local_nankan_trust_ready",
            side_effect=ValueError("blocked by trust guard"),
        ):
            exit_code = evaluate_script.main()

        self.assertEqual(exit_code, 1)


class RunEvaluateOutputSlugTest(unittest.TestCase):
    def test_derive_evaluation_output_slug_shortens_long_model_names(self) -> None:
        long_model_path = Path(
            "artifacts/models/"
            "catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_"
            "model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.joblib"
        )

        slug = evaluate_script._derive_evaluation_output_slug(
            "artifacts/runtime_configs/model_value_stack_local_nankan_value_blend_bootstrap_"
            "r20260416_local_nankan_value_blend_bootstrap_support_corrective_candidate_v1.yaml",
            long_model_path,
        )

        self.assertLessEqual(len(slug), 96)
        self.assertRegex(slug, r"^[a-z0-9_]+$")


class RunEvaluateWfProgressPayloadTest(unittest.TestCase):
    def test_build_wf_progress_payload_tracks_fold_state(self) -> None:
        summary = {
            "run_context": {
                "profile": "test_profile",
                "config": "configs/model.yaml",
                "data_config": "configs/data.yaml",
                "feature_config": "configs/features.yaml",
                "artifact_suffix": "r_test",
                "wf_mode": "full",
                "wf_scheme": "nested",
            },
            "wf_nested_test_roi_weighted": 1.25,
            "wf_nested_test_bets_total": 321,
        }

        payload = evaluate_script._build_wf_progress_payload(
            summary=summary,
            output_slug="example_slug",
            total_folds=5,
            completed_folds=2,
            status="running",
            current_fold=3,
            current_fold_state="optimizing",
            current_score_source="default",
            current_train_window={"start_date": "2024-01-01", "end_date": "2024-06-30"},
            current_valid_window={"start_date": "2024-07-01", "end_date": "2024-08-31"},
            current_test_window={"start_date": "2024-09-01", "end_date": "2024-10-31"},
            latest_completed_fold={"fold": 2, "test_roi": 1.1},
        )

        self.assertEqual(payload["status"], "running")
        self.assertEqual(payload["output_slug"], "example_slug")
        self.assertEqual(payload["target_folds"], 5)
        self.assertEqual(payload["completed_folds"], 2)
        self.assertEqual(payload["current_fold"], 3)
        self.assertEqual(payload["current_fold_state"], "optimizing")
        self.assertEqual(payload["current_score_source"], "default")
        self.assertEqual(payload["latest_completed_fold"], {"fold": 2, "test_roi": 1.1})
        self.assertIn("updated_at", payload)

    def test_build_wf_progress_payload_adds_final_metrics_on_completion(self) -> None:
        summary = {
            "run_context": {"wf_mode": "full", "wf_scheme": "nested"},
            "wf_nested_test_roi_weighted": 1.75,
            "wf_nested_test_bets_total": 544,
        }

        payload = evaluate_script._build_wf_progress_payload(
            summary=summary,
            output_slug="example_slug",
            total_folds=5,
            completed_folds=5,
            status="completed",
        )

        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["wf_nested_test_roi_weighted"], 1.75)
        self.assertEqual(payload["wf_nested_test_bets_total"], 544)


class RunEvaluateMarketDeviationMetricsTest(unittest.TestCase):
    def test_compute_market_deviation_metrics_returns_signal_summary(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "odds": 2.0, "is_win": 1},
                {"race_id": "r1", "odds": 4.0, "is_win": 0},
                {"race_id": "r2", "odds": 3.0, "is_win": 0},
                {"race_id": "r2", "odds": 1.5, "is_win": 1},
            ]
        )

        metrics = evaluate_script._compute_market_deviation_metrics(
            frame,
            np.array([0.8, -0.2, -0.1, 0.4]),
            label_col="is_win",
        )

        self.assertIsNone(metrics["market_deviation_metrics_skipped_reason"])
        self.assertGreater(metrics["alpha_target_std"], 0.0)
        self.assertGreater(metrics["pred_std"], 0.0)
        self.assertGreater(metrics["positive_signal_rate"], 0.0)
        self.assertLessEqual(metrics["positive_signal_rate"], 1.0)
        self.assertIsInstance(metrics["alpha_pred_corr"], float)

    def test_compute_market_deviation_metrics_reports_missing_columns(self) -> None:
        frame = pd.DataFrame(
            [
                {"race_id": "r1", "is_win": 1},
                {"race_id": "r1", "is_win": 0},
            ]
        )

        metrics = evaluate_script._compute_market_deviation_metrics(
            frame,
            np.array([0.1, -0.1]),
            label_col="is_win",
        )

        self.assertEqual(metrics["market_deviation_metrics_skipped_reason"], "missing_columns:odds")
        self.assertIsNone(metrics["alpha_pred_corr"])


if __name__ == "__main__":
    unittest.main()