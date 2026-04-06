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


if __name__ == "__main__":
    unittest.main()