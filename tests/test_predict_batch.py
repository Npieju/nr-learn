from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from racing_ml.serving.predict_batch import run_predict_from_frame


def _minimal_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"date": pd.Timestamp("2026-04-05"), "race_id": "r1", "horse_id": "h1", "horse_name": "A", "odds": 2.5, "f1": 0.9},
            {"date": pd.Timestamp("2026-04-05"), "race_id": "r1", "horse_id": "h2", "horse_name": "B", "odds": 30.0, "f1": 0.1},
            {"date": pd.Timestamp("2026-04-05"), "race_id": "r2", "horse_id": "h3", "horse_name": "C", "odds": 4.0, "f1": 0.8},
            {"date": pd.Timestamp("2026-04-05"), "race_id": "r2", "horse_id": "h4", "horse_name": "D", "odds": 7.5, "f1": 0.2},
        ]
    )


class PredictBatchContractTest(unittest.TestCase):
    def test_run_predict_from_frame_writes_policy_diagnostics_into_summary(self) -> None:
        frame = _minimal_frame()
        csv_writes: list[tuple[Path, pd.DataFrame]] = []
        json_writes: list[tuple[Path, dict[str, object]]] = []
        model_path = Path("/workspaces/nr-learn/artifacts/models/test_predict_model.joblib")

        def fake_prepare_scored_frame(input_frame: pd.DataFrame, scores: np.ndarray, *, odds_col: str | None, score_col: str) -> pd.DataFrame:
            output = input_frame.copy()
            output[score_col] = scores
            output["pred_rank"] = [1, 2, 1, 2]
            output["expected_value"] = [1.10, 0.80, 1.05, 0.70]
            output["ev_rank"] = [1, 2, 1, 2]
            return output

        def fake_annotate_runtime_policy(
            input_frame: pd.DataFrame,
            *,
            odds_col: str | None,
            policy_name: str,
            policy_config: dict[str, object],
            score_col: str,
        ) -> pd.DataFrame:
            output = input_frame.copy()
            output["policy_name"] = policy_name
            output["policy_strategy_kind"] = str(policy_config.get("strategy_kind"))
            output["policy_selected"] = [True, False, True, False]
            output["policy_selection_rank"] = [1, pd.NA, 1, pd.NA]
            output["policy_weight"] = [0.5, pd.NA, 0.5, pd.NA]
            output["policy_prob"] = [0.40, 0.05, 0.35, 0.10]
            output["policy_market_prob"] = [0.38, 0.03, 0.30, 0.08]
            output["policy_expected_value"] = [1.10, 0.80, 1.05, 0.70]
            output["policy_edge"] = [0.10, -0.20, 0.05, -0.30]
            output["policy_blend_weight"] = [0.8, 0.8, 0.8, 0.8]
            output["policy_min_prob"] = [0.05, 0.05, 0.05, 0.05]
            output["policy_odds_min"] = [1.0, 1.0, 1.0, 1.0]
            output["policy_odds_max"] = [25.0, 25.0, 25.0, 25.0]
            output["policy_top_k"] = [1, 1, 1, 1]
            output["policy_min_expected_value"] = [1.0, 1.0, 1.0, 1.0]
            output["policy_reject_reason_primary"] = [pd.NA, "odds_above_max", pd.NA, "expected_value_below_min_expected_value"]
            output["policy_reject_reasons"] = [pd.NA, "odds_above_max", pd.NA, "expected_value_below_min_expected_value"]
            return output

        with patch("racing_ml.serving.predict_batch.load_yaml", side_effect=[
            {"label": "is_win", "output": {"model_file": "artifacts/models/test_predict_model.joblib", "manifest_file": "artifacts/models/test_predict_model.manifest.json"}},
            {},
        ]), patch(
            "racing_ml.serving.predict_batch.resolve_output_artifacts",
            return_value=SimpleNamespace(model_path=model_path, manifest_path=Path("/workspaces/nr-learn/artifacts/models/test_predict_model.manifest.json")),
        ), patch(
            "racing_ml.serving.predict_batch.joblib.load",
            return_value=object(),
        ), patch(
            "racing_ml.serving.predict_batch.resolve_feature_selection",
            return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[]),
        ), patch(
            "racing_ml.serving.predict_batch.resolve_model_feature_selection",
            return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[]),
        ), patch(
            "racing_ml.serving.predict_batch.prepare_model_input_frame",
            side_effect=lambda input_frame, feature_columns, categorical_columns: input_frame[list(feature_columns)].copy(),
        ), patch(
            "racing_ml.serving.predict_batch.generate_prediction_outputs",
            return_value=SimpleNamespace(score=np.array([0.9, 0.1, 0.8, 0.2]), top3_probs=None),
        ), patch(
            "racing_ml.serving.predict_batch.resolve_odds_column",
            return_value="odds",
        ), patch(
            "racing_ml.serving.predict_batch.prepare_scored_frame",
            side_effect=fake_prepare_scored_frame,
        ), patch(
            "racing_ml.serving.predict_batch.resolve_runtime_policy",
            return_value=("runtime_portfolio_probe", {"strategy_kind": "portfolio"}),
        ), patch(
            "racing_ml.serving.predict_batch.annotate_runtime_policy",
            side_effect=fake_annotate_runtime_policy,
        ), patch(
            "racing_ml.serving.predict_batch.write_csv_file",
            side_effect=lambda path, data, **kwargs: csv_writes.append((path, data.copy())),
        ), patch(
            "racing_ml.serving.predict_batch.write_json",
            side_effect=lambda path, payload, **kwargs: json_writes.append((path, payload)),
        ), patch(
            "racing_ml.serving.predict_batch._plot_predictions",
            return_value=None,
        ), patch.object(Path, "exists", autospec=True, side_effect=lambda self: self == model_path):
            summary = run_predict_from_frame(
                model_config_path="configs/model.yaml",
                feature_config_path="configs/features.yaml",
                frame=frame,
                race_date="2026-04-05",
                profile_name="test_profile",
            )

        self.assertEqual(summary["policy_selected_rows"], 2)
        self.assertIsNotNone(summary["policy_diagnostics"])
        self.assertEqual(summary["policy_diagnostics"]["policy_selected_rows"], 2)
        self.assertEqual(summary["policy_diagnostics"]["likely_blocker_reason"], "odds_above_max")
        self.assertEqual(len(json_writes), 1)
        self.assertEqual(json_writes[0][1]["policy_diagnostics"]["rejected_rows"], 2)

    def test_run_predict_from_frame_includes_reject_reason_columns_in_csv(self) -> None:
        frame = _minimal_frame()
        csv_writes: list[tuple[Path, pd.DataFrame]] = []
        model_path = Path("/workspaces/nr-learn/artifacts/models/test_predict_model.joblib")

        def fake_prepare_scored_frame(input_frame: pd.DataFrame, scores: np.ndarray, *, odds_col: str | None, score_col: str) -> pd.DataFrame:
            output = input_frame.copy()
            output[score_col] = scores
            output["pred_rank"] = [1, 2, 1, 2]
            return output

        def fake_annotate_runtime_policy(
            input_frame: pd.DataFrame,
            *,
            odds_col: str | None,
            policy_name: str,
            policy_config: dict[str, object],
            score_col: str,
        ) -> pd.DataFrame:
            output = input_frame.copy()
            output["policy_name"] = policy_name
            output["policy_strategy_kind"] = str(policy_config.get("strategy_kind"))
            output["policy_selected"] = [False, False, False, False]
            output["policy_reject_reason_primary"] = ["prob_below_min_prob", "odds_above_max", "prob_below_min_prob", "ranked_below_top_k"]
            output["policy_reject_reasons"] = output["policy_reject_reason_primary"]
            return output

        with patch("racing_ml.serving.predict_batch.load_yaml", side_effect=[
            {"label": "is_win", "output": {"model_file": "artifacts/models/test_predict_model.joblib", "manifest_file": "artifacts/models/test_predict_model.manifest.json"}},
            {},
        ]), patch(
            "racing_ml.serving.predict_batch.resolve_output_artifacts",
            return_value=SimpleNamespace(model_path=model_path, manifest_path=Path("/workspaces/nr-learn/artifacts/models/test_predict_model.manifest.json")),
        ), patch(
            "racing_ml.serving.predict_batch.joblib.load",
            return_value=object(),
        ), patch(
            "racing_ml.serving.predict_batch.resolve_feature_selection",
            return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[]),
        ), patch(
            "racing_ml.serving.predict_batch.resolve_model_feature_selection",
            return_value=SimpleNamespace(feature_columns=["f1"], categorical_columns=[]),
        ), patch(
            "racing_ml.serving.predict_batch.prepare_model_input_frame",
            side_effect=lambda input_frame, feature_columns, categorical_columns: input_frame[list(feature_columns)].copy(),
        ), patch(
            "racing_ml.serving.predict_batch.generate_prediction_outputs",
            return_value=SimpleNamespace(score=np.array([0.9, 0.1, 0.8, 0.2]), top3_probs=None),
        ), patch(
            "racing_ml.serving.predict_batch.resolve_odds_column",
            return_value="odds",
        ), patch(
            "racing_ml.serving.predict_batch.prepare_scored_frame",
            side_effect=fake_prepare_scored_frame,
        ), patch(
            "racing_ml.serving.predict_batch.resolve_runtime_policy",
            return_value=("runtime_portfolio_probe", {"strategy_kind": "portfolio"}),
        ), patch(
            "racing_ml.serving.predict_batch.annotate_runtime_policy",
            side_effect=fake_annotate_runtime_policy,
        ), patch(
            "racing_ml.serving.predict_batch.write_csv_file",
            side_effect=lambda path, data, **kwargs: csv_writes.append((path, data.copy())),
        ), patch(
            "racing_ml.serving.predict_batch.write_json",
            return_value=None,
        ), patch(
            "racing_ml.serving.predict_batch._plot_predictions",
            return_value=None,
        ), patch.object(Path, "exists", autospec=True, side_effect=lambda self: self == model_path):
            run_predict_from_frame(
                model_config_path="configs/model.yaml",
                feature_config_path="configs/features.yaml",
                frame=frame,
                race_date="2026-04-05",
            )

        output = csv_writes[0][1]
        self.assertIn("policy_reject_reason_primary", output.columns)
        self.assertIn("policy_reject_reasons", output.columns)
        self.assertEqual(output.loc[output["horse_id"] == "h2", "policy_reject_reason_primary"].iloc[0], "odds_above_max")


if __name__ == "__main__":
    unittest.main()