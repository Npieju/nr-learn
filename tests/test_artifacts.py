from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from racing_ml.common.artifacts import (
    append_suffix_to_file_name,
    build_model_manifest,
    display_path,
    ensure_output_directory_path,
    ensure_output_file_path,
    write_csv_file,
    write_json,
    write_text_file,
)


class ArtifactHelpersTest(unittest.TestCase):
    def test_append_suffix_to_file_name_keeps_original_when_suffix_blank(self) -> None:
        self.assertEqual(append_suffix_to_file_name("report.json", None), "report.json")
        self.assertEqual(append_suffix_to_file_name("report.json", "  "), "report.json")

    def test_append_suffix_to_file_name_inserts_suffix_before_extension(self) -> None:
        self.assertEqual(
            append_suffix_to_file_name("reports/train_metrics.json", "20260321"),
            "reports/train_metrics_20260321.json",
        )

    def test_display_path_relativizes_absolute_paths_under_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            target = workspace_root / "artifacts" / "reports" / "summary.json"
            self.assertEqual(display_path(target, workspace_root), "artifacts/reports/summary.json")

    def test_output_path_guards_reject_wrong_kinds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            existing_dir = workspace_root / "reports"
            existing_dir.mkdir()
            existing_file = workspace_root / "metrics.json"
            existing_file.write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(IsADirectoryError, "must be a file path"):
                ensure_output_file_path(existing_dir, label="report", workspace_root=workspace_root)

            with self.assertRaisesRegex(NotADirectoryError, "must be a directory path"):
                ensure_output_directory_path(existing_file, label="report_dir", workspace_root=workspace_root)

    def test_write_helpers_create_parent_directories(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            json_path = workspace_root / "nested" / "metrics.json"
            text_path = workspace_root / "logs" / "summary.txt"
            csv_path = workspace_root / "tables" / "predictions.csv"

            write_json(json_path, {"accuracy": 0.75})
            write_text_file(text_path, "ok\n")
            write_csv_file(csv_path, pd.DataFrame([{"race_id": 1, "score": 0.12}]))

            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8"))["accuracy"], 0.75)
            self.assertEqual(text_path.read_text(encoding="utf-8"), "ok\n")
            self.assertIn("race_id,score", csv_path.read_text(encoding="utf-8"))

    def test_build_model_manifest_relativizes_paths_and_preserves_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            manifest = build_model_manifest(
                workspace_root=workspace_root,
                model_config_path="configs/model.yaml",
                data_config_path="configs/data.yaml",
                feature_config_path="configs/features.yaml",
                model_path="artifacts/models/model.joblib",
                report_path="artifacts/reports/train_metrics.json",
                task="classification",
                label_column="is_win",
                model_name="lightgbm",
                used_features=["odds", "horse_weight"],
                categorical_columns=["track_code"],
                metrics={"accuracy": 0.8},
                run_context={"rows": 100},
                leakage_audit={"status": "ok"},
                policy_constraints={"max_odds": 50},
                extra_metadata={"revision": "r1"},
            )

            self.assertEqual(manifest["model"]["path"], "artifacts/models/model.joblib")
            self.assertEqual(manifest["model"]["report_path"], "artifacts/reports/train_metrics.json")
            self.assertEqual(manifest["configs"]["model_config"], "configs/model.yaml")
            self.assertEqual(manifest["features"]["count"], 2)
            self.assertEqual(manifest["features"]["categorical_count"], 1)
            self.assertEqual(manifest["metadata"], {"revision": "r1"})


if __name__ == "__main__":
    unittest.main()