from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from racing_ml.common.artifacts import (
    absolutize_path,
    append_suffix_to_file_name,
    build_bundle_manifest,
    build_model_manifest,
    build_training_report_payload,
    display_path,
    derive_manifest_file_name,
    ensure_output_directory_path,
    ensure_output_file_path,
    relativize_path,
    resolve_component_from_config,
    resolve_output_artifacts,
    write_csv_file,
    write_json,
    write_text_file,
)


class ArtifactHelpersTest(unittest.TestCase):
    def _write_component_files(
        self,
        workspace_root: Path,
        *,
        include_manifest: bool = True,
        include_report: bool = True,
        manifest_metrics: dict[str, object] | None = None,
        report_metrics: dict[str, object] | None = None,
    ) -> Path:
        config_path = workspace_root / "configs" / "component.yaml"
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(
            """
task: regression
label: payout
model:
  name: catboost
output:
  model_dir: artifacts/models
  report_dir: artifacts/reports
  model_file: component.joblib
  report_file: component_metrics.json
evaluation:
  policy:
    min_rows: 100
""".strip()
            + "\n",
            encoding="utf-8",
        )

        model_path = workspace_root / "artifacts" / "models" / "component.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_text("model", encoding="utf-8")

        if include_report:
            report_path = workspace_root / "artifacts" / "reports" / "component_metrics.json"
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report_metrics or {"rmse": 1.23, "run_context": {"rows": 10}}),
                encoding="utf-8",
            )

        if include_manifest:
            manifest_path = workspace_root / "artifacts" / "models" / "component.manifest.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "metrics": manifest_metrics
                        or {"rmse": 0.99, "run_context": {"rows": 20}, "metadata": {"source": "manifest"}}
                    }
                ),
                encoding="utf-8",
            )

        return config_path

    def test_append_suffix_to_file_name_keeps_original_when_suffix_blank(self) -> None:
        self.assertEqual(append_suffix_to_file_name("report.json", None), "report.json")
        self.assertEqual(append_suffix_to_file_name("report.json", "  "), "report.json")

    def test_append_suffix_to_file_name_inserts_suffix_before_extension(self) -> None:
        self.assertEqual(
            append_suffix_to_file_name("reports/train_metrics.json", "20260321"),
            "reports/train_metrics_20260321.json",
        )

    def test_derive_manifest_file_name_uses_model_stem(self) -> None:
        self.assertEqual(derive_manifest_file_name("baseline_model.joblib"), "baseline_model.manifest.json")
        self.assertEqual(derive_manifest_file_name("models/stack.pkl"), "stack.manifest.json")

    def test_resolve_output_artifacts_uses_defaults(self) -> None:
        artifacts = resolve_output_artifacts()

        self.assertEqual(artifacts.model_dir, Path("artifacts/models"))
        self.assertEqual(artifacts.report_dir, Path("artifacts/reports"))
        self.assertEqual(artifacts.model_path, Path("artifacts/models/baseline_model.joblib"))
        self.assertEqual(artifacts.report_path, Path("artifacts/reports/train_metrics.json"))
        self.assertEqual(artifacts.manifest_path, Path("artifacts/models/baseline_model.manifest.json"))

    def test_resolve_output_artifacts_honors_custom_names(self) -> None:
        artifacts = resolve_output_artifacts(
            {
                "model_dir": "custom/models",
                "report_dir": "custom/reports",
                "model_file": "stack.pkl",
                "report_file": "summary.json",
                "manifest_file": "stack.custom.manifest.json",
            }
        )

        self.assertEqual(artifacts.model_path, Path("custom/models/stack.pkl"))
        self.assertEqual(artifacts.report_path, Path("custom/reports/summary.json"))
        self.assertEqual(artifacts.manifest_path, Path("custom/models/stack.custom.manifest.json"))

    def test_absolutize_and_relativize_path_handle_workspace_and_external_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            relative = absolutize_path("artifacts/models/model.joblib", workspace_root)
            external = workspace_root.parent / "outside.json"

            self.assertEqual(relative, workspace_root / "artifacts" / "models" / "model.joblib")
            self.assertEqual(relativize_path(relative, workspace_root), "artifacts/models/model.joblib")
            self.assertEqual(relativize_path(external, workspace_root), external.as_posix())
            self.assertIsNone(relativize_path(None, workspace_root))

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

    def test_build_training_report_payload_keeps_context_and_optional_sections(self) -> None:
        payload = build_training_report_payload(
            metrics={"accuracy": 0.8},
            run_context={"rows": 120},
            leakage_audit={"status": "ok"},
            policy_constraints={"max_odds": 50},
            extra_metadata={"revision": "r2"},
        )

        self.assertEqual(payload["accuracy"], 0.8)
        self.assertEqual(payload["run_context"], {"rows": 120})
        self.assertEqual(payload["leakage_audit"], {"status": "ok"})
        self.assertEqual(payload["policy_constraints"], {"max_odds": 50})
        self.assertEqual(payload["metadata"], {"revision": "r2"})

    def test_resolve_component_from_config_prefers_manifest_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            config_path = self._write_component_files(workspace_root)

            component = resolve_component_from_config(
                workspace_root=workspace_root,
                component_name="primary",
                config_path=config_path,
            )

            self.assertEqual(component["name"], "primary")
            self.assertEqual(component["task"], "regression")
            self.assertEqual(component["label_column"], "payout")
            self.assertEqual(component["model_name"], "catboost")
            self.assertEqual(component["model_path"], "artifacts/models/component.joblib")
            self.assertEqual(component["manifest_path"], "artifacts/models/component.manifest.json")
            self.assertEqual(component["report_path"], "artifacts/reports/component_metrics.json")
            self.assertEqual(component["metrics"], {"rmse": 0.99})
            self.assertEqual(component["evaluation_policy"], {"min_rows": 100})

    def test_resolve_component_from_config_falls_back_to_report_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            config_path = self._write_component_files(
                workspace_root,
                include_manifest=False,
                report_metrics={"rmse": 1.23, "metadata": {"source": "report"}, "policy_constraints": {"cap": 1}},
            )

            component = resolve_component_from_config(
                workspace_root=workspace_root,
                component_name="fallback",
                config_path=config_path,
            )

            self.assertIsNone(component["manifest_path"])
            self.assertEqual(component["metrics"], {"rmse": 1.23})

    def test_resolve_component_from_config_requires_model_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            workspace_root = Path(tmp_dir)
            config_path = workspace_root / "configs" / "component.yaml"
            config_path.parent.mkdir(parents=True, exist_ok=True)
            config_path.write_text(
                "output:\n  model_dir: artifacts/models\n  model_file: missing.joblib\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(FileNotFoundError, "Component 'missing' model file not found"):
                resolve_component_from_config(
                    workspace_root=workspace_root,
                    component_name="missing",
                    config_path=config_path,
                )

    def test_build_bundle_manifest_counts_components(self) -> None:
        manifest = build_bundle_manifest(
            bundle_name="stack_a",
            bundle_kind="ensemble",
            primary_component="win",
            components={"win": {"model_path": "artifacts/models/win.joblib"}, "place": {"model_path": "artifacts/models/place.joblib"}},
        )

        self.assertEqual(manifest["bundle_name"], "stack_a")
        self.assertEqual(manifest["primary_component"], "win")
        self.assertEqual(manifest["metadata"]["component_count"], 2)
        self.assertEqual(manifest["metadata"]["component_names"], ["win", "place"])


if __name__ == "__main__":
    unittest.main()