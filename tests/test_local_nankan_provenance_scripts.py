from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _load_script_module(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


capture_coverage_script = _load_script_module(
    "test_run_local_nankan_pre_race_capture_coverage",
    "scripts/run_local_nankan_pre_race_capture_coverage.py",
)
provenance_audit_script = _load_script_module(
    "test_run_local_nankan_provenance_audit",
    "scripts/run_local_nankan_provenance_audit.py",
)
source_timing_script = _load_script_module(
    "test_run_local_nankan_source_timing_audit",
    "scripts/run_local_nankan_source_timing_audit.py",
)


class LocalNankanProvenanceScriptsTest(unittest.TestCase):
    def test_pre_race_capture_coverage_main_writes_summary_and_date_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            race_card = tmp_path / "race_card.csv"
            race_result = tmp_path / "race_result.csv"
            baseline_summary = tmp_path / "baseline_summary.json"
            summary_output = tmp_path / "capture_summary.json"
            date_coverage_output = tmp_path / "capture_dates.csv"

            race_card_frame = pd.DataFrame([{"race_id": "r1"}, {"race_id": "r2"}])
            race_result_frame = pd.DataFrame([{"race_id": "r1"}])
            date_coverage_frame = pd.DataFrame([{"date": "2026-04-06", "pre_race_only_rows": 2}])
            summary_payload = {
                "status": "completed",
                "current_phase": "capturing_pre_race_pool",
                "recommended_action": "continue_pre_race_capture_loop",
                "pre_race_only_rows": 2,
                "pre_race_only_races": 2,
                "result_ready_races": 1,
                "pending_result_races": 1,
            }
            writes: dict[str, object] = {}

            with patch.object(
                capture_coverage_script.pd,
                "read_csv",
                side_effect=[race_card_frame, race_result_frame],
            ), patch.object(
                capture_coverage_script,
                "read_json",
                return_value={"pre_race_only_rows": 1},
            ), patch.object(
                capture_coverage_script,
                "build_pre_race_capture_coverage_summary",
                return_value=summary_payload.copy(),
            ) as build_summary, patch.object(
                capture_coverage_script,
                "build_pre_race_capture_date_coverage",
                return_value=date_coverage_frame,
            ) as build_dates, patch.object(
                capture_coverage_script,
                "write_json",
                side_effect=lambda path, payload, **kwargs: writes.__setitem__("summary", (path, payload)),
            ), patch.object(
                capture_coverage_script,
                "write_csv_file",
                side_effect=lambda path, frame, **kwargs: writes.__setitem__("date_coverage", (path, frame.copy())),
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_capture_coverage.py",
                    "--race-card-input",
                    str(race_card),
                    "--race-result-input",
                    str(race_result),
                    "--summary-output",
                    str(summary_output),
                    "--date-coverage-output",
                    str(date_coverage_output),
                    "--baseline-summary-input",
                    str(baseline_summary),
                ],
            ):
                baseline_summary.write_text("{}", encoding="utf-8")
                race_result.write_text("race_id\nr1\n", encoding="utf-8")
                exit_code = capture_coverage_script.main()

            self.assertEqual(exit_code, 0)
            build_summary.assert_called_once()
            build_dates.assert_called_once()
            summary_path, summary_written = writes["summary"]
            self.assertEqual(summary_path, summary_output)
            self.assertEqual(summary_written["read_order"][0], "status")
            self.assertEqual(summary_written["read_order"][3], "pre_race_only_rows")
            self.assertEqual(summary_written["read_order"][5], "pending_result_races")
            self.assertEqual(summary_written["race_card_input"], str(race_card))
            self.assertEqual(summary_written["race_result_input"], str(race_result))
            self.assertEqual(summary_written["baseline_summary_input"], str(baseline_summary))
            self.assertEqual(summary_written["date_coverage_output"], str(date_coverage_output))
            date_path, date_frame_written = writes["date_coverage"]
            self.assertEqual(date_path, date_coverage_output)
            self.assertEqual(len(date_frame_written), 1)

    def test_pre_race_capture_coverage_normalizes_nested_workspace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            race_card = tmp_path / "race_card.csv"
            race_result = tmp_path / "race_result.csv"
            summary_output = tmp_path / "capture_summary.json"
            date_coverage_output = tmp_path / "capture_dates.csv"

            writes: dict[str, object] = {}
            summary_payload = {
                "status": "completed",
                "current_phase": "capturing_pre_race_pool",
                "recommended_action": "continue_pre_race_capture_loop",
                "pre_race_only_rows": 2,
                "pre_race_only_races": 2,
                "result_ready_races": 1,
                "pending_result_races": 1,
                "baseline_comparison": {
                    "baseline_summary_input": str(ROOT / "artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json"),
                },
            }

            with patch.object(
                capture_coverage_script.pd,
                "read_csv",
                side_effect=[pd.DataFrame([{"race_id": "r1"}]), pd.DataFrame([{"race_id": "r1"}])],
            ), patch.object(
                capture_coverage_script,
                "build_pre_race_capture_coverage_summary",
                return_value=summary_payload,
            ), patch.object(
                capture_coverage_script,
                "build_pre_race_capture_date_coverage",
                return_value=pd.DataFrame([{"date": "2026-04-06", "pre_race_only_rows": 1}]),
            ), patch.object(
                capture_coverage_script,
                "write_json",
                side_effect=lambda path, payload, **kwargs: writes.__setitem__("summary", (path, payload)),
            ), patch.object(
                capture_coverage_script,
                "write_csv_file",
                return_value=None,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_capture_coverage.py",
                    "--race-card-input",
                    str(race_card),
                    "--race-result-input",
                    str(race_result),
                    "--summary-output",
                    str(summary_output),
                    "--date-coverage-output",
                    str(date_coverage_output),
                ],
            ):
                race_result.write_text("race_id\nr1\n", encoding="utf-8")
                exit_code = capture_coverage_script.main()

            self.assertEqual(exit_code, 0)
            _, summary_written = writes["summary"]
            self.assertEqual(summary_written["read_order"][0], "status")
            self.assertEqual(summary_written["read_order"][3], "pre_race_only_rows")
            self.assertEqual(summary_written["read_order"][5], "pending_result_races")
            self.assertEqual(
                summary_written["baseline_comparison"]["baseline_summary_input"],
                "artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json",
            )

    def test_source_timing_audit_normalizes_nested_workspace_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            race_card = tmp_path / "race_card.csv"
            race_result = tmp_path / "race_result.csv"
            summary_output = tmp_path / "source_timing_summary.json"
            date_output = tmp_path / "source_timing_dates.csv"
            year_output = tmp_path / "source_timing_years.csv"

            writes: dict[str, object] = {}
            summary_payload = {
                "status": "completed",
                "current_phase": "future_only_pre_race_capture_available",
                "historical_pre_race_recoverability": {
                    "result_ready_pre_race_rows": 0,
                },
                "audit_snapshot_path": str(ROOT / "artifacts/reports/local_nankan_source_timing_audit.json"),
            }

            with patch.object(
                source_timing_script.pd,
                "read_csv",
                side_effect=[pd.DataFrame([{"race_id": "r1"}]), pd.DataFrame([{"race_id": "r1"}])],
            ), patch.object(
                source_timing_script,
                "build_source_timing_audit_summary",
                return_value=(
                    summary_payload,
                    pd.DataFrame([{"date": "2026-04-06", "result_ready_pre_race_rows": 0}]),
                    pd.DataFrame([{"year": 2026, "result_ready_pre_race_rows": 0}]),
                ),
            ), patch.object(
                source_timing_script,
                "write_json",
                side_effect=lambda path, payload, **kwargs: writes.__setitem__("summary", (path, payload)),
            ), patch.object(
                source_timing_script,
                "write_csv_file",
                return_value=None,
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_source_timing_audit.py",
                    "--race-card-input",
                    str(race_card),
                    "--race-result-input",
                    str(race_result),
                    "--summary-output",
                    str(summary_output),
                    "--date-output",
                    str(date_output),
                    "--year-output",
                    str(year_output),
                ],
            ):
                race_result.write_text("race_id\nr1\n", encoding="utf-8")
                exit_code = source_timing_script.main()

            self.assertEqual(exit_code, 0)
            _, summary_written = writes["summary"]
            self.assertEqual(summary_written["read_order"][0], "status")
            self.assertEqual(summary_written["read_order"][3], "historical_pre_race_recoverability.result_ready_pre_race_rows")
            self.assertEqual(summary_written["read_order"][5], "historical_pre_race_recoverability.status")
            self.assertEqual(summary_written["audit_snapshot_path"], "artifacts/reports/local_nankan_source_timing_audit.json")

    def test_provenance_audit_main_writes_summary_and_optional_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            input_file = tmp_path / "local_nankan_primary.csv"
            summary_output = tmp_path / "provenance_summary.json"
            annotated_output = tmp_path / "annotated.csv"
            pre_race_output = tmp_path / "pre_race.csv"

            input_frame = pd.DataFrame([{"race_id": "r1", "market_timing_bucket": "pre_race"}])
            annotated_frame = pd.DataFrame([{"race_id": "r1", "market_timing_bucket": "pre_race"}])
            pre_race_frame = pd.DataFrame([{"race_id": "r1"}])
            summary_payload = {
                "pre_race_only_rows": 1,
                "unknown_rows": 0,
                "post_race_rows": 0,
            }
            csv_writes: list[tuple[Path, pd.DataFrame]] = []
            json_writes: list[tuple[Path, dict[str, object]]] = []

            with patch.object(
                provenance_audit_script.pd,
                "read_csv",
                return_value=input_frame,
            ), patch.object(
                provenance_audit_script,
                "annotate_market_timing_bucket",
                return_value=annotated_frame,
            ) as annotate_mock, patch.object(
                provenance_audit_script,
                "build_provenance_summary",
                return_value=summary_payload,
            ) as summary_mock, patch.object(
                provenance_audit_script,
                "filter_pre_race_only",
                return_value=pre_race_frame,
            ) as pre_race_mock, patch.object(
                provenance_audit_script,
                "write_json",
                side_effect=lambda path, payload, **kwargs: json_writes.append((path, payload)),
            ), patch.object(
                provenance_audit_script,
                "write_csv_file",
                side_effect=lambda path, frame, **kwargs: csv_writes.append((path, frame.copy())),
            ), patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_provenance_audit.py",
                    "--input-file",
                    str(input_file),
                    "--summary-output",
                    str(summary_output),
                    "--annotated-output",
                    str(annotated_output),
                    "--pre-race-output",
                    str(pre_race_output),
                ],
            ):
                exit_code = provenance_audit_script.main()

            self.assertEqual(exit_code, 0)
            annotate_mock.assert_called_once_with(input_frame)
            summary_mock.assert_called_once_with(annotated_frame)
            pre_race_mock.assert_called_once_with(annotated_frame)
            self.assertEqual(json_writes[0][0], summary_output)
            self.assertEqual(json_writes[0][1]["pre_race_only_rows"], 1)
            self.assertEqual([path for path, _ in csv_writes], [annotated_output, pre_race_output])

    def test_provenance_audit_normalizes_workspace_relative_artifact_paths(self) -> None:
        input_file = ROOT / "data/local_nankan/raw/local_nankan_primary.csv"
        summary_output = ROOT / "artifacts/tmp/provenance_summary.json"
        manifest_output = ROOT / "artifacts/tmp/provenance_manifest.json"
        annotated_output = ROOT / "artifacts/tmp/provenance_annotated.csv"
        pre_race_output = ROOT / "artifacts/tmp/provenance_pre_race.csv"

        json_writes: list[tuple[Path, dict[str, object]]] = []

        with patch.object(
            provenance_audit_script.pd,
            "read_csv",
            return_value=pd.DataFrame([{"race_id": "r1", "market_timing_bucket": "pre_race"}]),
        ), patch.object(
            provenance_audit_script,
            "annotate_market_timing_bucket",
            return_value=pd.DataFrame([{"race_id": "r1", "market_timing_bucket": "pre_race"}]),
        ), patch.object(
            provenance_audit_script,
            "build_provenance_summary",
            return_value={
                "pre_race_only_rows": 1,
                "unknown_rows": 0,
                "post_race_rows": 0,
                "source_manifest_path": str(ROOT / "artifacts/reports/local_nankan_provenance_audit.json"),
            },
        ), patch.object(
            provenance_audit_script,
            "filter_pre_race_only",
            return_value=pd.DataFrame([{"race_id": "r1"}]),
        ), patch.object(
            provenance_audit_script,
            "write_json",
            side_effect=lambda path, payload, **kwargs: json_writes.append((path, payload)),
        ), patch.object(
            provenance_audit_script,
            "write_csv_file",
            return_value=None,
        ), patch.object(
            sys,
            "argv",
            [
                "run_local_nankan_provenance_audit.py",
                "--input-file",
                str(input_file),
                "--summary-output",
                str(summary_output),
                "--manifest-output",
                str(manifest_output),
                "--annotated-output",
                str(annotated_output),
                "--pre-race-output",
                str(pre_race_output),
            ],
        ):
            exit_code = provenance_audit_script.main()

        self.assertEqual(exit_code, 0)
        manifest_written = json_writes[-1][1]
        self.assertEqual(manifest_written["artifacts"]["input_file"], "data/local_nankan/raw/local_nankan_primary.csv")
        self.assertEqual(manifest_written["artifacts"]["summary_output"], "artifacts/tmp/provenance_summary.json")
        self.assertEqual(manifest_written["artifacts"]["annotated_output"], "artifacts/tmp/provenance_annotated.csv")
        self.assertEqual(manifest_written["provenance_summary"]["source_manifest_path"], "artifacts/reports/local_nankan_provenance_audit.json")


if __name__ == "__main__":
    unittest.main()