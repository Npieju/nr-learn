from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


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


readiness_probe_script = _load_script_module(
    "test_run_local_nankan_pre_race_readiness_probe",
    "scripts/run_local_nankan_pre_race_readiness_probe.py",
)


class LocalNankanReadinessProbeScriptTest(unittest.TestCase):
    def test_readiness_probe_returns_not_ready_when_results_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            race_card = tmp_path / "race_card.csv"
            race_result = tmp_path / "race_result.csv"
            summary_output = tmp_path / "probe_summary.json"

            race_card.write_text(
                "race_id,date,card_snapshot_relation,odds_snapshot_relation\n"
                "r1,2026-04-06,pre_race,pre_race\n"
                "r2,2026-04-07,pre_race,pre_race\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_readiness_probe.py",
                    "--race-card-input",
                    str(race_card),
                    "--race-result-input",
                    str(race_result),
                    "--summary-output",
                    str(summary_output),
                ],
            ):
                exit_code = readiness_probe_script.main()

            self.assertEqual(exit_code, 2)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "not_ready")
            self.assertEqual(summary["current_phase"], "await_result_arrival")
            self.assertEqual(summary["recommended_action"], "wait_for_result_ready_pre_race_races")
            self.assertEqual(summary["materialization_summary"]["result_ready_races"], 0)
            self.assertEqual(summary["materialization_summary"]["pending_result_races"], 2)

    def test_readiness_probe_returns_ready_when_result_ready_race_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            race_card = tmp_path / "race_card.csv"
            race_result = tmp_path / "race_result.csv"
            summary_output = tmp_path / "probe_summary.json"

            race_card.write_text(
                "race_id,date,card_snapshot_relation,odds_snapshot_relation\n"
                "r1,2026-04-06,pre_race,pre_race\n"
                "r2,2026-04-07,pre_race,pre_race\n",
                encoding="utf-8",
            )
            race_result.write_text(
                "race_id\n"
                "r2\n",
                encoding="utf-8",
            )

            with patch.object(
                sys,
                "argv",
                [
                    "run_local_nankan_pre_race_readiness_probe.py",
                    "--race-card-input",
                    str(race_card),
                    "--race-result-input",
                    str(race_result),
                    "--summary-output",
                    str(summary_output),
                ],
            ):
                exit_code = readiness_probe_script.main()

            self.assertEqual(exit_code, 0)
            summary = json.loads(summary_output.read_text(encoding="utf-8"))
            self.assertEqual(summary["status"], "ready")
            self.assertEqual(summary["current_phase"], "ready_for_benchmark_handoff")
            self.assertEqual(summary["recommended_action"], "run_pre_race_benchmark_handoff")
            self.assertEqual(summary["materialization_summary"]["result_ready_races"], 1)
            self.assertEqual(summary["materialization_summary"]["pending_result_races"], 1)


if __name__ == "__main__":
    unittest.main()