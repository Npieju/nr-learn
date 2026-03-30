import unittest

from racing_ml.common.execution_capacity import assert_no_conflicting_heavy_processes
from racing_ml.common.execution_capacity import build_execution_capacity_status
from racing_ml.common.execution_capacity import find_conflicting_heavy_processes


class ExecutionCapacityTests(unittest.TestCase):
    def test_find_conflicting_heavy_processes_detects_other_heavy_jobs(self) -> None:
        process_table = "\n".join(
            [
                "PID PPID COMMAND",
                "100 1 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
                "101 1 /workspaces/nr-learn/.venv/bin/python scripts/run_collect_local_nankan.py --config configs/crawl_local_nankan_template.yaml --target pedigree",
                "102 1 /workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --config configs/model.yaml",
            ]
        )

        matches = find_conflicting_heavy_processes(
            current_pid=100,
            current_script_pattern="scripts/run_train.py",
            process_table=process_table,
        )

        self.assertEqual(
            [(item.kind, item.pid) for item in matches],
            [("local_nankan_collect", 101), ("evaluate", 102)],
        )

    def test_find_conflicting_heavy_processes_marks_same_script_duplicates(self) -> None:
        process_table = "\n".join(
            [
                "PID PPID COMMAND",
                "200 1 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
                "201 1 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
            ]
        )

        matches = find_conflicting_heavy_processes(
            current_pid=200,
            current_script_pattern="scripts/run_train.py",
            process_table=process_table,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].kind, "same_script")
        self.assertEqual(matches[0].pid, 201)

    def test_assert_no_conflicting_heavy_processes_raises_concise_error(self) -> None:
        process_table = "\n".join(
            [
                "PID PPID COMMAND",
                "300 1 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
                "301 1 /workspaces/nr-learn/.venv/bin/python scripts/run_collect_local_nankan.py --config configs/crawl_local_nankan_template.yaml --target pedigree",
            ]
        )

        with self.assertRaises(ValueError) as context:
            assert_no_conflicting_heavy_processes(
                current_pid=300,
                current_script_pattern="scripts/run_train.py",
                process_table=process_table,
            )

        message = str(context.exception)
        self.assertIn("resource-safe execution requires a quiet heavy-job lane", message)
        self.assertIn("local_nankan_collect:pid=301", message)

    def test_build_execution_capacity_status_returns_blocked_payload(self) -> None:
        process_table = "\n".join(
            [
                "PID PPID COMMAND",
                "400 1 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
                "401 1 /workspaces/nr-learn/.venv/bin/python scripts/run_backfill_local_nankan.py --crawl-config configs/crawl_local_nankan_template.yaml",
            ]
        )

        payload = build_execution_capacity_status(
            current_pid=400,
            current_script_pattern="scripts/run_train.py",
            process_table=process_table,
        )

        self.assertEqual(payload["status"], "blocked")
        self.assertEqual(payload["conflict_count"], 1)
        self.assertEqual(payload["conflicts"][0]["kind"], "local_nankan_backfill")
        self.assertEqual(payload["conflicts"][0]["pid"], 401)

    def test_find_conflicting_heavy_processes_ignores_ancestor_revision_gate(self) -> None:
        process_table = "\n".join(
            [
                "PID PPID COMMAND",
                "500 1 /workspaces/nr-learn/.venv/bin/python scripts/run_local_revision_gate.py --revision nar_baseline",
                "501 500 /workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py --revision nar_baseline",
                "502 501 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model_local_baseline.yaml",
                "503 1 /workspaces/nr-learn/.venv/bin/python scripts/run_collect_local_nankan.py --config configs/crawl_local_nankan_template.yaml",
            ]
        )

        matches = find_conflicting_heavy_processes(
            current_pid=502,
            current_script_pattern="scripts/run_train.py",
            process_table=process_table,
        )

        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].kind, "local_nankan_collect")
        self.assertEqual(matches[0].pid, 503)


if __name__ == "__main__":
    unittest.main()
