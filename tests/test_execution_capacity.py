from racing_ml.common.execution_capacity import assert_no_conflicting_heavy_processes
from racing_ml.common.execution_capacity import find_conflicting_heavy_processes


def test_find_conflicting_heavy_processes_detects_other_heavy_jobs():
    process_table = "\n".join(
        [
            "PID COMMAND",
            "100 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
            "101 /workspaces/nr-learn/.venv/bin/python scripts/run_collect_local_nankan.py --config configs/crawl_local_nankan_template.yaml --target pedigree",
            "102 /workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --config configs/model.yaml",
        ]
    )

    matches = find_conflicting_heavy_processes(
        current_pid=100,
        current_script_pattern="scripts/run_train.py",
        process_table=process_table,
    )

    assert [(item.kind, item.pid) for item in matches] == [
        ("local_nankan_collect", 101),
        ("evaluate", 102),
    ]


def test_find_conflicting_heavy_processes_marks_same_script_duplicates():
    process_table = "\n".join(
        [
            "PID COMMAND",
            "200 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
            "201 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
        ]
    )

    matches = find_conflicting_heavy_processes(
        current_pid=200,
        current_script_pattern="scripts/run_train.py",
        process_table=process_table,
    )

    assert len(matches) == 1
    assert matches[0].kind == "same_script"
    assert matches[0].pid == 201


def test_assert_no_conflicting_heavy_processes_raises_concise_error():
    process_table = "\n".join(
        [
            "PID COMMAND",
            "300 /workspaces/nr-learn/.venv/bin/python scripts/run_train.py --config configs/model.yaml",
            "301 /workspaces/nr-learn/.venv/bin/python scripts/run_collect_local_nankan.py --config configs/crawl_local_nankan_template.yaml --target pedigree",
        ]
    )

    try:
        assert_no_conflicting_heavy_processes(
            current_pid=300,
            current_script_pattern="scripts/run_train.py",
            process_table=process_table,
        )
    except ValueError as error:
        message = str(error)
    else:
        raise AssertionError("expected ValueError")

    assert "resource-safe execution requires a quiet heavy-job lane" in message
    assert "local_nankan_collect:pid=301" in message
