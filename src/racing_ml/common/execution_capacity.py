from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HeavyProcessMatch:
    pid: int
    kind: str
    command: str


_HEAVY_PROCESS_PATTERNS: tuple[tuple[str, str], ...] = (
    ("train", "scripts/run_train.py"),
    ("evaluate", "scripts/run_evaluate.py"),
    ("revision_gate", "scripts/run_revision_gate.py"),
    ("local_nankan_collect", "scripts/run_collect_local_nankan.py"),
    ("local_nankan_backfill", "scripts/run_backfill_local_nankan.py"),
)


def _read_process_table() -> str:
    result = subprocess.run(
        ["ps", "-eo", "pid,args"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def find_conflicting_heavy_processes(
    *,
    current_pid: int | None = None,
    current_script_pattern: str | None = None,
    process_table: str | None = None,
) -> list[HeavyProcessMatch]:
    current_pid = int(current_pid or os.getpid())
    process_table = process_table if process_table is not None else _read_process_table()
    matches: list[HeavyProcessMatch] = []

    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("PID "):
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        pid_text, command = parts
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid:
            continue
        for kind, pattern in _HEAVY_PROCESS_PATTERNS:
            if pattern not in command:
                continue
            if current_script_pattern and pattern == current_script_pattern:
                matches.append(HeavyProcessMatch(pid=pid, kind="same_script", command=command))
            else:
                matches.append(HeavyProcessMatch(pid=pid, kind=kind, command=command))
            break

    return matches


def assert_no_conflicting_heavy_processes(
    *,
    current_script_pattern: str,
    current_pid: int | None = None,
    process_table: str | None = None,
) -> None:
    conflicts = find_conflicting_heavy_processes(
        current_pid=current_pid,
        current_script_pattern=current_script_pattern,
        process_table=process_table,
    )
    if not conflicts:
        return

    summary = ", ".join(f"{item.kind}:pid={item.pid}" for item in conflicts[:5])
    raise ValueError(
        "resource-safe execution requires a quiet heavy-job lane; "
        f"found conflicting processes ({summary}). "
        "wait for them to finish or rerun with --allow-concurrent-heavy-jobs"
    )
