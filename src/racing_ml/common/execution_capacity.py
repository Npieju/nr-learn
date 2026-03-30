from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass


@dataclass(frozen=True)
class HeavyProcessMatch:
    pid: int
    kind: str
    command: str


@dataclass(frozen=True)
class _ProcessRow:
    pid: int
    ppid: int | None
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
        ["ps", "-eo", "pid,ppid,args"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _parse_process_rows(process_table: str) -> dict[int, _ProcessRow]:
    rows: dict[int, _ProcessRow] = {}

    for raw_line in process_table.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("PID "):
            continue
        parts = line.split(maxsplit=2)
        if len(parts) == 2:
            pid_text, command = parts
            ppid_text = None
        elif len(parts) == 3:
            pid_text, ppid_text, command = parts
        else:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        ppid: int | None
        try:
            ppid = int(ppid_text) if ppid_text is not None else None
        except ValueError:
            ppid = None
        rows[pid] = _ProcessRow(pid=pid, ppid=ppid, command=command)

    return rows


def _ancestor_pids(process_rows: dict[int, _ProcessRow], current_pid: int) -> set[int]:
    ancestors: set[int] = {current_pid}
    cursor = current_pid

    while True:
        row = process_rows.get(cursor)
        if row is None or row.ppid is None or row.ppid <= 1 or row.ppid in ancestors:
            break
        ancestors.add(row.ppid)
        cursor = row.ppid

    return ancestors


def find_conflicting_heavy_processes(
    *,
    current_pid: int | None = None,
    current_script_pattern: str | None = None,
    process_table: str | None = None,
) -> list[HeavyProcessMatch]:
    current_pid = int(current_pid or os.getpid())
    process_table = process_table if process_table is not None else _read_process_table()
    process_rows = _parse_process_rows(process_table)
    ignored_pids = _ancestor_pids(process_rows, current_pid)
    matches: list[HeavyProcessMatch] = []

    for pid, row in process_rows.items():
        if pid in ignored_pids:
            continue
        for kind, pattern in _HEAVY_PROCESS_PATTERNS:
            if pattern not in row.command:
                continue
            if current_script_pattern and pattern == current_script_pattern:
                matches.append(HeavyProcessMatch(pid=pid, kind="same_script", command=row.command))
            else:
                matches.append(HeavyProcessMatch(pid=pid, kind=kind, command=row.command))
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


def build_execution_capacity_status(
    *,
    current_script_pattern: str | None = None,
    current_pid: int | None = None,
    process_table: str | None = None,
) -> dict[str, object]:
    conflicts = find_conflicting_heavy_processes(
        current_pid=current_pid,
        current_script_pattern=current_script_pattern,
        process_table=process_table,
    )
    return {
        "status": "blocked" if conflicts else "ready",
        "current_script_pattern": current_script_pattern,
        "conflict_count": len(conflicts),
        "conflicts": [
            {
                "pid": item.pid,
                "kind": item.kind,
                "command": item.command,
            }
            for item in conflicts
        ],
    }
