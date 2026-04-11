from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.progress import ProgressBar, format_duration


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-future-wait-cycle {now}] {message}", flush=True)


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = list(streams)

    def write(self, data: str) -> int:
        alive_streams: list[object] = []
        for stream in self._streams:
            try:
                stream.write(data)
                alive_streams.append(stream)
            except (BrokenPipeError, OSError, ValueError):
                continue
        self._streams = alive_streams
        return len(data)

    def flush(self) -> None:
        alive_streams: list[object] = []
        for stream in self._streams:
            try:
                stream.flush()
                alive_streams.append(stream)
            except (BrokenPipeError, OSError, ValueError):
                continue
        self._streams = alive_streams

    def isatty(self) -> bool:
        if not self._streams:
            return False
        primary = self._streams[0]
        return bool(getattr(primary, "isatty", lambda: False)())


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _dict_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


DEFAULT_OPERATOR_BOARD_OUTPUT = "artifacts/reports/local_nankan_data_status_board.json"


def _display_path(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _parse_utc_iso(timestamp: str | None) -> datetime | None:
    if not timestamp:
        return None
    try:
        return datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    except ValueError:
        return None


def _elapsed_seconds_from_timestamps(started_at: str | None, finished_at: str | None) -> int:
    started = _parse_utc_iso(started_at)
    finished = _parse_utc_iso(finished_at)
    if started is None or finished is None:
        return 0
    return max(0, int((finished - started).total_seconds()))


def _build_current_timing(
    *,
    payload_finished_at: str,
    run_started_at: str,
    current_cycle_index: int | None,
    next_cycle_index: int | None,
    cycle_records: list[dict[str, Any]],
    wait_state: dict[str, Any] | None,
    cycle_state: dict[str, Any] | None,
) -> dict[str, Any]:
    if cycle_state is not None:
        return {
            "mode": "running_cycle",
            "started_at": cycle_state.get("started_at"),
            "updated_at": cycle_state.get("updated_at"),
            "elapsed_seconds": cycle_state.get("elapsed_seconds"),
            "current_cycle": current_cycle_index,
            "next_cycle": next_cycle_index,
        }

    if wait_state is not None:
        seconds_total = wait_state.get("seconds_total")
        seconds_remaining = wait_state.get("seconds_remaining")
        elapsed_seconds = None
        if isinstance(seconds_total, int) and isinstance(seconds_remaining, int):
            elapsed_seconds = max(0, seconds_total - seconds_remaining)
        return {
            "mode": "waiting_next_cycle",
            "started_at": wait_state.get("waiting_started_at"),
            "updated_at": wait_state.get("updated_at"),
            "elapsed_seconds": elapsed_seconds,
            "seconds_total": seconds_total,
            "seconds_remaining": seconds_remaining,
            "current_cycle": current_cycle_index,
            "next_cycle": next_cycle_index,
            "last_cycle_finished_at": cycle_records[-1].get("cycle_finished_at") if cycle_records else None,
        }

    latest_cycle = cycle_records[-1] if cycle_records else {}
    completed_started_at = latest_cycle.get("cycle_started_at") or run_started_at
    completed_finished_at = latest_cycle.get("cycle_finished_at") or payload_finished_at
    return {
        "mode": "completed",
        "started_at": completed_started_at,
        "updated_at": payload_finished_at,
        "elapsed_seconds": _elapsed_seconds_from_timestamps(completed_started_at, completed_finished_at),
        "finished_at": completed_finished_at,
        "current_cycle": current_cycle_index,
        "next_cycle": next_cycle_index,
    }


def _build_current_decision(
    *,
    current_cycle_index: int | None,
    current_readiness_summary: dict[str, Any] | None,
    final_status: str,
    final_phase: str,
    final_action: str,
    monitor_state: str,
    stopped_reason: str,
) -> dict[str, Any]:
    readiness = current_readiness_summary or {}
    effective_stop_reason = stopped_reason
    if monitor_state != "completed":
        effective_stop_reason = "running"
    return {
        "current_cycle": current_cycle_index,
        "monitor_state": monitor_state,
        "status": final_status,
        "current_phase": final_phase,
        "recommended_action": final_action,
        "benchmark_rerun_ready": readiness.get("benchmark_rerun_ready"),
        "bootstrap_status": readiness.get("bootstrap_status"),
        "result_ready_races": readiness.get("result_ready_races"),
        "pending_result_races": readiness.get("pending_result_races"),
        "stop_reason": effective_stop_reason,
    }


def _build_current_counts(current_readiness_summary: dict[str, Any] | None) -> dict[str, Any]:
    readiness = current_readiness_summary or {}
    return {
        "pre_race_only_rows": readiness.get("pre_race_only_rows"),
        "pre_race_only_races": readiness.get("pre_race_only_races"),
        "result_ready_races": readiness.get("result_ready_races"),
        "pending_result_races": readiness.get("pending_result_races"),
    }


def _build_current_flags(
    *,
    monitor_state: str,
    current_readiness_summary: dict[str, Any] | None,
    current_counts: dict[str, Any] | None,
) -> dict[str, bool]:
    readiness = current_readiness_summary or {}
    counts = current_counts or {}

    has_pre_race_only_rows = bool(counts.get("pre_race_only_rows"))
    has_pre_race_only_races = bool(counts.get("pre_race_only_races"))
    has_result_ready_races = bool(counts.get("result_ready_races"))
    has_pending_result_races = bool(counts.get("pending_result_races"))
    benchmark_rerun_ready = bool(readiness.get("benchmark_rerun_ready"))
    bootstrap_ready = (readiness.get("bootstrap_status") == "benchmark_ready")

    return {
        "has_pre_race_only_rows": has_pre_race_only_rows,
        "has_pre_race_only_races": has_pre_race_only_races,
        "has_result_ready_races": has_result_ready_races,
        "has_pending_result_races": has_pending_result_races,
        "benchmark_rerun_ready": benchmark_rerun_ready,
        "bootstrap_ready": bootstrap_ready,
        "blocked_on_result_arrival": has_pending_result_races and not has_result_ready_races,
        "cycle_in_flight": monitor_state == "running_cycle",
        "wait_in_flight": monitor_state == "waiting_next_cycle",
    }


def _build_current_blockers(
    *,
    current_decision: dict[str, Any] | None,
    current_counts: dict[str, Any] | None,
    current_flags: dict[str, Any] | None,
    current_statuses: dict[str, Any] | None,
    current_phases: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = current_decision or {}
    counts = current_counts or {}
    flags = current_flags or {}
    statuses = current_statuses or {}
    phases = current_phases or {}
    benchmark_observed = any(
        value is not None
        for value in (
            decision.get("benchmark_rerun_ready"),
            statuses.get("status_board"),
            phases.get("status_board"),
        )
    )
    bootstrap_observed = any(
        value is not None
        for value in (
            decision.get("bootstrap_status"),
            statuses.get("bootstrap_handoff"),
            phases.get("bootstrap_handoff"),
        )
    )

    details: list[dict[str, Any]] = []

    if bool(flags.get("has_pending_result_races")):
        details.append(
            {
                "code": "result_arrival_pending",
                "surface": "capture_loop",
                "status": statuses.get("capture_loop"),
                "phase": phases.get("capture_loop"),
                "pending_result_races": counts.get("pending_result_races"),
                "result_ready_races": counts.get("result_ready_races"),
            }
        )

    if benchmark_observed and not bool(flags.get("benchmark_rerun_ready")):
        details.append(
            {
                "code": "benchmark_rerun_not_ready",
                "surface": "status_board",
                "status": statuses.get("status_board"),
                "phase": phases.get("status_board"),
            }
        )

    bootstrap_status = decision.get("bootstrap_status")
    if bootstrap_observed and bootstrap_status not in {"benchmark_ready", "completed"}:
        details.append(
            {
                "code": "bootstrap_not_ready",
                "surface": "bootstrap_handoff",
                "status": statuses.get("bootstrap_handoff") or bootstrap_status,
                "phase": phases.get("bootstrap_handoff"),
            }
        )

    return {
        "primary_code": details[0]["code"] if details else None,
        "blocking_count": len(details),
        "codes": [detail["code"] for detail in details],
        "details": details,
        "observed_surfaces": {
            "status_board": benchmark_observed,
            "bootstrap_handoff": bootstrap_observed,
        },
    }


def _build_current_outcome(
    *,
    current_decision: dict[str, Any] | None,
    current_flags: dict[str, Any] | None,
    current_blockers: dict[str, Any] | None,
    current_focus: dict[str, Any] | None,
) -> dict[str, Any]:
    decision = current_decision or {}
    flags = current_flags or {}
    blockers = current_blockers or {}
    focus = current_focus or {}

    stop_reason = decision.get("stop_reason")
    primary_code = blockers.get("primary_code")
    blocking_details = blockers.get("details") if isinstance(blockers.get("details"), list) else []
    primary_detail = blocking_details[0] if blocking_details else {}

    outcome_state = "monitoring"
    summary_code = decision.get("monitor_state") or "monitoring"
    blocking_surface = None

    if bool(flags.get("bootstrap_ready")):
        outcome_state = "ready"
        summary_code = "bootstrap_ready"
    elif bool(flags.get("benchmark_rerun_ready")):
        outcome_state = "ready"
        summary_code = "benchmark_rerun_ready"
    elif primary_code:
        outcome_state = "blocked"
        summary_code = primary_code
        blocking_surface = primary_detail.get("surface")

    if stop_reason in {"bootstrap_completed", "bootstrap_failed"}:
        outcome_state = "completed"
        summary_code = str(stop_reason)
    elif stop_reason == "max_cycles_reached" and outcome_state == "monitoring":
        outcome_state = "completed"
        summary_code = "max_cycles_reached"

    return {
        "state": outcome_state,
        "summary_code": summary_code,
        "monitor_state": decision.get("monitor_state"),
        "blocking_surface": blocking_surface,
        "recommended_action": decision.get("recommended_action") or focus.get("recommended_action"),
        "current_surface": focus.get("current_surface"),
        "current_phase": focus.get("current_phase"),
        "stop_reason": stop_reason,
    }


def _build_current_statuses(
    *,
    final_status: str,
    monitor_state: str,
    current_surface_summaries: dict[str, Any] | None,
    current_readiness_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    summaries = current_surface_summaries or {}
    readiness = current_readiness_summary or {}
    capture_loop = summaries.get("capture_loop") if isinstance(summaries.get("capture_loop"), dict) else {}
    watcher = summaries.get("readiness_watcher") if isinstance(summaries.get("readiness_watcher"), dict) else {}
    bootstrap = summaries.get("bootstrap_handoff") if isinstance(summaries.get("bootstrap_handoff"), dict) else {}
    board = summaries.get("status_board") if isinstance(summaries.get("status_board"), dict) else {}
    readiness_cycle = summaries.get("readiness_cycle") if isinstance(summaries.get("readiness_cycle"), dict) else {}
    return {
        "monitor_state": monitor_state,
        "overall_status": final_status,
        "readiness_cycle": readiness_cycle.get("status") or readiness.get("wrapper_status"),
        "capture_loop": capture_loop.get("status"),
        "readiness_watcher": watcher.get("status"),
        "bootstrap_handoff": bootstrap.get("status") or readiness.get("bootstrap_status"),
        "status_board": board.get("status") or readiness.get("status_board_status"),
    }


def _build_current_phases(
    *,
    final_phase: str,
    monitor_phase: str,
    current_surface_summaries: dict[str, Any] | None,
) -> dict[str, Any]:
    summaries = current_surface_summaries or {}
    capture_loop = summaries.get("capture_loop") if isinstance(summaries.get("capture_loop"), dict) else {}
    watcher = summaries.get("readiness_watcher") if isinstance(summaries.get("readiness_watcher"), dict) else {}
    bootstrap = summaries.get("bootstrap_handoff") if isinstance(summaries.get("bootstrap_handoff"), dict) else {}
    board = summaries.get("status_board") if isinstance(summaries.get("status_board"), dict) else {}
    readiness_cycle = summaries.get("readiness_cycle") if isinstance(summaries.get("readiness_cycle"), dict) else {}
    return {
        "monitor_phase": monitor_phase,
        "overall_phase": final_phase,
        "readiness_cycle": readiness_cycle.get("current_phase"),
        "capture_loop": capture_loop.get("current_phase"),
        "readiness_watcher": watcher.get("current_phase"),
        "bootstrap_handoff": bootstrap.get("current_phase"),
        "status_board": board.get("current_phase"),
    }


def _build_current_focus(
    *,
    current_surface_summaries: dict[str, Any] | None,
    current_readiness_summary: dict[str, Any] | None,
    final_action: str,
) -> dict[str, Any]:
    summaries = current_surface_summaries or {}
    readiness = current_readiness_summary or {}
    readiness_cycle = summaries.get("readiness_cycle") if isinstance(summaries.get("readiness_cycle"), dict) else {}
    capture_loop = summaries.get("capture_loop") if isinstance(summaries.get("capture_loop"), dict) else {}
    watcher = summaries.get("readiness_watcher") if isinstance(summaries.get("readiness_watcher"), dict) else {}
    bootstrap = summaries.get("bootstrap_handoff") if isinstance(summaries.get("bootstrap_handoff"), dict) else {}
    board = summaries.get("status_board") if isinstance(summaries.get("status_board"), dict) else {}

    current_surface = "readiness_cycle"
    current_phase = readiness_cycle.get("current_phase")
    current_status = readiness_cycle.get("status") or readiness.get("wrapper_status")
    if capture_loop:
        current_surface = "capture_loop"
        current_phase = capture_loop.get("current_phase") or current_phase
        current_status = capture_loop.get("status") or current_status
    if watcher:
        current_surface = "readiness_watcher"
        current_phase = watcher.get("current_phase") or current_phase
        current_status = watcher.get("status") or current_status
    if bootstrap:
        current_surface = "bootstrap_handoff"
        current_phase = bootstrap.get("current_phase") or current_phase
        current_status = bootstrap.get("status") or current_status
    if board:
        current_surface = "status_board"
        current_phase = board.get("current_phase") or current_phase
        current_status = board.get("status") or current_status

    return {
        "current_surface": current_surface,
        "current_phase": current_phase,
        "status": current_status,
        "recommended_action": readiness.get("recommended_action") or final_action,
    }


def _build_current_progress(
    *,
    max_cycles: int,
    completed_cycles: int,
    monitor_state: str,
    current_cycle_index: int | None,
    next_cycle_index: int | None,
) -> dict[str, Any]:
    safe_max_cycles = max(1, int(max_cycles))
    remaining_cycles = max(0, safe_max_cycles - int(completed_cycles))
    completion_ratio = completed_cycles / safe_max_cycles
    return {
        "max_cycles": safe_max_cycles,
        "completed_cycles": completed_cycles,
        "remaining_cycles": remaining_cycles,
        "in_flight_cycle": current_cycle_index if monitor_state == "running_cycle" else None,
        "next_cycle": next_cycle_index,
        "completion_ratio": completion_ratio,
        "completion_percent": int(completion_ratio * 100),
    }


def _build_artifacts_from_cycle_record(cycle_record: dict[str, Any]) -> dict[str, Any]:
    return {
        "capture_loop_manifest": cycle_record.get("capture_loop_manifest"),
        "watcher_manifest": cycle_record.get("watcher_manifest"),
        "bootstrap_manifest": cycle_record.get("bootstrap_manifest"),
        "status_board_manifest": cycle_record.get("status_board_manifest"),
        "wrapper_manifest": cycle_record.get("wrapper_manifest"),
    }


def _artifact_key_for_surface(surface: str | None) -> str | None:
    mapping = {
        "capture_loop": "capture_loop_manifest",
        "readiness_watcher": "watcher_manifest",
        "bootstrap_handoff": "bootstrap_manifest",
        "status_board": "status_board_manifest",
        "readiness_cycle": "wrapper_manifest",
    }
    return mapping.get(str(surface)) if surface is not None else None


def _build_current_refs(
    *,
    current_artifacts: dict[str, Any] | None,
    current_focus: dict[str, Any] | None,
    current_blockers: dict[str, Any] | None,
) -> dict[str, Any]:
    artifacts = current_artifacts or {}
    focus = current_focus or {}
    blockers = current_blockers or {}
    blocker_details = blockers.get("details") if isinstance(blockers.get("details"), list) else []
    primary_blocker = blocker_details[0] if blocker_details else {}

    focus_surface = focus.get("current_surface")
    blocking_surface = primary_blocker.get("surface")
    focus_key = _artifact_key_for_surface(focus_surface)
    blocking_key = _artifact_key_for_surface(blocking_surface)

    return {
        "focus_surface": focus_surface,
        "focus_manifest": artifacts.get(focus_key) if focus_key else None,
        "blocking_surface": blocking_surface,
        "blocking_manifest": artifacts.get(blocking_key) if blocking_key else None,
        "status_board_manifest": artifacts.get("status_board_manifest"),
        "wrapper_manifest": artifacts.get("wrapper_manifest"),
    }


def _build_current_operator_card(
    *,
    current_outcome: dict[str, Any] | None,
    current_focus: dict[str, Any] | None,
    current_blockers: dict[str, Any] | None,
    current_refs: dict[str, Any] | None,
) -> dict[str, Any]:
    outcome = current_outcome or {}
    focus = current_focus or {}
    blockers = current_blockers or {}
    refs = current_refs or {}
    blocker_codes = blockers.get("codes") if isinstance(blockers.get("codes"), list) else []

    headline = str(outcome.get("summary_code") or outcome.get("state") or "monitoring")
    if headline == "result_arrival_pending":
        headline = "result arrival pending"
    elif headline == "benchmark_rerun_ready":
        headline = "benchmark rerun ready"
    elif headline == "bootstrap_ready":
        headline = "bootstrap ready"

    subtitle = f"focus={focus.get('current_surface')} phase={focus.get('current_phase')}"

    return {
        "headline": headline,
        "state": outcome.get("state"),
        "subtitle": subtitle,
        "recommended_action": outcome.get("recommended_action"),
        "focus_surface": focus.get("current_surface"),
        "focus_status": focus.get("status"),
        "blocking_surface": outcome.get("blocking_surface"),
        "blocking_codes": blocker_codes,
        "focus_manifest": refs.get("focus_manifest"),
        "blocking_manifest": refs.get("blocking_manifest"),
    }


def _build_current_surface_views(
    *,
    current_surface_summaries: dict[str, Any] | None,
    current_statuses: dict[str, Any] | None,
    current_phases: dict[str, Any] | None,
    current_artifacts: dict[str, Any] | None,
) -> dict[str, Any]:
    summaries = current_surface_summaries or {}
    statuses = current_statuses or {}
    phases = current_phases or {}
    artifacts = current_artifacts or {}
    surface_names = [
        "readiness_cycle",
        "capture_loop",
        "readiness_watcher",
        "bootstrap_handoff",
        "status_board",
    ]

    views: dict[str, Any] = {}
    for surface_name in surface_names:
        summary = summaries.get(surface_name) if isinstance(summaries.get(surface_name), dict) else {}
        artifact_key = _artifact_key_for_surface(surface_name)
        views[surface_name] = {
            "surface": surface_name,
            "status": statuses.get(surface_name),
            "current_phase": phases.get(surface_name),
            "manifest": artifacts.get(artifact_key) if artifact_key else None,
            "summary": summary,
        }
    return views


def _build_current_runtime(
    *,
    monitor_state: str,
    monitor_phase: str,
    current_timing: dict[str, Any] | None,
    current_progress: dict[str, Any] | None,
    wait_state: dict[str, Any] | None,
    current_cycle_index: int | None,
    next_cycle_index: int | None,
) -> dict[str, Any]:
    timing = current_timing or {}
    progress = current_progress or {}
    wait = wait_state or {}
    return {
        "monitor_state": monitor_state,
        "monitor_phase": monitor_phase,
        "mode": timing.get("mode"),
        "current_cycle": current_cycle_index,
        "next_cycle": next_cycle_index,
        "in_flight_cycle": progress.get("in_flight_cycle"),
        "completed_cycles": progress.get("completed_cycles"),
        "remaining_cycles": progress.get("remaining_cycles"),
        "completion_percent": progress.get("completion_percent"),
        "elapsed_seconds": timing.get("elapsed_seconds"),
        "seconds_remaining": timing.get("seconds_remaining") or wait.get("seconds_remaining"),
        "heartbeat_seconds": wait.get("heartbeat_seconds"),
        "updated_at": timing.get("updated_at") or wait.get("updated_at"),
    }


def _build_current_snapshot_meta(*, generated_at: str, execution_mode: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "generated_at": generated_at,
        "execution_role": "readiness_supervisor",
        "data_update_mode": "readiness_recheck_only",
        "execution_mode": execution_mode,
        "trigger_contract": "external_refresh_completed_only",
        "preferred_entrypoints": [
            "current_runtime",
            "current_operator_card",
            "current_surface_views",
            "current_outcome",
            "current_refs",
        ],
        "root_aliases_present": True,
    }


def _build_current_snapshot(
    *,
    payload_finished_at: str,
    run_started_at: str,
    current_cycle_index: int | None,
    next_cycle_index: int | None,
    cycle_records: list[dict[str, Any]],
    wait_state: dict[str, Any] | None,
    cycle_state: dict[str, Any] | None,
    current_artifacts: dict[str, Any] | None,
    current_surface_summaries: dict[str, Any] | None,
    current_readiness_summary: dict[str, Any] | None,
    final_status: str,
    final_phase: str,
    final_action: str,
    monitor_state: str,
    monitor_phase: str,
    max_cycles: int,
    completed_cycles: int,
    stopped_reason: str,
    execution_mode: str,
) -> dict[str, Any]:
    snapshot: dict[str, Any] = {}
    snapshot["current_timing"] = _build_current_timing(
        payload_finished_at=payload_finished_at,
        run_started_at=run_started_at,
        current_cycle_index=current_cycle_index,
        next_cycle_index=next_cycle_index,
        cycle_records=cycle_records,
        wait_state=wait_state,
        cycle_state=cycle_state,
    )
    if current_artifacts is not None:
        snapshot["current_artifacts"] = current_artifacts
    if current_surface_summaries is not None:
        snapshot["current_surface_summaries"] = current_surface_summaries
    if current_readiness_summary is not None:
        snapshot["current_readiness_summary"] = current_readiness_summary

    snapshot["current_decision"] = _build_current_decision(
        current_cycle_index=current_cycle_index,
        current_readiness_summary=current_readiness_summary,
        final_status=final_status,
        final_phase=final_phase,
        final_action=final_action,
        monitor_state=monitor_state,
        stopped_reason=stopped_reason,
    )
    snapshot["current_counts"] = _build_current_counts(current_readiness_summary)
    snapshot["current_flags"] = _build_current_flags(
        monitor_state=monitor_state,
        current_readiness_summary=current_readiness_summary,
        current_counts=snapshot["current_counts"],
    )
    snapshot["current_statuses"] = _build_current_statuses(
        final_status=final_status,
        monitor_state=monitor_state,
        current_surface_summaries=current_surface_summaries,
        current_readiness_summary=current_readiness_summary,
    )
    snapshot["current_phases"] = _build_current_phases(
        final_phase=final_phase,
        monitor_phase=monitor_phase,
        current_surface_summaries=current_surface_summaries,
    )
    snapshot["current_blockers"] = _build_current_blockers(
        current_decision=snapshot["current_decision"],
        current_counts=snapshot["current_counts"],
        current_flags=snapshot["current_flags"],
        current_statuses=snapshot["current_statuses"],
        current_phases=snapshot["current_phases"],
    )
    snapshot["current_focus"] = _build_current_focus(
        current_surface_summaries=current_surface_summaries,
        current_readiness_summary=current_readiness_summary,
        final_action=final_action,
    )
    snapshot["current_outcome"] = _build_current_outcome(
        current_decision=snapshot["current_decision"],
        current_flags=snapshot["current_flags"],
        current_blockers=snapshot["current_blockers"],
        current_focus=snapshot["current_focus"],
    )
    snapshot["current_refs"] = _build_current_refs(
        current_artifacts=current_artifacts,
        current_focus=snapshot["current_focus"],
        current_blockers=snapshot["current_blockers"],
    )
    snapshot["current_operator_card"] = _build_current_operator_card(
        current_outcome=snapshot["current_outcome"],
        current_focus=snapshot["current_focus"],
        current_blockers=snapshot["current_blockers"],
        current_refs=snapshot["current_refs"],
    )
    snapshot["current_surface_views"] = _build_current_surface_views(
        current_surface_summaries=current_surface_summaries,
        current_statuses=snapshot["current_statuses"],
        current_phases=snapshot["current_phases"],
        current_artifacts=current_artifacts,
    )
    snapshot["current_progress"] = _build_current_progress(
        max_cycles=max_cycles,
        completed_cycles=completed_cycles,
        monitor_state=monitor_state,
        current_cycle_index=current_cycle_index,
        next_cycle_index=next_cycle_index,
    )
    snapshot["current_runtime"] = _build_current_runtime(
        monitor_state=monitor_state,
        monitor_phase=monitor_phase,
        current_timing=snapshot["current_timing"],
        current_progress=snapshot["current_progress"],
        wait_state=wait_state,
        current_cycle_index=current_cycle_index,
        next_cycle_index=next_cycle_index,
    )
    snapshot["current_snapshot_meta"] = _build_current_snapshot_meta(
        generated_at=payload_finished_at,
        execution_mode=execution_mode,
    )
    return snapshot


def _run_command(
    *,
    label: str,
    command: list[str],
    heartbeat_seconds: int = 10,
    on_start: Callable[[str], None] | None = None,
    on_tick: Callable[[str, int], None] | None = None,
) -> dict[str, Any]:
    started_at = utc_now_iso()
    print(f"[local-nankan-future-wait-cycle] running {label}: {shlex.join(command)}", flush=True)
    started_perf = time.perf_counter()
    interval_seconds = max(1, int(heartbeat_seconds))
    if on_start is not None:
        on_start(started_at)
    log_progress(f"[local-nankan-future-wait-cycle] {label} child command started")
    process = subprocess.Popen(command, cwd=ROOT)
    while True:
        try:
            return_code = process.wait(timeout=interval_seconds)
            break
        except subprocess.TimeoutExpired:
            return_code = None
        elapsed_seconds = max(0, int(time.perf_counter() - started_perf))
        log_progress(
            f"[local-nankan-future-wait-cycle] {label} child command running... elapsed={format_duration(elapsed_seconds)}"
        )
        if on_tick is not None:
            on_tick(started_at, elapsed_seconds)
    elapsed_seconds = max(0, int(time.perf_counter() - started_perf))
    log_progress(f"[local-nankan-future-wait-cycle] {label} child command done in {format_duration(elapsed_seconds)}")
    return {
        "label": label,
        "command": command,
        "exit_code": int(return_code),
        "status": "completed" if return_code == 0 else "failed",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
    }


def _cycle_artifact_path(prefix: str, cycle_index: int, name: str) -> str:
    return f"artifacts/reports/{prefix}_cycle_{cycle_index:03d}_{name}.json"


def _effective_artifact_prefix(*, artifact_prefix: str, run_id: str | None) -> str:
    if not run_id:
        return artifact_prefix
    return f"{artifact_prefix}_{run_id}"


def _default_log_path(*, effective_artifact_prefix: str) -> Path:
    return ROOT / "artifacts" / "logs" / f"{effective_artifact_prefix}.log"


def _configure_live_log(log_path: Path) -> tuple[object, object, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = _TeeStream(sys.stdout, log_handle)
    sys.stderr = _TeeStream(sys.stderr, log_handle)
    print(f"[local-nankan-future-wait-cycle] live log file: {_display_path(log_path)}", flush=True)
    return log_handle, original_stdout, original_stderr


def _restore_live_log(log_state: tuple[object, object, object] | None) -> None:
    if log_state is None:
        return
    log_handle, original_stdout, original_stderr = log_state
    sys.stdout = original_stdout
    sys.stderr = original_stderr
    try:
        log_handle.close()
    except (OSError, ValueError):
        pass


def _run_manifest_path(*, manifest_output: str, run_id: str | None) -> str | None:
    if not run_id:
        return None
    manifest_path = Path(manifest_output)
    return str(manifest_path.with_name(f"{manifest_path.stem}_{run_id}{manifest_path.suffix}"))


def _write_wait_manifest(*, stable_manifest_path: Path, run_manifest_path: Path | None, payload: dict[str, Any]) -> None:
    write_json(stable_manifest_path, payload)
    if run_manifest_path is not None:
        write_json(run_manifest_path, payload)


def _build_operator_board_payload(
    *,
    board_payload: dict[str, Any],
    wait_payload: dict[str, Any],
    status_board_manifest: str,
    stable_manifest_path: Path,
    run_manifest_path: Path | None,
) -> dict[str, Any]:
    merged_payload = dict(board_payload)
    readiness_surfaces = dict(_dict_payload(board_payload.get("readiness_surfaces")))
    artifacts = dict(_dict_payload(board_payload.get("artifacts")))
    current_runtime = _dict_payload(wait_payload.get("current_runtime"))
    current_timing = _dict_payload(wait_payload.get("current_timing"))
    current_progress = _dict_payload(wait_payload.get("current_progress"))
    current_outcome = _dict_payload(wait_payload.get("current_outcome"))
    current_operator_card = _dict_payload(wait_payload.get("current_operator_card"))
    current_refs = _dict_payload(wait_payload.get("current_refs"))

    supervisor_manifest = _display_path(run_manifest_path or stable_manifest_path)
    supervisor_surface = {
        "status": wait_payload.get("status"),
        "monitor_state": wait_payload.get("monitor_state"),
        "monitor_phase": wait_payload.get("monitor_phase"),
        "stopped_reason": wait_payload.get("stopped_reason"),
        "completed_cycles": wait_payload.get("completed_cycles"),
        "max_cycles": wait_payload.get("max_cycles"),
        "current_cycle_index": wait_payload.get("current_cycle_index"),
        "next_cycle_index": wait_payload.get("next_cycle_index"),
        "wait_state": wait_payload.get("wait_state"),
        "cycle_state": wait_payload.get("cycle_state"),
        "current_runtime": current_runtime,
        "current_timing": current_timing,
        "current_progress": current_progress,
        "current_outcome": current_outcome,
        "current_operator_card": current_operator_card,
        "current_refs": current_refs,
        "manifest": supervisor_manifest,
        "status_board_manifest": status_board_manifest,
        "started_at": wait_payload.get("started_at"),
        "updated_at": wait_payload.get("updated_at") or wait_payload.get("finished_at"),
        "finished_at": wait_payload.get("finished_at"),
    }
    readiness_surfaces["readiness_supervisor"] = supervisor_surface

    artifacts["readiness_supervisor_manifest"] = supervisor_manifest
    artifacts["live_status_board_source"] = status_board_manifest

    highlights = [item for item in board_payload.get("highlights", []) if isinstance(item, str)]
    highlights.extend(
        [
            f"supervisor_monitor_state={supervisor_surface['monitor_state']}",
            f"supervisor_completed_cycles={supervisor_surface['completed_cycles']}",
            f"supervisor_seconds_remaining={current_runtime.get('seconds_remaining')}",
        ]
    )

    merged_payload["readiness_surfaces"] = readiness_surfaces
    merged_payload["artifacts"] = artifacts
    merged_payload["operator_runtime"] = {
        "surface": "readiness_supervisor",
        "monitor_state": supervisor_surface.get("monitor_state"),
        "monitor_phase": supervisor_surface.get("monitor_phase"),
        "current_runtime": current_runtime,
        "current_timing": current_timing,
        "current_progress": current_progress,
        "current_outcome": current_outcome,
        "current_operator_card": current_operator_card,
    }
    merged_payload["highlights"] = highlights
    return merged_payload


def _write_operator_board(
    *,
    operator_board_path: Path | None,
    latest_status_board_manifest: str | None,
    stable_manifest_path: Path,
    run_manifest_path: Path | None,
    wait_payload: dict[str, Any],
) -> None:
    if operator_board_path is None or latest_status_board_manifest is None:
        return
    board_payload = _read_json_dict(_resolve_path(latest_status_board_manifest))
    if not board_payload:
        return
    operator_board_payload = _build_operator_board_payload(
        board_payload=board_payload,
        wait_payload=wait_payload,
        status_board_manifest=latest_status_board_manifest,
        stable_manifest_path=stable_manifest_path,
        run_manifest_path=run_manifest_path,
    )
    write_json(operator_board_path, operator_board_payload)


def _build_wait_manifest_payload(
    *,
    run_started_at: str,
    final_status: str,
    final_phase: str,
    final_action: str,
    max_cycles: int,
    run_id: str | None,
    artifact_prefix: str,
    effective_artifact_prefix: str,
    run_manifest_output: str | None,
    log_path: Path,
    cycle_records: list[dict[str, Any]],
    wait_seconds: int,
    stopped_reason: str,
    oneshot: bool,
    wait_state: dict[str, Any] | None = None,
    cycle_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload_finished_at = utc_now_iso()
    monitor_state = "completed"
    monitor_phase = "completed"
    if cycle_state is not None:
        monitor_state = "running_cycle"
        monitor_phase = str(cycle_state.get("stage") or "running_cycle")
    elif wait_state is not None:
        monitor_state = "waiting_next_cycle"
        monitor_phase = "waiting_next_cycle"

    current_cycle_index = None
    next_cycle_index = None
    current_artifacts: dict[str, Any] | None = None
    current_surface_summaries: dict[str, Any] | None = None
    if cycle_state is not None:
        current_cycle_index = cycle_state.get("current_cycle")
        next_cycle_index = (int(current_cycle_index) + 1) if current_cycle_index is not None else None
        current_artifacts = cycle_state.get("artifacts") if isinstance(cycle_state.get("artifacts"), dict) else None
        current_surface_summaries = cycle_state.get("surface_summaries") if isinstance(cycle_state.get("surface_summaries"), dict) else None
    elif wait_state is not None:
        current_cycle_index = cycle_records[-1].get("cycle") if cycle_records else None
        next_cycle_index = wait_state.get("next_cycle") if isinstance(wait_state, dict) else None
        if cycle_records:
            current_artifacts = _build_artifacts_from_cycle_record(cycle_records[-1])
            current_surface_summaries = _build_surface_summaries_from_cycle_record(cycle_records[-1])
    elif cycle_records:
        current_cycle_index = cycle_records[-1].get("cycle")
        current_artifacts = _build_artifacts_from_cycle_record(cycle_records[-1])
        current_surface_summaries = _build_surface_summaries_from_cycle_record(cycle_records[-1])

    execution_mode = "oneshot" if oneshot else "bounded_wait_cycle"

    top_level_finished_at = payload_finished_at if monitor_state == "completed" else None

    payload = {
        "started_at": run_started_at,
        "updated_at": payload_finished_at,
        "finished_at": top_level_finished_at,
        "status": final_status,
        "current_phase": final_phase,
        "recommended_action": final_action,
        "monitor_state": monitor_state,
        "monitor_phase": monitor_phase,
        "current_cycle_index": current_cycle_index,
        "next_cycle_index": next_cycle_index,
        "max_cycles": max_cycles,
        "run_id": run_id,
        "artifact_prefix": artifact_prefix,
        "effective_artifact_prefix": effective_artifact_prefix,
        "run_manifest_output": run_manifest_output,
        "log_file": _display_path(log_path),
        "completed_cycles": len(cycle_records),
        "wait_seconds": wait_seconds,
        "stopped_reason": stopped_reason,
        "execution_role": "readiness_supervisor",
        "data_update_mode": "readiness_recheck_only",
        "execution_mode": execution_mode,
        "trigger_contract": "external_refresh_completed_only",
        "cycles": cycle_records,
    }
    if cycle_records:
        payload["latest_cycle"] = cycle_records[-1]
        payload["readiness_summary"] = _build_readiness_summary_from_cycle_record(
            cycle_records[-1],
            recommended_action=final_action,
        )
    current_readiness_summary = payload.get("readiness_summary") if isinstance(payload.get("readiness_summary"), dict) else None
    if wait_state is not None:
        payload["wait_state"] = wait_state
    if cycle_state is not None:
        payload["cycle_state"] = cycle_state
        payload["active_readiness_summary"] = _build_active_readiness_summary(cycle_state)
        current_readiness_summary = payload["active_readiness_summary"]
    payload.update(
        _build_current_snapshot(
            payload_finished_at=payload_finished_at,
            run_started_at=run_started_at,
            current_cycle_index=current_cycle_index,
            next_cycle_index=next_cycle_index,
            cycle_records=cycle_records,
            wait_state=wait_state,
            cycle_state=cycle_state,
            current_artifacts=current_artifacts,
            current_surface_summaries=current_surface_summaries,
            current_readiness_summary=current_readiness_summary,
            final_status=final_status,
            final_phase=final_phase,
            final_action=final_action,
            monitor_state=monitor_state,
            monitor_phase=monitor_phase,
            max_cycles=max_cycles,
            completed_cycles=len(cycle_records),
            stopped_reason=stopped_reason,
            execution_mode=execution_mode,
        )
    )
    payload["current_snapshot"] = {
        key: value for key, value in payload.items() if key.startswith("current_")
    }
    return payload


def _build_cycle_state(
    *,
    cycle_index: int,
    stage: str,
    label: str,
    command: list[str],
    started_at: str,
    elapsed_seconds: int,
    heartbeat_seconds: int,
    artifacts: dict[str, str],
    surface_summaries: dict[str, Any],
) -> dict[str, Any]:
    return {
        "active": True,
        "current_cycle": cycle_index,
        "stage": stage,
        "label": label,
        "command": command,
        "heartbeat_seconds": heartbeat_seconds,
        "started_at": started_at,
        "elapsed_seconds": elapsed_seconds,
        "updated_at": utc_now_iso(),
        "artifacts": artifacts,
        "surface_summaries": surface_summaries,
    }


def _build_cycle_surface_summaries(
    artifacts: dict[str, str],
    *,
    stage: str,
    label: str,
    elapsed_seconds: int,
) -> dict[str, Any]:
    summaries: dict[str, Any] = {
        "readiness_cycle": {
            "status": "running",
            "current_phase": stage,
            "label": label,
            "elapsed_seconds": elapsed_seconds,
        }
    }

    capture_payload = _read_json_dict(_resolve_path(artifacts["capture_loop_manifest"]))
    if capture_payload:
        latest_summary = capture_payload.get("latest_summary") if isinstance(capture_payload.get("latest_summary"), dict) else {}
        summaries["capture_loop"] = {
            "status": capture_payload.get("status"),
            "current_phase": capture_payload.get("current_phase"),
            "completed_passes": capture_payload.get("completed_passes"),
            "pre_race_only_rows": latest_summary.get("pre_race_only_rows"),
            "pre_race_only_races": latest_summary.get("pre_race_only_races"),
            "result_ready_races": latest_summary.get("result_ready_races"),
            "pending_result_races": latest_summary.get("pending_result_races"),
        }

    watcher_payload = _read_json_dict(_resolve_path(artifacts["watcher_manifest"]))
    if watcher_payload:
        summaries["readiness_watcher"] = {
            "status": watcher_payload.get("status"),
            "current_phase": watcher_payload.get("current_phase"),
            "attempts": watcher_payload.get("attempts"),
            "waited_seconds": watcher_payload.get("waited_seconds"),
        }

    bootstrap_payload = _read_json_dict(_resolve_path(artifacts["bootstrap_manifest"]))
    if bootstrap_payload:
        handoff_result = bootstrap_payload.get("handoff_command_result") if isinstance(bootstrap_payload.get("handoff_command_result"), dict) else {}
        summaries["bootstrap_handoff"] = {
            "status": bootstrap_payload.get("status"),
            "current_phase": bootstrap_payload.get("current_phase"),
            "handoff_exit_code": handoff_result.get("exit_code"),
        }

    board_payload = _read_json_dict(_resolve_path(artifacts["status_board_manifest"]))
    if board_payload:
        readiness = board_payload.get("readiness") if isinstance(board_payload.get("readiness"), dict) else {}
        summaries["status_board"] = {
            "status": board_payload.get("status"),
            "current_phase": board_payload.get("current_phase"),
            "benchmark_rerun_ready": readiness.get("benchmark_rerun_ready"),
        }

    wrapper_payload = _read_json_dict(_resolve_path(artifacts["wrapper_manifest"]))
    if wrapper_payload:
        summaries["readiness_cycle"] = {
            "status": wrapper_payload.get("status") or "running",
            "current_phase": wrapper_payload.get("current_phase") or stage,
            "recommended_action": wrapper_payload.get("recommended_action"),
            "label": label,
            "elapsed_seconds": elapsed_seconds,
        }

    return summaries


def _build_active_readiness_summary(cycle_state: dict[str, Any]) -> dict[str, Any]:
    surface_summaries = cycle_state.get("surface_summaries") if isinstance(cycle_state.get("surface_summaries"), dict) else {}
    capture_loop = surface_summaries.get("capture_loop") if isinstance(surface_summaries.get("capture_loop"), dict) else {}
    watcher = surface_summaries.get("readiness_watcher") if isinstance(surface_summaries.get("readiness_watcher"), dict) else {}
    bootstrap = surface_summaries.get("bootstrap_handoff") if isinstance(surface_summaries.get("bootstrap_handoff"), dict) else {}
    status_board = surface_summaries.get("status_board") if isinstance(surface_summaries.get("status_board"), dict) else {}
    readiness_cycle = surface_summaries.get("readiness_cycle") if isinstance(surface_summaries.get("readiness_cycle"), dict) else {}

    current_surface = "readiness_cycle"
    current_phase = readiness_cycle.get("current_phase")
    if capture_loop:
        current_surface = "capture_loop"
        current_phase = capture_loop.get("current_phase") or current_phase
    if watcher:
        current_surface = "readiness_watcher"
        current_phase = watcher.get("current_phase") or current_phase
    if bootstrap:
        current_surface = "bootstrap_handoff"
        current_phase = bootstrap.get("current_phase") or current_phase
    if status_board:
        current_surface = "status_board"
        current_phase = status_board.get("current_phase") or current_phase

    return {
        "cycle": cycle_state.get("current_cycle"),
        "stage": cycle_state.get("stage"),
        "current_surface": current_surface,
        "current_phase": current_phase,
        "status": readiness_cycle.get("status") or "running",
        "elapsed_seconds": cycle_state.get("elapsed_seconds"),
        "pre_race_only_rows": capture_loop.get("pre_race_only_rows"),
        "pre_race_only_races": capture_loop.get("pre_race_only_races"),
        "result_ready_races": capture_loop.get("result_ready_races"),
        "pending_result_races": capture_loop.get("pending_result_races"),
        "benchmark_rerun_ready": status_board.get("benchmark_rerun_ready"),
        "bootstrap_status": bootstrap.get("status"),
        "recommended_action": readiness_cycle.get("recommended_action"),
        "artifacts": cycle_state.get("artifacts"),
    }


def _build_surface_summaries_from_cycle_record(cycle_record: dict[str, Any]) -> dict[str, Any]:
    return _build_cycle_surface_summaries(
        _build_artifacts_from_cycle_record(cycle_record),
        stage=str(cycle_record.get("wrapper_current_phase") or "future_only_readiness_track"),
        label=f"cycle={cycle_record.get('cycle')}",
        elapsed_seconds=_elapsed_seconds_from_timestamps(
            cycle_record.get("cycle_started_at"),
            cycle_record.get("cycle_finished_at"),
        ),
    )


def _build_readiness_summary_from_cycle_record(cycle_record: dict[str, Any], *, recommended_action: str) -> dict[str, Any]:
    return {
        "cycle": cycle_record.get("cycle"),
        "wrapper_status": cycle_record.get("wrapper_status"),
        "status_board_status": cycle_record.get("status_board_status"),
        "benchmark_rerun_ready": cycle_record.get("benchmark_rerun_ready"),
        "pre_race_only_rows": cycle_record.get("pre_race_only_rows"),
        "pre_race_only_races": cycle_record.get("pre_race_only_races"),
        "result_ready_races": cycle_record.get("result_ready_races"),
        "pending_result_races": cycle_record.get("pending_result_races"),
        "bootstrap_status": cycle_record.get("bootstrap_status"),
        "cycle_finished_at": cycle_record.get("cycle_finished_at"),
        "recommended_action": recommended_action,
        "artifacts": _build_artifacts_from_cycle_record(cycle_record),
    }


def _should_run_bootstrap_followup(bootstrap_payload: dict[str, Any]) -> bool:
    return str(bootstrap_payload.get("status") or "") == "benchmark_ready"


def _build_bootstrap_followup_command(
    *,
    python_executable: str,
    bootstrap_handoff_script: str,
    artifact_prefix: str,
    cycle_index: int,
) -> tuple[list[str], dict[str, str]]:
    artifacts = {
        "wrapper_manifest": _cycle_artifact_path(artifact_prefix, cycle_index, "bootstrap_resume"),
        "handoff_manifest": _cycle_artifact_path(artifact_prefix, cycle_index, "pre_race_benchmark_handoff"),
        "pre_race_summary": _cycle_artifact_path(artifact_prefix, cycle_index, "pre_race_ready_summary"),
        "primary_manifest": _cycle_artifact_path(artifact_prefix, cycle_index, "pre_race_ready_primary_materialize"),
        "benchmark_manifest": _cycle_artifact_path(artifact_prefix, cycle_index, "pre_race_ready_benchmark_gate"),
        "log_prefix": f"{artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_resume",
        "bootstrap_revision": f"{artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_resume",
        "filtered_race_card_output": f"data/local_nankan_pre_race_ready/raw/{artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_resume_race_card.csv",
        "filtered_race_result_output": f"data/local_nankan_pre_race_ready/raw/{artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_resume_race_result.csv",
        "primary_output_file": f"data/local_nankan_pre_race_ready/raw/{artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_resume_primary.csv",
    }
    command = [
        python_executable,
        bootstrap_handoff_script,
        "--wrapper-manifest-output",
        artifacts["wrapper_manifest"],
        "--handoff-manifest-output",
        artifacts["handoff_manifest"],
        "--filtered-race-card-output",
        artifacts["filtered_race_card_output"],
        "--filtered-race-result-output",
        artifacts["filtered_race_result_output"],
        "--primary-output-file",
        artifacts["primary_output_file"],
        "--pre-race-summary-output",
        artifacts["pre_race_summary"],
        "--primary-manifest-file",
        artifacts["primary_manifest"],
        "--benchmark-manifest-output",
        artifacts["benchmark_manifest"],
        "--log-prefix",
        artifacts["log_prefix"],
        "--bootstrap-revision",
        artifacts["bootstrap_revision"],
        "--run-bootstrap",
    ]
    return command, artifacts


def _should_stop(wrapper_payload: dict[str, Any], board_payload: dict[str, Any]) -> tuple[bool, str]:
    readiness = board_payload.get("readiness") if isinstance(board_payload.get("readiness"), dict) else {}
    if bool(readiness.get("benchmark_rerun_ready")):
        return True, "benchmark_rerun_ready"

    capture_step = wrapper_payload.get("steps") if isinstance(wrapper_payload.get("steps"), dict) else {}
    capture_manifest = capture_step.get("capture_loop") if isinstance(capture_step.get("capture_loop"), dict) else {}
    capture_payload = capture_manifest.get("manifest") if isinstance(capture_manifest.get("manifest"), dict) else {}
    latest_summary = capture_payload.get("latest_summary") if isinstance(capture_payload.get("latest_summary"), dict) else {}
    result_ready_races = latest_summary.get("result_ready_races")
    if result_ready_races is not None and int(result_ready_races) > 0:
        return True, "result_ready_support_arrived"

    return False, "max_cycles_or_manual_stop"


def _wait_with_heartbeat(
    wait_seconds: int,
    *,
    interval_seconds: int,
    on_tick: Callable[[int], None] | None = None,
) -> None:
    sleep_seconds = max(0, int(wait_seconds))
    heartbeat_seconds = max(1, int(interval_seconds))
    if sleep_seconds == 0:
        if on_tick is not None:
            on_tick(0)
        log_progress("waiting before next cycle skipped total=0s")
        return
    print(f"[local-nankan-future-wait-cycle] waiting before next cycle seconds={sleep_seconds}", flush=True)
    remaining_seconds = sleep_seconds
    if on_tick is not None:
        on_tick(remaining_seconds)
    while remaining_seconds > 0:
        sleep_chunk = min(heartbeat_seconds, remaining_seconds)
        time.sleep(sleep_chunk)
        remaining_seconds = max(0, remaining_seconds - sleep_chunk)
        if remaining_seconds > 0:
            log_progress(
                f"waiting before next cycle running... remaining={remaining_seconds}s elapsed={sleep_seconds - remaining_seconds}s"
            )
        else:
            log_progress(f"waiting before next cycle done total={sleep_seconds}s")
        if on_tick is not None:
            on_tick(remaining_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat local Nankan future-only readiness cycles to observe result-arrival "
            "and benchmark/bootstrap handoff readiness. This supervisor does not ingest "
            "external data by itself."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--wrapper-script", default="scripts/run_local_nankan_future_only_readiness_cycle.py")
    parser.add_argument("--max-cycles", type=int, default=3)
    parser.add_argument("--wait-seconds", type=int, default=3600)
    parser.add_argument("--wait-heartbeat-seconds", type=int, default=60)
    parser.add_argument(
        "--oneshot",
        action="store_true",
        help=(
            "Run exactly one readiness cycle and exit without idle wait. Use this only "
            "when another ingest/event path already refreshes the underlying data and "
            "you only need a bounded re-check."
        ),
    )
    parser.add_argument("--artifact-prefix", default="local_nankan_future_only_wait_then_cycle_issue122")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--manifest-output", default="artifacts/reports/local_nankan_future_only_wait_then_cycle_issue122.json")
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--default-horizon-days", type=int, default=7)
    parser.add_argument("--max-passes", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=int, default=1)
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--bootstrap-handoff-script", default="scripts/run_local_nankan_result_ready_bootstrap_handoff.py")
    parser.add_argument("--run-bootstrap-on-ready", action="store_true")
    parser.add_argument("--operator-board-output", default=DEFAULT_OPERATOR_BOARD_OUTPUT)
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    run_manifest_output = _run_manifest_path(manifest_output=args.manifest_output, run_id=args.run_id)
    run_manifest_path = _resolve_path(run_manifest_output) if run_manifest_output else None
    if run_manifest_path is not None:
        run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    operator_board_path = _resolve_path(args.operator_board_output) if args.operator_board_output else None
    if operator_board_path is not None:
        operator_board_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=max(1, int(args.max_cycles)), prefix="[local-nankan-future-wait-cycle]", logger=log_progress, min_interval_sec=0.0)

    run_started_at = utc_now_iso()
    log_state: tuple[object, object, object] | None = None

    try:
        python_executable = str(_resolve_path(args.python_executable)) if Path(args.python_executable).is_absolute() else args.python_executable
        wrapper_script = str(_resolve_path(args.wrapper_script))
        bootstrap_handoff_script = str(_resolve_path(args.bootstrap_handoff_script))
        effective_artifact_prefix = _effective_artifact_prefix(artifact_prefix=args.artifact_prefix, run_id=args.run_id)
        log_path = _resolve_path(args.log_file) if args.log_file else _default_log_path(effective_artifact_prefix=effective_artifact_prefix)
        log_state = _configure_live_log(log_path)
        max_cycles = max(1, int(args.max_cycles))
        wait_seconds = max(0, int(args.wait_seconds))
        wait_heartbeat_seconds = max(1, int(args.wait_heartbeat_seconds))
        if args.oneshot:
            max_cycles = 1
            wait_seconds = 0

        progress.start(message=f"starting bounded wait-then-cycle max_cycles={max_cycles}")
        cycle_records: list[dict[str, Any]] = []
        stopped_reason = "max_cycles_reached"
        final_status = "partial"
        final_phase = "future_only_readiness_track"
        final_action = "capture_future_pre_race_rows_and_wait_for_results"
        cycle_heartbeat_seconds = 10

        for cycle_index in range(1, max_cycles + 1):
            capture_manifest = _cycle_artifact_path(effective_artifact_prefix, cycle_index, "pre_race_capture_loop")
            watcher_manifest = _cycle_artifact_path(effective_artifact_prefix, cycle_index, "readiness_watcher")
            bootstrap_manifest = _cycle_artifact_path(effective_artifact_prefix, cycle_index, "bootstrap_handoff")
            status_board_manifest = _cycle_artifact_path(effective_artifact_prefix, cycle_index, "status_board")
            wrapper_manifest = _cycle_artifact_path(effective_artifact_prefix, cycle_index, "readiness_cycle")
            cycle_artifacts = {
                "capture_loop_manifest": capture_manifest,
                "watcher_manifest": watcher_manifest,
                "bootstrap_manifest": bootstrap_manifest,
                "status_board_manifest": status_board_manifest,
                "wrapper_manifest": wrapper_manifest,
            }

            command = [
                python_executable,
                wrapper_script,
                "--default-horizon-days",
                str(args.default_horizon_days),
                "--max-passes",
                str(args.max_passes),
                "--poll-interval-seconds",
                str(args.poll_interval_seconds),
                "--capture-loop-manifest-output",
                capture_manifest,
                "--watcher-manifest-output",
                watcher_manifest,
                "--bootstrap-manifest-output",
                bootstrap_manifest,
                "--bootstrap-log-prefix",
                f"{effective_artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_handoff",
                "--bootstrap-revision",
                f"{effective_artifact_prefix}_cycle_{cycle_index:03d}_bootstrap_handoff",
                "--status-board-output",
                status_board_manifest,
                "--wrapper-manifest-output",
                wrapper_manifest,
            ]
            if args.include_completed:
                command.append("--include-completed")

            def _write_cycle_tick(started_at: str, elapsed_seconds: int, *, stage: str, label: str, command: list[str]) -> None:
                cycle_payload = _build_wait_manifest_payload(
                    run_started_at=run_started_at,
                    final_status=final_status,
                    final_phase=final_phase,
                    final_action=final_action,
                    max_cycles=max_cycles,
                    run_id=args.run_id,
                    artifact_prefix=args.artifact_prefix,
                    effective_artifact_prefix=effective_artifact_prefix,
                    run_manifest_output=run_manifest_output,
                    log_path=log_path,
                    cycle_records=cycle_records,
                    wait_seconds=wait_seconds,
                    stopped_reason="running",
                    oneshot=args.oneshot,
                    cycle_state=_build_cycle_state(
                        cycle_index=cycle_index,
                        stage=stage,
                        label=label,
                        command=command,
                        started_at=started_at,
                        elapsed_seconds=elapsed_seconds,
                        heartbeat_seconds=cycle_heartbeat_seconds,
                        artifacts=cycle_artifacts,
                        surface_summaries=_build_cycle_surface_summaries(
                            cycle_artifacts,
                            stage=stage,
                            label=label,
                            elapsed_seconds=elapsed_seconds,
                        ),
                    ),
                )
                _write_wait_manifest(stable_manifest_path=manifest_path, run_manifest_path=run_manifest_path, payload=cycle_payload)

            command_result = _run_command(
                label=f"cycle={cycle_index}",
                command=command,
                heartbeat_seconds=cycle_heartbeat_seconds,
                on_start=lambda started_at, label=f"cycle={cycle_index}", command=command: _write_cycle_tick(
                    started_at,
                    0,
                    stage="readiness_cycle",
                    label=label,
                    command=command,
                ),
                on_tick=lambda started_at, elapsed_seconds, label=f"cycle={cycle_index}", command=command: _write_cycle_tick(
                    started_at,
                    elapsed_seconds,
                    stage="readiness_cycle",
                    label=label,
                    command=command,
                ),
            )
            wrapper_payload = _read_json_dict(_resolve_path(wrapper_manifest))
            board_payload = _read_json_dict(_resolve_path(status_board_manifest))
            capture_payload = _read_json_dict(_resolve_path(capture_manifest))
            bootstrap_payload = _read_json_dict(_resolve_path(bootstrap_manifest))
            latest_summary = capture_payload.get("latest_summary") if isinstance(capture_payload.get("latest_summary"), dict) else {}
            readiness = board_payload.get("readiness") if isinstance(board_payload.get("readiness"), dict) else {}

            final_status = str(board_payload.get("status") or bootstrap_payload.get("status") or wrapper_payload.get("status") or final_status)
            final_phase = str(board_payload.get("current_phase") or bootstrap_payload.get("current_phase") or wrapper_payload.get("current_phase") or final_phase)
            final_action = str(board_payload.get("recommended_action") or bootstrap_payload.get("recommended_action") or wrapper_payload.get("recommended_action") or final_action)

            stop_now, stop_reason = _should_stop(wrapper_payload, board_payload)
            bootstrap_followup: dict[str, Any] | None = None
            if stop_now and args.run_bootstrap_on_ready and _should_run_bootstrap_followup(bootstrap_payload):
                followup_command, followup_artifacts = _build_bootstrap_followup_command(
                    python_executable=python_executable,
                    bootstrap_handoff_script=bootstrap_handoff_script,
                    artifact_prefix=effective_artifact_prefix,
                    cycle_index=cycle_index,
                )
                followup_result = _run_command(
                    label=f"bootstrap_resume_cycle={cycle_index}",
                    command=followup_command,
                    heartbeat_seconds=cycle_heartbeat_seconds,
                    on_start=lambda started_at, label=f"bootstrap_resume_cycle={cycle_index}", command=followup_command: _write_cycle_tick(
                        started_at,
                        0,
                        stage="bootstrap_followup",
                        label=label,
                        command=command,
                    ),
                    on_tick=lambda started_at, elapsed_seconds, label=f"bootstrap_resume_cycle={cycle_index}", command=followup_command: _write_cycle_tick(
                        started_at,
                        elapsed_seconds,
                        stage="bootstrap_followup",
                        label=label,
                        command=command,
                    ),
                )
                followup_wrapper = _read_json_dict(_resolve_path(followup_artifacts["wrapper_manifest"]))
                bootstrap_followup = {
                    "command_result": followup_result,
                    "artifacts": followup_artifacts,
                    "status": followup_wrapper.get("status"),
                    "current_phase": followup_wrapper.get("current_phase"),
                    "recommended_action": followup_wrapper.get("recommended_action"),
                }
                final_status = str(followup_wrapper.get("status") or final_status)
                final_phase = str(followup_wrapper.get("current_phase") or final_phase)
                final_action = str(followup_wrapper.get("recommended_action") or final_action)
                if str(followup_wrapper.get("status") or "") == "completed":
                    stop_reason = "bootstrap_completed"
                elif str(followup_wrapper.get("status") or "") == "failed":
                    stop_reason = "bootstrap_failed"

            cycle_records.append(
                {
                    "cycle": cycle_index,
                    "command_result": command_result,
                    "wrapper_manifest": wrapper_manifest,
                    "status_board_manifest": status_board_manifest,
                    "capture_loop_manifest": capture_manifest,
                    "watcher_manifest": watcher_manifest,
                    "bootstrap_manifest": bootstrap_manifest,
                    "wrapper_status": wrapper_payload.get("status"),
                    "wrapper_current_phase": wrapper_payload.get("current_phase"),
                    "status_board_status": board_payload.get("status"),
                    "benchmark_rerun_ready": readiness.get("benchmark_rerun_ready"),
                    "pre_race_only_rows": latest_summary.get("pre_race_only_rows"),
                    "pre_race_only_races": latest_summary.get("pre_race_only_races"),
                    "result_ready_races": latest_summary.get("result_ready_races"),
                    "pending_result_races": latest_summary.get("pending_result_races"),
                    "bootstrap_status": bootstrap_payload.get("status"),
                    "cycle_started_at": command_result.get("started_at"),
                    "cycle_finished_at": command_result.get("finished_at"),
                    "stop_reason": stop_reason,
                    "bootstrap_followup": bootstrap_followup,
                }
            )
            progress.update(current=cycle_index, message=f"cycle={cycle_index} pending={latest_summary.get('pending_result_races')} result_ready={latest_summary.get('result_ready_races')}")

            payload = _build_wait_manifest_payload(
                run_started_at=run_started_at,
                final_status=final_status,
                final_phase=final_phase,
                final_action=final_action,
                max_cycles=max_cycles,
                run_id=args.run_id,
                artifact_prefix=args.artifact_prefix,
                effective_artifact_prefix=effective_artifact_prefix,
                run_manifest_output=run_manifest_output,
                log_path=log_path,
                cycle_records=cycle_records,
                wait_seconds=wait_seconds,
                stopped_reason=stop_reason if stop_now else "running",
                oneshot=args.oneshot,
            )
            _write_wait_manifest(stable_manifest_path=manifest_path, run_manifest_path=run_manifest_path, payload=payload)
            _write_operator_board(
                operator_board_path=operator_board_path,
                latest_status_board_manifest=status_board_manifest,
                stable_manifest_path=manifest_path,
                run_manifest_path=run_manifest_path,
                wait_payload=payload,
            )

            if stop_now:
                stopped_reason = stop_reason
                break

            if cycle_index < max_cycles:
                waiting_started_at = utc_now_iso()

                def _write_wait_tick(remaining_seconds: int) -> None:
                    wait_payload = _build_wait_manifest_payload(
                        run_started_at=run_started_at,
                        final_status=final_status,
                        final_phase=final_phase,
                        final_action=final_action,
                        max_cycles=max_cycles,
                        run_id=args.run_id,
                        artifact_prefix=args.artifact_prefix,
                        effective_artifact_prefix=effective_artifact_prefix,
                        run_manifest_output=run_manifest_output,
                        log_path=log_path,
                        cycle_records=cycle_records,
                        wait_seconds=wait_seconds,
                        stopped_reason="running",
                        oneshot=args.oneshot,
                        wait_state={
                            "active": remaining_seconds > 0,
                            "seconds_total": wait_seconds,
                            "seconds_remaining": remaining_seconds,
                            "heartbeat_seconds": wait_heartbeat_seconds,
                            "next_cycle": cycle_index + 1,
                            "waiting_started_at": waiting_started_at,
                            "updated_at": utc_now_iso(),
                        },
                    )
                    _write_wait_manifest(stable_manifest_path=manifest_path, run_manifest_path=run_manifest_path, payload=wait_payload)
                    _write_operator_board(
                        operator_board_path=operator_board_path,
                        latest_status_board_manifest=status_board_manifest,
                        stable_manifest_path=manifest_path,
                        run_manifest_path=run_manifest_path,
                        wait_payload=wait_payload,
                    )

                _wait_with_heartbeat(
                    wait_seconds,
                    interval_seconds=wait_heartbeat_seconds,
                    on_tick=_write_wait_tick,
                )

        payload = _build_wait_manifest_payload(
            run_started_at=run_started_at,
            final_status=final_status,
            final_phase=final_phase,
            final_action=final_action,
            max_cycles=max_cycles,
            run_id=args.run_id,
            artifact_prefix=args.artifact_prefix,
            effective_artifact_prefix=effective_artifact_prefix,
            run_manifest_output=run_manifest_output,
            log_path=log_path,
            cycle_records=cycle_records,
            wait_seconds=wait_seconds,
            stopped_reason=stopped_reason,
            oneshot=args.oneshot,
        )
        _write_wait_manifest(stable_manifest_path=manifest_path, run_manifest_path=run_manifest_path, payload=payload)
        latest_status_board_manifest = cycle_records[-1].get("status_board_manifest") if cycle_records else None
        _write_operator_board(
            operator_board_path=operator_board_path,
            latest_status_board_manifest=str(latest_status_board_manifest) if latest_status_board_manifest else None,
            stable_manifest_path=manifest_path,
            run_manifest_path=run_manifest_path,
            wait_payload=payload,
        )
        progress.complete(message=f"wait-then-cycle manifest output={manifest_path}")
        return 0
    except KeyboardInterrupt:
        print("[local-nankan-future-wait-cycle] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-future-wait-cycle] failed: {error}")
        traceback.print_exc()
        return 1
    finally:
        _restore_live_log(log_state)


if __name__ == "__main__":
    raise SystemExit(main())