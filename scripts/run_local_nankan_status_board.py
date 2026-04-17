from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_COVERAGE_SNAPSHOT = "artifacts/reports/coverage_snapshot_local_nankan_current.json"
DEFAULT_BACKFILL_AGGREGATE = "artifacts/reports/local_nankan_backfill_race_card_20y.json"
DEFAULT_ARCHIVE_RUN = "artifacts/reports/local_nankan_archive_run.json"
DEFAULT_READINESS_PROBE_SUMMARY = "artifacts/reports/local_nankan_pre_race_readiness_probe_summary.json"
DEFAULT_PRE_RACE_HANDOFF_MANIFEST = "artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json"
DEFAULT_BOOTSTRAP_HANDOFF_MANIFEST = "artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json"
DEFAULT_READINESS_WATCHER_MANIFEST = "artifacts/reports/local_nankan_readiness_watcher_manifest.json"
DEFAULT_CAPTURE_LOOP_MANIFEST = "artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json"
DEFAULT_FOLLOWUP_ONESHOT_SCRIPT = "scripts/run_local_nankan_future_only_followup_oneshot.py"
DEFAULT_OUTPUT = "artifacts/reports/local_nankan_data_status_board.json"
BACKFILL_AGGREGATE_CANDIDATES = [
    "artifacts/reports/local_nankan_backfill_race_card_20y.json",
    "artifacts/reports/local_nankan_backfill_20y_windowed.json",
]
LIVE_TARGET_MANIFESTS = {
    "race_result": "artifacts/reports/local_nankan_crawl_manifest_race_result.json",
    "race_card": "artifacts/reports/local_nankan_crawl_manifest_race_card.json",
    "pedigree": "artifacts/reports/local_nankan_crawl_manifest_pedigree.json",
}


def log_progress(message: str) -> None:
    print(message, flush=True)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _display_path_value(path_text: object) -> str | None:
    if not isinstance(path_text, str) or not path_text:
        return None
    path = Path(path_text)
    if not path.is_absolute():
        return path_text
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return path_text


def _format_baseline_chain(initial_path: object, latest_path: object) -> str | None:
    initial_display = _display_path_value(initial_path)
    latest_display = _display_path_value(latest_path)
    if initial_display and latest_display:
        if initial_display == latest_display:
            return initial_display
        return f"{initial_display}->{latest_display}"
    return initial_display or latest_display


def _normalize_display_paths(value: object) -> object:
    if isinstance(value, dict):
        normalized: dict[str, object] = {}
        for key, child in value.items():
            if isinstance(child, str) and (
                key.endswith("_input")
                or key.endswith("_output")
                or key.endswith("_manifest")
                or key.endswith("_file")
                or key.endswith("_dir")
                or key.endswith("_path")
            ):
                normalized[key] = _display_path_value(child)
            else:
                normalized[key] = _normalize_display_paths(child)
        return normalized
    if isinstance(value, list):
        return [_normalize_display_paths(child) for child in value]
    if isinstance(value, str):
        return _display_path_value(value) or value
    return value


def _read_json(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _dict_payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_payload(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _resolve_backfill_aggregate(path_text: str) -> Path:
    requested = _resolve_path(path_text)
    if requested.exists():
        return requested
    for candidate in BACKFILL_AGGREGATE_CANDIDATES:
        candidate_path = _resolve_path(candidate)
        if candidate_path.exists():
            return candidate_path
    return requested


def _completed_targets_count(progress_payload: dict[str, object]) -> int:
    completed_targets = progress_payload.get("completed_targets")
    return len(completed_targets) if isinstance(completed_targets, list) else 0


def _build_live_collect_progress() -> dict[str, object]:
    payload: dict[str, object] = {}
    for target_name, manifest_path_text in LIVE_TARGET_MANIFESTS.items():
        target_payload = _read_json(_resolve_path(manifest_path_text))
        payload[target_name] = {
            "status": target_payload.get("status"),
            "requested_ids": target_payload.get("requested_ids"),
            "processed_ids": target_payload.get("processed_ids"),
            "parsed_ids": target_payload.get("parsed_ids"),
            "failure_count": target_payload.get("failure_count"),
            "rows_written": target_payload.get("rows_written"),
            "started_at": target_payload.get("started_at"),
            "finished_at": target_payload.get("finished_at"),
            "sample_ids": target_payload.get("sample_ids"),
        }
    return payload


def _build_readiness_surfaces(
    *,
    capture_loop_manifest_path: str,
    capture_loop_manifest: dict[str, object],
    readiness_probe_summary: dict[str, object],
    pre_race_handoff_manifest: dict[str, object],
    bootstrap_handoff_manifest: dict[str, object],
    readiness_watcher_manifest: dict[str, object],
) -> dict[str, object]:
    capture_loop_payload = _dict_payload(capture_loop_manifest)
    watcher_payload = _dict_payload(readiness_watcher_manifest)
    capture_pass_snapshots = _list_payload(capture_loop_payload.get("pass_snapshots"))
    latest_capture_pass = _dict_payload(capture_pass_snapshots[-1]) if capture_pass_snapshots else {}
    latest_capture_baseline = latest_capture_pass.get("baseline_summary_input")
    watcher_capture_loop = _dict_payload(watcher_payload.get("capture_loop_manifest"))
    probe_materialization = _dict_payload(readiness_probe_summary.get("materialization_summary"))
    probe_result_ready_races = readiness_probe_summary.get("result_ready_races")
    if probe_result_ready_races is None:
        probe_result_ready_races = probe_materialization.get("result_ready_races")
    probe_pending_result_races = readiness_probe_summary.get("pending_result_races")
    if probe_pending_result_races is None:
        probe_pending_result_races = probe_materialization.get("pending_result_races")
    probe_race_card_rows = readiness_probe_summary.get("race_card_rows")
    if probe_race_card_rows is None:
        probe_race_card_rows = probe_materialization.get("pre_race_only_rows")

    capture_execution_role = str(capture_loop_payload.get("execution_role") or "")
    capture_data_update_mode = str(capture_loop_payload.get("data_update_mode") or "")
    capture_trigger_contract = str(capture_loop_payload.get("trigger_contract") or "")
    latest_race_id_source_report = _dict_payload(capture_loop_payload.get("latest_race_id_source_report"))
    upstream_contract_ready = bool(
        capture_execution_role == "pre_race_capture_refresh_loop"
        and capture_data_update_mode == "capture_refresh_only"
        and capture_trigger_contract == "direct_capture_refresh"
    )

    return {
        "capture_loop": {
            "status": capture_loop_payload.get("status"),
            "current_phase": capture_loop_payload.get("current_phase"),
            "recommended_action": capture_loop_payload.get("recommended_action"),
            "completed_passes": capture_loop_payload.get("completed_passes"),
            "max_passes": capture_loop_payload.get("max_passes"),
            "initial_baseline_summary_input": _display_path_value(capture_loop_payload.get("initial_baseline_summary_input")),
            "latest_baseline_summary_input": _display_path_value(latest_capture_baseline),
            "snapshot_dir": _display_path_value(capture_loop_payload.get("snapshot_dir")),
            "latest_race_id_source_report": _normalize_display_paths(latest_race_id_source_report),
            "latest_summary": _normalize_display_paths(capture_loop_payload.get("latest_summary")),
        },
        "readiness_probe": {
            "status": readiness_probe_summary.get("status"),
            "current_phase": readiness_probe_summary.get("current_phase"),
            "recommended_action": readiness_probe_summary.get("recommended_action"),
            "result_ready_races": probe_result_ready_races,
            "pending_result_races": probe_pending_result_races,
            "race_card_rows": probe_race_card_rows,
            "historical_source_timing": _normalize_display_paths(readiness_probe_summary.get("historical_source_timing")),
        },
        "pre_race_handoff": {
            "status": pre_race_handoff_manifest.get("status"),
            "current_phase": pre_race_handoff_manifest.get("current_phase"),
            "recommended_action": pre_race_handoff_manifest.get("recommended_action"),
        },
        "bootstrap_handoff": {
            "status": bootstrap_handoff_manifest.get("status"),
            "current_phase": bootstrap_handoff_manifest.get("current_phase"),
            "recommended_action": bootstrap_handoff_manifest.get("recommended_action"),
            "evaluation_pointer_output": _display_path_value(bootstrap_handoff_manifest.get("evaluation_pointer_output")),
            "evaluation_pointer_payload": _normalize_display_paths(bootstrap_handoff_manifest.get("evaluation_pointer_payload")),
        },
        "readiness_watcher": {
            "status": watcher_payload.get("status"),
            "current_phase": watcher_payload.get("current_phase"),
            "recommended_action": watcher_payload.get("recommended_action"),
            "attempts": watcher_payload.get("attempts"),
            "timed_out": watcher_payload.get("timed_out"),
            "probe_summary": _normalize_display_paths(watcher_payload.get("probe_summary")),
            "capture_loop_manifest_output": watcher_payload.get("capture_loop_manifest_output"),
            "capture_initial_baseline_summary_input": _display_path_value(watcher_capture_loop.get("initial_baseline_summary_input")),
            "capture_latest_baseline_summary_input": _display_path_value(_dict_payload(_list_payload(watcher_capture_loop.get("pass_snapshots"))[-1]).get("baseline_summary_input") if _list_payload(watcher_capture_loop.get("pass_snapshots")) else None),
        },
        "followup_entrypoint": {
            "script": DEFAULT_FOLLOWUP_ONESHOT_SCRIPT,
            "execution_role": "readiness_followup_gate",
            "trigger_contract": "external_refresh_completed_only",
            "upstream_manifest": capture_loop_manifest_path,
            "upstream_execution_role": capture_loop_payload.get("execution_role"),
            "upstream_data_update_mode": capture_loop_payload.get("data_update_mode"),
            "upstream_trigger_contract": capture_loop_payload.get("trigger_contract"),
            "upstream_initial_baseline_summary_input": _display_path_value(capture_loop_payload.get("initial_baseline_summary_input")),
            "upstream_latest_baseline_summary_input": _display_path_value(latest_capture_baseline),
            "upstream_contract_ready": upstream_contract_ready,
            "recommended_mode": "dry_run_then_run",
            "read_order": [
                "status",
                "current_phase",
                "recommended_action",
                "upstream_refresh.upstream_fresh",
                "upstream_refresh.age_seconds",
                "upstream_refresh.contract_valid",
            ],
            "dry_run_command_preview": [
                "PYTHONPATH=src",
                ".venv/bin/python",
                DEFAULT_FOLLOWUP_ONESHOT_SCRIPT,
                "--upstream-manifest",
                capture_loop_manifest_path,
                "--dry-run",
            ],
            "run_command_preview": [
                "PYTHONPATH=src",
                ".venv/bin/python",
                DEFAULT_FOLLOWUP_ONESHOT_SCRIPT,
                "--upstream-manifest",
                capture_loop_manifest_path,
                "--run-bootstrap-on-ready",
            ],
        },
    }


def _derive_board_state(
    *,
    coverage: dict[str, object],
    backfill: dict[str, object],
    readiness_surfaces: dict[str, object],
) -> tuple[str, str, str | None]:
    capture_loop = _dict_payload(readiness_surfaces.get("capture_loop"))
    probe_payload = _dict_payload(readiness_surfaces.get("readiness_probe"))
    pre_race_handoff = _dict_payload(readiness_surfaces.get("pre_race_handoff"))
    bootstrap_handoff = _dict_payload(readiness_surfaces.get("bootstrap_handoff"))
    watcher_payload = _dict_payload(readiness_surfaces.get("readiness_watcher"))

    capture_loop_status = str(capture_loop.get("status") or "")
    probe_status = str(probe_payload.get("status") or "")
    pre_race_handoff_status = str(pre_race_handoff.get("status") or "")
    bootstrap_handoff_status = str(bootstrap_handoff.get("status") or "")
    watcher_status = str(watcher_payload.get("status") or "")

    # Future-only capture is the active operator path even while historical benchmark rerun remains blocked.
    if probe_status == "not_ready" and str(probe_payload.get("current_phase") or "") == "future_only_readiness_track":
        current_phase = str(
            watcher_payload.get("current_phase")
            or probe_payload.get("current_phase")
            or capture_loop.get("current_phase")
            or "future_only_readiness_track"
        )
        recommended_action = str(
            watcher_payload.get("recommended_action")
            or probe_payload.get("recommended_action")
            or capture_loop.get("recommended_action")
            or "capture_future_pre_race_rows_and_wait_for_results"
        )
        return "partial", current_phase, recommended_action

    if probe_status == "not_ready" or pre_race_handoff_status == "not_ready" or bootstrap_handoff_status == "not_ready":
        current_phase = str(
            bootstrap_handoff.get("current_phase")
            or pre_race_handoff.get("current_phase")
            or watcher_payload.get("current_phase")
            or probe_payload.get("current_phase")
            or "await_result_arrival"
        )
        recommended_action = str(
            bootstrap_handoff.get("recommended_action")
            or pre_race_handoff.get("recommended_action")
            or watcher_payload.get("recommended_action")
            or probe_payload.get("recommended_action")
            or "wait_for_result_ready_pre_race_races"
        )
        return "partial", current_phase, recommended_action

    if capture_loop_status in {"capturing", "ready", "empty"}:
        return (
            "partial",
            str(capture_loop.get("current_phase") or "capturing_pre_race_pool"),
            str(capture_loop.get("recommended_action") or "continue_recrawl_cadence_and_wait_for_results"),
        )

    if bootstrap_handoff_status == "benchmark_ready":
        return (
            "partial",
            str(bootstrap_handoff.get("current_phase") or "bootstrap_pending"),
            str(bootstrap_handoff.get("recommended_action") or "run_bootstrap_command_plan"),
        )

    if bootstrap_handoff_status == "completed":
        bootstrap_pointer = _dict_payload(bootstrap_handoff.get("evaluation_pointer_payload"))
        return (
            "completed",
            str(bootstrap_pointer.get("current_phase") or bootstrap_handoff.get("current_phase") or "bootstrap_completed"),
            str(bootstrap_pointer.get("recommended_action") or bootstrap_handoff.get("recommended_action") or "review_bootstrap_revision_outputs"),
        )

    return _derive_status(coverage=coverage, backfill=backfill)


def _build_effective_readiness(
    *,
    coverage_readiness: dict[str, object],
    readiness_surfaces: dict[str, object],
    current_phase: str,
    recommended_action: str | None,
) -> dict[str, object]:
    readiness = dict(coverage_readiness)
    capture_loop = _dict_payload(readiness_surfaces.get("capture_loop"))
    probe_payload = _dict_payload(readiness_surfaces.get("readiness_probe"))
    watcher_payload = _dict_payload(readiness_surfaces.get("readiness_watcher"))
    pre_race_handoff = _dict_payload(readiness_surfaces.get("pre_race_handoff"))
    bootstrap_handoff = _dict_payload(readiness_surfaces.get("bootstrap_handoff"))

    blocking_reasons: list[str] = []
    for label, payload in (
        ("capture_loop", capture_loop),
        ("readiness_probe", probe_payload),
        ("readiness_watcher", watcher_payload),
        ("pre_race_handoff", pre_race_handoff),
        ("bootstrap_handoff", bootstrap_handoff),
    ):
        status = str(payload.get("status") or "")
        if status == "not_ready":
            blocking_reasons.append(
                f"{label}:{payload.get('current_phase') or 'not_ready'}"
            )

    if blocking_reasons:
        readiness["benchmark_rerun_ready"] = False
        readiness["recommended_action"] = recommended_action
        readiness["reasons"] = blocking_reasons
        readiness["current_phase"] = current_phase
    return readiness


def _merge_live_target_states(
    target_states: dict[str, object],
    live_collect_progress: dict[str, object],
) -> dict[str, object]:
    merged: dict[str, object] = {**target_states}
    for target_name, live_value in live_collect_progress.items():
        live_payload = _dict_payload(live_value)
        if not live_payload:
            continue
        current_payload = _dict_payload(merged.get(target_name))
        merged[target_name] = {
            **current_payload,
            "present": True,
            "status": live_payload.get("status") or current_payload.get("status"),
            "requested_ids": live_payload.get("requested_ids"),
            "processed_ids": live_payload.get("processed_ids"),
            "parsed_ids": live_payload.get("parsed_ids"),
            "failure_count": live_payload.get("failure_count"),
            "rows_written": live_payload.get("rows_written"),
            "started_at": live_payload.get("started_at"),
            "finished_at": live_payload.get("finished_at"),
            "sample_ids": live_payload.get("sample_ids"),
        }
    return merged


def _backfill_process_state(backfill: dict[str, object]) -> tuple[str, str | None, int | None]:
    status = str(backfill.get("status") or "")
    pid = _optional_int(backfill.get("pid"))
    if status != "running":
        return status or "unknown", None, pid
    if pid is not None and _pid_is_running(pid):
        return "running", None, pid
    if pid is not None:
        return "stale", f"pid_not_running:{pid}", pid
    return "stale", "missing_pid", pid


def _derive_status(*, coverage: dict[str, object], backfill: dict[str, object]) -> tuple[str, str, str | None]:
    readiness = _dict_payload(coverage.get("readiness"))
    progress = _dict_payload(coverage.get("progress"))
    failed_targets = progress.get("failed_targets")
    current_stage = str(progress.get("current_stage") or "status_unknown")
    backfill_process_state, _, _ = _backfill_process_state(backfill)
    if bool(readiness.get("benchmark_rerun_ready")):
        return "completed", "ready_for_benchmark", None
    if backfill_process_state == "running":
        return "running", str(backfill.get("current_phase") or current_stage), str(backfill.get("recommended_action") or "monitor_local_nankan_backfill")
    if backfill_process_state == "stale":
        return "failed", str(backfill.get("current_phase") or current_stage), "inspect_local_nankan_backfill"
    if str(backfill.get("status") or "") == "failed":
        return "failed", current_stage, str(backfill.get("recommended_action") or "inspect_local_nankan_backfill")
    if isinstance(failed_targets, list) and failed_targets:
        return "partial", current_stage, str(readiness.get("recommended_action") or "inspect_local_crawl_failures")
    return "partial", current_stage, str(readiness.get("recommended_action") or backfill.get("recommended_action") or "inspect_local_nankan_status")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--coverage-snapshot", default=DEFAULT_COVERAGE_SNAPSHOT)
    parser.add_argument("--backfill-aggregate", default=DEFAULT_BACKFILL_AGGREGATE)
    parser.add_argument("--archive-run", default=DEFAULT_ARCHIVE_RUN)
    parser.add_argument("--capture-loop-manifest", default=DEFAULT_CAPTURE_LOOP_MANIFEST)
    parser.add_argument("--readiness-probe-summary", default=DEFAULT_READINESS_PROBE_SUMMARY)
    parser.add_argument("--pre-race-handoff-manifest", default=DEFAULT_PRE_RACE_HANDOFF_MANIFEST)
    parser.add_argument("--bootstrap-handoff-manifest", default=DEFAULT_BOOTSTRAP_HANDOFF_MANIFEST)
    parser.add_argument("--readiness-watcher-manifest", default=DEFAULT_READINESS_WATCHER_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

    progress = ProgressBar(total=3, prefix="[local-nankan-status-board]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message="loading local_nankan status inputs")

    with Heartbeat("[local-nankan-status-board]", "reading coverage and backfill manifests", logger=log_progress):
        coverage_payload = _read_json(_resolve_path(args.coverage_snapshot))
        backfill_path = _resolve_backfill_aggregate(args.backfill_aggregate)
        backfill_payload = _read_json(backfill_path)
        archive_payload = _read_json(_resolve_path(args.archive_run))
        capture_loop_manifest = _read_json(_resolve_path(args.capture_loop_manifest))
        readiness_probe_summary = _read_json(_resolve_path(args.readiness_probe_summary))
        pre_race_handoff_manifest = _read_json(_resolve_path(args.pre_race_handoff_manifest))
        bootstrap_handoff_manifest = _read_json(_resolve_path(args.bootstrap_handoff_manifest))
        readiness_watcher_manifest = _read_json(_resolve_path(args.readiness_watcher_manifest))
    progress.update(message="inputs loaded")

    coverage_readiness = _dict_payload(coverage_payload.get("readiness"))
    progress_payload = _dict_payload(coverage_payload.get("progress"))
    target_states = _dict_payload(coverage_payload.get("target_states"))
    external_outputs = _dict_payload(coverage_payload.get("external_outputs"))
    window_reports = _list_payload(backfill_payload.get("window_reports"))
    archive_reports = _list_payload(archive_payload.get("archive_reports"))
    backfill_process_state, backfill_stale_reason, backfill_pid = _backfill_process_state(backfill_payload)
    live_collect_progress = _build_live_collect_progress()
    target_states = _merge_live_target_states(target_states, live_collect_progress)
    readiness_surfaces = _build_readiness_surfaces(
        capture_loop_manifest_path=args.capture_loop_manifest,
        capture_loop_manifest=capture_loop_manifest,
        readiness_probe_summary=readiness_probe_summary,
        pre_race_handoff_manifest=pre_race_handoff_manifest,
        bootstrap_handoff_manifest=bootstrap_handoff_manifest,
        readiness_watcher_manifest=readiness_watcher_manifest,
    )
    status, current_phase, recommended_action = _derive_board_state(
        coverage=coverage_payload,
        backfill=backfill_payload,
        readiness_surfaces=readiness_surfaces,
    )
    readiness = _build_effective_readiness(
        coverage_readiness=coverage_readiness,
        readiness_surfaces=readiness_surfaces,
        current_phase=current_phase,
        recommended_action=recommended_action,
    )

    capture_baseline_chain = _format_baseline_chain(
        readiness_surfaces["capture_loop"].get("initial_baseline_summary_input"),
        readiness_surfaces["capture_loop"].get("latest_baseline_summary_input"),
    )
    latest_race_id_source_report = _dict_payload(readiness_surfaces["capture_loop"].get("latest_race_id_source_report"))

    payload = {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "artifacts": {
            "coverage_snapshot": args.coverage_snapshot,
            "backfill_aggregate": str(backfill_path.relative_to(ROOT)) if backfill_path.is_relative_to(ROOT) else str(backfill_path),
            "archive_run": args.archive_run,
            "capture_loop_manifest": args.capture_loop_manifest,
            "readiness_probe_summary": args.readiness_probe_summary,
            "pre_race_handoff_manifest": args.pre_race_handoff_manifest,
            "bootstrap_handoff_manifest": args.bootstrap_handoff_manifest,
            "readiness_watcher_manifest": args.readiness_watcher_manifest,
            "status_board": args.output,
        },
        "target_progress": progress_payload,
        "target_states": target_states,
        "live_collect_progress": live_collect_progress,
        "external_outputs": external_outputs,
        "window_progress": {
            "status": backfill_payload.get("status"),
            "process_state": backfill_process_state,
            "current_phase": backfill_payload.get("current_phase"),
            "date_order": backfill_payload.get("date_order"),
            "requested_window_count": backfill_payload.get("requested_window_count"),
            "selected_window_count": backfill_payload.get("selected_window_count"),
            "executed_window_count": backfill_payload.get("executed_window_count"),
            "resume_completed_window_count": backfill_payload.get("resume_completed_window_count"),
            "completed_window_count": backfill_payload.get("completed_window_count"),
            "remaining_window_count": backfill_payload.get("remaining_window_count"),
            "window_report_count": len(window_reports),
            "active_window": backfill_payload.get("active_window"),
            "active_window_date_window": backfill_payload.get("active_window_date_window"),
            "pid": backfill_pid,
            "pid_running": _pid_is_running(backfill_pid) if backfill_pid is not None else None,
            "stale_reason": backfill_stale_reason,
            "last_updated_at": backfill_payload.get("last_updated_at"),
        },
        "archive_progress": {
            "status": archive_payload.get("status"),
            "current_phase": archive_payload.get("current_phase"),
            "archive_report_count": len(archive_reports),
        },
        "read_order": [
            "status",
            "current_phase",
            "recommended_action",
            "readiness_surfaces.capture_loop.latest_race_id_source_report.upcoming_only",
            "readiness_surfaces.capture_loop.latest_race_id_source_report.as_of",
            "readiness_surfaces.capture_loop.latest_race_id_source_report.pre_filter_row_count",
            "readiness_surfaces.capture_loop.latest_race_id_source_report.filtered_out_count",
        ],
        "readiness_surfaces": readiness_surfaces,
        "readiness": readiness,
        "highlights": [
            f"current_phase={current_phase}",
            f"recommended_action={recommended_action}",
            f"completed_targets={_completed_targets_count(progress_payload)}",
            f"window_process_state={backfill_process_state}",
            f"date_order={backfill_payload.get('date_order')}",
            f"active_window={backfill_payload.get('active_window')}",
            f"race_card_processed_ids={_dict_payload(live_collect_progress.get('race_card')).get('processed_ids')}",
            f"remaining_windows={backfill_payload.get('remaining_window_count')}",
            f"capture_loop_status={readiness_surfaces['capture_loop'].get('status')}",
            f"probe_status={readiness_surfaces['readiness_probe'].get('status')}",
            f"handoff_status={readiness_surfaces['pre_race_handoff'].get('status')}",
            f"bootstrap_status={readiness_surfaces['bootstrap_handoff'].get('status')}",
            f"bootstrap_eval_pointer_status={_dict_payload(readiness_surfaces['bootstrap_handoff'].get('evaluation_pointer_payload')).get('status')}",
            f"watcher_status={readiness_surfaces['readiness_watcher'].get('status')}",
            f"capture_baseline_chain={capture_baseline_chain}",
            f"capture_upcoming_only={latest_race_id_source_report.get('upcoming_only')}",
            f"capture_as_of={latest_race_id_source_report.get('as_of')}",
            f"capture_pre_filter_rows={latest_race_id_source_report.get('pre_filter_row_count')}",
            f"capture_filtered_out={latest_race_id_source_report.get('filtered_out_count')}",
        ],
    }

    payload = _normalize_display_paths(payload)

    with Heartbeat("[local-nankan-status-board]", "writing status board", logger=log_progress):
        write_json(output_path, payload)
    progress.complete(message=f"saved status board path={args.output} status={status}")
    print(f"[local-nankan-status-board] saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
