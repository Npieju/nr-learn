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
    readiness_probe_summary: dict[str, object],
    pre_race_handoff_manifest: dict[str, object],
    bootstrap_handoff_manifest: dict[str, object],
    readiness_watcher_manifest: dict[str, object],
) -> dict[str, object]:
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

    return {
        "readiness_probe": {
            "status": readiness_probe_summary.get("status"),
            "recommended_action": readiness_probe_summary.get("recommended_action"),
            "result_ready_races": probe_result_ready_races,
            "pending_result_races": probe_pending_result_races,
            "race_card_rows": probe_race_card_rows,
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
        },
        "readiness_watcher": {
            "status": readiness_watcher_manifest.get("status"),
            "current_phase": readiness_watcher_manifest.get("current_phase"),
            "recommended_action": readiness_watcher_manifest.get("recommended_action"),
            "attempts": readiness_watcher_manifest.get("attempts"),
            "timed_out": readiness_watcher_manifest.get("timed_out"),
        },
    }


def _derive_board_state(
    *,
    coverage: dict[str, object],
    backfill: dict[str, object],
    readiness_surfaces: dict[str, object],
) -> tuple[str, str, str | None]:
    probe_payload = _dict_payload(readiness_surfaces.get("readiness_probe"))
    pre_race_handoff = _dict_payload(readiness_surfaces.get("pre_race_handoff"))
    bootstrap_handoff = _dict_payload(readiness_surfaces.get("bootstrap_handoff"))

    probe_status = str(probe_payload.get("status") or "")
    pre_race_handoff_status = str(pre_race_handoff.get("status") or "")
    bootstrap_handoff_status = str(bootstrap_handoff.get("status") or "")

    if probe_status == "not_ready" or pre_race_handoff_status == "not_ready" or bootstrap_handoff_status == "not_ready":
        current_phase = str(
            bootstrap_handoff.get("current_phase")
            or pre_race_handoff.get("current_phase")
            or probe_payload.get("current_phase")
            or "await_result_arrival"
        )
        recommended_action = str(
            bootstrap_handoff.get("recommended_action")
            or pre_race_handoff.get("recommended_action")
            or probe_payload.get("recommended_action")
            or "wait_for_result_ready_pre_race_races"
        )
        return "partial", current_phase, recommended_action

    if bootstrap_handoff_status == "benchmark_ready":
        return (
            "partial",
            str(bootstrap_handoff.get("current_phase") or "bootstrap_pending"),
            str(bootstrap_handoff.get("recommended_action") or "run_bootstrap_command_plan"),
        )

    if bootstrap_handoff_status == "completed":
        return (
            "completed",
            str(bootstrap_handoff.get("current_phase") or "bootstrap_completed"),
            str(bootstrap_handoff.get("recommended_action") or "review_bootstrap_revision_outputs"),
        )

    return _derive_status(coverage=coverage, backfill=backfill)


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
        readiness_probe_summary = _read_json(_resolve_path(args.readiness_probe_summary))
        pre_race_handoff_manifest = _read_json(_resolve_path(args.pre_race_handoff_manifest))
        bootstrap_handoff_manifest = _read_json(_resolve_path(args.bootstrap_handoff_manifest))
        readiness_watcher_manifest = _read_json(_resolve_path(args.readiness_watcher_manifest))
    progress.update(message="inputs loaded")

    readiness = _dict_payload(coverage_payload.get("readiness"))
    progress_payload = _dict_payload(coverage_payload.get("progress"))
    target_states = _dict_payload(coverage_payload.get("target_states"))
    external_outputs = _dict_payload(coverage_payload.get("external_outputs"))
    window_reports = _list_payload(backfill_payload.get("window_reports"))
    archive_reports = _list_payload(archive_payload.get("archive_reports"))
    backfill_process_state, backfill_stale_reason, backfill_pid = _backfill_process_state(backfill_payload)
    live_collect_progress = _build_live_collect_progress()
    target_states = _merge_live_target_states(target_states, live_collect_progress)
    readiness_surfaces = _build_readiness_surfaces(
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
            f"probe_status={readiness_surfaces['readiness_probe'].get('status')}",
            f"handoff_status={readiness_surfaces['pre_race_handoff'].get('status')}",
            f"bootstrap_status={readiness_surfaces['bootstrap_handoff'].get('status')}",
            f"watcher_status={readiness_surfaces['readiness_watcher'].get('status')}",
        ],
    }

    with Heartbeat("[local-nankan-status-board]", "writing status board", logger=log_progress):
        write_json(output_path, payload)
    progress.complete(message=f"saved status board path={args.output} status={status}")
    print(f"[local-nankan-status-board] saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
