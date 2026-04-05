from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_BACKFILL_MANIFEST = "artifacts/reports/netkeiba_backfill_manifest_2026_ytd.json"
DEFAULT_SNAPSHOT = "artifacts/reports/netkeiba_coverage_snapshot_2026_ytd.json"
DEFAULT_HANDOFF = "artifacts/reports/netkeiba_2026_live_handoff_manifest.json"
DEFAULT_OUTPUT = "artifacts/reports/netkeiba_2026_status_board.json"


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


def _target_summary(target_state: dict[str, object]) -> dict[str, object]:
    return {
        "status": target_state.get("status"),
        "requested_ids": target_state.get("requested_ids"),
        "processed_ids": target_state.get("processed_ids"),
        "parsed_ids": target_state.get("parsed_ids"),
        "failure_count": target_state.get("failure_count"),
        "rows_written": target_state.get("rows_written"),
        "started_at": target_state.get("started_at"),
        "finished_at": target_state.get("finished_at"),
    }


def _derive_status(
    *,
    backfill: dict[str, object],
    snapshot: dict[str, object],
    handoff: dict[str, object],
) -> tuple[str, str, str]:
    handoff_status = str(handoff.get("status") or "")
    handoff_phase = str(handoff.get("current_phase") or "")
    handoff_action = str(handoff.get("recommended_action") or "inspect_live_handoff_manifest")
    readiness = _dict_payload(snapshot.get("readiness"))
    progress = _dict_payload(snapshot.get("progress"))
    backfill_status = str(backfill.get("stopped_reason") or backfill.get("status") or "")

    if handoff_status == "completed":
        return "completed", handoff_phase or "live_predict_completed", handoff_action
    if handoff_status in {"handoff_failed", "failed"}:
        return "failed", handoff_phase or "live_predict_failed", handoff_action
    if handoff_status == "waiting":
        return "running", handoff_phase or "await_history_ready", handoff_action
    if bool(readiness.get("benchmark_rerun_ready")):
        return "ready", "history_ready_for_live_handoff", "run_netkeiba_2026_live_handoff"
    if str(progress.get("current_stage") or ""):
        return "running", str(progress.get("current_stage")), str(readiness.get("recommended_action") or "inspect_2026_snapshot")
    if backfill_status in {"completed", "running", "max_cycles_reached"}:
        return "running", "backfill_in_progress", "inspect_2026_backfill_manifest"
    return "partial", "status_unknown", "inspect_2026_manifests"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backfill-manifest", default=DEFAULT_BACKFILL_MANIFEST)
    parser.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    parser.add_argument("--handoff-manifest", default=DEFAULT_HANDOFF)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

    progress = ProgressBar(total=3, prefix="[netkeiba-2026-status-board]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message="loading 2026 YTD serving status inputs")

    with Heartbeat("[netkeiba-2026-status-board]", "reading manifests", logger=log_progress):
        backfill_payload = _read_json(_resolve_path(args.backfill_manifest))
        snapshot_payload = _read_json(_resolve_path(args.snapshot))
        handoff_payload = _read_json(_resolve_path(args.handoff_manifest))
    progress.update(message="inputs loaded")

    status, current_phase, recommended_action = _derive_status(
        backfill=backfill_payload,
        snapshot=snapshot_payload,
        handoff=handoff_payload,
    )
    snapshot_readiness = _dict_payload(snapshot_payload.get("readiness"))
    snapshot_progress = _dict_payload(snapshot_payload.get("progress"))
    external_outputs = _dict_payload(snapshot_payload.get("external_outputs"))
    target_states = _dict_payload(snapshot_payload.get("target_states"))
    crawl_lock = _dict_payload(snapshot_payload.get("crawl_lock"))
    cycles = _list_payload(backfill_payload.get("cycles"))
    last_cycle = _dict_payload(cycles[-1]) if cycles else {}
    completed_cycles = int(backfill_payload.get("completed_cycles") or 0)
    running_target_names = [
        name for name, payload in target_states.items() if _dict_payload(payload).get("status") == "running"
    ]
    active_cycle = completed_cycles + 1 if running_target_names else None

    payload = {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "artifacts": {
            "backfill_manifest": args.backfill_manifest,
            "snapshot": args.snapshot,
            "handoff_manifest": args.handoff_manifest,
            "status_board": args.output,
        },
        "backfill": {
            "stopped_reason": backfill_payload.get("stopped_reason"),
            "completed_cycles": completed_cycles,
            "active_cycle": active_cycle,
            "date_window": backfill_payload.get("date_window"),
            "crawl_lock": {
                "present": crawl_lock.get("present"),
                "pid": crawl_lock.get("pid"),
                "pid_running": crawl_lock.get("pid_running"),
                "started_at": crawl_lock.get("started_at"),
            },
            "current_targets": {
                "race_result": _target_summary(_dict_payload(target_states.get("race_result"))),
                "race_card": _target_summary(_dict_payload(target_states.get("race_card"))),
                "pedigree": _target_summary(_dict_payload(target_states.get("pedigree"))),
            },
            "last_cycle": {
                "cycle": last_cycle.get("cycle"),
                "finished_at": last_cycle.get("finished_at"),
                "race_id_count": _dict_payload(last_cycle.get("race_id_prep")).get("row_count"),
                "race_result_rows_written": _dict_payload(last_cycle.get("race_result")).get("rows_written"),
                "race_card_rows_written": _dict_payload(last_cycle.get("race_card")).get("rows_written"),
                "pedigree_rows_written": _dict_payload(last_cycle.get("pedigree")).get("rows_written"),
            },
        },
        "snapshot": {
            "readiness": snapshot_readiness,
            "progress": snapshot_progress,
            "external_outputs": external_outputs,
        },
        "handoff": {
            "status": handoff_payload.get("status"),
            "current_phase": handoff_payload.get("current_phase"),
            "recommended_action": handoff_payload.get("recommended_action"),
            "race_date": handoff_payload.get("race_date"),
            "history_ready_date": handoff_payload.get("history_ready_date"),
            "race_result_max_date": handoff_payload.get("race_result_max_date"),
            "race_card_max_date": handoff_payload.get("race_card_max_date"),
            "live_prediction_file": handoff_payload.get("live_prediction_file"),
            "live_report_file": handoff_payload.get("live_report_file"),
        },
        "highlights": [
            f"status={status}",
            f"current_phase={current_phase}",
            f"recommended_action={recommended_action}",
            f"completed_cycles={completed_cycles}",
            f"active_cycle={active_cycle}",
            f"snapshot_stage={snapshot_progress.get('current_stage')}",
            f"running_targets={','.join(running_target_names) if running_target_names else 'none'}",
            f"history_frontier_result={handoff_payload.get('race_result_max_date')}",
            f"history_frontier_race_card={handoff_payload.get('race_card_max_date')}",
        ],
    }

    with Heartbeat("[netkeiba-2026-status-board]", "writing status board", logger=log_progress):
        write_json(output_path, payload)
    progress.complete(message=f"saved status board path={args.output} status={status}")
    print(f"[netkeiba-2026-status-board] saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())