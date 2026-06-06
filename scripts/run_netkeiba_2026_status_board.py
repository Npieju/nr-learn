from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import pandas as pd

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
DEFAULT_LIVE_PUBLISH = "artifacts/reports/netkeiba_2026_live_publish_manifest.json"
DEFAULT_BENCHMARK_GATE = "artifacts/reports/netkeiba_2026_benchmark_gate.json"
DEFAULT_SERVING_COMPARE_DASHBOARD = "artifacts/reports/dashboard/serving_compare_dashboard_latest.json"
DEFAULT_OUTPUT = "artifacts/reports/netkeiba_2026_status_board.json"
DEFAULT_RACE_RESULT_PATH = "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
DEFAULT_RACE_CARD_PATH = "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"
DEFAULT_RACE_RESULT_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_race_result.json"
DEFAULT_RACE_CARD_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_race_card.json"
DEFAULT_PEDIGREE_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd_pedigree.json"
DEFAULT_CRAWL_LOCK_PATH = "artifacts/reports/netkeiba_crawl_manifest_2026_ytd.json.lock"


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


def _extract_external_max_date(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        header = pd.read_csv(path, nrows=0)
    except pd.errors.EmptyDataError:
        return None
    if "date" not in header.columns:
        return None
    frame = pd.read_csv(path, usecols=["date"], low_memory=False)
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.notna().any():
        return str(dates.max().normalize().date())
    return None


def _pid_running(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_live_crawl_lock(path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "present": False,
            "pid": None,
            "pid_running": False,
            "started_at": None,
        }
    payload = _read_json(path)
    raw_pid = payload.get("pid")
    try:
        pid = int(raw_pid) if raw_pid is not None else None
    except (TypeError, ValueError):
        pid = None
    return {
        "present": True,
        "pid": pid,
        "pid_running": _pid_running(pid),
        "started_at": payload.get("started_at"),
    }


def _date_gap_days(max_date_text: str | None, target_date_text: str | None) -> int | None:
    if not max_date_text or not target_date_text:
        return None
    max_date = pd.Timestamp(max_date_text)
    target_date = pd.Timestamp(target_date_text)
    return max(int((target_date - max_date).days), 0)


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


def _derive_live_artifact_path(prediction_file: str | None, suffix: str) -> Path | None:
    if not prediction_file:
        return None
    path = _resolve_path(prediction_file)
    if path.suffix:
        return path.with_suffix(suffix)
    return None


def _display_path_to_repo_path(path_text: object) -> str | None:
    if not path_text:
        return None
    path = _resolve_path(str(path_text))
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _latest_git_commit_for_paths(paths: list[str | None]) -> str | None:
    repo_paths = [path for path in paths if path]
    if not repo_paths:
        return None
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", *repo_paths],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    commit_sha = result.stdout.strip()
    return commit_sha or None


def _derive_status(
    *,
    backfill: dict[str, object],
    snapshot: dict[str, object],
    handoff: dict[str, object],
    benchmark_gate: dict[str, object],
) -> tuple[str, str, str]:
    handoff_status = str(handoff.get("status") or "")
    handoff_phase = str(handoff.get("current_phase") or "")
    handoff_action = str(handoff.get("recommended_action") or "inspect_live_handoff_manifest")
    benchmark_status = str(benchmark_gate.get("status") or "")
    benchmark_phase = str(benchmark_gate.get("current_phase") or "")
    benchmark_action = str(benchmark_gate.get("recommended_action") or "inspect_2026_benchmark_gate_manifest")
    readiness = _dict_payload(snapshot.get("readiness"))
    progress = _dict_payload(snapshot.get("progress"))
    backfill_status = str(backfill.get("stopped_reason") or backfill.get("status") or "")

    if handoff_status == "completed" and benchmark_status == "completed":
        return "completed", benchmark_phase or "benchmark_gate_completed", benchmark_action
    if handoff_status == "completed" and benchmark_status in {"running", "planned"}:
        return "running", benchmark_phase or "benchmark_gate_running", benchmark_action
    if handoff_status == "completed" and benchmark_status in {"not_ready", "snapshot_failed", "train_failed", "evaluate_failed", "failed", "interrupted"}:
        return "partial", benchmark_phase or "benchmark_gate_blocked", benchmark_action
    if handoff_status == "completed":
        return "handed_off", handoff_phase or "live_predict_completed", handoff_action
    if handoff_status == "timeout":
        return "partial", handoff_phase or "await_history_ready_timeout", handoff_action
    if handoff_status in {"handoff_failed", "failed"}:
        return "failed", handoff_phase or "live_predict_failed", handoff_action
    if handoff_status == "waiting":
        return "waiting", handoff_phase or "await_history_ready", handoff_action
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
    parser.add_argument("--live-publish-manifest", default=DEFAULT_LIVE_PUBLISH)
    parser.add_argument("--benchmark-gate-manifest", default=DEFAULT_BENCHMARK_GATE)
    parser.add_argument("--serving-compare-dashboard-summary", default=None)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--race-result-manifest", default=DEFAULT_RACE_RESULT_MANIFEST)
    parser.add_argument("--race-card-manifest", default=DEFAULT_RACE_CARD_MANIFEST)
    parser.add_argument("--pedigree-manifest", default=DEFAULT_PEDIGREE_MANIFEST)
    parser.add_argument("--crawl-lock-path", default=DEFAULT_CRAWL_LOCK_PATH)
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
        live_publish_payload = _read_json(_resolve_path(args.live_publish_manifest))
        benchmark_gate_payload = _read_json(_resolve_path(args.benchmark_gate_manifest))
        serving_compare_payload = _read_json(_resolve_path(args.serving_compare_dashboard_summary)) if args.serving_compare_dashboard_summary else {}
        race_result_target_payload = _read_json(_resolve_path(args.race_result_manifest))
        race_card_target_payload = _read_json(_resolve_path(args.race_card_manifest))
        pedigree_target_payload = _read_json(_resolve_path(args.pedigree_manifest))
    progress.update(message="inputs loaded")

    status, current_phase, recommended_action = _derive_status(
        backfill=backfill_payload,
        snapshot=snapshot_payload,
        handoff=handoff_payload,
        benchmark_gate=benchmark_gate_payload,
    )
    snapshot_readiness = _dict_payload(snapshot_payload.get("readiness"))
    snapshot_progress = _dict_payload(snapshot_payload.get("progress"))
    external_outputs = _dict_payload(snapshot_payload.get("external_outputs"))
    target_states = _dict_payload(snapshot_payload.get("target_states"))
    live_target_states = {
        "race_result": race_result_target_payload or _dict_payload(target_states.get("race_result")),
        "race_card": race_card_target_payload or _dict_payload(target_states.get("race_card")),
        "pedigree": pedigree_target_payload or _dict_payload(target_states.get("pedigree")),
    }
    crawl_lock = _read_live_crawl_lock(_resolve_path(args.crawl_lock_path))
    live_race_result_max_date = _extract_external_max_date(_resolve_path(args.race_result_path))
    live_race_card_max_date = _extract_external_max_date(_resolve_path(args.race_card_path))
    history_ready_date = str(handoff_payload.get("history_ready_date") or "") or None
    race_result_gap_days = _date_gap_days(live_race_result_max_date or handoff_payload.get("race_result_max_date"), history_ready_date)
    race_card_gap_days = _date_gap_days(live_race_card_max_date or handoff_payload.get("race_card_max_date"), history_ready_date)
    history_dates_ready = (
        race_result_gap_days == 0 and race_card_gap_days == 0
        if race_result_gap_days is not None and race_card_gap_days is not None
        else False
    )
    limiting_history_target = None
    if race_result_gap_days is not None or race_card_gap_days is not None:
        result_gap = race_result_gap_days if race_result_gap_days is not None else -1
        card_gap = race_card_gap_days if race_card_gap_days is not None else -1
        limiting_history_target = "race_result" if result_gap >= card_gap else "race_card"
    cycles = _list_payload(backfill_payload.get("cycles"))
    last_cycle = _dict_payload(cycles[-1]) if cycles else {}
    completed_cycles = int(backfill_payload.get("completed_cycles") or 0)
    running_target_names = [
        name for name, payload in live_target_states.items() if _dict_payload(payload).get("status") == "running"
    ]
    active_cycle = completed_cycles + 1 if running_target_names else None
    prediction_file = handoff_payload.get("live_prediction_file")
    live_summary_payload = _read_json(_derive_live_artifact_path(prediction_file, ".summary.json")) if prediction_file else {}
    live_runtime_payload = _read_json(_derive_live_artifact_path(prediction_file, ".live.json")) if prediction_file else {}
    publish_payload = _dict_payload(live_publish_payload)
    publish_handoff = _dict_payload(publish_payload.get("handoff"))
    publish_pages = _dict_payload(publish_payload.get("pages"))
    publish_git = _dict_payload(publish_payload.get("git"))
    publish_page_repo_path = _display_path_to_repo_path(publish_pages.get("target_page"))
    publish_data_repo_path = _display_path_to_repo_path(publish_pages.get("data_file"))
    publish_pages_commit_sha = _latest_git_commit_for_paths([publish_page_repo_path, publish_data_repo_path])
    if not prediction_file and publish_handoff.get("prediction_file"):
        prediction_file = publish_handoff.get("prediction_file")
        live_summary_payload = _read_json(_derive_live_artifact_path(prediction_file, ".summary.json")) if prediction_file else {}
        live_runtime_payload = _read_json(_derive_live_artifact_path(prediction_file, ".live.json")) if prediction_file else {}
    benchmark_gate_status = str(benchmark_gate_payload.get("status") or "")
    serving_compare_compare = _dict_payload(serving_compare_payload.get("compare"))
    serving_compare_bankroll = _dict_payload(serving_compare_payload.get("bankroll"))
    serving_compare_left = _dict_payload(serving_compare_payload.get("left"))
    serving_compare_right = _dict_payload(serving_compare_payload.get("right"))

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
            "live_publish_manifest": args.live_publish_manifest,
            "benchmark_gate_manifest": args.benchmark_gate_manifest,
            "serving_compare_dashboard_summary": args.serving_compare_dashboard_summary,
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
                "race_result": _target_summary(_dict_payload(live_target_states.get("race_result"))),
                "race_card": _target_summary(_dict_payload(live_target_states.get("race_card"))),
                "pedigree": _target_summary(_dict_payload(live_target_states.get("pedigree"))),
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
            "history_ready_date": history_ready_date,
            "race_result_max_date": live_race_result_max_date or handoff_payload.get("race_result_max_date"),
            "race_card_max_date": live_race_card_max_date or handoff_payload.get("race_card_max_date"),
            "race_result_gap_days": race_result_gap_days,
            "race_card_gap_days": race_card_gap_days,
            "history_dates_ready": history_dates_ready,
            "limiting_history_target": limiting_history_target,
            "live_prediction_file": handoff_payload.get("live_prediction_file"),
            "live_report_file": handoff_payload.get("live_report_file"),
        },
        "live_outputs": {
            "prediction_file": prediction_file,
            "summary_file": live_summary_payload.get("summary_file") or str(_derive_live_artifact_path(prediction_file, ".summary.json").relative_to(ROOT)) if prediction_file and _derive_live_artifact_path(prediction_file, ".summary.json") is not None else None,
            "runtime_manifest_file": str(_derive_live_artifact_path(prediction_file, ".live.json").relative_to(ROOT)) if prediction_file and _derive_live_artifact_path(prediction_file, ".live.json") is not None else None,
            "policy_name": live_summary_payload.get("policy_name"),
            "policy_selected_rows": live_summary_payload.get("policy_selected_rows"),
            "records": live_summary_payload.get("records") or live_runtime_payload.get("record_count"),
            "num_races": live_summary_payload.get("num_races") or live_runtime_payload.get("race_count"),
            "odds_official_datetime_max": live_runtime_payload.get("odds_official_datetime_max"),
        },
        "publish": {
            "status": publish_payload.get("status"),
            "current_phase": publish_payload.get("current_phase"),
            "recommended_action": publish_payload.get("recommended_action"),
            "manifest": args.live_publish_manifest,
            "mode": publish_payload.get("mode"),
            "odds_refresh": publish_payload.get("odds_refresh"),
            "page_url": publish_pages.get("page_url"),
            "target_page": publish_pages.get("target_page"),
            "data_file": publish_pages.get("data_file"),
            "git_status": publish_git.get("status"),
            "commit_sha": publish_git.get("commit_sha"),
            "pages_commit_sha": publish_pages_commit_sha,
            "pages_commit_short": publish_pages_commit_sha[:7] if publish_pages_commit_sha else None,
        },
        "benchmark_gate": {
            "status": benchmark_gate_payload.get("status"),
            "current_phase": benchmark_gate_payload.get("current_phase"),
            "recommended_action": benchmark_gate_payload.get("recommended_action"),
            "manifest": args.benchmark_gate_manifest,
            "completed_step": benchmark_gate_payload.get("completed_step"),
            "train_status": _dict_payload(benchmark_gate_payload.get("train")).get("status"),
            "evaluate_status": _dict_payload(benchmark_gate_payload.get("evaluate")).get("status"),
            "readiness": _dict_payload(benchmark_gate_payload.get("readiness")),
        },
        "serving_compare": {
            "summary_file": args.serving_compare_dashboard_summary,
            "status": serving_compare_payload.get("status"),
            "recommended_action": serving_compare_payload.get("recommended_action"),
            "window_label": serving_compare_payload.get("window_label"),
            "prediction_backend": serving_compare_payload.get("prediction_backend"),
            "date_count": serving_compare_payload.get("date_count"),
            "left_profile": serving_compare_left.get("profile"),
            "right_profile": serving_compare_right.get("profile"),
            "left_total_policy_bets": serving_compare_compare.get("left_total_policy_bets"),
            "right_total_policy_bets": serving_compare_compare.get("right_total_policy_bets"),
            "right_minus_left_total_policy_net": serving_compare_compare.get("right_minus_left_total_policy_net"),
            "right_minus_left_pure_final_bankroll": serving_compare_bankroll.get("right_minus_left_pure_final_bankroll"),
            "best_selected_label": _dict_payload(serving_compare_bankroll.get("best_result")).get("selected_label"),
            "best_final_bankroll": _dict_payload(serving_compare_bankroll.get("best_result")).get("final_bankroll"),
        },
        "highlights": [
            f"status={status}",
            f"current_phase={current_phase}",
            f"recommended_action={recommended_action}",
            f"completed_cycles={completed_cycles}",
            f"active_cycle={active_cycle}",
            f"snapshot_stage={snapshot_progress.get('current_stage')}",
            f"running_targets={','.join(running_target_names) if running_target_names else 'none'}",
            f"history_frontier_result={live_race_result_max_date or handoff_payload.get('race_result_max_date')}",
            f"history_frontier_race_card={live_race_card_max_date or handoff_payload.get('race_card_max_date')}",
            f"history_gap_days_result={race_result_gap_days}",
            f"history_gap_days_race_card={race_card_gap_days}",
            f"limiting_history_target={limiting_history_target}",
            f"benchmark_gate_status={benchmark_gate_status or 'not_run'}",
            f"policy_selected_rows={live_summary_payload.get('policy_selected_rows')}",
            f"live_num_races={live_summary_payload.get('num_races') or live_runtime_payload.get('race_count')}",
            f"publish_status={publish_payload.get('status')}",
            f"publish_git_status={publish_git.get('status')}",
            f"publish_pages_commit={publish_pages_commit_sha[:7] if publish_pages_commit_sha else None}",
            f"serving_compare_window={serving_compare_payload.get('window_label')}",
            f"serving_compare_net_delta={serving_compare_compare.get('right_minus_left_total_policy_net')}",
            f"serving_compare_bankroll_delta={serving_compare_bankroll.get('right_minus_left_pure_final_bankroll')}",
            f"serving_compare_best={_dict_payload(serving_compare_bankroll.get('best_result')).get('selected_label')}",
        ],
    }

    with Heartbeat("[netkeiba-2026-status-board]", "writing status board", logger=log_progress):
        write_json(output_path, payload)
    progress.complete(message=f"saved status board path={args.output} status={status}")
    print(f"[netkeiba-2026-status-board] saved: {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
