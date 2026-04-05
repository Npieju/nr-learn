from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_SNAPSHOT_SCRIPT = "scripts/run_netkeiba_2026_ytd_snapshot.py"
DEFAULT_LIVE_SCRIPT = "scripts/run_jra_live_predict.py"
DEFAULT_SNAPSHOT_OUTPUT = "artifacts/reports/netkeiba_coverage_snapshot_2026_ytd.json"
DEFAULT_WRAPPER_MANIFEST = "artifacts/reports/netkeiba_2026_live_handoff_manifest.json"
DEFAULT_PROFILE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_PATH = "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
DEFAULT_RACE_CARD_PATH = "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-2026-live-handoff {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[netkeiba-2026-live-handoff] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[netkeiba-2026-live-handoff]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _read_json_dict(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _extract_external_max_date(path: Path) -> pd.Timestamp | None:
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
        return dates.max().normalize()
    return None


def _completed_like(status: object) -> bool:
    normalized = str(status or "").strip()
    return normalized in {"completed", "completed_untracked"}


def _build_manifest(
    *,
    status: str,
    current_phase: str,
    recommended_action: str,
    attempts: int,
    waited_seconds: int,
    race_date: str,
    history_ready_date: str,
    snapshot_payload: dict[str, Any],
    race_result_max_date: str | None,
    race_card_max_date: str | None,
    live_exit_code: int | None = None,
    live_command: list[str] | None = None,
    live_prediction_file: str | None = None,
    live_report_file: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    highlights = [
        f"status={status}",
        f"current_phase={current_phase}",
        f"next operator action: {recommended_action}",
    ]
    if race_result_max_date or race_card_max_date:
        highlights.append(
            f"history frontier result={race_result_max_date or 'unknown'} race_card={race_card_max_date or 'unknown'} target={history_ready_date}"
        )
    if live_prediction_file:
        highlights.append(f"prediction_file={live_prediction_file}")
    if error:
        highlights.append(error)
    return {
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "attempts": int(attempts),
        "waited_seconds": int(waited_seconds),
        "race_date": race_date,
        "history_ready_date": history_ready_date,
        "race_result_max_date": race_result_max_date,
        "race_card_max_date": race_card_max_date,
        "live_exit_code": live_exit_code,
        "live_command": live_command,
        "live_prediction_file": live_prediction_file,
        "live_report_file": live_report_file,
        "snapshot_readiness": snapshot_payload.get("readiness") if isinstance(snapshot_payload, dict) else {},
        "snapshot_progress": snapshot_payload.get("progress") if isinstance(snapshot_payload, dict) else {},
        "error": error,
        "highlights": highlights,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--snapshot-script", default=DEFAULT_SNAPSHOT_SCRIPT)
    parser.add_argument("--snapshot-output", default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--live-script", default=DEFAULT_LIVE_SCRIPT)
    parser.add_argument("--wrapper-manifest-output", default=DEFAULT_WRAPPER_MANIFEST)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--race-date", required=True)
    parser.add_argument("--headline-contains", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--history-lag-days", type=int, default=1)
    parser.add_argument("--wait-for-ready", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=300)
    parser.add_argument("--refresh-live-crawl", action="store_true")
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    args = parser.parse_args()

    wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)
    wrapper_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=3, prefix="[netkeiba-2026-live-handoff]", logger=log_progress, min_interval_sec=0.0)

    try:
        target_race_date = pd.Timestamp(args.race_date).normalize()
        history_ready_date = (target_race_date - pd.Timedelta(days=max(int(args.history_lag_days), 0))).normalize()
        snapshot_command = [str(args.python_executable), str(_resolve_path(args.snapshot_script))]
        live_command = [
            str(args.python_executable),
            str(_resolve_path(args.live_script)),
            "--profile",
            args.profile,
            "--race-date",
            args.race_date,
        ]
        if args.headline_contains:
            live_command.extend(["--headline-contains", args.headline_contains])
        if args.limit is not None:
            live_command.extend(["--limit", str(args.limit)])
        if args.refresh_live_crawl:
            live_command.append("--refresh")

        wait_started = time.monotonic()
        attempts = 0
        progress.start(message=f"starting race_date={args.race_date} history_ready_date={history_ready_date.date()}")

        while True:
            attempts += 1
            snapshot_exit = _run_command(label=f"snapshot attempt={attempts}", command=snapshot_command)
            snapshot_payload = _read_json_dict(_resolve_path(args.snapshot_output))
            race_result_max_date = _extract_external_max_date(_resolve_path(args.race_result_path))
            race_card_max_date = _extract_external_max_date(_resolve_path(args.race_card_path))
            readiness = snapshot_payload.get("readiness") if isinstance(snapshot_payload.get("readiness"), dict) else {}
            target_states = snapshot_payload.get("target_states") if isinstance(snapshot_payload.get("target_states"), dict) else {}
            race_targets_completed = _completed_like(target_states.get("race_result", {}).get("status")) and _completed_like(target_states.get("race_card", {}).get("status"))
            history_dates_ready = (
                race_result_max_date is not None
                and race_card_max_date is not None
                and race_result_max_date >= history_ready_date
                and race_card_max_date >= history_ready_date
            )
            snapshot_consistent = bool(readiness.get("snapshot_consistent"))
            ready = bool(snapshot_exit == 0 and race_targets_completed and snapshot_consistent and history_dates_ready)
            waited_seconds = int(max(0, time.monotonic() - wait_started))
            progress.update(
                current=1,
                message=(
                    f"snapshot attempt={attempts} ready={ready} race_targets_completed={race_targets_completed} "
                    f"snapshot_consistent={snapshot_consistent} result_max={race_result_max_date} race_card_max={race_card_max_date}"
                ),
            )

            if ready:
                live_exit = _run_command(label="live_predict", command=live_command)
                prediction_file = None
                report_file = None
                if live_exit == 0:
                    date_tag = pd.Timestamp(args.race_date).strftime("%Y%m%d")
                    prediction_file = f"artifacts/predictions/predictions_{date_tag}_jra_live.csv"
                    report_file = f"artifacts/predictions/predictions_{date_tag}_jra_live.report.md"
                manifest = _build_manifest(
                    status="completed" if live_exit == 0 else "handoff_failed",
                    current_phase="live_predict_completed" if live_exit == 0 else "live_predict_failed",
                    recommended_action="review_live_prediction_outputs" if live_exit == 0 else "inspect_live_predict_failure",
                    attempts=attempts,
                    waited_seconds=waited_seconds,
                    race_date=args.race_date,
                    history_ready_date=str(history_ready_date.date()),
                    snapshot_payload=snapshot_payload,
                    race_result_max_date=str(race_result_max_date.date()) if race_result_max_date is not None else None,
                    race_card_max_date=str(race_card_max_date.date()) if race_card_max_date is not None else None,
                    live_exit_code=live_exit,
                    live_command=live_command,
                    live_prediction_file=prediction_file,
                    live_report_file=report_file,
                )
                write_json(wrapper_manifest_path, manifest)
                progress.complete(message=f"handoff completed output={wrapper_manifest_path}")
                return 0 if live_exit == 0 else 1

            timed_out = (not args.wait_for_ready) or (
                args.max_wait_seconds > 0 and waited_seconds >= int(args.max_wait_seconds)
            )
            if timed_out:
                manifest = _build_manifest(
                    status="waiting",
                    current_phase="await_history_ready",
                    recommended_action="wait_for_2026_history_frontier",
                    attempts=attempts,
                    waited_seconds=waited_seconds,
                    race_date=args.race_date,
                    history_ready_date=str(history_ready_date.date()),
                    snapshot_payload=snapshot_payload,
                    race_result_max_date=str(race_result_max_date.date()) if race_result_max_date is not None else None,
                    race_card_max_date=str(race_card_max_date.date()) if race_card_max_date is not None else None,
                    live_command=live_command,
                    error=(
                        f"snapshot_exit={snapshot_exit} race_targets_completed={race_targets_completed} "
                        f"snapshot_consistent={snapshot_consistent} history_dates_ready={history_dates_ready}"
                    ),
                )
                write_json(wrapper_manifest_path, manifest)
                progress.complete(message=f"not ready output={wrapper_manifest_path}")
                return 2

            sleep_seconds = max(1, int(args.poll_interval_seconds))
            print(f"[netkeiba-2026-live-handoff] waiting for readiness sleep_seconds={sleep_seconds}", flush=True)
            time.sleep(sleep_seconds)
    except KeyboardInterrupt:
        print("[netkeiba-2026-live-handoff] interrupted by user")
        return 130
    except Exception as error:
        print(f"[netkeiba-2026-live-handoff] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())