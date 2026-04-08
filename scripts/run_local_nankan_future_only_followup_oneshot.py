from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_WAIT_CYCLE_SCRIPT = "scripts/run_local_nankan_future_only_wait_then_cycle.py"
DEFAULT_UPSTREAM_MANIFEST = "artifacts/reports/local_nankan_pre_race_capture_loop_issue122_cycle.json"
DEFAULT_WAIT_CYCLE_MANIFEST = "artifacts/reports/local_nankan_future_only_wait_then_cycle_issue122.json"
DEFAULT_OUTPUT = "artifacts/reports/local_nankan_future_only_followup_oneshot_issue122.json"
DEFAULT_ARTIFACT_PREFIX = "local_nankan_future_only_followup_oneshot_issue122"
DEFAULT_LOG_DIR = "artifacts/logs"
EXPECTED_UPSTREAM_EXECUTION_ROLE = "pre_race_capture_refresh_loop"
EXPECTED_UPSTREAM_DATA_UPDATE_MODE = "capture_refresh_only"
EXPECTED_UPSTREAM_TRIGGER_CONTRACT = "direct_capture_refresh"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-followup-oneshot {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


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


def _extract_upstream_observed_at(upstream_path: Path, upstream_payload: dict[str, Any]) -> str | None:
    for key in ("finished_at", "updated_at", "observed_at", "started_at"):
        observed_at = upstream_payload.get(key)
        if isinstance(observed_at, str) and observed_at:
            return observed_at
    if not upstream_path.exists():
        return None
    observed_at = datetime.fromtimestamp(upstream_path.stat().st_mtime, tz=timezone.utc)
    return observed_at.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _compute_age_seconds_against(observed_at: str | None, *, reference_at: str | None) -> int | None:
    parsed = _parse_utc_iso(observed_at)
    reference = _parse_utc_iso(reference_at)
    if parsed is None or reference is None:
        return None
    return max(0, int((reference - parsed).total_seconds()))


def _evaluate_upstream_contract(upstream_payload: dict[str, Any]) -> tuple[bool, list[str]]:
    mismatches: list[str] = []
    if str(upstream_payload.get("execution_role") or "") != EXPECTED_UPSTREAM_EXECUTION_ROLE:
        mismatches.append(
            f"execution_role must be {EXPECTED_UPSTREAM_EXECUTION_ROLE}"
        )
    if str(upstream_payload.get("data_update_mode") or "") != EXPECTED_UPSTREAM_DATA_UPDATE_MODE:
        mismatches.append(
            f"data_update_mode must be {EXPECTED_UPSTREAM_DATA_UPDATE_MODE}"
        )
    if str(upstream_payload.get("trigger_contract") or "") != EXPECTED_UPSTREAM_TRIGGER_CONTRACT:
        mismatches.append(
            f"trigger_contract must be {EXPECTED_UPSTREAM_TRIGGER_CONTRACT}"
        )
    return len(mismatches) == 0, mismatches


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[local-nankan-followup-oneshot] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[local-nankan-followup-oneshot]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _build_output_payload(
    *,
    observed_at: str,
    status: str,
    current_phase: str,
    recommended_action: str,
    upstream_path: Path,
    upstream_payload: dict[str, Any],
    upstream_observed_at: str | None,
    upstream_age_seconds: int | None,
    max_upstream_age_seconds: int,
    run_id: str,
    artifact_prefix: str,
    log_path: Path,
    wait_cycle_manifest_output: str,
    child_command: list[str],
    child_exit_code: int | None = None,
    wait_cycle_payload: dict[str, Any] | None = None,
    dry_run: bool = False,
    run_bootstrap_on_ready: bool = False,
) -> dict[str, Any]:
    upstream_contract_valid, upstream_contract_errors = _evaluate_upstream_contract(upstream_payload)
    upstream_fresh = bool(
        upstream_path.exists()
        and upstream_payload
        and upstream_contract_valid
        and upstream_age_seconds is not None
        and upstream_age_seconds <= int(max_upstream_age_seconds)
    )
    child_launch_allowed = upstream_fresh and status not in {"not_ready"}
    highlights = [
        f"upstream_exists={str(upstream_path.exists()).lower()}",
        f"upstream_fresh={str(upstream_fresh).lower()}",
        f"status={status}",
        f"current_phase={current_phase}",
    ]
    if child_exit_code is not None:
        highlights.append(f"followup_exit_code={int(child_exit_code)}")
    elif dry_run:
        highlights.append("followup_exit_code=planned")
    else:
        highlights.append("followup_exit_code=blocked")

    payload = {
        "observed_at": observed_at,
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "execution_role": "readiness_followup_gate",
        "trigger_contract": "external_refresh_completed_only",
        "read_order": [
            "status",
            "current_phase",
            "recommended_action",
            "upstream_refresh.upstream_fresh",
            "upstream_refresh.age_seconds",
            "followup_command.exit_code",
            "wait_cycle_followup.status",
        ],
        "highlights": highlights,
        "dry_run": bool(dry_run),
        "run_id": run_id,
        "artifact_prefix": artifact_prefix,
        "run_bootstrap_on_ready": bool(run_bootstrap_on_ready),
        "upstream_fresh": upstream_fresh,
        "child_launch_allowed": child_launch_allowed,
        "upstream_refresh": {
            "manifest": _display_path(upstream_path),
            "exists": upstream_path.exists(),
            "status": upstream_payload.get("status"),
            "current_phase": upstream_payload.get("current_phase"),
            "recommended_action": upstream_payload.get("recommended_action"),
            "execution_role": upstream_payload.get("execution_role"),
            "data_update_mode": upstream_payload.get("data_update_mode"),
            "trigger_contract": upstream_payload.get("trigger_contract"),
            "observed_at": upstream_observed_at,
            "age_seconds": upstream_age_seconds,
            "max_allowed_age_seconds": int(max_upstream_age_seconds),
            "upstream_fresh": upstream_fresh,
            "contract_valid": upstream_contract_valid,
            "contract_errors": upstream_contract_errors,
        },
        "followup_command": {
            "wait_cycle_manifest_output": wait_cycle_manifest_output,
            "log_file": _display_path(log_path),
            "command": child_command,
        },
    }
    if child_exit_code is not None:
        payload["followup_command"]["exit_code"] = int(child_exit_code)
    if wait_cycle_payload is not None:
        payload["wait_cycle_followup"] = wait_cycle_payload
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local Nankan readiness-only oneshot follow-up only after a fresh "
            "upstream refresh artifact has been observed."
        )
    )
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--wait-cycle-script", default=DEFAULT_WAIT_CYCLE_SCRIPT)
    parser.add_argument("--upstream-manifest", default=DEFAULT_UPSTREAM_MANIFEST)
    parser.add_argument("--max-upstream-age-seconds", type=int, default=7200)
    parser.add_argument("--wait-cycle-manifest-output", default=DEFAULT_WAIT_CYCLE_MANIFEST)
    parser.add_argument("--artifact-prefix", default=DEFAULT_ARTIFACT_PREFIX)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--log-dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--run-bootstrap-on-ready", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    upstream_path = _resolve_path(args.upstream_manifest)
    log_dir = _resolve_path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=3, prefix="[local-nankan-followup-oneshot]", logger=log_progress, min_interval_sec=0.0)

    try:
        progress.start(message=f"checking upstream refresh manifest={_display_path(upstream_path)}")
        upstream_payload = _read_json_dict(upstream_path)
        upstream_observed_at = _extract_upstream_observed_at(upstream_path, upstream_payload)
        evaluation_observed_at = utc_now_iso()
        upstream_age_seconds = _compute_age_seconds_against(
            upstream_observed_at,
            reference_at=evaluation_observed_at,
        )
        progress.update(current=1, message=f"upstream exists={upstream_path.exists()} age_seconds={upstream_age_seconds}")

        run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
        log_path = log_dir / f"{args.artifact_prefix}_{run_id}.log"
        child_command = [
            args.python_executable,
            str(_resolve_path(args.wait_cycle_script)),
            "--oneshot",
            "--run-id",
            run_id,
            "--artifact-prefix",
            args.artifact_prefix,
            "--manifest-output",
            args.wait_cycle_manifest_output,
            "--log-file",
            str(log_path),
        ]
        if args.run_bootstrap_on_ready:
            child_command.append("--run-bootstrap-on-ready")

        if not upstream_path.exists() or not upstream_payload:
            payload = _build_output_payload(
                observed_at=evaluation_observed_at,
                status="not_ready",
                current_phase="await_external_refresh_completion",
                recommended_action="produce_fresh_upstream_refresh_artifact_before_followup_oneshot",
                upstream_path=upstream_path,
                upstream_payload=upstream_payload,
                upstream_observed_at=upstream_observed_at,
                upstream_age_seconds=upstream_age_seconds,
                max_upstream_age_seconds=args.max_upstream_age_seconds,
                run_id=run_id,
                artifact_prefix=args.artifact_prefix,
                log_path=log_path,
                wait_cycle_manifest_output=args.wait_cycle_manifest_output,
                child_command=child_command,
                dry_run=args.dry_run,
                run_bootstrap_on_ready=args.run_bootstrap_on_ready,
            )
            write_json(output_path, payload)
            progress.complete(message=f"follow-up blocked output={_display_path(output_path)}")
            return 2

        upstream_contract_valid, _ = _evaluate_upstream_contract(upstream_payload)
        if not upstream_contract_valid:
            payload = _build_output_payload(
                observed_at=evaluation_observed_at,
                status="not_ready",
                current_phase="invalid_upstream_refresh_contract",
                recommended_action="rerun_capture_refresh_with_self_describing_manifest_before_followup_oneshot",
                upstream_path=upstream_path,
                upstream_payload=upstream_payload,
                upstream_observed_at=upstream_observed_at,
                upstream_age_seconds=upstream_age_seconds,
                max_upstream_age_seconds=args.max_upstream_age_seconds,
                run_id=run_id,
                artifact_prefix=args.artifact_prefix,
                log_path=log_path,
                wait_cycle_manifest_output=args.wait_cycle_manifest_output,
                child_command=child_command,
                dry_run=args.dry_run,
                run_bootstrap_on_ready=args.run_bootstrap_on_ready,
            )
            write_json(output_path, payload)
            progress.complete(message=f"follow-up blocked invalid_upstream output={_display_path(output_path)}")
            return 2

        if upstream_age_seconds is None or upstream_age_seconds > int(args.max_upstream_age_seconds):
            payload = _build_output_payload(
                observed_at=evaluation_observed_at,
                status="not_ready",
                current_phase="await_fresh_external_refresh_completion",
                recommended_action="refresh_upstream_artifact_then_rerun_followup_oneshot",
                upstream_path=upstream_path,
                upstream_payload=upstream_payload,
                upstream_observed_at=upstream_observed_at,
                upstream_age_seconds=upstream_age_seconds,
                max_upstream_age_seconds=args.max_upstream_age_seconds,
                run_id=run_id,
                artifact_prefix=args.artifact_prefix,
                log_path=log_path,
                wait_cycle_manifest_output=args.wait_cycle_manifest_output,
                child_command=child_command,
                dry_run=args.dry_run,
                run_bootstrap_on_ready=args.run_bootstrap_on_ready,
            )
            write_json(output_path, payload)
            progress.complete(message=f"follow-up blocked stale_upstream output={_display_path(output_path)}")
            return 2

        if args.dry_run:
            payload = _build_output_payload(
                observed_at=evaluation_observed_at,
                status="dry_run",
                current_phase="followup_plan_ready",
                recommended_action="run_followup_oneshot",
                upstream_path=upstream_path,
                upstream_payload=upstream_payload,
                upstream_observed_at=upstream_observed_at,
                upstream_age_seconds=upstream_age_seconds,
                max_upstream_age_seconds=args.max_upstream_age_seconds,
                run_id=run_id,
                artifact_prefix=args.artifact_prefix,
                log_path=log_path,
                wait_cycle_manifest_output=args.wait_cycle_manifest_output,
                child_command=child_command,
                dry_run=True,
                run_bootstrap_on_ready=args.run_bootstrap_on_ready,
            )
            write_json(output_path, payload)
            progress.complete(message=f"follow-up plan ready output={_display_path(output_path)}")
            return 0

        child_exit_code = _run_command(label="followup_oneshot", command=child_command)
        progress.update(current=2, message=f"followup oneshot exit_code={child_exit_code}")
        wait_cycle_payload = _read_json_dict(_resolve_path(args.wait_cycle_manifest_output))

        payload = _build_output_payload(
            observed_at=evaluation_observed_at,
            status=str(wait_cycle_payload.get("status") or ("completed" if child_exit_code == 0 else "failed")),
            current_phase=str(wait_cycle_payload.get("current_phase") or "followup_oneshot_completed"),
            recommended_action=str(wait_cycle_payload.get("recommended_action") or "inspect_wait_cycle_followup_manifest"),
            upstream_path=upstream_path,
            upstream_payload=upstream_payload,
            upstream_observed_at=upstream_observed_at,
            upstream_age_seconds=upstream_age_seconds,
            max_upstream_age_seconds=args.max_upstream_age_seconds,
            run_id=run_id,
            artifact_prefix=args.artifact_prefix,
            log_path=log_path,
            wait_cycle_manifest_output=args.wait_cycle_manifest_output,
            child_command=child_command,
            child_exit_code=child_exit_code,
            wait_cycle_payload=wait_cycle_payload,
            dry_run=False,
            run_bootstrap_on_ready=args.run_bootstrap_on_ready,
        )
        write_json(output_path, payload)
        progress.complete(message=f"follow-up oneshot output={_display_path(output_path)}")
        return int(child_exit_code)
    except KeyboardInterrupt:
        print("[local-nankan-followup-oneshot] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-followup-oneshot] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())