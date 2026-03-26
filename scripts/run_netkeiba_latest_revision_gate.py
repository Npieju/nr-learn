from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_RACE_RESULT_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2025_missing_result_race_result.json"
DEFAULT_RACE_CARD_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2025_backfill_race_card.json"
DEFAULT_PEDIGREE_MANIFEST = "artifacts/reports/netkeiba_crawl_manifest_2025_pedigree_pedigree.json"
DEFAULT_CRAWL_LOCK_PATH = "artifacts/reports/netkeiba_crawl_manifest_2025_pedigree.json.lock"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-latest-gate {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _safe_write_manifest(path: Path, payload: dict[str, object]) -> None:
    if path.exists() and path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _run_command(command: list[str], *, label: str) -> dict[str, object]:
    started_at = utc_now_iso()
    printable = shlex.join(command)
    print(f"[netkeiba-latest-gate] running {label}: {printable}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    finished_at = utc_now_iso()
    return {
        "label": label,
        "command": command,
        "status": "completed" if result.returncode == 0 else "failed",
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _collect_revision_gate_artifacts(revision_slug: str) -> dict[str, object]:
    report_dir = ROOT / "artifacts" / "reports"
    revision_manifest_path = report_dir / f"revision_gate_{revision_slug}.json"
    promotion_report_path = report_dir / f"promotion_gate_{revision_slug}.json"

    payload: dict[str, object] = {
        "revision_manifest": artifact_display_path(revision_manifest_path, workspace_root=ROOT),
        "promotion_report": artifact_display_path(promotion_report_path, workspace_root=ROOT),
        "revision_manifest_present": revision_manifest_path.exists(),
        "promotion_report_present": promotion_report_path.exists(),
    }

    if revision_manifest_path.exists():
        revision_manifest = read_json(revision_manifest_path)
        if isinstance(revision_manifest, dict):
            payload["revision_manifest_payload"] = revision_manifest

    if promotion_report_path.exists():
        promotion_report = read_json(promotion_report_path)
        if isinstance(promotion_report, dict):
            payload["promotion_report_payload"] = promotion_report

    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default="current_best_eval_2025_latest")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--evaluate-max-rows", type=int, default=200000)
    parser.add_argument("--evaluate-pre-feature-max-rows", type=int, default=300000)
    parser.add_argument("--evaluate-wf-mode", choices=["off", "fast", "full"], default="full")
    parser.add_argument("--evaluate-wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--promotion-min-feasible-folds", type=int, default=1)
    parser.add_argument("--snapshot-output", default="artifacts/reports/netkeiba_coverage_snapshot_2025_latest.json")
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--race-result-manifest", default=DEFAULT_RACE_RESULT_MANIFEST)
    parser.add_argument("--race-card-manifest", default=DEFAULT_RACE_CARD_MANIFEST)
    parser.add_argument("--pedigree-manifest", default=DEFAULT_PEDIGREE_MANIFEST)
    parser.add_argument("--crawl-lock-path", default=DEFAULT_CRAWL_LOCK_PATH)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_slug = args.revision or f"latest_{time.strftime('%Y%m%d_%H%M%S')}"
    manifest_path = _resolve_path(
        args.manifest_output or f"artifacts/reports/netkeiba_latest_revision_gate_{revision_slug}.json"
    )
    snapshot_path = _resolve_path(args.snapshot_output)

    payload: dict[str, object] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "waiting_readiness",
        "profile": args.profile,
        "revision": revision_slug,
        "evaluation": {
            "max_rows": int(args.evaluate_max_rows),
            "pre_feature_max_rows": int(args.evaluate_pre_feature_max_rows)
            if args.evaluate_pre_feature_max_rows is not None
            else None,
            "wf_mode": args.evaluate_wf_mode,
            "wf_scheme": args.evaluate_wf_scheme,
        },
        "snapshot_output": artifact_display_path(snapshot_path, workspace_root=ROOT),
        "target_manifests": {
            "race_result": args.race_result_manifest,
            "race_card": args.race_card_manifest,
            "pedigree": args.pedigree_manifest,
        },
        "crawl_lock_path": args.crawl_lock_path,
        "dry_run": bool(args.dry_run),
    }

    try:
        artifact_ensure_output_file_path(manifest_path, label="manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(snapshot_path, label="snapshot output", workspace_root=ROOT)
        _safe_write_manifest(manifest_path, payload)

        progress = ProgressBar(total=3, prefix="[netkeiba-latest-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"profile={args.profile} revision={revision_slug}")

        snapshot_command = [
            sys.executable,
            str(ROOT / "scripts/run_netkeiba_coverage_snapshot.py"),
            "--config",
            "configs/data_2025_latest.yaml",
            "--tail-rows",
            str(args.tail_rows),
            "--output",
            artifact_display_path(snapshot_path, workspace_root=ROOT),
            "--race-result-manifest",
            args.race_result_manifest,
            "--race-card-manifest",
            args.race_card_manifest,
            "--pedigree-manifest",
            args.pedigree_manifest,
            "--crawl-lock-path",
            args.crawl_lock_path,
        ]
        deadline = time.monotonic() + args.wait_timeout_seconds if args.wait_timeout_seconds > 0 else None

        while True:
            with Heartbeat("[netkeiba-latest-gate]", "running readiness snapshot", logger=log_progress):
                snapshot_result = _run_command(snapshot_command, label="snapshot")
            payload["snapshot"] = snapshot_result
            if int(snapshot_result["exit_code"]) != 0:
                payload["status"] = "snapshot_failed"
                payload["finished_at"] = utc_now_iso()
                _safe_write_manifest(manifest_path, payload)
                return int(snapshot_result["exit_code"]) or 1

            snapshot_payload = read_json(snapshot_path)
            readiness = dict(snapshot_payload.get("readiness", {})) if isinstance(snapshot_payload, dict) else {}
            payload["readiness"] = readiness
            payload["target_states"] = snapshot_payload.get("target_states", {}) if isinstance(snapshot_payload, dict) else {}
            payload["alignment"] = snapshot_payload.get("alignment", {}) if isinstance(snapshot_payload, dict) else {}
            _safe_write_manifest(manifest_path, payload)

            if bool(readiness.get("benchmark_rerun_ready", False)):
                progress.update(current=1, message="snapshot readiness confirmed")
                break

            if args.wait_timeout_seconds == 0:
                payload["status"] = "not_ready"
                payload["finished_at"] = utc_now_iso()
                _safe_write_manifest(manifest_path, payload)
                print(
                    "[netkeiba-latest-gate] "
                    f"not ready: action={readiness.get('recommended_action')} reasons={readiness.get('reasons')}",
                    flush=True,
                )
                return 2

            if deadline is not None and time.monotonic() >= deadline:
                payload["status"] = "timeout"
                payload["finished_at"] = utc_now_iso()
                _safe_write_manifest(manifest_path, payload)
                print("[netkeiba-latest-gate] timeout waiting for readiness", flush=True)
                return 124

            print(
                "[netkeiba-latest-gate] "
                f"waiting for readiness action={readiness.get('recommended_action')} reasons={readiness.get('reasons')}",
                flush=True,
            )
            time.sleep(max(args.poll_seconds, 1))

        revision_command = [
            sys.executable,
            str(ROOT / "scripts/run_revision_gate.py"),
            "--profile",
            args.profile,
            "--revision",
            revision_slug,
            "--evaluate-max-rows",
            str(args.evaluate_max_rows),
            "--evaluate-wf-mode",
            args.evaluate_wf_mode,
            "--evaluate-wf-scheme",
            args.evaluate_wf_scheme,
            "--promotion-min-feasible-folds",
            str(args.promotion_min_feasible_folds),
        ]
        if args.evaluate_pre_feature_max_rows is not None:
            revision_command.extend(
                ["--evaluate-pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)]
            )
        if args.dry_run:
            revision_command.append("--dry-run")

        payload["status"] = "running_revision_gate"
        payload["revision_gate_command"] = revision_command
        payload["revision_gate_artifacts"] = _collect_revision_gate_artifacts(revision_slug)
        _safe_write_manifest(manifest_path, payload)

        with Heartbeat("[netkeiba-latest-gate]", "running revision gate", logger=log_progress):
            revision_result = _run_command(revision_command, label="revision_gate")
        payload["revision_gate"] = revision_result
        payload["revision_gate_artifacts"] = _collect_revision_gate_artifacts(revision_slug)
        if int(revision_result["exit_code"]) != 0:
            revision_gate_payload = payload.get("revision_gate_artifacts", {})
            revision_manifest_payload = {}
            if isinstance(revision_gate_payload, dict):
                candidate = revision_gate_payload.get("revision_manifest_payload")
                if isinstance(candidate, dict):
                    revision_manifest_payload = candidate

            if revision_manifest_payload.get("status") == "block":
                payload["status"] = "revision_gate_blocked"
                payload["decision"] = revision_manifest_payload.get("decision")
            else:
                payload["status"] = "revision_gate_failed"
            payload["finished_at"] = utc_now_iso()
            _safe_write_manifest(manifest_path, payload)
            return int(revision_result["exit_code"]) or 1

        payload["status"] = "completed"
        payload["decision"] = "promote"
        payload["finished_at"] = utc_now_iso()
        _safe_write_manifest(manifest_path, payload)
        progress.complete(message="latest revision gate completed")
        print("[netkeiba-latest-gate] completed", flush=True)
        return 0
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at"] = utc_now_iso()
        _safe_write_manifest(manifest_path, payload)
        print("[netkeiba-latest-gate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-latest-gate] failed: {error}")
        return 1
    except Exception as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-latest-gate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())