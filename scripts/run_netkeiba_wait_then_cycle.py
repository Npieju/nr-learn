import argparse
import json
import os
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

from racing_ml.common.artifacts import absolutize_path, read_json, relativize_path, utc_now_iso, write_json
from racing_ml.common.config import load_yaml


def _resolve_path(path_text: str | Path) -> Path:
    return absolutize_path(path_text, ROOT)


def _build_lock_path(base_path: Path) -> Path:
    if not base_path.suffix:
        return base_path.parent / f"{base_path.name}.lock"
    return base_path.with_name(f"{base_path.stem}{base_path.suffix}.lock")


def _read_lock_payload(lock_path: Path) -> dict[str, Any]:
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


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


def _read_manifest_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _summarize_backfill_state(path: Path) -> dict[str, Any]:
    payload = _read_manifest_dict(path)
    cycles = payload.get("cycles")
    last_cycle = cycles[-1] if isinstance(cycles, list) and cycles else {}
    last_post_cycle = last_cycle.get("post_cycle_command") if isinstance(last_cycle, dict) else {}
    return {
        "path": relativize_path(path, ROOT),
        "exists": path.exists(),
        "stopped_reason": payload.get("stopped_reason"),
        "completed_cycles": payload.get("completed_cycles"),
        "last_cycle": last_cycle.get("cycle") if isinstance(last_cycle, dict) else None,
        "last_cycle_finished_at": last_cycle.get("finished_at") if isinstance(last_cycle, dict) else None,
        "last_post_cycle_status": last_post_cycle.get("status") if isinstance(last_post_cycle, dict) else None,
        "last_post_cycle_exit_code": last_post_cycle.get("exit_code") if isinstance(last_post_cycle, dict) else None,
    }


def _summarize_pedigree_state(path: Path) -> dict[str, Any]:
    payload = _read_manifest_dict(path)
    return {
        "path": relativize_path(path, ROOT),
        "exists": path.exists(),
        "status": payload.get("status"),
        "started_at": payload.get("started_at"),
        "processed_ids": payload.get("processed_ids"),
        "requested_ids": payload.get("requested_ids"),
        "rows_written": payload.get("rows_written"),
        "existing_rows_merged": payload.get("existing_rows_merged"),
    }


def _build_gate_command(args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_netkeiba_benchmark_gate.py"),
        "--data-config",
        args.data_config,
        "--model-config",
        args.model_config,
        "--feature-config",
        args.feature_config,
        "--tail-rows",
        str(args.tail_rows),
        "--snapshot-output",
        args.snapshot_output,
        "--manifest-output",
        args.gate_manifest_output,
        "--max-rows",
        str(args.max_rows),
        "--wf-mode",
        args.wf_mode,
        "--wf-scheme",
        args.wf_scheme,
    ]
    if args.skip_train:
        command.append("--skip-train")
    if args.skip_evaluate:
        command.append("--skip-evaluate")
    return command


def _build_backfill_command(args: argparse.Namespace, post_cycle_command: str) -> list[str]:
    command = [
        sys.executable,
        str(ROOT / "scripts/run_backfill_netkeiba.py"),
        "--data-config",
        args.data_config,
        "--crawl-config",
        args.crawl_config,
        "--date-order",
        args.date_order,
        "--race-batch-size",
        str(args.race_batch_size),
        "--pedigree-batch-size",
        str(args.pedigree_batch_size),
        "--max-cycles",
        "1",
        "--post-cycle-command",
        post_cycle_command,
        "--stop-on-post-cycle-failure",
        "--manifest-file",
        args.backfill_manifest_output,
    ]
    if args.start_date:
        command.extend(["--start-date", args.start_date])
    if args.end_date:
        command.extend(["--end-date", args.end_date])
    return command


def _run_command(command: list[str], *, label: str) -> dict[str, Any]:
    started_at = utc_now_iso()
    printable = shlex.join(command)
    print(f"[netkeiba-wait-cycle] running {label}: {printable}", flush=True)
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


def _build_wait_observation(
    *,
    lock_path: Path,
    active_backfill_manifest_path: Path,
    active_pedigree_manifest_path: Path,
    wait_started_at: float,
) -> dict[str, Any]:
    now = time.monotonic()
    return {
        "observed_at": utc_now_iso(),
        "elapsed_seconds": int(now - wait_started_at),
        "lock_exists": lock_path.exists(),
        "lock": _read_lock_payload(lock_path),
        "active_backfill": _summarize_backfill_state(active_backfill_manifest_path),
        "active_pedigree": _summarize_pedigree_state(active_pedigree_manifest_path),
    }


def _write_wait_manifest(path: Path, payload: dict[str, Any]) -> None:
    write_json(path, payload)


def _extract_last_post_cycle_exit_code(backfill_manifest_path: Path) -> int | None:
    payload = _read_manifest_dict(backfill_manifest_path)
    cycles = payload.get("cycles")
    if not isinstance(cycles, list) or not cycles:
        return None
    last_cycle = cycles[-1]
    if not isinstance(last_cycle, dict):
        return None
    post_cycle = last_cycle.get("post_cycle_command")
    if not isinstance(post_cycle, dict):
        return None
    exit_code = post_cycle.get("exit_code")
    return int(exit_code) if exit_code is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--crawl-config", default="configs/crawl_netkeiba_template.yaml")
    parser.add_argument("--model-config", default="configs/model_catboost_fundamental_enriched.yaml")
    parser.add_argument("--feature-config", default="configs/features_catboost_fundamental_enriched.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="desc")
    parser.add_argument("--race-batch-size", type=int, default=100)
    parser.add_argument("--pedigree-batch-size", type=int, default=500)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--wait-timeout-seconds", type=int, default=0)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument(
        "--wait-manifest-output",
        default="artifacts/reports/netkeiba_wait_then_cycle_manifest.json",
    )
    parser.add_argument(
        "--backfill-manifest-output",
        default="artifacts/reports/netkeiba_backfill_handoff_manifest.json",
    )
    parser.add_argument(
        "--snapshot-output",
        default="artifacts/reports/netkeiba_coverage_snapshot.json",
    )
    parser.add_argument(
        "--gate-manifest-output",
        default="artifacts/reports/netkeiba_benchmark_gate_manifest.json",
    )
    parser.add_argument(
        "--observe-backfill-manifest",
        default="artifacts/reports/netkeiba_backfill_manifest.json",
    )
    parser.add_argument(
        "--observe-pedigree-manifest",
        default="artifacts/reports/netkeiba_crawl_manifest_pedigree.json",
    )
    args = parser.parse_args()

    wait_manifest_path = _resolve_path(args.wait_manifest_output)
    backfill_manifest_path = _resolve_path(args.backfill_manifest_output)
    gate_manifest_path = _resolve_path(args.gate_manifest_output)
    snapshot_path = _resolve_path(args.snapshot_output)
    active_backfill_manifest_path = _resolve_path(args.observe_backfill_manifest)
    active_pedigree_manifest_path = _resolve_path(args.observe_pedigree_manifest)

    crawl_config = load_yaml(ROOT / args.crawl_config)
    crawl_cfg = crawl_config.get("crawl", crawl_config)
    crawl_manifest_path = _resolve_path(crawl_cfg.get("manifest_file", "artifacts/reports/netkeiba_crawl_manifest.json"))
    lock_path = _build_lock_path(crawl_manifest_path)

    payload: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "waiting",
        "configs": {
            "data_config": args.data_config,
            "crawl_config": args.crawl_config,
            "model_config": args.model_config,
            "feature_config": args.feature_config,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "date_order": args.date_order,
            "race_batch_size": int(args.race_batch_size),
            "pedigree_batch_size": int(args.pedigree_batch_size),
            "poll_seconds": int(args.poll_seconds),
            "wait_timeout_seconds": int(args.wait_timeout_seconds),
            "tail_rows": int(args.tail_rows),
            "max_rows": int(args.max_rows),
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "skip_train": bool(args.skip_train),
            "skip_evaluate": bool(args.skip_evaluate),
        },
        "paths": {
            "lock_file": relativize_path(lock_path, ROOT),
            "crawl_manifest": relativize_path(crawl_manifest_path, ROOT),
            "active_backfill_manifest": relativize_path(active_backfill_manifest_path, ROOT),
            "active_pedigree_manifest": relativize_path(active_pedigree_manifest_path, ROOT),
            "backfill_manifest": relativize_path(backfill_manifest_path, ROOT),
            "snapshot_output": relativize_path(snapshot_path, ROOT),
            "gate_manifest": relativize_path(gate_manifest_path, ROOT),
        },
    }
    _write_wait_manifest(wait_manifest_path, payload)

    try:
        wait_started_at = time.monotonic()
        deadline = wait_started_at + args.wait_timeout_seconds if args.wait_timeout_seconds > 0 else None

        while True:
            observation = _build_wait_observation(
                lock_path=lock_path,
                active_backfill_manifest_path=active_backfill_manifest_path,
                active_pedigree_manifest_path=active_pedigree_manifest_path,
                wait_started_at=wait_started_at,
            )
            payload["wait"] = observation
            _write_wait_manifest(wait_manifest_path, payload)

            if not lock_path.exists():
                break

            lock_payload = observation.get("lock")
            pid = int(lock_payload.get("pid", 0) or 0) if isinstance(lock_payload, dict) else 0
            if pid > 0 and not _pid_is_running(pid):
                print(
                    f"[netkeiba-wait-cycle] removing stale lock: {lock_path} pid={pid}",
                    flush=True,
                )
                try:
                    lock_path.unlink()
                    continue
                except OSError:
                    pass

            pedigree_state = observation.get("active_pedigree") if isinstance(observation, dict) else {}
            processed_ids = pedigree_state.get("processed_ids") if isinstance(pedigree_state, dict) else None
            requested_ids = pedigree_state.get("requested_ids") if isinstance(pedigree_state, dict) else None
            print(
                "[netkeiba-wait-cycle] "
                f"waiting for lock release pid={pid or 'unknown'} "
                f"pedigree={processed_ids}/{requested_ids}",
                flush=True,
            )

            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for netkeiba lock: {lock_path}")
            time.sleep(max(args.poll_seconds, 1))

        gate_command = _build_gate_command(args)
        post_cycle_command = shlex.join(gate_command)
        backfill_command = _build_backfill_command(args, post_cycle_command)

        payload["status"] = "running_backfill"
        payload["gate_command"] = gate_command
        payload["backfill_command"] = backfill_command
        _write_wait_manifest(wait_manifest_path, payload)

        backfill_result = _run_command(backfill_command, label="handoff_backfill_cycle")
        payload["backfill"] = backfill_result
        payload["backfill_summary"] = _summarize_backfill_state(backfill_manifest_path)
        payload["gate_summary"] = _read_manifest_dict(gate_manifest_path)

        if int(backfill_result.get("exit_code", 1)) != 0:
            payload["status"] = "backfill_failed"
            payload["finished_at"] = utc_now_iso()
            _write_wait_manifest(wait_manifest_path, payload)
            return int(backfill_result.get("exit_code", 1)) or 1

        post_cycle_exit_code = _extract_last_post_cycle_exit_code(backfill_manifest_path)
        if post_cycle_exit_code is not None and post_cycle_exit_code != 0:
            gate_status = None
            if isinstance(payload.get("gate_summary"), dict):
                gate_status = payload["gate_summary"].get("status")
            payload["status"] = "gate_failed" if gate_status != "not_ready" else "gate_not_ready"
            payload["finished_at"] = utc_now_iso()
            _write_wait_manifest(wait_manifest_path, payload)
            return int(post_cycle_exit_code)

        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        _write_wait_manifest(wait_manifest_path, payload)
        print("[netkeiba-wait-cycle] completed", flush=True)
        return 0
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at"] = utc_now_iso()
        _write_wait_manifest(wait_manifest_path, payload)
        print("[netkeiba-wait-cycle] interrupted by user")
        return 130
    except TimeoutError as error:
        payload["status"] = "timeout"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _write_wait_manifest(wait_manifest_path, payload)
        print(f"[netkeiba-wait-cycle] timeout: {error}")
        return 124
    except Exception as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _write_wait_manifest(wait_manifest_path, payload)
        print(f"[netkeiba-wait-cycle] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())