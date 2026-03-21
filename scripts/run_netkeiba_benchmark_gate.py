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


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-benchmark-gate {now}] {message}", flush=True)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _safe_write_manifest(path: Path, payload: dict[str, object]) -> None:
    if path.exists() and path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _run_command(command: list[str], *, cwd: Path, label: str) -> dict[str, object]:
    started_at = utc_now_iso()
    printable = shlex.join(command)
    print(f"[netkeiba-benchmark-gate] running {label}: {printable}", flush=True)
    result = subprocess.run(command, cwd=cwd, check=False)
    finished_at = utc_now_iso()
    return {
        "label": label,
        "command": command,
        "status": "completed" if result.returncode == 0 else "failed",
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--snapshot-output", default="artifacts/reports/netkeiba_coverage_snapshot.json")
    parser.add_argument("--manifest-output", default="artifacts/reports/netkeiba_benchmark_gate_manifest.json")
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest_output)
    snapshot_path = _resolve_path(args.snapshot_output)
    payload: dict[str, object] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "configs": {
            "data_config": args.data_config,
            "model_config": args.model_config,
            "feature_config": args.feature_config,
            "tail_rows": int(args.tail_rows),
            "max_rows": int(args.max_rows),
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "skip_train": bool(args.skip_train),
            "skip_evaluate": bool(args.skip_evaluate),
        },
    }

    try:
        artifact_ensure_output_file_path(manifest_path, label="manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(snapshot_path, label="snapshot output", workspace_root=ROOT)
        _safe_write_manifest(manifest_path, payload)

        total_steps = 3 + int(not args.skip_train) + int(not args.skip_evaluate)
        progress = ProgressBar(total=total_steps, prefix="[netkeiba-benchmark-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="gate manifest initialized")
        snapshot_command = [
            sys.executable,
            str(ROOT / "scripts/run_netkeiba_coverage_snapshot.py"),
            "--config",
            args.data_config,
            "--tail-rows",
            str(args.tail_rows),
            "--output",
            str(snapshot_path),
        ]
        with Heartbeat("[netkeiba-benchmark-gate]", "running snapshot", logger=log_progress):
            snapshot_result = _run_command(snapshot_command, cwd=ROOT, label="snapshot")
        payload["snapshot"] = snapshot_result
        if int(snapshot_result["exit_code"]) != 0:
            payload["status"] = "snapshot_failed"
            payload["finished_at"] = utc_now_iso()
            _safe_write_manifest(manifest_path, payload)
            return int(snapshot_result["exit_code"]) or 1
        progress.update(message="snapshot completed")

        snapshot_payload = read_json(snapshot_path)
        readiness = dict(snapshot_payload.get("readiness", {})) if isinstance(snapshot_payload, dict) else {}
        latest_tail = dict(snapshot_payload.get("coverage", {}).get("latest_tail", {})) if isinstance(snapshot_payload, dict) else {}
        payload["readiness"] = readiness
        payload["coverage_summary"] = {
            "latest_tail_horse_key_ratio": latest_tail.get("horse_key", {}).get("non_null_ratio"),
            "latest_tail_breeder_ratio": latest_tail.get("breeder_name", {}).get("non_null_ratio"),
            "latest_tail_sire_ratio": latest_tail.get("sire_name", {}).get("non_null_ratio"),
        }

        if not bool(readiness.get("benchmark_rerun_ready", False)):
            payload["status"] = "not_ready"
            payload["finished_at"] = utc_now_iso()
            _safe_write_manifest(manifest_path, payload)
            progress.complete(message="snapshot says not ready")
            print(
                "[netkeiba-benchmark-gate] "
                f"not ready: action={readiness.get('recommended_action')} reasons={readiness.get('reasons')}",
                flush=True,
            )
            return 2

        if not args.skip_train:
            train_command = [
                sys.executable,
                str(ROOT / "scripts/run_train.py"),
                "--config",
                args.model_config,
                "--data-config",
                args.data_config,
                "--feature-config",
                args.feature_config,
            ]
            with Heartbeat("[netkeiba-benchmark-gate]", "running train", logger=log_progress):
                train_result = _run_command(train_command, cwd=ROOT, label="train")
            payload["train"] = train_result
            if int(train_result["exit_code"]) != 0:
                payload["status"] = "train_failed"
                payload["finished_at"] = utc_now_iso()
                _safe_write_manifest(manifest_path, payload)
                return int(train_result["exit_code"]) or 1
            progress.update(message="train completed")

        if not args.skip_evaluate:
            evaluate_command = [
                sys.executable,
                str(ROOT / "scripts/run_evaluate.py"),
                "--config",
                args.model_config,
                "--data-config",
                args.data_config,
                "--feature-config",
                args.feature_config,
                "--max-rows",
                str(args.max_rows),
                "--wf-mode",
                args.wf_mode,
                "--wf-scheme",
                args.wf_scheme,
            ]
            with Heartbeat("[netkeiba-benchmark-gate]", "running evaluate", logger=log_progress):
                evaluate_result = _run_command(evaluate_command, cwd=ROOT, label="evaluate")
            payload["evaluate"] = evaluate_result
            if int(evaluate_result["exit_code"]) != 0:
                payload["status"] = "evaluate_failed"
                payload["finished_at"] = utc_now_iso()
                _safe_write_manifest(manifest_path, payload)
                return int(evaluate_result["exit_code"]) or 1
            progress.update(message="evaluate completed")

        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        with Heartbeat("[netkeiba-benchmark-gate]", "writing gate manifest", logger=log_progress):
            _safe_write_manifest(manifest_path, payload)
        progress.complete(message="benchmark gate completed")
        print("[netkeiba-benchmark-gate] completed", flush=True)
        return 0
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at"] = utc_now_iso()
        _safe_write_manifest(manifest_path, payload)
        print("[netkeiba-benchmark-gate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-benchmark-gate] failed: {error}")
        return 1
    except Exception as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-benchmark-gate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())