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


DEFAULT_UNIVERSE = "jra"
DEFAULT_SOURCE_SCOPE = "netkeiba"
DEFAULT_SCHEMA_VERSION = "netkeiba.benchmark_gate.v1"
DEFAULT_RACE_RESULT_PATH = "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
DEFAULT_RACE_CARD_PATH = "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"
DEFAULT_PEDIGREE_PATH = "data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv"


def _set_step(payload: dict[str, object], step_name: str) -> None:
    payload["completed_step"] = step_name


def _require_text(value: str, *, field_name: str, error_code: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{error_code}:{field_name} must not be empty")
    return normalized


def _set_failure(
    payload: dict[str, object],
    *,
    status: str,
    error_code: str,
    error_message: str,
    recommended_action: str,
) -> None:
    payload["status"] = status
    payload["finished_at"] = utc_now_iso()
    payload["error_code"] = error_code
    payload["error_message"] = error_message
    payload["recommended_action"] = recommended_action


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
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--baseline-reference", default=None)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest_output)
    snapshot_path = _resolve_path(args.snapshot_output)
    universe = DEFAULT_UNIVERSE
    source_scope = DEFAULT_SOURCE_SCOPE
    schema_version = DEFAULT_SCHEMA_VERSION
    baseline_reference = None
    payload: dict[str, object] = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "artifact_type": "benchmark_gate_manifest",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "completed_step": "init",
        "universe": DEFAULT_UNIVERSE,
        "source_scope": DEFAULT_SOURCE_SCOPE,
        "baseline_reference": None,
        "error_code": None,
        "error_message": None,
        "recommended_action": None,
        "configs": {
            "data_config": args.data_config,
            "model_config": args.model_config,
            "feature_config": args.feature_config,
            "tail_rows": int(args.tail_rows),
            "max_rows": int(args.max_rows),
            "pre_feature_max_rows": int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "skip_train": bool(args.skip_train),
            "skip_evaluate": bool(args.skip_evaluate),
            "race_result_path": args.race_result_path,
            "race_card_path": args.race_card_path,
            "pedigree_path": args.pedigree_path,
        },
    }

    try:
        universe = _require_text(args.universe, field_name="universe", error_code="missing_universe")
        source_scope = _require_text(args.source_scope, field_name="source_scope", error_code="missing_source_scope")
        schema_version = _require_text(args.schema_version, field_name="schema_version", error_code="missing_schema_version")
        baseline_reference = str(args.baseline_reference).strip() or None
        payload.update(
            {
                "schema_version": schema_version,
                "universe": universe,
                "source_scope": source_scope,
                "baseline_reference": baseline_reference,
            }
        )
        _set_step(payload, "init_manifest")
        artifact_ensure_output_file_path(manifest_path, label="manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(snapshot_path, label="snapshot output", workspace_root=ROOT)
        _safe_write_manifest(manifest_path, payload)

        total_steps = 3 + int(not args.skip_train) + int(not args.skip_evaluate)
        progress = ProgressBar(total=total_steps, prefix="[netkeiba-benchmark-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="gate manifest initialized")
        _set_step(payload, "run_snapshot")
        snapshot_command = [
            sys.executable,
            str(ROOT / "scripts/run_netkeiba_coverage_snapshot.py"),
            "--config",
            args.data_config,
            "--tail-rows",
            str(args.tail_rows),
            "--output",
            str(snapshot_path),
            "--universe",
            universe,
            "--source-scope",
            source_scope,
            "--schema-version",
            "netkeiba.coverage_snapshot.v1",
            "--race-result-path",
            args.race_result_path,
            "--race-card-path",
            args.race_card_path,
            "--pedigree-path",
            args.pedigree_path,
        ]
        if baseline_reference is not None:
            snapshot_command.extend(["--baseline-reference", baseline_reference])
        with Heartbeat("[netkeiba-benchmark-gate]", "running snapshot", logger=log_progress):
            snapshot_result = _run_command(snapshot_command, cwd=ROOT, label="snapshot")
        payload["snapshot"] = snapshot_result
        if int(snapshot_result["exit_code"]) != 0:
            _set_failure(
                payload,
                status="snapshot_failed",
                error_code="snapshot_failed",
                error_message="snapshot command returned non-zero exit code",
                recommended_action="inspect_snapshot_manifest",
            )
            _safe_write_manifest(manifest_path, payload)
            return int(snapshot_result["exit_code"]) or 1
        progress.update(message="snapshot completed")

        snapshot_payload = read_json(snapshot_path)
        readiness = dict(snapshot_payload.get("readiness", {})) if isinstance(snapshot_payload, dict) else {}
        latest_tail = dict(snapshot_payload.get("coverage", {}).get("latest_tail", {})) if isinstance(snapshot_payload, dict) else {}
        integrity_summary = dict(snapshot_payload.get("integrity_summary", {})) if isinstance(snapshot_payload, dict) else {}
        payload["readiness"] = readiness
        payload["coverage_summary"] = {
            "latest_tail_horse_key_ratio": latest_tail.get("horse_key", {}).get("non_null_ratio"),
            "latest_tail_breeder_ratio": latest_tail.get("breeder_name", {}).get("non_null_ratio"),
            "latest_tail_sire_ratio": latest_tail.get("sire_name", {}).get("non_null_ratio"),
        }
        payload["integrity_summary"] = integrity_summary

        _set_step(payload, "validate_readiness")
        if not bool(readiness.get("benchmark_rerun_ready", False)):
            _set_failure(
                payload,
                status="not_ready",
                error_code="snapshot_not_ready",
                error_message="snapshot readiness did not reach benchmark_rerun_ready",
                recommended_action=str(readiness.get("recommended_action") or "inspect_snapshot_readiness"),
            )
            _safe_write_manifest(manifest_path, payload)
            progress.complete(message="snapshot says not ready")
            print(
                "[netkeiba-benchmark-gate] "
                f"not ready: action={readiness.get('recommended_action')} reasons={readiness.get('reasons')}",
                flush=True,
            )
            return 2

        if not args.skip_train:
            _set_step(payload, "run_train")
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
                _set_failure(
                    payload,
                    status="train_failed",
                    error_code="train_failed",
                    error_message="train command returned non-zero exit code",
                    recommended_action="inspect_train_logs",
                )
                _safe_write_manifest(manifest_path, payload)
                return int(train_result["exit_code"]) or 1
            progress.update(message="train completed")

        if not args.skip_evaluate:
            _set_step(payload, "run_evaluate")
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
            if args.pre_feature_max_rows is not None:
                evaluate_command.extend(["--pre-feature-max-rows", str(args.pre_feature_max_rows)])
            with Heartbeat("[netkeiba-benchmark-gate]", "running evaluate", logger=log_progress):
                evaluate_result = _run_command(evaluate_command, cwd=ROOT, label="evaluate")
            payload["evaluate"] = evaluate_result
            if int(evaluate_result["exit_code"]) != 0:
                _set_failure(
                    payload,
                    status="evaluate_failed",
                    error_code="evaluate_failed",
                    error_message="evaluate command returned non-zero exit code",
                    recommended_action="inspect_evaluate_logs",
                )
                _safe_write_manifest(manifest_path, payload)
                return int(evaluate_result["exit_code"]) or 1
            progress.update(message="evaluate completed")

        _set_step(payload, "write_manifest")
        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        payload["error_code"] = None
        payload["error_message"] = None
        payload["recommended_action"] = None
        with Heartbeat("[netkeiba-benchmark-gate]", "writing gate manifest", logger=log_progress):
            _safe_write_manifest(manifest_path, payload)
        _set_step(payload, "completed")
        _safe_write_manifest(manifest_path, payload)
        progress.complete(message="benchmark gate completed")
        print("[netkeiba-benchmark-gate] completed", flush=True)
        return 0
    except KeyboardInterrupt:
        _set_failure(
            payload,
            status="interrupted",
            error_code="interrupted",
            error_message="interrupted by user",
            recommended_action="rerun_benchmark_gate",
        )
        _safe_write_manifest(manifest_path, payload)
        print("[netkeiba-benchmark-gate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        error_text = str(error)
        error_code, _, message = error_text.partition(":")
        if error_code not in {
            "missing_universe",
            "missing_source_scope",
            "missing_schema_version",
        }:
            error_code = "invalid_output_path" if isinstance(error, IsADirectoryError) else "gate_failed"
            message = error_text
        _set_failure(
            payload,
            status="failed",
            error_code=error_code,
            error_message=message or error_text,
            recommended_action="inspect_gate_inputs",
        )
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-benchmark-gate] failed: {error}")
        return 1
    except Exception as error:
        _set_failure(
            payload,
            status="failed",
            error_code="gate_failed",
            error_message=str(error),
            recommended_action="inspect_gate_traceback",
        )
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-benchmark-gate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())