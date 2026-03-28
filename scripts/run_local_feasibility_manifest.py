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


DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_local_baseline.yaml"
DEFAULT_FEATURE_CONFIG = "configs/features_local_baseline.yaml"
DEFAULT_SNAPSHOT_OUTPUT = "artifacts/reports/coverage_snapshot_local_nankan.json"
DEFAULT_VALIDATION_OUTPUT = "artifacts/reports/data_source_validation_local_nankan.json"
DEFAULT_FEATURE_GAP_SUMMARY_OUTPUT = "artifacts/reports/feature_gap_summary_local_nankan.json"
DEFAULT_FEATURE_GAP_FEATURE_OUTPUT = "artifacts/reports/feature_gap_feature_coverage_local_nankan.csv"
DEFAULT_FEATURE_GAP_RAW_OUTPUT = "artifacts/reports/feature_gap_raw_column_coverage_local_nankan.csv"
DEFAULT_EVALUATION_POINTER_OUTPUT = "artifacts/reports/evaluation_local_nankan_pointer.json"
DEFAULT_MANIFEST_OUTPUT = "artifacts/reports/local_feasibility_manifest_local_nankan.json"
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-feasibility {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _safe_write_manifest(path: Path, payload: dict[str, object]) -> None:
    if path.exists() and path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _set_step(payload: dict[str, object], step_name: str) -> None:
    payload["completed_step"] = step_name


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


def _run_command(command: list[str], *, label: str) -> dict[str, object]:
    started_at = utc_now_iso()
    printable = shlex.join(command)
    print(f"[local-feasibility] running {label}: {printable}", flush=True)
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


def _read_optional_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--model-artifact-suffix", default=None)
    parser.add_argument("--snapshot-output", default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--validation-output", default=DEFAULT_VALIDATION_OUTPUT)
    parser.add_argument("--feature-gap-summary-output", default=DEFAULT_FEATURE_GAP_SUMMARY_OUTPUT)
    parser.add_argument("--feature-gap-feature-output", default=DEFAULT_FEATURE_GAP_FEATURE_OUTPUT)
    parser.add_argument("--feature-gap-raw-output", default=DEFAULT_FEATURE_GAP_RAW_OUTPUT)
    parser.add_argument("--evaluation-output", default=DEFAULT_EVALUATION_POINTER_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--skip-feature-gap", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest_output)
    snapshot_path = _resolve_path(args.snapshot_output)
    validation_path = _resolve_path(args.validation_output)
    feature_gap_summary_path = _resolve_path(args.feature_gap_summary_output)
    feature_gap_feature_path = _resolve_path(args.feature_gap_feature_output)
    feature_gap_raw_path = _resolve_path(args.feature_gap_raw_output)
    evaluation_path = _resolve_path(args.evaluation_output)

    payload: dict[str, object] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "completed_step": "init",
        "universe": args.universe,
        "source_scope": args.source_scope,
        "baseline_reference": args.baseline_reference,
        "error_code": None,
        "error_message": None,
        "recommended_action": None,
        "run_context": {
            "data_config": args.data_config,
            "model_config": args.model_config,
            "feature_config": args.feature_config,
            "tail_rows": int(args.tail_rows),
            "max_rows": int(args.max_rows),
            "coverage_threshold": float(args.coverage_threshold),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "model_artifact_suffix": args.model_artifact_suffix,
            "skip_validation": bool(args.skip_validation),
            "skip_feature_gap": bool(args.skip_feature_gap),
            "skip_evaluate": bool(args.skip_evaluate),
            "dry_run": bool(args.dry_run),
        },
        "artifacts": {
            "snapshot": artifact_display_path(snapshot_path, workspace_root=ROOT),
            "validation": artifact_display_path(validation_path, workspace_root=ROOT),
            "feature_gap_summary": artifact_display_path(feature_gap_summary_path, workspace_root=ROOT),
            "feature_gap_feature": artifact_display_path(feature_gap_feature_path, workspace_root=ROOT),
            "feature_gap_raw": artifact_display_path(feature_gap_raw_path, workspace_root=ROOT),
            "evaluation_pointer": artifact_display_path(evaluation_path, workspace_root=ROOT),
        },
    }

    try:
        artifact_ensure_output_file_path(manifest_path, label="manifest output", workspace_root=ROOT)
        _safe_write_manifest(manifest_path, payload)

        total_steps = 1 + int(not args.skip_validation) + int(not args.skip_feature_gap) + int(not args.skip_evaluate)
        progress = ProgressBar(total=total_steps, prefix="[local-feasibility]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"universe={args.universe}")

        snapshot_command = [
            sys.executable,
            str(ROOT / "scripts/run_local_coverage_snapshot.py"),
            "--data-config",
            args.data_config,
            "--tail-rows",
            str(args.tail_rows),
            "--output",
            args.snapshot_output,
            "--universe",
            args.universe,
            "--source-scope",
            args.source_scope,
            "--baseline-reference",
            args.baseline_reference,
        ]
        _set_step(payload, "run_snapshot")
        payload["snapshot"] = {"command": snapshot_command}
        _safe_write_manifest(manifest_path, payload)
        if args.dry_run:
            payload["snapshot"]["status"] = "planned"
        else:
            with Heartbeat("[local-feasibility]", "running snapshot", logger=log_progress):
                snapshot_result = _run_command(snapshot_command, label="snapshot")
            payload["snapshot"] = snapshot_result
            snapshot_payload = _read_optional_json(snapshot_path)
            if snapshot_payload is not None:
                payload["snapshot_payload"] = snapshot_payload
                readiness = snapshot_payload.get("readiness")
                if isinstance(readiness, dict):
                    payload["readiness"] = readiness
            if int(snapshot_result["exit_code"]) != 0:
                _set_failure(
                    payload,
                    status="snapshot_failed",
                    error_code="snapshot_failed",
                    error_message="local snapshot command returned non-zero exit code",
                    recommended_action="inspect_local_snapshot",
                )
                _safe_write_manifest(manifest_path, payload)
                return int(snapshot_result["exit_code"]) or 1
        progress.update(message="snapshot handled")

        if not args.skip_validation:
            validation_command = [
                sys.executable,
                str(ROOT / "scripts/run_local_data_source_validation.py"),
                "--config",
                args.data_config,
                "--output",
                args.validation_output,
            ]
            _set_step(payload, "run_validation")
            payload["validation"] = {"command": validation_command}
            _safe_write_manifest(manifest_path, payload)
            if args.dry_run:
                payload["validation"]["status"] = "planned"
            else:
                with Heartbeat("[local-feasibility]", "running validation", logger=log_progress):
                    validation_result = _run_command(validation_command, label="validation")
                payload["validation"] = validation_result
                validation_payload = _read_optional_json(validation_path)
                if validation_payload is not None:
                    payload["validation_payload"] = validation_payload
                if int(validation_result["exit_code"]) != 0:
                    _set_failure(
                        payload,
                        status="validation_failed",
                        error_code="validation_failed",
                        error_message="local data validation returned non-zero exit code",
                        recommended_action="inspect_local_data_validation",
                    )
                    _safe_write_manifest(manifest_path, payload)
                    return int(validation_result["exit_code"]) or 1
            progress.update(message="validation handled")

        if not args.skip_feature_gap:
            feature_gap_command = [
                sys.executable,
                str(ROOT / "scripts/run_local_feature_gap_report.py"),
                "--config",
                args.data_config,
                "--feature-config",
                args.feature_config,
                "--model-config",
                args.model_config,
                "--template-config",
                args.data_config,
                "--max-rows",
                str(args.max_rows),
                "--coverage-threshold",
                str(args.coverage_threshold),
                "--summary-output",
                args.feature_gap_summary_output,
                "--feature-output",
                args.feature_gap_feature_output,
                "--raw-output",
                args.feature_gap_raw_output,
            ]
            _set_step(payload, "run_feature_gap")
            payload["feature_gap"] = {"command": feature_gap_command}
            _safe_write_manifest(manifest_path, payload)
            if args.dry_run:
                payload["feature_gap"]["status"] = "planned"
            else:
                with Heartbeat("[local-feasibility]", "running feature gap", logger=log_progress):
                    feature_gap_result = _run_command(feature_gap_command, label="feature_gap")
                payload["feature_gap"] = feature_gap_result
                feature_gap_payload = _read_optional_json(feature_gap_summary_path)
                if feature_gap_payload is not None:
                    payload["feature_gap_payload"] = feature_gap_payload
                if int(feature_gap_result["exit_code"]) != 0:
                    _set_failure(
                        payload,
                        status="feature_gap_failed",
                        error_code="feature_gap_failed",
                        error_message="local feature gap returned non-zero exit code",
                        recommended_action="inspect_local_feature_gap",
                    )
                    _safe_write_manifest(manifest_path, payload)
                    return int(feature_gap_result["exit_code"]) or 1
            progress.update(message="feature gap handled")

        if not args.skip_evaluate:
            evaluate_command = [
                sys.executable,
                str(ROOT / "scripts/run_local_evaluate.py"),
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
                "--output",
                args.evaluation_output,
                "--universe",
                args.universe,
                "--source-scope",
                args.source_scope,
                "--baseline-reference",
                args.baseline_reference,
            ]
            if args.start_date:
                evaluate_command.extend(["--start-date", args.start_date])
            if args.end_date:
                evaluate_command.extend(["--end-date", args.end_date])
            if args.model_artifact_suffix:
                evaluate_command.extend(["--model-artifact-suffix", args.model_artifact_suffix])
            _set_step(payload, "run_evaluate")
            payload["evaluation"] = {"command": evaluate_command}
            _safe_write_manifest(manifest_path, payload)
            if args.dry_run:
                payload["evaluation"]["status"] = "planned"
            else:
                with Heartbeat("[local-feasibility]", "running evaluate", logger=log_progress):
                    evaluation_result = _run_command(evaluate_command, label="evaluation")
                payload["evaluation"] = evaluation_result
                evaluation_payload = _read_optional_json(evaluation_path)
                if evaluation_payload is not None:
                    payload["evaluation_payload"] = evaluation_payload
                if int(evaluation_result["exit_code"]) != 0:
                    _set_failure(
                        payload,
                        status="evaluation_failed",
                        error_code="evaluation_failed",
                        error_message="local evaluate returned non-zero exit code",
                        recommended_action="inspect_local_evaluation",
                    )
                    _safe_write_manifest(manifest_path, payload)
                    return int(evaluation_result["exit_code"]) or 1
            progress.update(message="evaluation handled")

        _set_step(payload, "completed")
        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        _safe_write_manifest(manifest_path, payload)
        progress.complete(message="local feasibility manifest completed")
        print(f"[local-feasibility] manifest saved: {manifest_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        _set_failure(
            payload,
            status="interrupted",
            error_code="interrupted",
            error_message="interrupted by user",
            recommended_action="rerun_local_feasibility",
        )
        _safe_write_manifest(manifest_path, payload)
        print("[local-feasibility] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        _set_failure(
            payload,
            status="failed",
            error_code="local_feasibility_failed",
            error_message=str(error),
            recommended_action="inspect_local_feasibility_inputs",
        )
        _safe_write_manifest(manifest_path, payload)
        print(f"[local-feasibility] failed: {error}", flush=True)
        return 1
    except Exception as error:
        _set_failure(
            payload,
            status="failed",
            error_code="local_feasibility_failed",
            error_message=str(error),
            recommended_action="inspect_local_feasibility_traceback",
        )
        _safe_write_manifest(manifest_path, payload)
        print(f"[local-feasibility] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())