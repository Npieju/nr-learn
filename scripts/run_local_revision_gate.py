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
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-revision-gate {now}] {message}", flush=True)


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
    print(f"[local-revision-gate] running {label}: {printable}", flush=True)
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


def _normalize_revision_slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("revision must not be empty")
    return normalized


def _build_evaluation_pointer_payload(
    *,
    revision_slug: str,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    config_path: str,
    data_config_path: str,
    feature_config_path: str,
    evaluate_command: list[str],
    evaluate_result: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "started_at": evaluate_result.get("started_at", utc_now_iso()),
        "finished_at": evaluate_result.get("finished_at", utc_now_iso()),
        "status": "completed" if int(evaluate_result.get("exit_code", 1)) == 0 else "failed",
        "revision": revision_slug,
        "universe": universe,
        "source_scope": source_scope,
        "baseline_reference": baseline_reference,
        "run_context": {
            "config": config_path,
            "data_config": data_config_path,
            "feature_config": feature_config_path,
        },
        "evaluate_command": evaluate_command,
        "exit_code": int(evaluate_result.get("exit_code", 1)),
    }

    manifest_path = ROOT / "artifacts/reports/evaluation_manifest.json"
    summary_path = ROOT / "artifacts/reports/evaluation_summary.json"
    if manifest_path.exists():
        manifest = read_json(manifest_path)
        if isinstance(manifest, dict):
            payload["latest_manifest"] = artifact_display_path(manifest_path, workspace_root=ROOT)
            payload["latest_manifest_payload"] = manifest
            output_files = manifest.get("output_files")
            if isinstance(output_files, dict):
                payload["output_files"] = output_files
    if summary_path.exists():
        payload["latest_summary"] = artifact_display_path(summary_path, workspace_root=ROOT)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--evaluate-max-rows", type=int, default=120000)
    parser.add_argument("--evaluate-pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--evaluate-start-date", default=None)
    parser.add_argument("--evaluate-end-date", default=None)
    parser.add_argument("--evaluate-wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--evaluate-wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--promotion-min-feasible-folds", type=int, default=1)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--lineage-output", default=None)
    parser.add_argument("--benchmark-manifest-output", default=None)
    parser.add_argument("--snapshot-output", default=None)
    parser.add_argument("--evaluation-pointer-output", default=None)
    parser.add_argument("--promotion-output", default=None)
    parser.add_argument("--revision-manifest-output", default=None)
    parser.add_argument("--wf-summary-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_revision_slug(revision_value)
    snapshot_output = args.snapshot_output or f"artifacts/reports/coverage_snapshot_{revision_slug}.json"
    benchmark_manifest_output = args.benchmark_manifest_output or f"artifacts/reports/benchmark_gate_{revision_slug}.json"
    evaluation_pointer_output = args.evaluation_pointer_output or f"artifacts/reports/evaluation_{revision_slug}_pointer.json"
    promotion_output = args.promotion_output or f"artifacts/reports/promotion_gate_{revision_slug}.json"
    revision_manifest_output = args.revision_manifest_output or f"artifacts/reports/revision_gate_{revision_slug}.json"
    wf_summary_output = args.wf_summary_output or f"artifacts/reports/wf_feasibility_diag_{revision_slug}.json"
    lineage_output = args.lineage_output or f"artifacts/reports/local_revision_gate_{revision_slug}.json"

    lineage_path = _resolve_path(lineage_output)
    snapshot_path = _resolve_path(snapshot_output)
    benchmark_manifest_path = _resolve_path(benchmark_manifest_output)
    evaluation_pointer_path = _resolve_path(evaluation_pointer_output)
    promotion_path = _resolve_path(promotion_output)
    revision_manifest_path = _resolve_path(revision_manifest_output)
    wf_summary_path = _resolve_path(wf_summary_output)

    payload: dict[str, object] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running",
        "completed_step": "init",
        "revision": revision_slug,
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
            "evaluate_max_rows": int(args.evaluate_max_rows),
            "evaluate_pre_feature_max_rows": int(args.evaluate_pre_feature_max_rows) if args.evaluate_pre_feature_max_rows is not None else None,
            "evaluate_start_date": args.evaluate_start_date,
            "evaluate_end_date": args.evaluate_end_date,
            "evaluate_wf_mode": args.evaluate_wf_mode,
            "evaluate_wf_scheme": args.evaluate_wf_scheme,
            "promotion_min_feasible_folds": int(args.promotion_min_feasible_folds),
            "dry_run": bool(args.dry_run),
        },
        "artifacts": {
            "snapshot": artifact_display_path(snapshot_path, workspace_root=ROOT),
            "benchmark_manifest": artifact_display_path(benchmark_manifest_path, workspace_root=ROOT),
            "evaluation_pointer": artifact_display_path(evaluation_pointer_path, workspace_root=ROOT),
            "promotion_output": artifact_display_path(promotion_path, workspace_root=ROOT),
            "revision_manifest": artifact_display_path(revision_manifest_path, workspace_root=ROOT),
            "wf_summary": artifact_display_path(wf_summary_path, workspace_root=ROOT),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        },
    }

    try:
        artifact_ensure_output_file_path(lineage_path, label="lineage output", workspace_root=ROOT)
        _safe_write_manifest(lineage_path, payload)

        progress = ProgressBar(total=3, prefix="[local-revision-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"revision={revision_slug} universe={args.universe}")

        benchmark_command = [
            sys.executable,
            str(ROOT / "scripts/run_local_benchmark_gate.py"),
            "--data-config",
            args.data_config,
            "--model-config",
            args.model_config,
            "--feature-config",
            args.feature_config,
            "--tail-rows",
            str(args.tail_rows),
            "--snapshot-output",
            snapshot_output,
            "--manifest-output",
            benchmark_manifest_output,
            "--max-rows",
            str(args.evaluate_max_rows),
            "--universe",
            args.universe,
            "--source-scope",
            args.source_scope,
            "--baseline-reference",
            args.baseline_reference,
            "--skip-train",
            "--skip-evaluate",
        ]
        if args.evaluate_pre_feature_max_rows is not None:
            benchmark_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])

        _set_step(payload, "run_benchmark_gate")
        payload["benchmark_gate"] = {"command": benchmark_command}
        _safe_write_manifest(lineage_path, payload)
        if args.dry_run:
            payload["benchmark_gate"]["status"] = "planned"
        else:
            with Heartbeat("[local-revision-gate]", "running benchmark gate", logger=log_progress):
                benchmark_result = _run_command(benchmark_command, label="benchmark_gate")
            payload["benchmark_gate"] = benchmark_result
            benchmark_payload = _read_optional_json(benchmark_manifest_path)
            if benchmark_payload is not None:
                payload["benchmark_gate_payload"] = benchmark_payload
            if int(benchmark_result["exit_code"]) != 0:
                error_code = "benchmark_gate_blocked" if int(benchmark_result["exit_code"]) == 2 else "benchmark_gate_failed"
                _set_failure(
                    payload,
                    status="benchmark_gate_failed",
                    error_code=error_code,
                    error_message="local benchmark gate returned non-zero exit code",
                    recommended_action="inspect_local_benchmark_gate",
                )
                _safe_write_manifest(lineage_path, payload)
                return int(benchmark_result["exit_code"]) or 1
        progress.update(message="benchmark gate handled")

        revision_command = [
            sys.executable,
            str(ROOT / "scripts/run_revision_gate.py"),
            "--config",
            args.model_config,
            "--data-config",
            args.data_config,
            "--feature-config",
            args.feature_config,
            "--revision",
            revision_slug,
            "--train-artifact-suffix",
            revision_slug,
            "--evaluate-max-rows",
            str(args.evaluate_max_rows),
            "--evaluate-wf-mode",
            args.evaluate_wf_mode,
            "--evaluate-wf-scheme",
            args.evaluate_wf_scheme,
            "--promotion-min-feasible-folds",
            str(args.promotion_min_feasible_folds),
            "--promotion-output",
            promotion_output,
            "--wf-summary-output",
            wf_summary_output,
            "--manifest-output",
            revision_manifest_output,
        ]
        if args.evaluate_pre_feature_max_rows is not None:
            revision_command.extend(["--evaluate-pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
        if args.evaluate_start_date:
            revision_command.extend(["--evaluate-start-date", args.evaluate_start_date])
        if args.evaluate_end_date:
            revision_command.extend(["--evaluate-end-date", args.evaluate_end_date])
        if args.dry_run:
            revision_command.append("--dry-run")

        _set_step(payload, "run_revision_gate")
        payload["revision_gate"] = {"command": revision_command}
        _safe_write_manifest(lineage_path, payload)
        if args.dry_run:
            payload["revision_gate"]["status"] = "planned"
        else:
            with Heartbeat("[local-revision-gate]", "running revision gate", logger=log_progress):
                revision_result = _run_command(revision_command, label="revision_gate")
            payload["revision_gate"] = revision_result
            revision_manifest_payload = _read_optional_json(revision_manifest_path)
            promotion_payload = _read_optional_json(promotion_path)
            if revision_manifest_payload is not None:
                payload["revision_manifest_payload"] = revision_manifest_payload
            if promotion_payload is not None:
                payload["promotion_payload"] = promotion_payload
            if int(revision_result["exit_code"]) != 0:
                revision_status = str((revision_manifest_payload or {}).get("status", "failed"))
                decision = str((revision_manifest_payload or {}).get("decision", "error"))
                error_code = "revision_gate_blocked" if revision_status == "block" or decision == "block" else "revision_gate_failed"
                _set_failure(
                    payload,
                    status="revision_gate_failed",
                    error_code=error_code,
                    error_message="local revision gate returned non-zero exit code",
                    recommended_action="inspect_local_revision_manifest",
                )
                _safe_write_manifest(lineage_path, payload)
                return int(revision_result["exit_code"]) or 1
        progress.update(message="revision gate handled")

        pointer_command = [
            sys.executable,
            str(ROOT / "scripts/run_local_evaluate.py"),
            "--config",
            args.model_config,
            "--data-config",
            args.data_config,
            "--feature-config",
            args.feature_config,
            "--max-rows",
            str(args.evaluate_max_rows),
            "--output",
            evaluation_pointer_output,
            "--universe",
            args.universe,
            "--source-scope",
            args.source_scope,
            "--baseline-reference",
            args.baseline_reference,
            "--model-artifact-suffix",
            revision_slug,
        ]
        if args.evaluate_start_date:
            pointer_command.extend(["--start-date", args.evaluate_start_date])
        if args.evaluate_end_date:
            pointer_command.extend(["--end-date", args.evaluate_end_date])

        _set_step(payload, "write_evaluation_pointer")
        payload["evaluation_pointer"] = {"command": pointer_command}
        _safe_write_manifest(lineage_path, payload)
        if args.dry_run:
            payload["evaluation_pointer"]["status"] = "planned"
        else:
            evaluation_result = {
                "label": "evaluation_pointer",
                "command": pointer_command,
                "status": "completed",
                "exit_code": 0,
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
            }
            pointer_payload = _build_evaluation_pointer_payload(
                revision_slug=revision_slug,
                universe=args.universe,
                source_scope=args.source_scope,
                baseline_reference=args.baseline_reference,
                config_path=args.model_config,
                data_config_path=args.data_config,
                feature_config_path=args.feature_config,
                evaluate_command=pointer_command,
                evaluate_result=evaluation_result,
            )
            write_json(evaluation_pointer_path, pointer_payload)
            payload["evaluation_pointer"] = evaluation_result
            payload["evaluation_pointer_payload"] = pointer_payload
        progress.update(message="evaluation pointer handled")

        _set_step(payload, "completed")
        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        _safe_write_manifest(lineage_path, payload)
        progress.complete(message="local revision gate completed")
        print(f"[local-revision-gate] lineage manifest saved: {lineage_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        _set_failure(
            payload,
            status="interrupted",
            error_code="interrupted",
            error_message="interrupted by user",
            recommended_action="rerun_local_revision_gate",
        )
        _safe_write_manifest(lineage_path, payload)
        print("[local-revision-gate] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        _set_failure(
            payload,
            status="failed",
            error_code="local_revision_gate_failed",
            error_message=str(error),
            recommended_action="inspect_local_revision_gate_inputs",
        )
        _safe_write_manifest(lineage_path, payload)
        print(f"[local-revision-gate] failed: {error}", flush=True)
        return 1
    except Exception as error:
        _set_failure(
            payload,
            status="failed",
            error_code="local_revision_gate_failed",
            error_message=str(error),
            recommended_action="inspect_local_revision_gate_traceback",
        )
        _safe_write_manifest(lineage_path, payload)
        print(f"[local-revision-gate] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())