from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import re
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
from racing_ml.data.local_nankan_bootstrap import (
    build_value_blend_bootstrap_command_plan,
    materialize_local_nankan_bootstrap_runtime_configs,
    resolve_python_executable,
)


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[nar-bootstrap-handoff {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _sanitize_label(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "step"


def _build_log_path(*, log_dir: Path, log_prefix: str, revision: str, label: str) -> Path:
    safe_parts: list[str] = []
    for value in (log_prefix, revision, label):
        safe_value = _sanitize_label(value)
        if safe_value and safe_value not in safe_parts:
            safe_parts.append(safe_value)
    stem = "_".join(safe_parts) or "step"
    suffix = ".log"
    max_name_length = 240
    max_stem_length = max_name_length - len(suffix)
    if len(stem) > max_stem_length:
        digest = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:12]
        keep_length = max_stem_length - len(digest) - 1
        stem = f"{stem[:keep_length].rstrip('_')}_{digest}"
    return log_dir / f"{stem}{suffix}"


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _display_path_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, Path):
        return _display_path(value)
    if isinstance(value, str):
        return _display_path(_resolve_path(value)) if Path(value).is_absolute() else value
    return value


def _normalize_display_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_display_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_display_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_display_paths(item) for item in value]
    return _display_path_value(value)


def _run_command(*, label: str, command: list[str], log_path: Path | None = None) -> dict[str, Any]:
    started_at = utc_now_iso()
    print(f"[nar-bootstrap-handoff] running {label}: {shlex.join(command)}", flush=True)
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as handle:
            handle.write(f"started_at: {started_at}\n")
            handle.write(f"label: {label}\n")
            handle.write(f"command: {shlex.join(command)}\n\n")
            handle.flush()
            with Heartbeat("[nar-bootstrap-handoff]", f"{label} child command", logger=log_progress):
                result = subprocess.run(command, cwd=ROOT, check=False, stdout=handle, stderr=subprocess.STDOUT)
    else:
        with Heartbeat("[nar-bootstrap-handoff]", f"{label} child command", logger=log_progress):
            result = subprocess.run(command, cwd=ROOT, check=False)
    return {
        "label": label,
        "command": _normalize_display_paths(list(command)),
        "exit_code": int(result.returncode),
        "status": "completed" if result.returncode == 0 else "failed",
        "started_at": started_at,
        "finished_at": utc_now_iso(),
        "log_file": _display_path(log_path) if log_path is not None else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", default=None)
    parser.add_argument("--handoff-script", default="scripts/run_local_nankan_pre_race_benchmark_handoff.py")
    parser.add_argument("--data-config", default="configs/data_local_nankan_pre_race_ready.yaml")
    parser.add_argument("--benchmark-model-config", default="configs/model_local_baseline_wf_runtime_narrow.yaml")
    parser.add_argument("--benchmark-feature-config", default="configs/features_local_baseline.yaml")
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--pedigree-input", default="data/external/local_nankan/pedigree/local_pedigree.csv")
    parser.add_argument("--filtered-race-card-output", default="data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv")
    parser.add_argument("--filtered-race-result-output", default="data/local_nankan_pre_race_ready/raw/local_nankan_race_result_pre_race_ready.csv")
    parser.add_argument("--primary-output-file", default="data/local_nankan_pre_race_ready/raw/local_nankan_primary_pre_race_ready.csv")
    parser.add_argument("--pre-race-summary-output", default="artifacts/reports/local_nankan_pre_race_ready_summary.json")
    parser.add_argument("--primary-manifest-file", default="artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json")
    parser.add_argument("--benchmark-manifest-output", default="artifacts/reports/benchmark_gate_local_nankan_pre_race_ready.json")
    parser.add_argument("--handoff-manifest-output", default="artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json")
    parser.add_argument("--wrapper-manifest-output", default="artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json")
    parser.add_argument("--source-timing-summary-input", default="artifacts/reports/local_nankan_source_timing_audit.json")
    parser.add_argument("--wait-for-results", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--run-bootstrap", action="store_true")
    parser.add_argument("--runtime-config-dir", default="artifacts/runtime_configs")
    parser.add_argument("--bootstrap-feature-config", default="configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml")
    parser.add_argument("--bootstrap-win-config", default="configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml")
    parser.add_argument("--bootstrap-roi-config", default="configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml")
    parser.add_argument("--bootstrap-stack-config", default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml")
    parser.add_argument("--bootstrap-revision", default="r20260404_local_nankan_value_blend_bootstrap_pre_race_ready_v1")
    parser.add_argument("--log-dir", default="artifacts/logs")
    parser.add_argument("--log-prefix", default="local_nankan_result_ready_bootstrap_handoff")
    args = parser.parse_args()

    progress = ProgressBar(total=3, prefix="[nar-bootstrap-handoff]", logger=log_progress, min_interval_sec=0.0)
    wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)
    wrapper_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        python_executable = resolve_python_executable(workspace_root=ROOT, fallback=args.python_executable or sys.executable)
        progress.start(message="starting result-ready bootstrap handoff")

        runtime_configs = _normalize_display_paths(materialize_local_nankan_bootstrap_runtime_configs(
            workspace_root=ROOT,
            revision=args.bootstrap_revision,
            win_config=args.bootstrap_win_config,
            roi_config=args.bootstrap_roi_config,
            stack_config=args.bootstrap_stack_config,
            output_dir=args.runtime_config_dir,
        ))
        log_dir = _resolve_path(args.log_dir)
        handoff_log_path = _build_log_path(
            log_dir=log_dir,
            log_prefix=args.log_prefix,
            revision=args.bootstrap_revision,
            label="pre_race_benchmark_handoff",
        )

        handoff_command = [
            python_executable,
            str(_resolve_path(args.handoff_script)),
            "--data-config",
            args.data_config,
            "--model-config",
            args.benchmark_model_config,
            "--feature-config",
            args.benchmark_feature_config,
            "--race-card-input",
            args.race_card_input,
            "--race-result-input",
            args.race_result_input,
            "--pedigree-input",
            args.pedigree_input,
            "--filtered-race-card-output",
            args.filtered_race_card_output,
            "--filtered-race-result-output",
            args.filtered_race_result_output,
            "--primary-output-file",
            args.primary_output_file,
            "--pre-race-summary-output",
            args.pre_race_summary_output,
            "--primary-manifest-file",
            args.primary_manifest_file,
            "--benchmark-manifest-output",
            args.benchmark_manifest_output,
            "--wrapper-manifest-output",
            args.handoff_manifest_output,
            "--source-timing-summary-input",
            args.source_timing_summary_input,
            "--wf-mode",
            "off",
        ]
        if args.wait_for_results:
            handoff_command.extend(
                [
                    "--wait-for-results",
                    "--max-wait-seconds",
                    str(args.max_wait_seconds),
                    "--poll-interval-seconds",
                    str(args.poll_interval_seconds),
                ]
            )

        handoff_result = _run_command(label="pre_race_benchmark_handoff", command=handoff_command, log_path=handoff_log_path)
        handoff_manifest = _read_json_dict(_resolve_path(args.handoff_manifest_output))
        handoff_exit = int(handoff_result["exit_code"])
        progress.update(current=1, message=f"handoff exit_code={handoff_exit} status={handoff_manifest.get('status')}")

        base_command_plan = build_value_blend_bootstrap_command_plan(
            workspace_root=ROOT,
            python_executable=python_executable,
            data_config=args.data_config,
            feature_config=args.bootstrap_feature_config,
            win_config=runtime_configs["win_config"],
            roi_config=runtime_configs["roi_config"],
            stack_config=runtime_configs["stack_config"],
            revision=args.bootstrap_revision,
        )
        command_plan = _normalize_display_paths([
            {
                **step,
                "log_file": _display_path(
                    _build_log_path(
                        log_dir=log_dir,
                        log_prefix=args.log_prefix,
                        revision=args.bootstrap_revision,
                        label=str(step["label"]),
                    )
                ),
            }
            for step in base_command_plan
        ])

        if handoff_exit == 2 or str(handoff_manifest.get("status")) == "not_ready":
            manifest = _normalize_display_paths({
                "status": "not_ready",
                "current_phase": str(handoff_manifest.get("current_phase") or "await_result_arrival"),
                "recommended_action": str(handoff_manifest.get("recommended_action") or "wait_for_result_ready_pre_race_races"),
                "handoff_command_result": handoff_result,
                "handoff_manifest": handoff_manifest,
                "runtime_configs": runtime_configs,
                "log_dir": _display_path(log_dir),
                "bootstrap_command_plan": command_plan,
            })
            write_json(wrapper_manifest_path, manifest)
            progress.complete(message=f"not ready output={wrapper_manifest_path}")
            return 2

        if handoff_exit != 0 or str(handoff_manifest.get("status")) != "completed":
            manifest = _normalize_display_paths({
                "status": "failed",
                "current_phase": "benchmark_handoff",
                "recommended_action": "inspect_handoff_manifest",
                "handoff_command_result": handoff_result,
                "handoff_manifest": handoff_manifest,
                "runtime_configs": runtime_configs,
                "log_dir": _display_path(log_dir),
                "bootstrap_command_plan": command_plan,
            })
            write_json(wrapper_manifest_path, manifest)
            return 1

        progress.update(current=2, message="benchmark handoff completed; bootstrap surface ready")

        if not args.run_bootstrap:
            manifest = _normalize_display_paths({
                "status": "benchmark_ready",
                "current_phase": "bootstrap_pending",
                "recommended_action": "run_bootstrap_command_plan",
                "handoff_command_result": handoff_result,
                "handoff_manifest": handoff_manifest,
                "runtime_configs": runtime_configs,
                "log_dir": _display_path(log_dir),
                "bootstrap_command_plan": command_plan,
            })
            write_json(wrapper_manifest_path, manifest)
            progress.complete(message=f"bootstrap plan ready output={wrapper_manifest_path}")
            return 0

        bootstrap_runs: list[dict[str, Any]] = []
        for step in command_plan:
            step_result = _run_command(
                label=str(step["label"]),
                command=list(step["command"]),
                log_path=_resolve_path(str(step["log_file"])),
            )
            bootstrap_runs.append(step_result)
            exit_code = int(step_result["exit_code"])
            if exit_code != 0:
                manifest = _normalize_display_paths({
                    "status": "failed",
                    "current_phase": str(step["label"]),
                    "recommended_action": "inspect_bootstrap_child_command",
                    "handoff_command_result": handoff_result,
                    "handoff_manifest": handoff_manifest,
                    "runtime_configs": runtime_configs,
                    "log_dir": _display_path(log_dir),
                    "bootstrap_runs": bootstrap_runs,
                })
                write_json(wrapper_manifest_path, manifest)
                return 1

        manifest = _normalize_display_paths({
            "status": "completed",
            "current_phase": "bootstrap_completed",
            "recommended_action": "review_bootstrap_revision_outputs",
            "handoff_command_result": handoff_result,
            "handoff_manifest": handoff_manifest,
            "runtime_configs": runtime_configs,
            "log_dir": _display_path(log_dir),
            "bootstrap_runs": bootstrap_runs,
        })
        write_json(wrapper_manifest_path, manifest)
        progress.complete(message=f"bootstrap completed output={wrapper_manifest_path}")
        return 0
    except KeyboardInterrupt:
        print("[nar-bootstrap-handoff] interrupted by user")
        return 130
    except Exception as error:
        print(f"[nar-bootstrap-handoff] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
