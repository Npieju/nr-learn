from __future__ import annotations

import argparse
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

from racing_ml.common.artifacts import read_json, write_json
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


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[nar-bootstrap-handoff] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[nar-bootstrap-handoff]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


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
    args = parser.parse_args()

    progress = ProgressBar(total=3, prefix="[nar-bootstrap-handoff]", logger=log_progress, min_interval_sec=0.0)
    wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)
    wrapper_manifest_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        python_executable = resolve_python_executable(workspace_root=ROOT, fallback=args.python_executable or sys.executable)
        progress.start(message="starting result-ready bootstrap handoff")

        runtime_configs = materialize_local_nankan_bootstrap_runtime_configs(
            workspace_root=ROOT,
            revision=args.bootstrap_revision,
            win_config=args.bootstrap_win_config,
            roi_config=args.bootstrap_roi_config,
            stack_config=args.bootstrap_stack_config,
            output_dir=args.runtime_config_dir,
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

        handoff_exit = _run_command(label="pre_race_benchmark_handoff", command=handoff_command)
        handoff_manifest = _read_json_dict(_resolve_path(args.handoff_manifest_output))
        progress.update(current=1, message=f"handoff exit_code={handoff_exit} status={handoff_manifest.get('status')}")

        command_plan = build_value_blend_bootstrap_command_plan(
            workspace_root=ROOT,
            python_executable=python_executable,
            data_config=args.data_config,
            feature_config=args.bootstrap_feature_config,
            win_config=runtime_configs["win_config"],
            roi_config=runtime_configs["roi_config"],
            stack_config=runtime_configs["stack_config"],
            revision=args.bootstrap_revision,
        )

        if handoff_exit == 2 or str(handoff_manifest.get("status")) == "not_ready":
            manifest = {
                "status": "not_ready",
                "current_phase": "await_result_arrival",
                "recommended_action": "wait_for_result_ready_pre_race_races",
                "handoff_manifest": handoff_manifest,
                "runtime_configs": runtime_configs,
                "bootstrap_command_plan": command_plan,
            }
            write_json(wrapper_manifest_path, manifest)
            progress.complete(message=f"not ready output={wrapper_manifest_path}")
            return 2

        if handoff_exit != 0 or str(handoff_manifest.get("status")) != "completed":
            manifest = {
                "status": "failed",
                "current_phase": "benchmark_handoff",
                "recommended_action": "inspect_handoff_manifest",
                "handoff_manifest": handoff_manifest,
                "runtime_configs": runtime_configs,
                "bootstrap_command_plan": command_plan,
            }
            write_json(wrapper_manifest_path, manifest)
            return 1

        progress.update(current=2, message="benchmark handoff completed; bootstrap surface ready")

        if not args.run_bootstrap:
            manifest = {
                "status": "benchmark_ready",
                "current_phase": "bootstrap_pending",
                "recommended_action": "run_bootstrap_command_plan",
                "handoff_manifest": handoff_manifest,
                "runtime_configs": runtime_configs,
                "bootstrap_command_plan": command_plan,
            }
            write_json(wrapper_manifest_path, manifest)
            progress.complete(message=f"bootstrap plan ready output={wrapper_manifest_path}")
            return 0

        bootstrap_runs: list[dict[str, Any]] = []
        for step in command_plan:
            exit_code = _run_command(label=str(step["label"]), command=list(step["command"]))
            bootstrap_runs.append(
                {
                    "label": str(step["label"]),
                    "command": list(step["command"]),
                    "exit_code": int(exit_code),
                }
            )
            if exit_code != 0:
                manifest = {
                    "status": "failed",
                    "current_phase": str(step["label"]),
                    "recommended_action": "inspect_bootstrap_child_command",
                    "handoff_manifest": handoff_manifest,
                    "runtime_configs": runtime_configs,
                    "bootstrap_runs": bootstrap_runs,
                }
                write_json(wrapper_manifest_path, manifest)
                return 1

        manifest = {
            "status": "completed",
            "current_phase": "bootstrap_completed",
            "recommended_action": "review_bootstrap_revision_outputs",
            "handoff_manifest": handoff_manifest,
            "runtime_configs": runtime_configs,
            "bootstrap_runs": bootstrap_runs,
        }
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
