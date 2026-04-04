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


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-pre-race-handoff {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[local-nankan-pre-race-handoff] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[local-nankan-pre-race-handoff]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _read_json_dict(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _build_not_ready_manifest(
    *,
    pre_race_summary: dict[str, Any],
    benchmark_manifest_output: Path,
    attempts: int,
    waited_seconds: int,
    timed_out: bool,
) -> dict[str, Any]:
    return {
        "status": "not_ready",
        "current_phase": str(pre_race_summary.get("current_phase") or "await_result_arrival"),
        "recommended_action": str(pre_race_summary.get("recommended_action") or "wait_for_result_ready_pre_race_races"),
        "pre_race_summary": pre_race_summary,
        "benchmark_manifest_output": str(benchmark_manifest_output),
        "attempts": int(attempts),
        "waited_seconds": int(waited_seconds),
        "timed_out": bool(timed_out),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data_local_nankan.yaml")
    parser.add_argument("--model-config", default="configs/model_local_baseline.yaml")
    parser.add_argument("--feature-config", default="configs/features_local_baseline.yaml")
    parser.add_argument("--race-card-input", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--race-result-input", default="data/external/local_nankan/results/local_race_result.csv")
    parser.add_argument("--pedigree-input", default="data/external/local_nankan/pedigree/local_pedigree.csv")
    parser.add_argument("--filtered-race-card-output", default="data/local_nankan/raw/local_nankan_race_card_pre_race_ready.csv")
    parser.add_argument("--filtered-race-result-output", default="data/local_nankan/raw/local_nankan_race_result_pre_race_ready.csv")
    parser.add_argument("--primary-output-file", default="data/local_nankan/raw/local_nankan_primary_pre_race_ready.csv")
    parser.add_argument("--pre-race-summary-output", default="artifacts/reports/local_nankan_pre_race_ready_summary.json")
    parser.add_argument("--primary-manifest-file", default="artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json")
    parser.add_argument("--benchmark-manifest-output", default="artifacts/reports/benchmark_gate_local_nankan_pre_race_ready.json")
    parser.add_argument("--wrapper-manifest-output", default="artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json")
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--wait-for-results", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=60)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    progress = ProgressBar(total=3, prefix="[local-nankan-pre-race-handoff]", logger=log_progress, min_interval_sec=0.0)
    wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)

    try:
        progress.start(message="starting pre-race benchmark handoff")

        pre_race_command = [
            sys.executable,
            str(ROOT / "scripts/run_materialize_local_nankan_pre_race_primary.py"),
            "--data-config",
            args.data_config,
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
            "--manifest-file",
            args.primary_manifest_file,
            "--summary-output",
            args.pre_race_summary_output,
        ]
        benchmark_manifest_output = _resolve_path(args.benchmark_manifest_output)
        wait_started = time.monotonic()
        attempts = 0
        pre_race_exit = 1
        pre_race_summary: dict[str, Any] = {}
        while True:
            attempts += 1
            pre_race_exit = _run_command(label=f"pre_race_primary attempt={attempts}", command=pre_race_command)
            pre_race_summary = _read_json_dict(_resolve_path(args.pre_race_summary_output))
            progress.update(current=1, message=f"pre-race primary attempt={attempts} exit_code={pre_race_exit} status={pre_race_summary.get('status')}")

            if pre_race_exit not in {2} and pre_race_summary.get("status") != "not_ready":
                break

            waited_seconds = int(max(0, time.monotonic() - wait_started))
            timed_out = (not args.wait_for_results) or (args.max_wait_seconds >= 0 and waited_seconds >= max(0, args.max_wait_seconds))
            if timed_out:
                wrapper_manifest = _build_not_ready_manifest(
                    pre_race_summary=pre_race_summary,
                    benchmark_manifest_output=benchmark_manifest_output,
                    attempts=attempts,
                    waited_seconds=waited_seconds,
                    timed_out=True,
                )
                write_json(wrapper_manifest_path, wrapper_manifest)
                progress.complete(message=f"not ready output={wrapper_manifest_path}")
                return 2

            pending_races = (
                pre_race_summary.get("materialization_summary", {}).get("pending_result_races")
                if isinstance(pre_race_summary.get("materialization_summary"), dict)
                else None
            )
            sleep_seconds = max(1, int(args.poll_interval_seconds))
            print(
                "[local-nankan-pre-race-handoff] "
                f"waiting for results pending_races={pending_races} sleep_seconds={sleep_seconds}",
                flush=True,
            )
            time.sleep(sleep_seconds)

        if pre_race_exit != 0:
            wrapper_manifest = {
                "status": "failed",
                "current_phase": "pre_race_primary",
                "recommended_action": "inspect_pre_race_primary_summary",
                "pre_race_summary": pre_race_summary,
            }
            write_json(wrapper_manifest_path, wrapper_manifest)
            return 1

        benchmark_command = [
            sys.executable,
            str(ROOT / "scripts/run_local_benchmark_gate.py"),
            "--data-config",
            args.data_config,
            "--model-config",
            args.model_config,
            "--feature-config",
            args.feature_config,
            "--race-result-path",
            args.filtered_race_result_output,
            "--race-card-path",
            args.filtered_race_card_output,
            "--pedigree-path",
            args.pedigree_input,
            "--materialize-primary-before-gate",
            "--materialize-output-file",
            args.primary_output_file,
            "--materialize-manifest-file",
            args.primary_manifest_file,
            "--manifest-output",
            args.benchmark_manifest_output,
            "--tail-rows",
            str(args.tail_rows),
            "--max-rows",
            str(args.max_rows),
            "--wf-mode",
            args.wf_mode,
            "--wf-scheme",
            args.wf_scheme,
        ]
        if args.pre_feature_max_rows is not None:
            benchmark_command.extend(["--pre-feature-max-rows", str(args.pre_feature_max_rows)])
        if args.skip_train:
            benchmark_command.append("--skip-train")
        if args.skip_evaluate:
            benchmark_command.append("--skip-evaluate")

        benchmark_exit = _run_command(label="benchmark_gate", command=benchmark_command)
        benchmark_manifest = _read_json_dict(_resolve_path(args.benchmark_manifest_output))
        progress.update(current=2, message=f"benchmark gate exit_code={benchmark_exit} status={benchmark_manifest.get('status')}")

        wrapper_manifest = {
            "status": "completed" if benchmark_exit == 0 else "failed",
            "current_phase": "benchmark_gate",
            "recommended_action": "review_benchmark_manifest" if benchmark_exit == 0 else "inspect_benchmark_manifest",
            "pre_race_summary": pre_race_summary,
            "benchmark_manifest": benchmark_manifest,
        }
        write_json(wrapper_manifest_path, wrapper_manifest)
        progress.complete(message=f"handoff completed output={wrapper_manifest_path}")
        return 0 if benchmark_exit == 0 else 1
    except KeyboardInterrupt:
        print("[local-nankan-pre-race-handoff] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-pre-race-handoff] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
