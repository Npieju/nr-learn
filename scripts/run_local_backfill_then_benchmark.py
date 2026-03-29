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

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_CRAWL_CONFIG = "configs/crawl_local_nankan_template.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_local_baseline.yaml"
DEFAULT_FEATURE_CONFIG = "configs/features_local_baseline.yaml"
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_PATH = "data/external/local_nankan/results/local_race_result.csv"
DEFAULT_RACE_CARD_PATH = "data/external/local_nankan/racecard/local_racecard.csv"
DEFAULT_PEDIGREE_PATH = "data/external/local_nankan/pedigree/local_pedigree.csv"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-backfill-benchmark {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _safe_write_manifest(path: Path, payload: dict[str, Any]) -> None:
    if path.exists() and path.is_dir():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, payload)


def _read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _dict_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _run_command(command: list[str], *, label: str) -> dict[str, Any]:
    started_at = utc_now_iso()
    printable = shlex.join(command)
    print(f"[local-backfill-benchmark] running {label}: {printable}", flush=True)
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


def _planned_step(*, label: str, command: list[str], output_path: Path | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "label": label,
        "command": command,
        "status": "planned",
    }
    if output_path is not None:
        payload["output"] = artifact_display_path(output_path, workspace_root=ROOT)
    return payload


def _read_order() -> list[str]:
    return [
        "local_backfill_then_benchmark",
        "backfill",
        "materialize",
        "benchmark_gate",
        "data_preflight",
        "snapshot",
    ]


def _current_phase(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "")
    if status == "planned":
        return "planned"
    if status == "running_backfill":
        return "backfill"
    if status == "backfill_failed":
        return "backfill"
    if status == "backfill_not_ready":
        return str((payload.get("backfill_summary") or {}).get("current_phase") or "backfill")
    if status == "running_benchmark_gate":
        return "benchmark_gate"
    if status == "gate_not_ready":
        return "benchmark_gate"
    if status == "gate_failed":
        return "benchmark_gate"
    if status == "completed":
        return "completed"
    if status == "interrupted":
        return "interrupted"
    return "init_manifest"


def _recommended_action(payload: dict[str, Any]) -> str:
    status = str(payload.get("status") or "")
    backfill_summary = _dict_payload(payload.get("backfill_summary"))
    gate_summary = _dict_payload(payload.get("gate_summary"))
    if status == "planned":
        return "run_local_backfill_then_benchmark"
    if status == "backfill_failed":
        return "inspect_local_backfill"
    if status == "backfill_not_ready":
        return str(backfill_summary.get("recommended_action") or "inspect_local_backfill")
    if status == "gate_not_ready":
        return str(gate_summary.get("recommended_action") or "inspect_local_benchmark_gate")
    if status == "gate_failed":
        return "inspect_local_benchmark_gate"
    if status == "completed":
        return "review_local_benchmark_manifest"
    if status == "interrupted":
        return "rerun_local_backfill_then_benchmark"
    return "inspect_local_backfill_then_benchmark"


def _highlights(payload: dict[str, Any]) -> list[str]:
    status = str(payload.get("status") or "")
    recommended_action = str(payload.get("recommended_action") or _recommended_action(payload))
    backfill_summary = _dict_payload(payload.get("backfill_summary"))
    gate_summary = _dict_payload(payload.get("gate_summary"))

    if status == "planned":
        return [
            "wrapper will run local backfill with materialize_after_collect before launching the local benchmark gate",
            "the same smoke-ready external outputs can therefore drive both primary raw promotion and benchmark preflight from one entrypoint",
            f"next operator action: {recommended_action}",
        ]

    if status == "completed":
        return [
            "local backfill handoff reached benchmark completion",
            f"backfill status={backfill_summary.get('status')}, gate status={gate_summary.get('status')}",
            f"next operator action: {recommended_action}",
        ]

    if status in {"backfill_failed", "backfill_not_ready"}:
        return [
            f"wrapper stopped during backfill with status={backfill_summary.get('status')}",
            str(backfill_summary.get("highlights") or backfill_summary.get("stopped_reason") or status),
            f"next operator action: {recommended_action}",
        ]

    if status in {"gate_not_ready", "gate_failed"}:
        return [
            f"wrapper reached benchmark gate with status={gate_summary.get('status') or status}",
            str(gate_summary.get("highlights") or gate_summary.get("error_message") or status),
            f"next operator action: {recommended_action}",
        ]

    return [
        f"wrapper is in progress at {_current_phase(payload)}",
        f"next operator action: {recommended_action}",
    ]


def _refresh_summary_fields(payload: dict[str, Any]) -> None:
    payload["read_order"] = _read_order()
    payload["current_phase"] = _current_phase(payload)
    payload["recommended_action"] = _recommended_action(payload)
    payload["highlights"] = _highlights(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--crawl-config", default=DEFAULT_CRAWL_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--race-id-source", choices=["seed_file", "race_list"], default="seed_file")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="desc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--materialize-output-file", default=None)
    parser.add_argument(
        "--wrapper-manifest-output",
        default="artifacts/reports/local_backfill_then_benchmark_manifest.json",
    )
    parser.add_argument(
        "--backfill-manifest-output",
        default="artifacts/reports/local_nankan_backfill_handoff_manifest.json",
    )
    parser.add_argument(
        "--materialize-manifest-output",
        default="artifacts/reports/local_nankan_primary_handoff_manifest.json",
    )
    parser.add_argument(
        "--snapshot-output",
        default="artifacts/reports/coverage_snapshot_local_nankan_handoff.json",
    )
    parser.add_argument(
        "--preflight-output",
        default="artifacts/reports/data_preflight_local_nankan_handoff.json",
    )
    parser.add_argument(
        "--benchmark-manifest-output",
        default="artifacts/reports/benchmark_gate_local_nankan_handoff.json",
    )
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)
    backfill_manifest_path = _resolve_path(args.backfill_manifest_output)
    materialize_manifest_path = _resolve_path(args.materialize_manifest_output)
    snapshot_path = _resolve_path(args.snapshot_output)
    preflight_path = _resolve_path(args.preflight_output)
    benchmark_manifest_path = _resolve_path(args.benchmark_manifest_output)

    backfill_command = [
        sys.executable,
        str(ROOT / "scripts/run_backfill_local_nankan.py"),
        "--crawl-config",
        args.crawl_config,
        "--data-config",
        args.data_config,
        "--race-id-source",
        args.race_id_source,
        "--date-order",
        args.date_order,
        "--max-cycles",
        str(args.max_cycles),
        "--manifest-file",
        args.backfill_manifest_output,
        "--materialize-after-collect",
        "--race-result-path",
        args.race_result_path,
        "--race-card-path",
        args.race_card_path,
        "--pedigree-path",
        args.pedigree_path,
        "--materialize-manifest-file",
        args.materialize_manifest_output,
    ]
    if args.seed_file:
        backfill_command.extend(["--seed-file", args.seed_file])
    if args.start_date:
        backfill_command.extend(["--start-date", args.start_date])
    if args.end_date:
        backfill_command.extend(["--end-date", args.end_date])
    if args.limit is not None:
        backfill_command.extend(["--limit", str(args.limit)])
    if args.materialize_output_file:
        backfill_command.extend(["--materialize-output-file", args.materialize_output_file])

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
        args.snapshot_output,
        "--manifest-output",
        args.benchmark_manifest_output,
        "--max-rows",
        str(args.max_rows),
        "--wf-mode",
        args.wf_mode,
        "--wf-scheme",
        args.wf_scheme,
        "--universe",
        args.universe,
        "--source-scope",
        args.source_scope,
        "--baseline-reference",
        args.baseline_reference,
        "--race-result-path",
        args.race_result_path,
        "--race-card-path",
        args.race_card_path,
        "--pedigree-path",
        args.pedigree_path,
        "--preflight-output",
        args.preflight_output,
    ]
    if args.pre_feature_max_rows is not None:
        benchmark_command.extend(["--pre-feature-max-rows", str(args.pre_feature_max_rows)])
    if args.skip_train:
        benchmark_command.append("--skip-train")
    if args.skip_evaluate:
        benchmark_command.append("--skip-evaluate")

    payload: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "planned" if args.dry_run else "running_backfill",
        "configs": {
            "data_config": args.data_config,
            "crawl_config": args.crawl_config,
            "model_config": args.model_config,
            "feature_config": args.feature_config,
            "seed_file": args.seed_file,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "date_order": args.date_order,
            "limit": args.limit,
            "max_cycles": int(args.max_cycles),
            "tail_rows": int(args.tail_rows),
            "max_rows": int(args.max_rows),
            "pre_feature_max_rows": int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "universe": args.universe,
            "source_scope": args.source_scope,
            "baseline_reference": args.baseline_reference,
            "skip_train": bool(args.skip_train),
            "skip_evaluate": bool(args.skip_evaluate),
        },
        "artifacts": {
            "wrapper_manifest": artifact_display_path(wrapper_manifest_path, workspace_root=ROOT),
            "backfill_manifest": artifact_display_path(backfill_manifest_path, workspace_root=ROOT),
            "materialize_manifest": artifact_display_path(materialize_manifest_path, workspace_root=ROOT),
            "benchmark_manifest": artifact_display_path(benchmark_manifest_path, workspace_root=ROOT),
            "preflight_manifest": artifact_display_path(preflight_path, workspace_root=ROOT),
            "snapshot": artifact_display_path(snapshot_path, workspace_root=ROOT),
        },
    }
    _refresh_summary_fields(payload)

    try:
        artifact_ensure_output_file_path(wrapper_manifest_path, label="wrapper manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(backfill_manifest_path, label="backfill manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(materialize_manifest_path, label="materialize manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(benchmark_manifest_path, label="benchmark manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(preflight_path, label="preflight output", workspace_root=ROOT)
        artifact_ensure_output_file_path(snapshot_path, label="snapshot output", workspace_root=ROOT)

        if args.dry_run:
            payload["backfill"] = _planned_step(
                label="backfill",
                command=backfill_command + ["--dry-run"],
                output_path=backfill_manifest_path,
            )
            payload["benchmark_gate"] = _planned_step(
                label="benchmark_gate",
                command=benchmark_command,
                output_path=benchmark_manifest_path,
            )
            payload["materialize"] = {
                "status": "planned",
                "manifest_file": artifact_display_path(materialize_manifest_path, workspace_root=ROOT),
            }
            payload["finished_at"] = utc_now_iso()
            _refresh_summary_fields(payload)
            _safe_write_manifest(wrapper_manifest_path, payload)
            print(f"[local-backfill-benchmark] planned manifest saved: {wrapper_manifest_path}", flush=True)
            return 0

        _safe_write_manifest(wrapper_manifest_path, payload)
        progress = ProgressBar(total=3, prefix="[local-backfill-benchmark]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="wrapper manifest initialized")

        with Heartbeat("[local-backfill-benchmark]", "running local backfill handoff", logger=log_progress):
            backfill_result = _run_command(backfill_command, label="backfill")
        payload["backfill"] = backfill_result
        payload["backfill_summary"] = _read_optional_json(backfill_manifest_path)
        payload["materialize_summary"] = _read_optional_json(materialize_manifest_path)
        if int(backfill_result.get("exit_code", 1)) != 0:
            payload["status"] = "backfill_failed"
            payload["finished_at"] = utc_now_iso()
            _refresh_summary_fields(payload)
            _safe_write_manifest(wrapper_manifest_path, payload)
            return int(backfill_result.get("exit_code", 1)) or 1
        progress.update(current=1, message="backfill handoff completed")

        backfill_summary = _dict_payload(payload.get("backfill_summary"))
        backfill_phase = str(backfill_summary.get("current_phase") or "")
        if backfill_phase != "materialized_primary_raw":
            payload["status"] = "backfill_not_ready"
            payload["finished_at"] = utc_now_iso()
            _refresh_summary_fields(payload)
            _safe_write_manifest(wrapper_manifest_path, payload)
            return 2

        payload["status"] = "running_benchmark_gate"
        _refresh_summary_fields(payload)
        _safe_write_manifest(wrapper_manifest_path, payload)

        with Heartbeat("[local-backfill-benchmark]", "running local benchmark gate", logger=log_progress):
            gate_result = _run_command(benchmark_command, label="benchmark_gate")
        payload["benchmark_gate"] = gate_result
        payload["gate_summary"] = _read_optional_json(benchmark_manifest_path)
        payload["preflight_summary"] = _read_optional_json(preflight_path)
        if int(gate_result.get("exit_code", 1)) != 0:
            payload["status"] = "gate_not_ready" if int(gate_result.get("exit_code", 1)) == 2 else "gate_failed"
            payload["finished_at"] = utc_now_iso()
            _refresh_summary_fields(payload)
            _safe_write_manifest(wrapper_manifest_path, payload)
            return int(gate_result.get("exit_code", 1)) or 1
        progress.update(current=2, message="benchmark gate completed")

        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        _refresh_summary_fields(payload)
        _safe_write_manifest(wrapper_manifest_path, payload)
        progress.complete(message="local handoff completed")
        print(f"[local-backfill-benchmark] manifest saved: {wrapper_manifest_path}", flush=True)
        return 0
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at"] = utc_now_iso()
        _refresh_summary_fields(payload)
        _safe_write_manifest(wrapper_manifest_path, payload)
        print("[local-backfill-benchmark] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _refresh_summary_fields(payload)
        _safe_write_manifest(wrapper_manifest_path, payload)
        print(f"[local-backfill-benchmark] failed: {error}", flush=True)
        return 1
    except Exception as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error"] = str(error)
        _refresh_summary_fields(payload)
        _safe_write_manifest(wrapper_manifest_path, payload)
        print(f"[local-backfill-benchmark] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())