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
    print(f"[local-revision-gate {now}] {message}", flush=True)


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    def isatty(self) -> bool:
        primary = self._streams[0]
        return bool(getattr(primary, "isatty", lambda: False)())


def _default_log_path(*, revision_slug: str) -> Path:
    return ROOT / "artifacts" / "logs" / f"local_revision_gate_{revision_slug}.log"


def _configure_live_log(log_path: Path) -> None:
    artifact_ensure_output_file_path(log_path, label="run log", workspace_root=ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_handle)
    sys.stderr = _TeeStream(sys.stderr, log_handle)
    print(
        f"[local-revision-gate] live log file: {artifact_display_path(log_path, workspace_root=ROOT)}",
        flush=True,
    )


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


def _as_int(value: object, default: int = 1) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        return int(default)
    except (TypeError, ValueError):
        return int(default)


def _dict_payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


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
    exit_code = _as_int(evaluate_result.get("exit_code", 1), default=1)
    payload: dict[str, object] = {
        "started_at": evaluate_result.get("started_at", utc_now_iso()),
        "finished_at": evaluate_result.get("finished_at", utc_now_iso()),
        "status": "completed" if exit_code == 0 else "failed",
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
        "exit_code": exit_code,
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


def _planned_command_step(*, label: str, command: list[str], output_path: Path | None = None) -> dict[str, object]:
    payload: dict[str, object] = {
        "label": label,
        "command": command,
        "status": "planned",
    }
    if output_path is not None:
        payload["output"] = artifact_display_path(output_path, workspace_root=ROOT)
    return payload


def _planned_data_preflight_payload(
    *,
    data_config_path: str,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    output_path: Path,
) -> dict[str, object]:
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
        "completed_step": "planned",
        "artifact_type": "dataset_source_preflight",
        "config": data_config_path,
        "universe": universe,
        "source_scope": source_scope,
        "baseline_reference": baseline_reference,
        "error_code": None,
        "error_message": None,
        "recommended_action": "populate_primary_raw_dir",
        "artifacts": {
            "preflight_manifest": artifact_display_path(output_path, workspace_root=ROOT),
        },
        "readiness": {
            "benchmark_rerun_ready": False,
            "recommended_action": "populate_primary_raw_dir",
            "reasons": [
                "Primary local raw dataset readiness will be checked by local benchmark gate before training and evaluation run.",
            ],
            "primary_dataset_status": "unknown",
            "missing_required_append_tables": [],
            "missing_required_supplemental_tables": [],
        },
    }


def _planned_primary_materialize_payload(
    *,
    data_config_path: str,
    race_result_path: str,
    race_card_path: str,
    pedigree_path: str,
    manifest_output_path: Path,
    output_path: Path | None,
) -> dict[str, object]:
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
        "current_phase": "planned",
        "recommended_action": "populate_external_results",
        "output_file": artifact_display_path(output_path, workspace_root=ROOT) if output_path is not None else None,
        "manifest_file": artifact_display_path(manifest_output_path, workspace_root=ROOT),
        "source_files": {
            "race_result": race_result_path,
            "race_card": race_card_path,
            "pedigree": pedigree_path,
        },
        "row_count": 0,
        "columns": [],
        "highlights": [
            f"primary raw materialize will inspect race_result at {race_result_path}",
            f"materialize manifest path: {artifact_display_path(manifest_output_path, workspace_root=ROOT)}",
            f"data config: {data_config_path}",
        ],
    }


def _planned_benchmark_gate_payload(
    *,
    data_config_path: str,
    model_config_path: str,
    feature_config_path: str,
    tail_rows: int,
    evaluate_max_rows: int,
    evaluate_pre_feature_max_rows: int | None,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    snapshot_output_path: Path,
    manifest_output_path: Path,
    preflight_output_path: Path,
    race_result_path: str,
    race_card_path: str,
    pedigree_path: str,
    materialize_primary_before_gate: bool,
    materialize_manifest_output_path: Path | None,
    materialize_output_path: Path | None,
) -> dict[str, object]:
    payload = {
        "schema_version": "local.benchmark_gate.v1",
        "artifact_type": "benchmark_gate_manifest",
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
        "completed_step": "planned",
        "universe": universe,
        "source_scope": source_scope,
        "baseline_reference": baseline_reference,
        "error_code": None,
        "error_message": None,
        "recommended_action": "populate_primary_raw_dir",
        "configs": {
            "data_config": data_config_path,
            "model_config": model_config_path,
            "feature_config": feature_config_path,
            "tail_rows": int(tail_rows),
            "max_rows": int(evaluate_max_rows),
            "pre_feature_max_rows": int(evaluate_pre_feature_max_rows) if evaluate_pre_feature_max_rows is not None else None,
            "wf_mode": "off",
            "wf_scheme": "nested",
            "skip_train": True,
            "skip_evaluate": True,
            "race_result_path": race_result_path,
            "race_card_path": race_card_path,
            "pedigree_path": pedigree_path,
            "materialize_primary_before_gate": materialize_primary_before_gate,
            "preflight_output": artifact_display_path(preflight_output_path, workspace_root=ROOT),
        },
        "artifacts": {
            "snapshot": artifact_display_path(snapshot_output_path, workspace_root=ROOT),
            "benchmark_manifest": artifact_display_path(manifest_output_path, workspace_root=ROOT),
            "data_preflight": artifact_display_path(preflight_output_path, workspace_root=ROOT),
        },
        "preflight": {
            "status": "planned",
            "output": artifact_display_path(preflight_output_path, workspace_root=ROOT),
        },
    }
    if materialize_primary_before_gate and materialize_manifest_output_path is not None:
        payload["artifacts"]["primary_materialize_manifest"] = artifact_display_path(
            materialize_manifest_output_path,
            workspace_root=ROOT,
        )
        payload["primary_materialize"] = _planned_primary_materialize_payload(
            data_config_path=data_config_path,
            race_result_path=race_result_path,
            race_card_path=race_card_path,
            pedigree_path=pedigree_path,
            manifest_output_path=materialize_manifest_output_path,
            output_path=materialize_output_path,
        )
    return payload


def _planned_backfill_handoff_payload(
    *,
    crawl_config_path: str,
    data_config_path: str,
    model_config_path: str,
    feature_config_path: str,
    seed_file: str | None,
    start_date: str | None,
    end_date: str | None,
    date_order: str,
    limit: int | None,
    max_cycles: int,
    tail_rows: int,
    evaluate_max_rows: int,
    evaluate_pre_feature_max_rows: int | None,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    race_result_path: str,
    race_card_path: str,
    pedigree_path: str,
    wrapper_manifest_output_path: Path,
    backfill_manifest_output_path: Path,
    materialize_manifest_output_path: Path,
    benchmark_manifest_output_path: Path,
    preflight_output_path: Path,
    snapshot_output_path: Path,
) -> dict[str, object]:
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
        "current_phase": "planned",
        "recommended_action": "run_local_backfill_then_benchmark",
        "configs": {
            "crawl_config": crawl_config_path,
            "data_config": data_config_path,
            "model_config": model_config_path,
            "feature_config": feature_config_path,
            "seed_file": seed_file,
            "start_date": start_date,
            "end_date": end_date,
            "date_order": date_order,
            "limit": limit,
            "max_cycles": int(max_cycles),
            "tail_rows": int(tail_rows),
            "max_rows": int(evaluate_max_rows),
            "pre_feature_max_rows": int(evaluate_pre_feature_max_rows) if evaluate_pre_feature_max_rows is not None else None,
            "wf_mode": "off",
            "wf_scheme": "nested",
            "universe": universe,
            "source_scope": source_scope,
            "baseline_reference": baseline_reference,
            "skip_train": True,
            "skip_evaluate": True,
            "race_result_path": race_result_path,
            "race_card_path": race_card_path,
            "pedigree_path": pedigree_path,
        },
        "artifacts": {
            "wrapper_manifest": artifact_display_path(wrapper_manifest_output_path, workspace_root=ROOT),
            "backfill_manifest": artifact_display_path(backfill_manifest_output_path, workspace_root=ROOT),
            "materialize_manifest": artifact_display_path(materialize_manifest_output_path, workspace_root=ROOT),
            "benchmark_manifest": artifact_display_path(benchmark_manifest_output_path, workspace_root=ROOT),
            "data_preflight": artifact_display_path(preflight_output_path, workspace_root=ROOT),
            "snapshot": artifact_display_path(snapshot_output_path, workspace_root=ROOT),
        },
        "read_order": [
            "local_backfill_then_benchmark",
            "backfill",
            "materialize",
            "benchmark_gate",
            "data_preflight",
            "snapshot",
        ],
        "highlights": [
            "revision gate will first run local backfill handoff so Phase 0 readiness can advance before benchmark/revision lineage continues",
            f"wrapper manifest path: {artifact_display_path(wrapper_manifest_output_path, workspace_root=ROOT)}",
            f"benchmark manifest path: {artifact_display_path(benchmark_manifest_output_path, workspace_root=ROOT)}",
        ],
    }


def _planned_revision_manifest_payload(
    *,
    revision_slug: str,
    config_path: str,
    data_config_path: str,
    feature_config_path: str,
    train_artifact_suffix: str,
    evaluate_max_rows: int,
    evaluate_pre_feature_max_rows: int | None,
    evaluate_start_date: str | None,
    evaluate_end_date: str | None,
    evaluate_wf_mode: str,
    evaluate_wf_scheme: str,
    promotion_min_feasible_folds: int,
    promotion_output_path: Path,
    wf_summary_output_path: Path,
    manifest_output_path: Path,
    train_command: list[str],
    evaluate_command: list[str],
    wf_command: list[str],
    promotion_command: list[str],
) -> dict[str, object]:
    return {
        "generated_at": utc_now_iso(),
        "started_at": utc_now_iso(),
        "revision": revision_slug,
        "status": "planned",
        "decision": "not_run",
        "profile": None,
        "config": config_path,
        "data_config": data_config_path,
        "feature_config": feature_config_path,
        "train_artifact_suffix": train_artifact_suffix,
        "training": {
            "skipped": False,
            "max_train_rows": None,
            "max_valid_rows": None,
        },
        "evaluation": {
            "model_artifact_suffix": None,
            "max_rows": int(evaluate_max_rows),
            "pre_feature_max_rows": int(evaluate_pre_feature_max_rows) if evaluate_pre_feature_max_rows is not None else None,
            "start_date": evaluate_start_date,
            "end_date": evaluate_end_date,
            "wf_mode": evaluate_wf_mode,
            "wf_scheme": evaluate_wf_scheme,
        },
        "promotion_gate": {
            "min_feasible_folds": int(promotion_min_feasible_folds),
            "output": artifact_display_path(promotion_output_path, workspace_root=ROOT),
            "summary": None,
            "formal_benchmark": None,
        },
        "steps": [
            {"name": "train", "command": train_command, "status": "planned"},
            {"name": "evaluate", "command": evaluate_command, "status": "planned"},
            {"name": "wf_feasibility", "command": wf_command, "status": "planned"},
            {"name": "promotion_gate", "command": promotion_command, "status": "planned"},
        ],
        "artifacts": {
            "wf_summary": artifact_display_path(wf_summary_output_path, workspace_root=ROOT),
            "promotion_report": artifact_display_path(promotion_output_path, workspace_root=ROOT),
            "evaluation_manifest": "artifacts/reports/evaluation_manifest.json",
            "evaluation_summary": "artifacts/reports/evaluation_summary.json",
            "revision_manifest": artifact_display_path(manifest_output_path, workspace_root=ROOT),
        },
        "dry_run": True,
    }


def _planned_promotion_payload(*, output_path: Path, min_feasible_folds: int) -> dict[str, object]:
    return {
        "status": "planned",
        "decision": "not_run",
        "recommended_action": "run_revision_gate",
        "min_feasible_folds": int(min_feasible_folds),
        "output": artifact_display_path(output_path, workspace_root=ROOT),
        "summary": None,
        "formal_benchmark": None,
    }


def _planned_evaluation_pointer_payload(
    *,
    revision_slug: str,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    config_path: str,
    data_config_path: str,
    feature_config_path: str,
    evaluate_command: list[str],
) -> dict[str, object]:
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
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
        "exit_code": None,
        "latest_manifest": "artifacts/reports/evaluation_manifest.json",
        "latest_summary": "artifacts/reports/evaluation_summary.json",
        "output_files": None,
    }


def _planned_highlights(*, revision_slug: str, materialize_primary_before_gate: bool, backfill_before_benchmark: bool) -> list[str]:
    highlights = [
        f"local benchmark gate will validate source readiness before training for {revision_slug}",
        "revision gate will reuse the same revision slug across training, evaluation, WF, and promotion artifacts",
        "evaluation pointer will be written after revision gate so downstream public snapshot and mixed manifests can follow the lineage",
    ]
    if backfill_before_benchmark:
        highlights.insert(1, "revision gate will first run local backfill handoff so prepare/collect/materialize can advance before benchmark gate")
    if materialize_primary_before_gate:
        highlights.insert(1, "benchmark gate will attempt primary raw materialize before preflight so existing external local outputs can satisfy raw readiness")
    return highlights


def _backfill_handoff_phase(payload: dict[str, object]) -> str | None:
    handoff_payload = _dict_payload(payload.get("backfill_handoff_payload"))
    phase = handoff_payload.get("current_phase")
    if isinstance(phase, str) and phase.strip():
        return phase
    return None


def _current_phase(payload: dict[str, object]) -> str:
    status = str(payload.get("status") or "")
    completed_step = str(payload.get("completed_step") or "")

    if status == "planned":
        return "planned"
    if status == "completed":
        return "completed"
    if status == "backfill_handoff_blocked":
        return str(_backfill_handoff_phase(payload) or "backfill")
    if status == "backfill_handoff_failed":
        return str(_backfill_handoff_phase(payload) or "backfill")
    if status == "benchmark_gate_blocked":
        return "benchmark_gate"
    if status == "benchmark_gate_failed":
        return "benchmark_gate"
    if status == "revision_gate_failed":
        return "revision_gate"
    if status == "failed":
        if completed_step == "run_benchmark_gate":
            return "benchmark_gate"
        if completed_step == "run_revision_gate":
            return "revision_gate"
        if completed_step == "write_evaluation_pointer":
            return "evaluation_pointer"
        return "local_revision_gate_failed"
    if status == "interrupted":
        if completed_step == "run_benchmark_gate":
            return "benchmark_gate"
        if completed_step == "run_revision_gate":
            return "revision_gate"
        if completed_step == "write_evaluation_pointer":
            return "evaluation_pointer"
        return "interrupted"

    phase_map = {
        "init": "init_manifest",
        "run_benchmark_gate": "benchmark_gate",
        "run_revision_gate": "revision_gate",
        "write_evaluation_pointer": "evaluation_pointer",
        "planned": "planned",
    }
    return phase_map.get(completed_step, completed_step or "init_manifest")


def _recommended_action(payload: dict[str, object]) -> str | None:
    status = str(payload.get("status") or "")
    if status == "planned":
        return "run_local_revision_gate"
    if status == "completed":
        return "review_local_public_snapshot"
    if status == "backfill_handoff_blocked":
        handoff_payload = _dict_payload(payload.get("backfill_handoff_payload"))
        return str(handoff_payload.get("recommended_action") or "run_local_backfill_then_benchmark")
    if status == "backfill_handoff_failed":
        return "inspect_local_backfill_then_benchmark"
    if status == "benchmark_gate_blocked":
        benchmark_payload = _dict_payload(payload.get("benchmark_gate_payload"))
        return str(benchmark_payload.get("recommended_action") or payload.get("recommended_action") or "inspect_local_benchmark_gate")
    if status == "benchmark_gate_failed":
        return "inspect_local_benchmark_gate"
    if status == "revision_gate_failed":
        return str(payload.get("recommended_action") or "inspect_local_revision_manifest")
    if status == "interrupted":
        return "rerun_local_revision_gate"
    if status == "failed":
        current_phase = str(payload.get("current_phase") or _current_phase(payload))
        if current_phase == "evaluation_pointer":
            return "inspect_local_evaluation_pointer"
        return str(payload.get("recommended_action") or "inspect_local_revision_gate_inputs")
    return str(payload.get("recommended_action") or "inspect_local_revision_gate")


def _highlights(payload: dict[str, object]) -> list[str]:
    status = str(payload.get("status") or "")
    revision = str(payload.get("revision") or "unknown_revision")
    current_phase = str(payload.get("current_phase") or _current_phase(payload))
    recommended_action = str(payload.get("recommended_action") or _recommended_action(payload) or "inspect_local_revision_gate")
    run_context = _dict_payload(payload.get("run_context"))

    if status == "planned":
        return _planned_highlights(
            revision_slug=revision,
            materialize_primary_before_gate=bool(run_context.get("materialize_primary_before_gate")),
            backfill_before_benchmark=bool(run_context.get("backfill_before_benchmark")),
        ) + [f"next operator action: {recommended_action}"]

    if status == "completed":
        highlights = [f"local revision lineage completed for revision={revision}"]
        benchmark_payload = _dict_payload(payload.get("benchmark_gate_payload"))
        benchmark_status = benchmark_payload.get("status")
        if benchmark_status is not None:
            highlights.append(f"benchmark gate status={benchmark_status}")
        promotion_payload = _dict_payload(payload.get("promotion_payload"))
        promotion_decision = promotion_payload.get("decision")
        promotion_status = promotion_payload.get("status")
        if promotion_decision is not None or promotion_status is not None:
            highlights.append(
                f"promotion status={promotion_status if promotion_status is not None else 'unknown'}, decision={promotion_decision if promotion_decision is not None else 'unknown'}"
            )
        highlights.append(f"next operator action: {recommended_action}")
        return highlights

    if status in {"backfill_handoff_blocked", "backfill_handoff_failed"}:
        handoff_payload = _dict_payload(payload.get("backfill_handoff_payload"))
        highlights = [f"local revision lineage stopped during {current_phase} for revision={revision}"]
        handoff_status = handoff_payload.get("status")
        if handoff_status is not None:
            highlights.append(f"backfill handoff status={handoff_status}")
        error_message = str(payload.get("error_message") or handoff_payload.get("recommended_action") or status)
        highlights.append(error_message)
        highlights.append(f"next operator action: {recommended_action}")
        return highlights

    error_message = str(payload.get("error_message") or status)
    highlights = [f"local revision lineage stopped during {current_phase} for revision={revision}"]
    highlights.append(error_message)
    highlights.append(f"next operator action: {recommended_action}")
    return highlights


def _refresh_summary_fields(payload: dict[str, object]) -> None:
    payload["current_phase"] = _current_phase(payload)
    payload["recommended_action"] = _recommended_action(payload)
    payload["highlights"] = _highlights(payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--crawl-config", default=DEFAULT_CRAWL_CONFIG)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
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
    parser.add_argument("--evaluate-max-rows", type=int, default=120000)
    parser.add_argument("--evaluate-pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--evaluate-start-date", default=None)
    parser.add_argument("--evaluate-end-date", default=None)
    parser.add_argument("--evaluate-wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--evaluate-wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--wf-max-silent-seconds", type=float, default=None)
    parser.add_argument("--wf-max-fold-elapsed-seconds", type=float, default=None)
    parser.add_argument("--allow-wf-soft-block", action="store_true")
    parser.add_argument("--promotion-min-feasible-folds", type=int, default=1)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--materialize-primary-before-gate", action="store_true")
    parser.add_argument("--backfill-before-benchmark", action="store_true")
    parser.add_argument("--materialize-output-file", default=None)
    parser.add_argument("--materialize-manifest-output", default=None)
    parser.add_argument("--backfill-wrapper-output", default=None)
    parser.add_argument("--backfill-manifest-output", default=None)
    parser.add_argument("--lineage-output", default=None)
    parser.add_argument("--benchmark-manifest-output", default=None)
    parser.add_argument("--snapshot-output", default=None)
    parser.add_argument("--evaluation-pointer-output", default=None)
    parser.add_argument("--data-preflight-output", default=None)
    parser.add_argument("--promotion-output", default=None)
    parser.add_argument("--revision-manifest-output", default=None)
    parser.add_argument("--wf-summary-output", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_revision_slug(revision_value)
    log_path = _resolve_path(args.log_file) if args.log_file else _default_log_path(revision_slug=revision_slug)
    _configure_live_log(log_path)
    snapshot_output = args.snapshot_output or f"artifacts/reports/coverage_snapshot_{revision_slug}.json"
    benchmark_manifest_output = args.benchmark_manifest_output or f"artifacts/reports/benchmark_gate_{revision_slug}.json"
    data_preflight_output = args.data_preflight_output or f"artifacts/reports/data_preflight_{revision_slug}.json"
    evaluation_pointer_output = args.evaluation_pointer_output or f"artifacts/reports/evaluation_{revision_slug}_pointer.json"
    promotion_output = args.promotion_output or f"artifacts/reports/promotion_gate_{revision_slug}.json"
    revision_manifest_output = args.revision_manifest_output or f"artifacts/reports/revision_gate_{revision_slug}.json"
    wf_summary_output = args.wf_summary_output or f"artifacts/reports/wf_feasibility_diag_{revision_slug}.json"
    materialize_manifest_output = args.materialize_manifest_output or f"artifacts/reports/local_nankan_primary_materialize_{revision_slug}.json"
    backfill_wrapper_output = args.backfill_wrapper_output or f"artifacts/reports/local_backfill_then_benchmark_{revision_slug}.json"
    backfill_manifest_output = args.backfill_manifest_output or f"artifacts/reports/local_nankan_backfill_{revision_slug}.json"
    lineage_output = args.lineage_output or f"artifacts/reports/local_revision_gate_{revision_slug}.json"

    lineage_path = _resolve_path(lineage_output)
    snapshot_path = _resolve_path(snapshot_output)
    benchmark_manifest_path = _resolve_path(benchmark_manifest_output)
    data_preflight_path = _resolve_path(data_preflight_output)
    evaluation_pointer_path = _resolve_path(evaluation_pointer_output)
    promotion_path = _resolve_path(promotion_output)
    revision_manifest_path = _resolve_path(revision_manifest_output)
    wf_summary_path = _resolve_path(wf_summary_output)
    materialize_manifest_path = _resolve_path(materialize_manifest_output)
    backfill_wrapper_path = _resolve_path(backfill_wrapper_output)
    backfill_manifest_path = _resolve_path(backfill_manifest_output)
    materialize_output_path = _resolve_path(args.materialize_output_file) if args.materialize_output_file else None

    payload: dict[str, object] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "planned" if args.dry_run else "running",
        "completed_step": "init",
        "revision": revision_slug,
        "universe": args.universe,
        "source_scope": args.source_scope,
        "baseline_reference": args.baseline_reference,
        "error_code": None,
        "error_message": None,
        "recommended_action": None,
        "run_context": {
            "crawl_config": args.crawl_config,
            "data_config": args.data_config,
            "model_config": args.model_config,
            "feature_config": args.feature_config,
            "seed_file": args.seed_file,
            "race_id_source": args.race_id_source,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "date_order": args.date_order,
            "limit": int(args.limit) if args.limit is not None else None,
            "max_cycles": int(args.max_cycles),
            "tail_rows": int(args.tail_rows),
            "evaluate_max_rows": int(args.evaluate_max_rows),
            "evaluate_pre_feature_max_rows": int(args.evaluate_pre_feature_max_rows) if args.evaluate_pre_feature_max_rows is not None else None,
            "evaluate_start_date": args.evaluate_start_date,
            "evaluate_end_date": args.evaluate_end_date,
            "evaluate_wf_mode": args.evaluate_wf_mode,
            "evaluate_wf_scheme": args.evaluate_wf_scheme,
            "allow_wf_soft_block": bool(args.allow_wf_soft_block),
            "promotion_min_feasible_folds": int(args.promotion_min_feasible_folds),
            "race_result_path": args.race_result_path,
            "race_card_path": args.race_card_path,
            "pedigree_path": args.pedigree_path,
            "materialize_primary_before_gate": bool(args.materialize_primary_before_gate),
            "backfill_before_benchmark": bool(args.backfill_before_benchmark),
            "backfill_wrapper_output": artifact_display_path(backfill_wrapper_path, workspace_root=ROOT),
            "backfill_manifest_output": artifact_display_path(backfill_manifest_path, workspace_root=ROOT),
            "materialize_output_file": artifact_display_path(materialize_output_path, workspace_root=ROOT) if materialize_output_path is not None else None,
            "materialize_manifest_output": artifact_display_path(materialize_manifest_path, workspace_root=ROOT),
            "dry_run": bool(args.dry_run),
        },
        "artifacts": {
            "run_log": artifact_display_path(log_path, workspace_root=ROOT),
            "backfill_wrapper_manifest": artifact_display_path(backfill_wrapper_path, workspace_root=ROOT),
            "backfill_manifest": artifact_display_path(backfill_manifest_path, workspace_root=ROOT),
            "snapshot": artifact_display_path(snapshot_path, workspace_root=ROOT),
            "benchmark_manifest": artifact_display_path(benchmark_manifest_path, workspace_root=ROOT),
            "data_preflight": artifact_display_path(data_preflight_path, workspace_root=ROOT),
            "primary_materialize_manifest": artifact_display_path(materialize_manifest_path, workspace_root=ROOT),
            "evaluation_pointer": artifact_display_path(evaluation_pointer_path, workspace_root=ROOT),
            "promotion_output": artifact_display_path(promotion_path, workspace_root=ROOT),
            "revision_manifest": artifact_display_path(revision_manifest_path, workspace_root=ROOT),
            "wf_summary": artifact_display_path(wf_summary_path, workspace_root=ROOT),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        },
        "read_order": [
            "local_revision_gate",
            *( ["local_backfill_then_benchmark", "backfill", "materialize"] if args.backfill_before_benchmark else []),
            *(["primary_materialize"] if args.materialize_primary_before_gate else []),
            "benchmark_gate",
            "data_preflight",
            "revision_gate",
            "promotion_gate",
            "evaluation_pointer",
        ],
    }
    _refresh_summary_fields(payload)

    try:
        artifact_ensure_output_file_path(lineage_path, label="lineage output", workspace_root=ROOT)
        if args.backfill_before_benchmark:
            artifact_ensure_output_file_path(backfill_wrapper_path, label="backfill wrapper output", workspace_root=ROOT)
            artifact_ensure_output_file_path(backfill_manifest_path, label="backfill manifest output", workspace_root=ROOT)
        _refresh_summary_fields(payload)
        _safe_write_manifest(lineage_path, payload)

        progress = ProgressBar(total=3, prefix="[local-revision-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"revision={revision_slug} universe={args.universe}")

        if args.backfill_before_benchmark:
            benchmark_command = [
                sys.executable,
                str(ROOT / "scripts/run_local_backfill_then_benchmark.py"),
                "--crawl-config",
                args.crawl_config,
                "--data-config",
                args.data_config,
                "--model-config",
                args.model_config,
                "--feature-config",
                args.feature_config,
                "--race-id-source",
                args.race_id_source,
                "--date-order",
                args.date_order,
                "--max-cycles",
                str(args.max_cycles),
                "--tail-rows",
                str(args.tail_rows),
                "--snapshot-output",
                snapshot_output,
                "--benchmark-manifest-output",
                benchmark_manifest_output,
                "--preflight-output",
                data_preflight_output,
                "--max-rows",
                str(args.evaluate_max_rows),
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
                "--wrapper-manifest-output",
                backfill_wrapper_output,
                "--backfill-manifest-output",
                backfill_manifest_output,
                "--materialize-manifest-output",
                materialize_manifest_output,
                "--skip-train",
                "--skip-evaluate",
            ]
            if args.seed_file:
                benchmark_command.extend(["--seed-file", args.seed_file])
            if args.start_date:
                benchmark_command.extend(["--start-date", args.start_date])
            if args.end_date:
                benchmark_command.extend(["--end-date", args.end_date])
            if args.limit is not None:
                benchmark_command.extend(["--limit", str(args.limit)])
            if args.evaluate_pre_feature_max_rows is not None:
                benchmark_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
            if args.materialize_output_file:
                benchmark_command.extend(["--materialize-output-file", args.materialize_output_file])
        else:
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
                "--preflight-output",
                data_preflight_output,
                "--max-rows",
                str(args.evaluate_max_rows),
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
                "--skip-train",
                "--skip-evaluate",
            ]
            if args.evaluate_pre_feature_max_rows is not None:
                benchmark_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
            if args.materialize_primary_before_gate:
                benchmark_command.extend([
                    "--materialize-primary-before-gate",
                    "--materialize-manifest-file",
                    materialize_manifest_output,
                ])
                if args.materialize_output_file:
                    benchmark_command.extend(["--materialize-output-file", args.materialize_output_file])

        _set_step(payload, "run_benchmark_gate")
        payload["benchmark_gate"] = _planned_command_step(
            label="benchmark_gate",
            command=benchmark_command,
            output_path=benchmark_manifest_path,
        )
        _refresh_summary_fields(payload)
        _safe_write_manifest(lineage_path, payload)
        if args.dry_run:
            if args.backfill_before_benchmark:
                payload["backfill_handoff_payload"] = _planned_backfill_handoff_payload(
                    crawl_config_path=args.crawl_config,
                    data_config_path=args.data_config,
                    model_config_path=args.model_config,
                    feature_config_path=args.feature_config,
                    seed_file=args.seed_file,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    date_order=args.date_order,
                    limit=args.limit,
                    max_cycles=args.max_cycles,
                    tail_rows=args.tail_rows,
                    evaluate_max_rows=args.evaluate_max_rows,
                    evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                    universe=args.universe,
                    source_scope=args.source_scope,
                    baseline_reference=args.baseline_reference,
                    race_result_path=args.race_result_path,
                    race_card_path=args.race_card_path,
                    pedigree_path=args.pedigree_path,
                    wrapper_manifest_output_path=backfill_wrapper_path,
                    backfill_manifest_output_path=backfill_manifest_path,
                    materialize_manifest_output_path=materialize_manifest_path,
                    benchmark_manifest_output_path=benchmark_manifest_path,
                    preflight_output_path=data_preflight_path,
                    snapshot_output_path=snapshot_path,
                )
            payload["benchmark_gate_payload"] = _planned_benchmark_gate_payload(
                data_config_path=args.data_config,
                model_config_path=args.model_config,
                feature_config_path=args.feature_config,
                tail_rows=args.tail_rows,
                evaluate_max_rows=args.evaluate_max_rows,
                evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                universe=args.universe,
                source_scope=args.source_scope,
                baseline_reference=args.baseline_reference,
                snapshot_output_path=snapshot_path,
                manifest_output_path=benchmark_manifest_path,
                preflight_output_path=data_preflight_path,
                race_result_path=args.race_result_path,
                race_card_path=args.race_card_path,
                pedigree_path=args.pedigree_path,
                materialize_primary_before_gate=bool(args.materialize_primary_before_gate),
                materialize_manifest_output_path=materialize_manifest_path,
                materialize_output_path=materialize_output_path,
            )
            payload["data_preflight_payload"] = _planned_data_preflight_payload(
                data_config_path=args.data_config,
                universe=args.universe,
                source_scope=args.source_scope,
                baseline_reference=args.baseline_reference,
                output_path=data_preflight_path,
            )
            if args.materialize_primary_before_gate or args.backfill_before_benchmark:
                payload["primary_materialize_payload"] = _planned_primary_materialize_payload(
                    data_config_path=args.data_config,
                    race_result_path=args.race_result_path,
                    race_card_path=args.race_card_path,
                    pedigree_path=args.pedigree_path,
                    manifest_output_path=materialize_manifest_path,
                    output_path=materialize_output_path,
                )
            _refresh_summary_fields(payload)
        else:
            with Heartbeat("[local-revision-gate]", "running benchmark gate", logger=log_progress):
                benchmark_result = _run_command(benchmark_command, label="benchmark_gate")
            payload["benchmark_gate"] = benchmark_result
            backfill_handoff_payload = _read_optional_json(backfill_wrapper_path) if args.backfill_before_benchmark else None
            backfill_summary_payload = _read_optional_json(backfill_manifest_path) if args.backfill_before_benchmark else None
            benchmark_payload = _read_optional_json(benchmark_manifest_path)
            data_preflight_payload = _read_optional_json(data_preflight_path)
            primary_materialize_payload = _read_optional_json(materialize_manifest_path) if (args.materialize_primary_before_gate or args.backfill_before_benchmark) else None
            if backfill_handoff_payload is not None:
                payload["backfill_handoff_payload"] = backfill_handoff_payload
            if backfill_summary_payload is not None:
                payload["backfill_summary_payload"] = backfill_summary_payload
            if benchmark_payload is not None:
                payload["benchmark_gate_payload"] = benchmark_payload
            if data_preflight_payload is not None:
                payload["data_preflight_payload"] = data_preflight_payload
            if primary_materialize_payload is not None:
                payload["primary_materialize_payload"] = primary_materialize_payload
            benchmark_exit_code = _as_int(benchmark_result.get("exit_code", 1), default=1)
            if benchmark_exit_code != 0:
                if args.backfill_before_benchmark and isinstance(backfill_handoff_payload, dict):
                    handoff_status = str(backfill_handoff_payload.get("status") or "")
                    if benchmark_exit_code == 2 and handoff_status in {"backfill_not_ready", "running_backfill"}:
                        _set_failure(
                            payload,
                            status="backfill_handoff_blocked",
                            error_code="backfill_handoff_blocked",
                            error_message="local backfill handoff stopped before benchmark gate became ready",
                            recommended_action=str(backfill_handoff_payload.get("recommended_action") or "run_local_backfill_then_benchmark"),
                        )
                        _refresh_summary_fields(payload)
                        _safe_write_manifest(lineage_path, payload)
                        return benchmark_exit_code or 1
                error_code = "benchmark_gate_blocked" if benchmark_exit_code == 2 else "benchmark_gate_failed"
                _set_failure(
                    payload,
                    status="benchmark_gate_blocked" if benchmark_exit_code == 2 else "benchmark_gate_failed",
                    error_code=error_code,
                    error_message="local benchmark gate returned non-zero exit code",
                    recommended_action=str((benchmark_payload or {}).get("recommended_action") or "inspect_local_benchmark_gate"),
                )
                _refresh_summary_fields(payload)
                _safe_write_manifest(lineage_path, payload)
                return benchmark_exit_code or 1
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
        if args.wf_max_silent_seconds is not None:
            revision_command.extend(["--wf-max-silent-seconds", str(args.wf_max_silent_seconds)])
        if args.wf_max_fold_elapsed_seconds is not None:
            revision_command.extend(["--wf-max-fold-elapsed-seconds", str(args.wf_max_fold_elapsed_seconds)])
        if args.dry_run:
            revision_command.append("--dry-run")
        if args.allow_wf_soft_block:
            revision_command.append("--allow-wf-soft-block")

        train_command = [
            sys.executable,
            "scripts/run_train.py",
            "--config",
            args.model_config,
            "--data-config",
            args.data_config,
            "--feature-config",
            args.feature_config,
            "--artifact-suffix",
            revision_slug,
        ]
        evaluate_command = [
            sys.executable,
            "scripts/run_evaluate.py",
            "--config",
            args.model_config,
            "--data-config",
            args.data_config,
            "--feature-config",
            args.feature_config,
            "--artifact-suffix",
            revision_slug,
            "--max-rows",
            str(args.evaluate_max_rows),
            "--wf-mode",
            args.evaluate_wf_mode,
            "--wf-scheme",
            args.evaluate_wf_scheme,
        ]
        wf_command = [
            sys.executable,
            "scripts/run_wf_feasibility_diag.py",
            "--config",
            args.model_config,
            "--data-config",
            args.data_config,
            "--feature-config",
            args.feature_config,
            "--artifact-suffix",
            revision_slug,
            "--wf-mode",
            args.evaluate_wf_mode,
            "--wf-scheme",
            args.evaluate_wf_scheme,
        ]
        if args.wf_max_silent_seconds is not None:
            wf_command.extend(["--max-silent-seconds", str(args.wf_max_silent_seconds)])
        if args.wf_max_fold_elapsed_seconds is not None:
            wf_command.extend(["--max-fold-elapsed-seconds", str(args.wf_max_fold_elapsed_seconds)])
        promotion_command = [
            sys.executable,
            "scripts/run_promotion_gate.py",
            "--evaluation-manifest",
            "artifacts/reports/evaluation_manifest.json",
            "--wf-summary",
            artifact_display_path(wf_summary_path, workspace_root=ROOT),
            "--min-feasible-folds",
            str(args.promotion_min_feasible_folds),
            "--output",
            artifact_display_path(promotion_path, workspace_root=ROOT),
        ]
        if args.evaluate_pre_feature_max_rows is not None:
            evaluate_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
            wf_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
        if args.evaluate_start_date:
            evaluate_command.extend(["--start-date", args.evaluate_start_date])
            wf_command.extend(["--start-date", args.evaluate_start_date])
        if args.evaluate_end_date:
            evaluate_command.extend(["--end-date", args.evaluate_end_date])
            wf_command.extend(["--end-date", args.evaluate_end_date])

        _set_step(payload, "run_revision_gate")
        payload["revision_gate"] = _planned_command_step(
            label="revision_gate",
            command=revision_command,
            output_path=revision_manifest_path,
        )
        _refresh_summary_fields(payload)
        _safe_write_manifest(lineage_path, payload)
        if args.dry_run:
            payload["revision_manifest_payload"] = _planned_revision_manifest_payload(
                revision_slug=revision_slug,
                config_path=args.model_config,
                data_config_path=args.data_config,
                feature_config_path=args.feature_config,
                train_artifact_suffix=revision_slug,
                evaluate_max_rows=args.evaluate_max_rows,
                evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                evaluate_start_date=args.evaluate_start_date,
                evaluate_end_date=args.evaluate_end_date,
                evaluate_wf_mode=args.evaluate_wf_mode,
                evaluate_wf_scheme=args.evaluate_wf_scheme,
                promotion_min_feasible_folds=args.promotion_min_feasible_folds,
                promotion_output_path=promotion_path,
                wf_summary_output_path=wf_summary_path,
                manifest_output_path=revision_manifest_path,
                train_command=train_command,
                evaluate_command=evaluate_command,
                wf_command=wf_command,
                promotion_command=promotion_command,
            )
            payload["promotion_payload"] = _planned_promotion_payload(
                output_path=promotion_path,
                min_feasible_folds=args.promotion_min_feasible_folds,
            )
            _refresh_summary_fields(payload)
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
            revision_exit_code = _as_int(revision_result.get("exit_code", 1), default=1)
            if revision_exit_code != 0:
                revision_status = str((revision_manifest_payload or {}).get("status", "failed"))
                decision = str((revision_manifest_payload or {}).get("decision", "error"))
                if revision_status == "block" or decision == "hold":
                    progress.update(message="revision gate blocked but lineage will continue with evaluation pointer")
                else:
                    error_code = str(
                        (revision_manifest_payload or {}).get("error_code")
                        or ("revision_gate_blocked" if revision_status == "block" or decision == "block" else "revision_gate_failed")
                    )
                    _set_failure(
                        payload,
                        status="revision_gate_failed",
                        error_code=error_code,
                        error_message=str(
                            (revision_manifest_payload or {}).get("error_message")
                            or "local revision gate returned non-zero exit code"
                        ),
                        recommended_action=str(
                            (revision_manifest_payload or {}).get("recommended_action")
                            or "inspect_local_revision_manifest"
                        ),
                    )
                    _refresh_summary_fields(payload)
                    _safe_write_manifest(lineage_path, payload)
                    return revision_exit_code or 1
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
        payload["evaluation_pointer"] = _planned_command_step(
            label="evaluation_pointer",
            command=pointer_command,
            output_path=evaluation_pointer_path,
        )
        _refresh_summary_fields(payload)
        _safe_write_manifest(lineage_path, payload)
        if args.dry_run:
            payload["evaluation_pointer_payload"] = _planned_evaluation_pointer_payload(
                revision_slug=revision_slug,
                universe=args.universe,
                source_scope=args.source_scope,
                baseline_reference=args.baseline_reference,
                config_path=args.model_config,
                data_config_path=args.data_config,
                feature_config_path=args.feature_config,
                evaluate_command=pointer_command,
            )
            _refresh_summary_fields(payload)
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
            _refresh_summary_fields(payload)
        progress.update(message="evaluation pointer handled")

        if args.dry_run:
            _set_step(payload, "planned")
            payload["status"] = "planned"
            payload["finished_at"] = utc_now_iso()
            _refresh_summary_fields(payload)
            _safe_write_manifest(lineage_path, payload)
            progress.complete(message="dry-run plan prepared")
            print(f"[local-revision-gate] planned manifest saved: {lineage_path}", flush=True)
            return 0

        _set_step(payload, "completed")
        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        _refresh_summary_fields(payload)
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
        _refresh_summary_fields(payload)
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
        _refresh_summary_fields(payload)
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
        _refresh_summary_fields(payload)
        _safe_write_manifest(lineage_path, payload)
        print(f"[local-revision-gate] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
