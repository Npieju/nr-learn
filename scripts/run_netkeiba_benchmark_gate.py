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
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import inspect_dataset_sources


DEFAULT_UNIVERSE = "jra"
DEFAULT_SOURCE_SCOPE = "netkeiba"
DEFAULT_SCHEMA_VERSION = "netkeiba.benchmark_gate.v1"
DEFAULT_RACE_RESULT_PATH = "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
DEFAULT_RACE_CARD_PATH = "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"
DEFAULT_PEDIGREE_PATH = "data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv"


def _healthy_table_status(status: object, *, optional: bool) -> bool:
    normalized = str(status or "")
    if normalized in {"ok", "ok_materialized"}:
        return True
    return optional and normalized == "optional_missing"


def _recommended_action_for_primary(report: dict[str, object]) -> str:
    if not bool(report.get("raw_dir_exists")):
        return "populate_primary_raw_dir"
    return "add_primary_csv_to_raw_dir"


def _preflight_readiness(report: dict[str, object]) -> tuple[str, str | None, str | None, dict[str, object]]:
    primary = report.get("primary_dataset") if isinstance(report.get("primary_dataset"), dict) else {}
    append_tables = [row for row in (report.get("append_tables") or []) if isinstance(row, dict)]
    supplemental_tables = [row for row in (report.get("supplemental_tables") or []) if isinstance(row, dict)]

    missing_required_append = [
        str(row.get("name") or "")
        for row in append_tables
        if not _healthy_table_status(row.get("status"), optional=bool(row.get("optional")))
    ]
    missing_required_supplemental = [
        str(row.get("name") or "")
        for row in supplemental_tables
        if not _healthy_table_status(row.get("status"), optional=bool(row.get("optional")))
    ]

    readiness = {
        "benchmark_rerun_ready": True,
        "recommended_action": "run_local_benchmark",
        "reasons": [],
        "primary_dataset_status": primary.get("status"),
        "missing_required_append_tables": missing_required_append,
        "missing_required_supplemental_tables": missing_required_supplemental,
    }

    if str(primary.get("status") or "") != "ok":
        readiness["benchmark_rerun_ready"] = False
        readiness["recommended_action"] = _recommended_action_for_primary(report)
        readiness["reasons"] = [str(primary.get("error") or "primary dataset is missing")]
        return "not_ready", "primary_dataset_missing", str(primary.get("error") or "primary dataset is missing"), readiness

    if missing_required_append:
        readiness["benchmark_rerun_ready"] = False
        readiness["recommended_action"] = "populate_required_append_tables"
        readiness["reasons"] = [f"required append tables missing: {', '.join(missing_required_append)}"]
        return "not_ready", "required_append_tables_missing", readiness["reasons"][0], readiness

    if missing_required_supplemental:
        readiness["benchmark_rerun_ready"] = False
        readiness["recommended_action"] = "populate_required_supplemental_tables"
        readiness["reasons"] = [f"required supplemental tables missing: {', '.join(missing_required_supplemental)}"]
        return "not_ready", "required_supplemental_tables_missing", readiness["reasons"][0], readiness

    return "ready", None, None, readiness


def _build_preflight_payload(
    *,
    config_path: str,
    universe: str,
    source_scope: str,
    baseline_reference: str | None,
    output_path: Path,
    report: dict[str, object],
    status: str,
    error_code: str | None,
    error_message: str | None,
    recommended_action: str | None,
    readiness: dict[str, object],
) -> dict[str, object]:
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": status,
        "completed_step": "completed",
        "artifact_type": "dataset_source_preflight",
        "config": config_path,
        "universe": universe,
        "source_scope": source_scope,
        "baseline_reference": baseline_reference,
        "error_code": error_code,
        "error_message": error_message,
        "recommended_action": recommended_action,
        "artifacts": {
            "preflight_manifest": artifact_display_path(output_path, workspace_root=ROOT),
        },
        "readiness": readiness,
        "source_report": report,
    }


def _set_step(payload: dict[str, object], step_name: str) -> None:
    payload["completed_step"] = step_name


def _require_text(value: str, *, field_name: str, error_code: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{error_code}:{field_name} must not be empty")
    return normalized


def _normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    return normalized or None


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


def _read_order(*, include_preflight: bool, include_train: bool, include_evaluate: bool) -> list[str]:
    order = ["benchmark_gate_manifest"]
    if include_preflight:
        order.append("preflight")
    order.append("snapshot")
    if include_train:
        order.append("train")
    if include_evaluate:
        order.append("evaluate")
    return order


def _current_phase(payload: dict[str, object]) -> str:
    status = str(payload.get("status") or "")
    completed_step = str(payload.get("completed_step") or "")

    if status == "completed":
        return "completed"

    if status == "not_ready":
        if completed_step == "preflight_sources":
            return "preflight"
        return "validate_readiness"

    if status == "snapshot_failed":
        return "snapshot"
    if status == "train_failed":
        return "train"
    if status == "evaluate_failed":
        return "evaluate"
    if status == "interrupted":
        return completed_step or "interrupted"
    if status == "failed":
        if completed_step in {"run_snapshot", "validate_readiness", "run_train", "run_evaluate", "preflight_sources"}:
            phase_map = {
                "preflight_sources": "preflight",
                "run_snapshot": "snapshot",
                "validate_readiness": "validate_readiness",
                "run_train": "train",
                "run_evaluate": "evaluate",
            }
            return phase_map.get(completed_step, completed_step)
        return completed_step or "failed"

    phase_map = {
        "init": "init_manifest",
        "init_manifest": "init_manifest",
        "preflight_sources": "preflight",
        "run_snapshot": "snapshot",
        "validate_readiness": "validate_readiness",
        "run_train": "train",
        "run_evaluate": "evaluate",
        "write_manifest": "write_manifest",
    }
    return phase_map.get(completed_step, completed_step or "init_manifest")


def _highlights(payload: dict[str, object]) -> list[str]:
    status = str(payload.get("status") or "")
    current_phase = str(payload.get("current_phase") or _current_phase(payload))
    recommended_action = str(payload.get("recommended_action") or "inspect_benchmark_gate_manifest")
    error_code = str(payload.get("error_code") or "")
    universe = str(payload.get("universe") or "unknown")
    include_train = not bool(((payload.get("configs") or {}) if isinstance(payload.get("configs"), dict) else {}).get("skip_train"))
    include_evaluate = not bool(((payload.get("configs") or {}) if isinstance(payload.get("configs"), dict) else {}).get("skip_evaluate"))

    if status == "completed":
        highlights = [
            f"benchmark gate completed for universe={universe}",
            "snapshot readiness reached benchmark_rerun_ready=true",
        ]
        if include_train and include_evaluate:
            highlights.append("train and evaluate completed; downstream evaluation artifacts are ready")
        elif include_train and not include_evaluate:
            highlights.append("train completed and evaluate was skipped by configuration")
        elif include_evaluate:
            highlights.append("evaluate completed after readiness check while train was skipped by configuration")
        else:
            highlights.append("train and evaluate were skipped by configuration after readiness check")
        return highlights

    if status in {"not_ready", "snapshot_failed", "train_failed", "evaluate_failed", "failed", "interrupted"}:
        message = str(payload.get("error_message") or status)
        summary = f"benchmark gate stopped during {current_phase} for universe={universe}"
        if status == "not_ready":
            summary = f"benchmark gate is not ready during {current_phase} for universe={universe}"
        highlights = [summary]
        if error_code:
            highlights.append(f"error_code={error_code}: {message}")
        else:
            highlights.append(message)
        highlights.append(f"next operator action: {recommended_action}")
        return highlights

    return [
        f"benchmark gate is in progress at {current_phase} for universe={universe}",
        f"next operator action: {recommended_action}",
    ]


def _refresh_summary_fields(payload: dict[str, object]) -> None:
    configs = payload.get("configs") if isinstance(payload.get("configs"), dict) else {}
    include_preflight = bool(configs.get("preflight_output"))
    include_train = not bool(configs.get("skip_train"))
    include_evaluate = not bool(configs.get("skip_evaluate"))
    payload["read_order"] = _read_order(
        include_preflight=include_preflight,
        include_train=include_train,
        include_evaluate=include_evaluate,
    )
    payload["current_phase"] = _current_phase(payload)
    payload["highlights"] = _highlights(payload)


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
    parser.add_argument("--preflight-output", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_path(args.manifest_output)
    snapshot_path = _resolve_path(args.snapshot_output)
    preflight_path = _resolve_path(args.preflight_output) if args.preflight_output else None
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
            "preflight_output": artifact_display_path(preflight_path, workspace_root=ROOT) if preflight_path is not None else None,
        },
    }
    _refresh_summary_fields(payload)

    try:
        universe = _require_text(args.universe, field_name="universe", error_code="missing_universe")
        source_scope = _require_text(args.source_scope, field_name="source_scope", error_code="missing_source_scope")
        schema_version = _require_text(args.schema_version, field_name="schema_version", error_code="missing_schema_version")
        baseline_reference = _normalize_optional_text(args.baseline_reference)
        payload.update(
            {
                "schema_version": schema_version,
                "universe": universe,
                "source_scope": source_scope,
                "baseline_reference": baseline_reference,
            }
        )
        _refresh_summary_fields(payload)
        _set_step(payload, "init_manifest")
        _refresh_summary_fields(payload)
        artifact_ensure_output_file_path(manifest_path, label="manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(snapshot_path, label="snapshot output", workspace_root=ROOT)
        if preflight_path is not None:
            artifact_ensure_output_file_path(preflight_path, label="preflight output", workspace_root=ROOT)
        _refresh_summary_fields(payload)
        _safe_write_manifest(manifest_path, payload)

        total_steps = 3 + int(preflight_path is not None) + int(not args.skip_train) + int(not args.skip_evaluate)
        progress = ProgressBar(total=total_steps, prefix="[netkeiba-benchmark-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="gate manifest initialized")

        if preflight_path is not None:
            _set_step(payload, "preflight_sources")
            data_cfg = load_yaml(ROOT / args.data_config)
            dataset_cfg = data_cfg.get("dataset", {}) if isinstance(data_cfg, dict) else {}
            raw_dir = dataset_cfg.get("raw_dir", "data/raw") if isinstance(dataset_cfg, dict) else "data/raw"
            with Heartbeat("[netkeiba-benchmark-gate]", "running source preflight", logger=log_progress):
                preflight_report = inspect_dataset_sources(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
            preflight_status, preflight_error_code, preflight_error_message, preflight_readiness = _preflight_readiness(preflight_report)
            preflight_payload = _build_preflight_payload(
                config_path=args.data_config,
                universe=universe,
                source_scope=source_scope,
                baseline_reference=baseline_reference,
                output_path=preflight_path,
                report=preflight_report,
                status=preflight_status,
                error_code=preflight_error_code,
                error_message=preflight_error_message,
                recommended_action=preflight_readiness.get("recommended_action") if isinstance(preflight_readiness, dict) else None,
                readiness=preflight_readiness,
            )
            write_json(preflight_path, preflight_payload)
            payload["preflight"] = {
                "status": preflight_status,
                "output": artifact_display_path(preflight_path, workspace_root=ROOT),
            }
            payload["preflight_payload"] = preflight_payload
            payload["readiness"] = preflight_readiness
            if preflight_status != "ready":
                _set_failure(
                    payload,
                    status="not_ready",
                    error_code=str(preflight_error_code or "source_preflight_not_ready"),
                    error_message=str(preflight_error_message or "source preflight reported not ready"),
                    recommended_action=str(preflight_readiness.get("recommended_action") or "inspect_source_preflight"),
                )
                _refresh_summary_fields(payload)
                _safe_write_manifest(manifest_path, payload)
                progress.complete(message="source preflight says not ready")
                print(
                    "[netkeiba-benchmark-gate] "
                    f"preflight not ready: action={preflight_readiness.get('recommended_action')} reasons={preflight_readiness.get('reasons')}",
                    flush=True,
                )
                return 2
            progress.update(message="source preflight completed")

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
            _refresh_summary_fields(payload)
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
            _refresh_summary_fields(payload)
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
                _refresh_summary_fields(payload)
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
                _refresh_summary_fields(payload)
                _safe_write_manifest(manifest_path, payload)
                return int(evaluate_result["exit_code"]) or 1
            progress.update(message="evaluate completed")

        _set_step(payload, "write_manifest")
        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        payload["error_code"] = None
        payload["error_message"] = None
        payload["recommended_action"] = None
        _refresh_summary_fields(payload)
        with Heartbeat("[netkeiba-benchmark-gate]", "writing gate manifest", logger=log_progress):
            _safe_write_manifest(manifest_path, payload)
        _set_step(payload, "completed")
        _refresh_summary_fields(payload)
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
        _refresh_summary_fields(payload)
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
            "primary_dataset_missing",
            "required_append_tables_missing",
            "required_supplemental_tables_missing",
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
        _refresh_summary_fields(payload)
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
        _refresh_summary_fields(payload)
        _safe_write_manifest(manifest_path, payload)
        print(f"[netkeiba-benchmark-gate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())