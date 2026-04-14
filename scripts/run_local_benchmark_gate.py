from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml

DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_MODEL_CONFIG = "configs/model_local_baseline.yaml"
DEFAULT_FEATURE_CONFIG = "configs/features_local_baseline.yaml"
DEFAULT_SNAPSHOT_OUTPUT = "artifacts/reports/coverage_snapshot_local_nankan.json"
DEFAULT_MANIFEST_OUTPUT = "artifacts/reports/benchmark_gate_local_nankan.json"
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_SCHEMA_VERSION = "local.benchmark_gate.v1"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RACE_RESULT_PATH = "data/external/local_nankan/results/local_race_result.csv"
DEFAULT_RACE_CARD_PATH = "data/external/local_nankan/racecard/local_racecard.csv"
DEFAULT_PEDIGREE_PATH = "data/external/local_nankan/pedigree/local_pedigree.csv"
DEFAULT_PREFLIGHT_OUTPUT = "artifacts/reports/data_preflight_local_nankan.json"
DEFAULT_PRIMARY_MATERIALIZE_MANIFEST = "artifacts/reports/local_nankan_primary_materialize_manifest.json"
DEFAULT_PROVENANCE_SUMMARY_OUTPUT = "artifacts/reports/local_nankan_provenance_summary.json"
DEFAULT_PROVENANCE_MANIFEST_OUTPUT = "artifacts/reports/local_nankan_provenance_audit.json"


def log_progress(message: str) -> None:
    print(message, flush=True)


def _run_command(*, label: str, command: list[str]) -> subprocess.CompletedProcess[bytes]:
    print(f"[local-benchmark-gate] running {label}: {' '.join(command)}", flush=True)
    with Heartbeat("[local-benchmark-gate]", f"{label} child command", logger=log_progress):
        return subprocess.run(command, cwd=ROOT, check=False)


def _load_generated_files(manifest_file: str) -> dict[str, str]:
    manifest_path = Path(manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = ROOT / manifest_path
    if not manifest_path.exists():
        return {}
    payload = json.loads(manifest_path.read_text())
    generated_files = payload.get("generated_files")
    return generated_files if isinstance(generated_files, dict) else {}


def _resolve_primary_input_path(*, data_config_path: str, override_path: str | None) -> Path:
    if override_path:
        path = Path(override_path)
        return path if path.is_absolute() else (ROOT / path)
    config_payload = load_yaml(ROOT / data_config_path)
    dataset_cfg = config_payload.get("dataset") if isinstance(config_payload.get("dataset"), dict) else config_payload
    raw_dir = dataset_cfg.get("raw_dir", "data/local_nankan/raw") if isinstance(dataset_cfg, dict) else "data/local_nankan/raw"
    raw_path = Path(str(raw_dir))
    if not raw_path.is_absolute():
        raw_path = ROOT / raw_path
    return raw_path / "local_nankan_primary.csv"


def _display_path_value(value: object) -> object:
    if not isinstance(value, str) or not value.startswith("/"):
        return value
    try:
        path = Path(value)
    except (TypeError, ValueError):
        return value
    if not path.is_absolute():
        return value
    try:
        path.relative_to(ROOT)
    except ValueError:
        return value
    return artifact_display_path(path, workspace_root=ROOT)


def _normalize_display_paths(value: object) -> object:
    if isinstance(value, dict):
        return {key: _normalize_display_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_display_paths(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_display_paths(item) for item in value]
    return _display_path_value(value)


def _write_provenance_block_outputs(
    *,
    args: argparse.Namespace,
    provenance_manifest_path: Path,
    provenance_summary_path: Path,
) -> None:
    provenance_payload = json.loads(provenance_manifest_path.read_text(encoding="utf-8")) if provenance_manifest_path.exists() else {}
    readiness = provenance_payload.get("readiness") if isinstance(provenance_payload.get("readiness"), dict) else {}
    recommended_action = str(readiness.get("recommended_action") or provenance_payload.get("recommended_action") or "inspect_local_nankan_provenance_audit")
    blocking_reasons = readiness.get("blocking_reasons") if isinstance(readiness.get("blocking_reasons"), list) else []
    benchmark_manifest = {
        "schema_version": args.schema_version,
        "artifact_type": "benchmark_gate_manifest",
        "started_at": None,
        "finished_at": None,
        "status": "not_ready",
        "completed_step": "provenance_preflight",
        "current_phase": "provenance_preflight",
        "universe": args.universe,
        "source_scope": args.source_scope,
        "baseline_reference": args.baseline_reference,
        "error_code": "market_provenance_not_ready",
        "error_message": str(provenance_payload.get("error_message") or "local market provenance trust gate blocked benchmark"),
        "recommended_action": recommended_action,
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
            "preflight_output": args.preflight_output,
            "provenance_summary_output": args.provenance_summary_output,
            "provenance_manifest_output": args.provenance_manifest_output,
        },
        "readiness": {
            "benchmark_rerun_ready": False,
            "recommended_action": recommended_action,
            "reasons": blocking_reasons or ["strict local market provenance trust gate blocked benchmark rerun"],
        },
        "provenance_audit": provenance_payload,
        "artifacts": {
            "manifest": args.manifest_output,
            "preflight_manifest": args.preflight_output,
            "provenance_summary": args.provenance_summary_output,
            "provenance_manifest": args.provenance_manifest_output,
        },
        "highlights": [
            "local benchmark gate stopped before snapshot because strict market provenance trust requirements were not satisfied",
            f"next operator action: {recommended_action}",
        ],
    }
    preflight_payload = {
        "started_at": None,
        "finished_at": None,
        "status": "not_ready",
        "completed_step": "provenance_preflight",
        "artifact_type": "dataset_source_preflight",
        "config": args.data_config,
        "universe": args.universe,
        "source_scope": args.source_scope,
        "baseline_reference": args.baseline_reference,
        "error_code": "market_provenance_not_ready",
        "error_message": str(provenance_payload.get("error_message") or "local market provenance trust gate blocked benchmark"),
        "recommended_action": recommended_action,
        "artifacts": {
            "preflight_manifest": args.preflight_output,
            "provenance_summary": args.provenance_summary_output,
            "provenance_manifest": args.provenance_manifest_output,
        },
        "readiness": {
            "benchmark_rerun_ready": False,
            "recommended_action": recommended_action,
            "reasons": blocking_reasons or ["strict local market provenance trust gate blocked benchmark rerun"],
        },
        "source_report": {
            "local_market_provenance_summary": json.loads(provenance_summary_path.read_text(encoding="utf-8")) if provenance_summary_path.exists() else None,
        },
    }
    write_json(ROOT / args.manifest_output, _normalize_display_paths(benchmark_manifest))
    write_json(ROOT / args.preflight_output, _normalize_display_paths(preflight_payload))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--model-config", default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--snapshot-output", default=DEFAULT_SNAPSHOT_OUTPUT)
    parser.add_argument("--manifest-output", default=DEFAULT_MANIFEST_OUTPUT)
    parser.add_argument("--max-rows", type=int, default=200000)
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="off")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--preflight-output", default=DEFAULT_PREFLIGHT_OUTPUT)
    parser.add_argument("--materialize-primary-before-gate", action="store_true")
    parser.add_argument("--materialize-output-file", default=None)
    parser.add_argument("--materialize-manifest-file", default=DEFAULT_PRIMARY_MATERIALIZE_MANIFEST)
    parser.add_argument("--skip-provenance-audit", action="store_true")
    parser.add_argument("--provenance-input-file", default=None)
    parser.add_argument("--provenance-summary-output", default=DEFAULT_PROVENANCE_SUMMARY_OUTPUT)
    parser.add_argument("--provenance-manifest-output", default=DEFAULT_PROVENANCE_MANIFEST_OUTPUT)
    parser.add_argument("--provenance-max-unknown-ratio", type=float, default=0.0)
    parser.add_argument("--provenance-max-post-race-ratio", type=float, default=0.0)
    parser.add_argument("--provenance-min-pre-race-rows", type=int, default=1)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    args = parser.parse_args()
    progress = ProgressBar(total=3, prefix="[local-benchmark-gate]", logger=log_progress, min_interval_sec=0.0)
    progress.start(
        message=(
            f"starting data_config={args.data_config} model_config={args.model_config} "
            f"feature_config={args.feature_config}"
        )
    )

    if args.materialize_primary_before_gate:
        materialize_command = [
            sys.executable,
            str(ROOT / "scripts/run_materialize_local_nankan_primary.py"),
            "--data-config",
            args.data_config,
            "--race-result-path",
            args.race_result_path,
            "--race-card-path",
            args.race_card_path,
            "--pedigree-path",
            args.pedigree_path,
            "--manifest-file",
            args.materialize_manifest_file,
        ]
        if args.materialize_output_file:
            materialize_command.extend(["--output-file", args.materialize_output_file])
        materialize_result = _run_command(label="primary_materialize", command=materialize_command)
        progress.update(message=f"primary materialize finished exit_code={int(materialize_result.returncode)}")
        if int(materialize_result.returncode) not in {0, 2}:
            return int(materialize_result.returncode)
        if int(materialize_result.returncode) == 2:
            print(
                "[local-benchmark-gate] primary materialize not ready; continuing to benchmark gate so preflight can emit the formal blocker",
                flush=True,
            )

    race_card_path = args.race_card_path
    pedigree_path = args.pedigree_path
    if args.materialize_primary_before_gate:
        with Heartbeat("[local-benchmark-gate]", "loading generated primary files", logger=log_progress):
            generated_files = _load_generated_files(args.materialize_manifest_file)
        progress.update(message="resolved generated primary file paths")
        race_card_path = str(generated_files.get("local_nankan_race_card") or race_card_path)
        pedigree_path = str(generated_files.get("local_nankan_pedigree") or pedigree_path)
    else:
        progress.update(message="primary materialize skipped")

    provenance_input_path = _resolve_primary_input_path(
        data_config_path=args.data_config,
        override_path=args.provenance_input_file or args.materialize_output_file,
    )
    provenance_manifest_path = ROOT / args.provenance_manifest_output
    provenance_summary_path = ROOT / args.provenance_summary_output
    if args.skip_provenance_audit:
        progress.update(message="provenance audit skipped by configuration")
    elif not provenance_input_path.exists():
        progress.update(message=f"provenance input missing; defer to source preflight path input={provenance_input_path}")
    else:
        provenance_command = [
            sys.executable,
            str(ROOT / "scripts/run_local_nankan_provenance_audit.py"),
            "--input-file",
            str(provenance_input_path),
            "--summary-output",
            args.provenance_summary_output,
            "--manifest-output",
            args.provenance_manifest_output,
            "--fail-on-missing-columns",
            "--max-unknown-ratio",
            str(args.provenance_max_unknown_ratio),
            "--max-post-race-ratio",
            str(args.provenance_max_post_race_ratio),
            "--min-pre-race-rows",
            str(args.provenance_min_pre_race_rows),
        ]
        provenance_result = _run_command(label="provenance_audit", command=provenance_command)
        progress.update(message=f"provenance audit finished exit_code={int(provenance_result.returncode)}")
        if int(provenance_result.returncode) == 2:
            _write_provenance_block_outputs(
                args=args,
                provenance_manifest_path=provenance_manifest_path,
                provenance_summary_path=provenance_summary_path,
            )
            print(
                "[local-benchmark-gate] strict market provenance trust gate blocked benchmark rerun; see provenance manifest for details",
                flush=True,
            )
            return 2
        if int(provenance_result.returncode) != 0:
            return int(provenance_result.returncode)

    command = [
        sys.executable,
        str(ROOT / "scripts/run_netkeiba_benchmark_gate.py"),
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
        args.manifest_output,
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
        "--schema-version",
        args.schema_version,
        "--baseline-reference",
        args.baseline_reference,
        "--race-result-path",
        args.race_result_path,
        "--race-card-path",
        race_card_path,
        "--pedigree-path",
        pedigree_path,
        "--preflight-output",
        args.preflight_output,
    ]
    if args.pre_feature_max_rows is not None:
        command.extend(["--pre-feature-max-rows", str(args.pre_feature_max_rows)])
    if args.skip_train:
        command.append("--skip-train")
    if args.skip_evaluate:
        command.append("--skip-evaluate")

    result = _run_command(label="benchmark_gate", command=command)
    progress.complete(message=f"benchmark gate finished exit_code={int(result.returncode)} manifest={args.manifest_output}")
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
