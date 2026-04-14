from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.local_nankan_trust import resolve_local_nankan_trust_block
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_CONFIG = "configs/model_local_baseline.yaml"
DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_FEATURE_CONFIG = "configs/features_local_baseline.yaml"
DEFAULT_POINTER_OUTPUT = "artifacts/reports/evaluation_local_nankan_pointer.json"
DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"


def log_progress(message: str) -> None:
    print(message, flush=True)


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _display_path_value(value: object) -> object:
    if not isinstance(value, str):
        return value
    root_text = str(ROOT)
    if value == root_text:
        return "."
    root_prefix = f"{root_text}/"
    if root_prefix in value:
        value = value.replace(root_prefix, "")
    if not value.startswith("/"):
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
    if isinstance(value, Path):
        return artifact_display_path(value, workspace_root=ROOT)
    return _display_path_value(value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--feature-config", default=DEFAULT_FEATURE_CONFIG)
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--model-artifact-suffix", default=None)
    parser.add_argument("--output", default=DEFAULT_POINTER_OUTPUT)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    args = parser.parse_args()
    progress = ProgressBar(total=3, prefix="[local-evaluate]", logger=log_progress, min_interval_sec=0.0)
    progress.start(
        message=(
            f"starting config={args.config} data_config={args.data_config} "
            f"max_rows={args.max_rows} wf_mode={args.wf_mode}"
        )
    )

    data_config = load_yaml(_resolve_path(args.data_config))
    trust_block = resolve_local_nankan_trust_block(
        workspace_root=ROOT,
        data_config=data_config,
        data_config_path=args.data_config,
        allow_diagnostic_override=False,
        command_name="evaluate",
    )
    if trust_block is not None:
        progress.update(message="historical local Nankan trust preflight blocked evaluate")
        pointer_path = _resolve_path(args.output)
        artifact_ensure_output_file_path(pointer_path, label="output", workspace_root=ROOT)
        payload: dict[str, object] = {
            "started_at": utc_now_iso(),
            "status": "blocked_by_trust",
            "universe": args.universe,
            "source_scope": args.source_scope,
            "baseline_reference": args.baseline_reference,
            "run_context": {
                "config": args.config,
                "data_config": args.data_config,
                "feature_config": args.feature_config,
                "max_rows": int(args.max_rows),
                "start_date": args.start_date,
                "end_date": args.end_date,
                "wf_mode": args.wf_mode,
                "wf_scheme": args.wf_scheme,
                "model_artifact_suffix": args.model_artifact_suffix,
            },
            "evaluate_command": None,
            "exit_code": 1,
            "error_code": str(trust_block.get("error_code") or "historical_local_nankan_trust_not_ready"),
            "error_message": str(trust_block.get("error_message") or "historical local Nankan trust is not ready"),
            "recommended_action": str(trust_block.get("recommended_action") or "inspect_local_nankan_provenance_audit"),
            "trust_context": {
                "strict_trust_ready": bool(trust_block.get("strict_trust_ready", False)),
                "pre_race": trust_block.get("pre_race"),
                "post_race": trust_block.get("post_race"),
                "unknown": trust_block.get("unknown"),
                "provenance_manifest": artifact_display_path(
                    Path(str(trust_block.get("provenance_manifest_path"))),
                    workspace_root=ROOT,
                ),
            },
            "finished_at": utc_now_iso(),
        }
        with Heartbeat("[local-evaluate]", "writing local evaluation pointer", logger=log_progress):
            write_json(pointer_path, _normalize_display_paths(payload))
        progress.complete(message=f"trust block pointer saved path={artifact_display_path(pointer_path, workspace_root=ROOT)}")
        print(f"[local-evaluate] blocked: {payload['error_message']}", flush=True)
        return 1

    command = [
        sys.executable,
        str(ROOT / "scripts/run_evaluate.py"),
        "--config",
        args.config,
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
    if args.start_date:
        command.extend(["--start-date", args.start_date])
    if args.end_date:
        command.extend(["--end-date", args.end_date])
    if args.model_artifact_suffix:
        command.extend(["--model-artifact-suffix", args.model_artifact_suffix])

    progress.update(message="launching delegated evaluate command")
    print(f"[local-evaluate] running: {' '.join(command)}", flush=True)
    with Heartbeat("[local-evaluate]", "evaluate child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    progress.update(message=f"evaluate command finished exit_code={int(result.returncode)}")

    pointer_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(pointer_path, label="output", workspace_root=ROOT)
    payload: dict[str, object] = {
        "started_at": utc_now_iso(),
        "status": "completed" if result.returncode == 0 else "failed",
        "universe": args.universe,
        "source_scope": args.source_scope,
        "baseline_reference": args.baseline_reference,
        "run_context": {
            "config": args.config,
            "data_config": args.data_config,
            "feature_config": args.feature_config,
            "max_rows": int(args.max_rows),
            "start_date": args.start_date,
            "end_date": args.end_date,
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "model_artifact_suffix": args.model_artifact_suffix,
        },
        "evaluate_command": command,
        "exit_code": int(result.returncode),
        "finished_at": utc_now_iso(),
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

    with Heartbeat("[local-evaluate]", "writing local evaluation pointer", logger=log_progress):
        write_json(pointer_path, _normalize_display_paths(payload))
    progress.complete(message=f"pointer saved path={artifact_display_path(pointer_path, workspace_root=ROOT)}")
    print(f"[local-evaluate] pointer saved: {pointer_path}", flush=True)
    return int(result.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
