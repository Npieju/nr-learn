from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json
from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles, resolve_model_run_profile
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[revision-gate {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _normalize_revision_slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("revision must not be empty")
    return normalized


def _sanitize_output_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def _derive_date_window_slug(start_date: str | None, end_date: str | None) -> str:
    if not start_date and not end_date:
        return ""

    start_token = _sanitize_output_slug((start_date or "start_auto").replace("-", ""))
    end_token = _sanitize_output_slug((end_date or "end_auto").replace("-", ""))
    return f"_{start_token}_{end_token}"


def _derive_wf_slug(wf_mode: str, wf_scheme: str) -> str:
    return f"_wf_{_sanitize_output_slug(wf_mode)}_{_sanitize_output_slug(wf_scheme)}"


def _derive_wf_output_slug(config_path: str) -> str:
    stem = Path(config_path).stem.strip()
    if stem.endswith("_model"):
        stem = stem[: -len("_model")]
    if stem.startswith("model_"):
        stem = stem[len("model_") :]
    return _sanitize_output_slug(stem)


def _extend_profile_or_config_args(
    command: list[str],
    *,
    profile: str | None,
    config_path: str,
    data_config_path: str,
    feature_config_path: str,
) -> None:
    if profile:
        command.extend(["--profile", profile])
        return
    command.extend(
        [
            "--config",
            config_path,
            "--data-config",
            data_config_path,
            "--feature-config",
            feature_config_path,
        ]
    )


def _run_command(command: list[str], *, label: str) -> None:
    log_progress(f"running {label}: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def _run_command_result(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    log_progress(f"running {label}: {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=False)


def _planned_step(name: str, command: list[str]) -> dict[str, object]:
    return {"name": name, "command": command, "status": "planned"}


def _read_json_dict(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _build_manifest_payload(
    *,
    revision_slug: str,
    started_at: str,
    status: str,
    decision: str,
    resolved_profile: str | None,
    config_path: str,
    data_config_path: str,
    feature_config_path: str,
    train_artifact_suffix: str,
    skip_train: bool,
    train_max_train_rows: int | None,
    train_max_valid_rows: int | None,
    evaluate_model_artifact_suffix: str | None,
    evaluate_max_rows: int,
    evaluate_pre_feature_max_rows: int | None,
    evaluate_start_date: str | None,
    evaluate_end_date: str | None,
    evaluate_wf_mode: str,
    evaluate_wf_scheme: str,
    wf_summary_output: Path,
    promotion_min_feasible_folds: int,
    promotion_output: Path,
    manifest_output: Path,
    executed_steps: list[dict[str, object]],
    promotion_report: dict[str, object] | None = None,
    dry_run: bool = False,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "generated_at": utc_now_iso(),
        "started_at": started_at,
        "revision": revision_slug,
        "status": status,
        "decision": decision,
        "profile": resolved_profile,
        "config": config_path,
        "data_config": data_config_path,
        "feature_config": feature_config_path,
        "train_artifact_suffix": train_artifact_suffix,
        "training": {
            "skipped": bool(skip_train),
            "max_train_rows": int(train_max_train_rows) if train_max_train_rows is not None else None,
            "max_valid_rows": int(train_max_valid_rows) if train_max_valid_rows is not None else None,
        },
        "evaluation": {
            "model_artifact_suffix": evaluate_model_artifact_suffix,
            "max_rows": int(evaluate_max_rows),
            "pre_feature_max_rows": int(evaluate_pre_feature_max_rows) if evaluate_pre_feature_max_rows is not None else None,
            "start_date": evaluate_start_date,
            "end_date": evaluate_end_date,
            "wf_mode": evaluate_wf_mode,
            "wf_scheme": evaluate_wf_scheme,
        },
        "promotion_gate": {
            "min_feasible_folds": int(promotion_min_feasible_folds),
            "output": artifact_display_path(promotion_output, workspace_root=ROOT),
            "summary": (promotion_report or {}).get("summary") if isinstance(promotion_report, dict) else None,
            "formal_benchmark": (promotion_report or {}).get("formal_benchmark") if isinstance(promotion_report, dict) else None,
        },
        "steps": executed_steps,
        "artifacts": {
            "wf_summary": artifact_display_path(wf_summary_output, workspace_root=ROOT),
            "promotion_report": artifact_display_path(promotion_output, workspace_root=ROOT),
            "evaluation_manifest": "artifacts/reports/evaluation_manifest.json",
            "evaluation_summary": "artifacts/reports/evaluation_summary.json",
            "revision_manifest": artifact_display_path(manifest_output, workspace_root=ROOT),
        },
    }
    if dry_run:
        payload["dry_run"] = True
    return payload


def _write_manifest(manifest_output: Path, payload: dict[str, object], *, label: str) -> None:
    with Heartbeat("[revision-gate]", label, logger=log_progress):
        write_json(manifest_output, payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--revision", required=False, default=None)
    parser.add_argument("--train-artifact-suffix", default=None)
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--train-max-train-rows", type=int, default=None)
    parser.add_argument("--train-max-valid-rows", type=int, default=None)
    parser.add_argument("--evaluate-model-artifact-suffix", default=None)
    parser.add_argument("--evaluate-max-rows", type=int, default=120000)
    parser.add_argument("--evaluate-pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--evaluate-start-date", default=None)
    parser.add_argument("--evaluate-end-date", default=None)
    parser.add_argument("--evaluate-wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--evaluate-wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--promotion-min-feasible-folds", type=int, default=1)
    parser.add_argument("--promotion-output", default=None)
    parser.add_argument("--wf-summary-output", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        if args.list_profiles:
            print(format_model_run_profiles())
            return 0

        if args.profile and any(value is not None for value in (args.config, args.data_config, args.feature_config)):
            raise ValueError("--profile cannot be combined with --config, --data-config, or --feature-config")

        resolved_profile, config_path, data_config_path, feature_config_path = resolve_model_run_profile(
            args.profile,
            default_model_config=args.config or "configs/model.yaml",
            default_data_config=args.data_config or "configs/data.yaml",
            default_feature_config=args.feature_config or "configs/features.yaml",
        )
        revision_value = args.revision or f"revision_{time.strftime('%Y%m%d_%H%M%S')}"
        revision_slug = _normalize_revision_slug(revision_value)
        train_artifact_suffix = str(args.train_artifact_suffix or revision_slug).strip()
        if not train_artifact_suffix:
            raise ValueError("train artifact suffix must not be empty")

        report_dir = ROOT / "artifacts" / "reports"
        promotion_output = _resolve_path(args.promotion_output) if args.promotion_output else report_dir / f"promotion_gate_{revision_slug}.json"
        wf_summary_output = (
            _resolve_path(args.wf_summary_output)
            if args.wf_summary_output
            else report_dir
            / (
                f"wf_feasibility_diag_{_derive_wf_output_slug(config_path)}"
                f"{_derive_date_window_slug(args.evaluate_start_date, args.evaluate_end_date)}"
                f"{_derive_wf_slug(args.evaluate_wf_mode, args.evaluate_wf_scheme)}.json"
            )
        )
        manifest_output = _resolve_path(args.manifest_output) if args.manifest_output else report_dir / f"revision_gate_{revision_slug}.json"
        artifact_ensure_output_file_path(promotion_output, label="promotion output", workspace_root=ROOT)
        artifact_ensure_output_file_path(wf_summary_output, label="wf summary output", workspace_root=ROOT)
        artifact_ensure_output_file_path(manifest_output, label="manifest output", workspace_root=ROOT)
        manifest_output.parent.mkdir(parents=True, exist_ok=True)
        promotion_output.parent.mkdir(parents=True, exist_ok=True)
        wf_summary_output.parent.mkdir(parents=True, exist_ok=True)

        progress = ProgressBar(total=4, prefix="[revision-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"starting revision={revision_slug} profile={resolved_profile or 'custom'} config={config_path} "
                f"data_config={data_config_path} feature_config={feature_config_path} "
                f"train_max_train_rows={args.train_max_train_rows or 'config'} train_max_valid_rows={args.train_max_valid_rows or 'config'} "
                f"skip_train={args.skip_train} evaluate_model_artifact_suffix={args.evaluate_model_artifact_suffix or 'none'}"
            )
        )

        train_command = [sys.executable, "scripts/run_train.py"]
        _extend_profile_or_config_args(
            train_command,
            profile=resolved_profile,
            config_path=config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
        )
        train_command.extend(["--artifact-suffix", train_artifact_suffix])
        if args.train_max_train_rows is not None:
            train_command.extend(["--max-train-rows", str(args.train_max_train_rows)])
        if args.train_max_valid_rows is not None:
            train_command.extend(["--max-valid-rows", str(args.train_max_valid_rows)])

        evaluate_command = [
            sys.executable,
            "scripts/run_evaluate.py",
            "--max-rows",
            str(args.evaluate_max_rows),
            "--wf-mode",
            args.evaluate_wf_mode,
            "--wf-scheme",
            args.evaluate_wf_scheme,
        ]
        _extend_profile_or_config_args(
            evaluate_command,
            profile=resolved_profile,
            config_path=config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
        )
        evaluate_command.extend(["--artifact-suffix", train_artifact_suffix])
        if args.evaluate_model_artifact_suffix:
            evaluate_command.extend(["--model-artifact-suffix", str(args.evaluate_model_artifact_suffix)])
        if args.evaluate_pre_feature_max_rows is not None:
            evaluate_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
        if args.evaluate_start_date:
            evaluate_command.extend(["--start-date", str(args.evaluate_start_date)])
        if args.evaluate_end_date:
            evaluate_command.extend(["--end-date", str(args.evaluate_end_date)])

        wf_command = [
            sys.executable,
            "scripts/run_wf_feasibility_diag.py",
            "--wf-mode",
            args.evaluate_wf_mode,
            "--wf-scheme",
            args.evaluate_wf_scheme,
        ]
        _extend_profile_or_config_args(
            wf_command,
            profile=resolved_profile,
            config_path=config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
        )
        wf_command.extend(["--artifact-suffix", train_artifact_suffix])
        if args.evaluate_model_artifact_suffix:
            wf_command.extend(["--model-artifact-suffix", str(args.evaluate_model_artifact_suffix)])
        if args.evaluate_pre_feature_max_rows is not None:
            wf_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
        if args.evaluate_start_date:
            wf_command.extend(["--start-date", str(args.evaluate_start_date)])
        if args.evaluate_end_date:
            wf_command.extend(["--end-date", str(args.evaluate_end_date)])

        promotion_command = [
            sys.executable,
            "scripts/run_promotion_gate.py",
            "--evaluation-manifest",
            "artifacts/reports/evaluation_manifest.json",
            "--wf-summary",
            artifact_display_path(wf_summary_output, workspace_root=ROOT),
            "--min-feasible-folds",
            str(args.promotion_min_feasible_folds),
            "--output",
            artifact_display_path(promotion_output, workspace_root=ROOT),
        ]

        started_at = utc_now_iso()
        executed_steps: list[dict[str, object]] = []
        status = "pass"
        decision = "promote"

        if args.dry_run:
            executed_steps = [
                _planned_step("train", train_command),
                _planned_step("evaluate", evaluate_command),
                _planned_step("wf_feasibility", wf_command),
                _planned_step("promotion_gate", promotion_command),
            ]
            status = "dry_run"
            decision = "not_run"
            progress.update(message=f"planned train artifact_suffix={train_artifact_suffix}")
            progress.update(message="planned evaluation")
            progress.update(message="planned wf feasibility")

            manifest_payload = _build_manifest_payload(
                revision_slug=revision_slug,
                started_at=started_at,
                status=status,
                decision=decision,
                resolved_profile=resolved_profile,
                config_path=config_path,
                data_config_path=data_config_path,
                feature_config_path=feature_config_path,
                train_artifact_suffix=train_artifact_suffix,
                skip_train=bool(args.skip_train),
                train_max_train_rows=args.train_max_train_rows,
                train_max_valid_rows=args.train_max_valid_rows,
                evaluate_model_artifact_suffix=args.evaluate_model_artifact_suffix,
                evaluate_max_rows=args.evaluate_max_rows,
                evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                evaluate_start_date=args.evaluate_start_date,
                evaluate_end_date=args.evaluate_end_date,
                evaluate_wf_mode=args.evaluate_wf_mode,
                evaluate_wf_scheme=args.evaluate_wf_scheme,
                wf_summary_output=wf_summary_output,
                promotion_min_feasible_folds=args.promotion_min_feasible_folds,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=None,
                dry_run=True,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing dry-run manifest")
            progress.complete(message="dry-run plan prepared")
            print(f"[revision-gate] dry-run manifest saved: {manifest_output}")
            print(f"[revision-gate] train command: {' '.join(train_command)}")
            print(f"[revision-gate] evaluate command: {' '.join(evaluate_command)}")
            print(f"[revision-gate] wf command: {' '.join(wf_command)}")
            print(f"[revision-gate] promotion command: {' '.join(promotion_command)}")
            return 0

        if args.skip_train:
            executed_steps.append(
                {
                    "name": "train",
                    "command": train_command,
                    "status": "skipped",
                    "reason": "skip_train",
                }
            )
            progress.update(message=f"train skipped artifact_suffix={train_artifact_suffix}")
        else:
            train_result = _run_command_result(train_command, label="train")
            executed_steps.append({
                "name": "train",
                "command": train_command,
                "status": "completed" if train_result.returncode == 0 else "failed",
                "return_code": int(train_result.returncode),
            })
            if train_result.returncode != 0:
                status = "failed"
                decision = "error"
                manifest_payload = _build_manifest_payload(
                    revision_slug=revision_slug,
                    started_at=started_at,
                    status=status,
                    decision=decision,
                    resolved_profile=resolved_profile,
                    config_path=config_path,
                    data_config_path=data_config_path,
                    feature_config_path=feature_config_path,
                    train_artifact_suffix=train_artifact_suffix,
                    skip_train=bool(args.skip_train),
                    train_max_train_rows=args.train_max_train_rows,
                    train_max_valid_rows=args.train_max_valid_rows,
                    evaluate_model_artifact_suffix=args.evaluate_model_artifact_suffix,
                    evaluate_max_rows=args.evaluate_max_rows,
                    evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                    evaluate_start_date=args.evaluate_start_date,
                    evaluate_end_date=args.evaluate_end_date,
                    evaluate_wf_mode=args.evaluate_wf_mode,
                    evaluate_wf_scheme=args.evaluate_wf_scheme,
                    promotion_min_feasible_folds=args.promotion_min_feasible_folds,
                    promotion_output=promotion_output,
                    manifest_output=manifest_output,
                    executed_steps=executed_steps,
                    promotion_report=None,
                )
                _write_manifest(manifest_output, manifest_payload, label="writing failed revision manifest")
                print(f"[revision-gate] manifest saved: {manifest_output}")
                print("[revision-gate] decision: error")
                return int(train_result.returncode) or 1
            progress.update(message=f"train completed artifact_suffix={train_artifact_suffix}")

        evaluate_result = _run_command_result(evaluate_command, label="evaluate")
        executed_steps.append({
            "name": "evaluate",
            "command": evaluate_command,
            "status": "completed" if evaluate_result.returncode == 0 else "failed",
            "return_code": int(evaluate_result.returncode),
        })
        if evaluate_result.returncode != 0:
            status = "failed"
            decision = "error"
            manifest_payload = _build_manifest_payload(
                revision_slug=revision_slug,
                started_at=started_at,
                status=status,
                decision=decision,
                resolved_profile=resolved_profile,
                config_path=config_path,
                data_config_path=data_config_path,
                feature_config_path=feature_config_path,
                train_artifact_suffix=train_artifact_suffix,
                skip_train=bool(args.skip_train),
                train_max_train_rows=args.train_max_train_rows,
                train_max_valid_rows=args.train_max_valid_rows,
                evaluate_model_artifact_suffix=args.evaluate_model_artifact_suffix,
                evaluate_max_rows=args.evaluate_max_rows,
                evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                evaluate_start_date=args.evaluate_start_date,
                evaluate_end_date=args.evaluate_end_date,
                evaluate_wf_mode=args.evaluate_wf_mode,
                evaluate_wf_scheme=args.evaluate_wf_scheme,
                wf_summary_output=wf_summary_output,
                promotion_min_feasible_folds=args.promotion_min_feasible_folds,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=None,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing failed revision manifest")
            print(f"[revision-gate] manifest saved: {manifest_output}")
            print("[revision-gate] decision: error")
            return int(evaluate_result.returncode) or 1
        progress.update(message="evaluation completed")

        wf_result = _run_command_result(wf_command, label="wf_feasibility")
        executed_steps.append({
            "name": "wf_feasibility",
            "command": wf_command,
            "status": "completed" if wf_result.returncode == 0 else "failed",
            "return_code": int(wf_result.returncode),
        })
        if wf_result.returncode != 0:
            status = "failed"
            decision = "error"
            manifest_payload = _build_manifest_payload(
                revision_slug=revision_slug,
                started_at=started_at,
                status=status,
                decision=decision,
                resolved_profile=resolved_profile,
                config_path=config_path,
                data_config_path=data_config_path,
                feature_config_path=feature_config_path,
                train_artifact_suffix=train_artifact_suffix,
                skip_train=bool(args.skip_train),
                train_max_train_rows=args.train_max_train_rows,
                train_max_valid_rows=args.train_max_valid_rows,
                evaluate_model_artifact_suffix=args.evaluate_model_artifact_suffix,
                evaluate_max_rows=args.evaluate_max_rows,
                evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
                evaluate_start_date=args.evaluate_start_date,
                evaluate_end_date=args.evaluate_end_date,
                evaluate_wf_mode=args.evaluate_wf_mode,
                evaluate_wf_scheme=args.evaluate_wf_scheme,
                wf_summary_output=wf_summary_output,
                promotion_min_feasible_folds=args.promotion_min_feasible_folds,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=None,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing failed revision manifest")
            print(f"[revision-gate] manifest saved: {manifest_output}")
            print("[revision-gate] decision: error")
            return int(wf_result.returncode) or 1
        progress.update(message="wf feasibility completed")

        promotion_result = subprocess.run(promotion_command, cwd=ROOT, check=False)
        executed_steps.append(
            {
                "name": "promotion_gate",
                "command": promotion_command,
                "status": "completed" if promotion_result.returncode == 0 else "blocked",
                "return_code": int(promotion_result.returncode),
            }
        )
        if promotion_result.returncode != 0:
            status = "block"
            decision = "hold"
        progress.complete(message=f"promotion gate finished status={status}")

        promotion_report = _read_json_dict(promotion_output)

        manifest_payload = _build_manifest_payload(
            revision_slug=revision_slug,
            started_at=started_at,
            status=status,
            decision=decision,
            resolved_profile=resolved_profile,
            config_path=config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
            train_artifact_suffix=train_artifact_suffix,
            skip_train=bool(args.skip_train),
            train_max_train_rows=args.train_max_train_rows,
            train_max_valid_rows=args.train_max_valid_rows,
            evaluate_model_artifact_suffix=args.evaluate_model_artifact_suffix,
            evaluate_max_rows=args.evaluate_max_rows,
            evaluate_pre_feature_max_rows=args.evaluate_pre_feature_max_rows,
            evaluate_start_date=args.evaluate_start_date,
            evaluate_end_date=args.evaluate_end_date,
            evaluate_wf_mode=args.evaluate_wf_mode,
            evaluate_wf_scheme=args.evaluate_wf_scheme,
            wf_summary_output=wf_summary_output,
            promotion_min_feasible_folds=args.promotion_min_feasible_folds,
            promotion_output=promotion_output,
            manifest_output=manifest_output,
            executed_steps=executed_steps,
            promotion_report=promotion_report,
        )
        _write_manifest(manifest_output, manifest_payload, label="writing revision manifest")
        print(f"[revision-gate] manifest saved: {manifest_output}")
        print(f"[revision-gate] decision: {decision}")
        return 0 if decision == "promote" else 1
    except KeyboardInterrupt:
        print("[revision-gate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[revision-gate] failed: {error}")
        return 1
    except Exception as error:
        print(f"[revision-gate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
