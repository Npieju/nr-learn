from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
NO_MODEL_ARTIFACT_SUFFIX = "__NO_MODEL_ARTIFACT_SUFFIX__"
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


class _TeeStream:
    def __init__(self, *streams: object) -> None:
        self._streams = list(streams)

    def write(self, data: str) -> int:
        alive_streams: list[object] = []
        for stream in self._streams:
            try:
                stream.write(data)
                alive_streams.append(stream)
            except (BrokenPipeError, OSError, ValueError):
                continue
        self._streams = alive_streams
        return len(data)

    def flush(self) -> None:
        alive_streams: list[object] = []
        for stream in self._streams:
            try:
                stream.flush()
                alive_streams.append(stream)
            except (BrokenPipeError, OSError, ValueError):
                continue
        self._streams = alive_streams

    def isatty(self) -> bool:
        if not self._streams:
            return False
        primary = self._streams[0]
        return bool(getattr(primary, "isatty", lambda: False)())


def _default_log_path(*, revision_slug: str) -> Path:
    return ROOT / "artifacts" / "logs" / f"revision_gate_{revision_slug}.log"


def _configure_live_log(log_path: Path) -> None:
    artifact_ensure_output_file_path(log_path, label="run log", workspace_root=ROOT)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_handle = log_path.open("a", encoding="utf-8", buffering=1)
    sys.stdout = _TeeStream(sys.stdout, log_handle)
    sys.stderr = _TeeStream(sys.stderr, log_handle)
    print(
        f"[revision-gate] live log file: {artifact_display_path(log_path, workspace_root=ROOT)}",
        flush=True,
    )


def _python_command(script_path: str) -> list[str]:
    return [sys.executable, "-u", script_path]


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


def _forward_stream(
    stream: object,
    *,
    sink: list[str],
    target: object,
) -> None:
    if stream is None:
        return
    try:
        for chunk in iter(stream.readline, ""):
            if not chunk:
                break
            sink.append(chunk)
            print(chunk, end="", file=target, flush=True)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _run_command_result(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    log_progress(f"running {label}: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=1,
    )
    stdout_chunks: list[str] = []
    stderr_chunks: list[str] = []
    stdout_thread = threading.Thread(
        target=_forward_stream,
        args=(process.stdout,),
        kwargs={"sink": stdout_chunks, "target": sys.stdout},
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_forward_stream,
        args=(process.stderr,),
        kwargs={"sink": stderr_chunks, "target": sys.stderr},
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()
    return_code = process.wait()
    stdout_thread.join()
    stderr_thread.join()
    return subprocess.CompletedProcess(
        args=command,
        returncode=return_code,
        stdout="".join(stdout_chunks),
        stderr="".join(stderr_chunks),
    )


def _planned_step(
    name: str,
    command: list[str],
    *,
    outputs: dict[str, object] | None = None,
    note: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {"name": name, "command": command, "status": "planned"}
    if outputs:
        payload["outputs"] = outputs
    if note:
        payload["note"] = note
    return payload


def _read_order() -> list[str]:
    return [
        "revision_gate",
        "train",
        "evaluate",
        "wf_feasibility",
        "promotion_gate",
    ]


def _planned_highlights(*, revision_slug: str, skip_train: bool) -> list[str]:
    highlights = []
    if skip_train:
        highlights.append(f"train step is marked skipped; revision {revision_slug} will reuse an existing model artifact")
    else:
        highlights.append(f"revision {revision_slug} will train a new artifact before evaluation and promotion gate")
    highlights.append("evaluation step writes the canonical evaluation manifest and summary consumed by promotion gate")
    highlights.append("matching wf feasibility summary is generated before promotion gate so benchmark support can be judged without a separate manual step")
    return highlights


def _read_json_dict(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


_CHALLENGER_EQUIVALENCE_FIELDS: tuple[str, ...] = (
    "stability_assessment",
    "auc",
    "top1_roi",
    "ev_top1_roi",
    "wf_nested_actual_folds",
    "wf_nested_test_roi_weighted",
    "wf_nested_test_roi_mean",
    "wf_nested_test_bets_total",
)


def _values_equivalent(candidate_value: object, anchor_value: object, *, tolerance: float) -> bool:
    if isinstance(candidate_value, (int, float)) and isinstance(anchor_value, (int, float)):
        return abs(float(candidate_value) - float(anchor_value)) <= tolerance
    return candidate_value == anchor_value


def _resolve_versioned_evaluation_summary_path() -> Path:
    manifest_path = ROOT / "artifacts" / "reports" / "evaluation_manifest.json"
    manifest_payload = _read_json_dict(manifest_path)
    if isinstance(manifest_payload, dict):
        output_files = manifest_payload.get("output_files")
        if isinstance(output_files, dict):
            versioned_summary = output_files.get("versioned_summary")
            if isinstance(versioned_summary, str) and versioned_summary.strip():
                return _resolve_path(versioned_summary)
    return ROOT / "artifacts" / "reports" / "evaluation_summary.json"


def _build_challenger_equivalence_report(
    *,
    anchor_summary_path: Path,
    candidate_summary_path: Path,
    tolerance: float,
) -> dict[str, object]:
    anchor_payload = _read_json_dict(anchor_summary_path)
    candidate_payload = _read_json_dict(candidate_summary_path)
    report: dict[str, object] = {
        "anchor_summary": artifact_display_path(anchor_summary_path, workspace_root=ROOT),
        "candidate_summary": artifact_display_path(candidate_summary_path, workspace_root=ROOT),
        "fields": list(_CHALLENGER_EQUIVALENCE_FIELDS),
        "tolerance": float(tolerance),
    }
    if anchor_payload is None:
        report["status"] = "unavailable"
        report["reason"] = "anchor_summary_missing_or_invalid"
        return report
    if candidate_payload is None:
        report["status"] = "unavailable"
        report["reason"] = "candidate_summary_missing_or_invalid"
        return report

    comparisons: list[dict[str, object]] = []
    all_equivalent = True
    for field_name in _CHALLENGER_EQUIVALENCE_FIELDS:
        anchor_value = anchor_payload.get(field_name)
        candidate_value = candidate_payload.get(field_name)
        equal = _values_equivalent(candidate_value, anchor_value, tolerance=tolerance)
        all_equivalent = all_equivalent and equal
        comparisons.append(
            {
                "field": field_name,
                "anchor_value": anchor_value,
                "candidate_value": candidate_value,
                "equivalent": equal,
            }
        )
    report["status"] = "equivalent" if all_equivalent else "different"
    report["comparisons"] = comparisons
    return report


def _build_evaluation_promotion_alignment_report(*, evaluation_summary_path: Path) -> dict[str, object]:
    payload = _read_json_dict(evaluation_summary_path)
    report: dict[str, object] = {
        "evaluation_summary": artifact_display_path(evaluation_summary_path, workspace_root=ROOT),
    }
    if payload is None:
        report["status"] = "unavailable"
        report["reason"] = "evaluation_summary_missing_or_invalid"
        return report

    folds = payload.get("wf_nested_folds")
    if not isinstance(folds, list) or not folds:
        report["status"] = "clear"
        report["reason"] = "wf_nested_folds_unavailable"
        return report

    actual_folds = payload.get("wf_nested_actual_folds")
    if not isinstance(actual_folds, int):
        actual_folds = None

    strategy_kinds: list[str] = []
    all_no_bet = True
    for item in folds:
        if not isinstance(item, dict):
            all_no_bet = False
            continue
        strategy_kind = str(item.get("strategy_kind") or "")
        strategy_kinds.append(strategy_kind)
        if strategy_kind != "no_bet":
            all_no_bet = False

    bets_total_raw = payload.get("wf_nested_test_bets_total")
    bets_total = int(bets_total_raw) if isinstance(bets_total_raw, (int, float)) else None
    report["wf_nested_actual_folds"] = actual_folds
    report["wf_nested_fold_count"] = len(folds)
    report["wf_nested_strategy_kinds"] = strategy_kinds
    report["wf_nested_test_bets_total"] = bets_total

    if all_no_bet and bets_total == 0:
        report["status"] = "triggered"
        report["reason"] = "evaluation_nested_all_no_bet"
    else:
        report["status"] = "clear"
        report["reason"] = "evaluation_nested_has_feasible_policy"
    return report


def _promotion_alignment_highlights(report: dict[str, object]) -> list[str]:
    status = str(report.get("status") or "")
    if status == "triggered":
        fold_count = report.get("wf_nested_actual_folds")
        strategy_kinds = report.get("wf_nested_strategy_kinds") or []
        bets_total = report.get("wf_nested_test_bets_total")
        return [
            "evaluation nested walk-forward produced no feasible betting policy across all completed folds",
            f"evaluation nested strategies={strategy_kinds} folds={fold_count} wf_nested_test_bets_total={bets_total}",
            "formal promotion is short-circuited because wf feasibility would otherwise uplift a no-bet evaluation candidate",
        ]
    if status == "clear":
        return ["evaluation nested walk-forward produced at least one feasible policy; conservative short-circuit not triggered"]
    reason = str(report.get("reason") or "promotion_alignment_check_unavailable")
    return [f"evaluation promotion alignment check unavailable: {reason}"]


def _challenger_equivalence_highlights(report: dict[str, object]) -> list[str]:
    status = str(report.get("status") or "")
    if status == "equivalent":
        return [
            "challenger evaluation summary is equivalent to the anchor across the configured decisive fields",
            "continuing to wf feasibility and promotion gate is unlikely to change the candidate role unless later diagnostics diverge",
        ]
    if status == "different":
        differences = [
            str(item.get("field"))
            for item in report.get("comparisons", [])
            if isinstance(item, dict) and not bool(item.get("equivalent"))
        ]
        joined = ", ".join(differences[:5]) if differences else "configured fields"
        return [f"challenger evaluation summary differs from the anchor on {joined}"]
    reason = str(report.get("reason") or "equivalence_check_unavailable")
    return [f"challenger equivalence check unavailable: {reason}"]


def _build_lock_path(manifest_output: Path, *, revision_slug: str) -> Path:
    if manifest_output.name == f"revision_gate_{revision_slug}.json":
        return manifest_output.with_suffix(manifest_output.suffix + ".lock")
    return manifest_output.parent / f"revision_gate_{revision_slug}.lock"


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_lock_payload(lock_path: Path) -> dict[str, object]:
    if not lock_path.exists():
        return {}
    try:
        text = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return {}
    if not text:
        return {}
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _acquire_run_lock(
    *,
    lock_path: Path,
    revision_slug: str,
    manifest_output: Path,
) -> dict[str, object]:
    payload = {
        "pid": os.getpid(),
        "revision": revision_slug,
        "manifest_output": artifact_display_path(manifest_output, workspace_root=ROOT),
        "started_at": utc_now_iso(),
    }

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing = _read_lock_payload(lock_path)
            existing_pid = int(existing.get("pid", 0) or 0) if isinstance(existing, dict) else 0
            if existing and not _pid_is_running(existing_pid):
                try:
                    lock_path.unlink()
                    continue
                except FileNotFoundError:
                    continue
            return existing
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
            return {}


def _release_run_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        return


def _duplicate_run_highlights(*, revision_slug: str, lock_path: Path, existing: dict[str, object]) -> list[str]:
    active_pid = existing.get("pid")
    started_at = existing.get("started_at")
    manifest_output = existing.get("manifest_output")
    highlights = [
        f"revision {revision_slug} is already running in another process",
        f"active lock path: {artifact_display_path(lock_path, workspace_root=ROOT)}",
    ]
    if active_pid is not None:
        highlights.append(f"active pid={active_pid}")
    if started_at is not None:
        highlights.append(f"active run started_at={started_at}")
    if manifest_output is not None:
        highlights.append(f"active manifest={manifest_output}")
    return highlights


def _classify_step_failure(*, step_name: str, result: subprocess.CompletedProcess[str]) -> dict[str, object]:
    combined_output = "\n".join(
        part.strip()
        for part in [result.stdout or "", result.stderr or ""]
        if isinstance(part, str) and part.strip()
    )

    if (
        step_name == "train"
        and "Time split is empty. Dataset has too few dated samples." in combined_output
        and (
            "configured_train_window_has_no_rows" in combined_output
            or "configured_valid_window_has_no_rows" in combined_output
        )
    ):
        return {
            "error_code": "insufficient_split_date_coverage",
            "error_message": "revision gate training failed because the available local dates do not satisfy the configured train/valid split windows",
            "recommended_action": "expand_local_date_window_or_align_split_window",
            "current_phase": "train",
            "highlights": [
                "revision training could not populate the configured train/valid windows from the available local dates",
                "expand the local seed/date window or align the split window before rerunning revision gate",
            ],
        }

    if step_name == "train" and "Time split is empty. Dataset has too few dated samples." in combined_output:
        return {
            "error_code": "insufficient_dated_samples",
            "error_message": "revision gate training failed because the dataset has too few dated samples for a time split",
            "recommended_action": "expand_local_date_window",
            "current_phase": "train",
            "highlights": [
                "revision training could not build a time split from the available local rows",
                "expand the local seed/date window before rerunning revision gate",
            ],
        }

    if step_name == "wf_feasibility" and (
        "No nested walk-forward slices available for the requested window" in combined_output
        or "No walk-forward slices available for the requested window" in combined_output
    ):
        return {
            "error_code": "insufficient_wf_window_coverage",
            "error_message": "revision gate walk-forward feasibility could not build slices from the available local date window",
            "recommended_action": "expand_local_date_window_or_relax_wf_scheme",
            "current_phase": "wf_feasibility",
            "highlights": [
                "walk-forward feasibility reached model inference but could not form enough chronological slices",
                "expand the local date window or switch to a less demanding walk-forward scheme before rerunning revision gate",
            ],
        }

    if step_name == "wf_feasibility" and "Odds column is required for feasibility diagnostic" in combined_output:
        return {
            "error_code": "missing_market_odds",
            "error_message": "revision gate walk-forward feasibility could not run because the dataset does not include historical odds",
            "recommended_action": "populate_historical_odds_or_accept_formal_block",
            "current_phase": "wf_feasibility",
            "highlights": [
                "walk-forward feasibility reached feature/model loading but the local dataset has no odds column for policy simulation",
                "populate historical odds in the local raw pipeline before rerunning if promotion-grade policy diagnostics are required",
            ],
        }

    return {
        "error_code": f"{step_name}_failed",
        "error_message": f"revision gate {step_name} step returned non-zero exit code",
        "recommended_action": "inspect_revision_gate_failure",
        "current_phase": step_name,
        "highlights": [
            f"revision {step_name} step failed",
            "inspect command output and generated manifests before retrying",
        ],
    }


def _failure_highlights(failure_details: dict[str, object]) -> list[str]:
    highlights = failure_details.get("highlights")
    if not isinstance(highlights, list):
        return []
    return [str(item) for item in highlights if isinstance(item, str)]


def _write_soft_block_wf_summary(
    *,
    wf_summary_output: Path,
    config_path: str,
    data_config_path: str,
    feature_config_path: str,
    artifact_suffix: str,
    model_artifact_suffix: str | None,
    start_date: str | None,
    end_date: str | None,
    wf_mode: str,
    wf_scheme: str,
    failure_details: dict[str, object],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "run_context": {
            "config": config_path,
            "data_config": data_config_path,
            "feature_config": feature_config_path,
            "loaded_rows": None,
            "data_load_strategy": None,
            "primary_source_rows_total": None,
            "pre_feature_max_rows": None,
            "pre_feature_rows": None,
            "start_date": start_date,
            "end_date": end_date,
            "wf_mode": wf_mode,
            "wf_scheme": wf_scheme,
            "rows": None,
            "races": None,
            "feature_count": None,
            "categorical_feature_count": None,
            "artifact_suffix": artifact_suffix,
            "model_artifact_suffix": model_artifact_suffix or artifact_suffix,
            "model_path": None,
        },
        "policy_constraints": {},
        "policy_search": {},
        "stability_assessment": "probe_only",
        "stability_guardrail": {
            "assessment": "probe_only",
            "is_representative": False,
            "warnings": _failure_highlights(failure_details),
            "failed_checks": [],
            "skipped_checks": [],
            "observed": {},
            "thresholds": {},
            "notes": [
                "synthetic walk-forward summary was emitted because the requested date window could not form nested slices",
            ],
        },
        "error_code": failure_details.get("error_code"),
        "error_message": failure_details.get("error_message"),
        "recommended_action": failure_details.get("recommended_action"),
        "current_phase": failure_details.get("current_phase"),
        "highlights": _failure_highlights(failure_details),
        "folds": [],
    }
    write_json(wf_summary_output, payload)
    return payload


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
    promotion_min_formal_weighted_roi: float | None = None,
    promotion_report: dict[str, object] | None = None,
    challenger_equivalence: dict[str, object] | None = None,
    error_code: str | None = None,
    error_message: str | None = None,
    recommended_action: str | None = None,
    current_phase: str | None = None,
    highlights: list[str] | None = None,
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
            "min_formal_weighted_roi": float(promotion_min_formal_weighted_roi) if promotion_min_formal_weighted_roi is not None else None,
            "output": artifact_display_path(promotion_output, workspace_root=ROOT),
            "summary": (promotion_report or {}).get("summary") if isinstance(promotion_report, dict) else None,
            "formal_benchmark": (promotion_report or {}).get("formal_benchmark") if isinstance(promotion_report, dict) else None,
        },
        "challenger_equivalence": challenger_equivalence,
        "read_order": _read_order(),
        "steps": executed_steps,
        "artifacts": {
            "wf_summary": artifact_display_path(wf_summary_output, workspace_root=ROOT),
            "promotion_report": artifact_display_path(promotion_output, workspace_root=ROOT),
            "evaluation_manifest": "artifacts/reports/evaluation_manifest.json",
            "evaluation_summary": "artifacts/reports/evaluation_summary.json",
            "revision_manifest": artifact_display_path(manifest_output, workspace_root=ROOT),
        },
    }
    if error_code is not None:
        payload["error_code"] = error_code
    if error_message is not None:
        payload["error_message"] = error_message
    if recommended_action is not None:
        payload["recommended_action"] = recommended_action
    if current_phase is not None:
        payload["current_phase"] = current_phase
    if highlights is not None:
        payload["highlights"] = highlights
    if dry_run:
        payload["dry_run"] = True
    return payload


def _build_running_manifest_payload(
    *,
    revision_slug: str,
    started_at: str,
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
    promotion_min_formal_weighted_roi: float | None = None,
    challenger_equivalence: dict[str, object] | None = None,
) -> dict[str, object]:
    initial_phase = "evaluate" if skip_train else "train"
    return _build_manifest_payload(
        revision_slug=revision_slug,
        started_at=started_at,
        status="running",
        decision="in_progress",
        resolved_profile=resolved_profile,
        config_path=config_path,
        data_config_path=data_config_path,
        feature_config_path=feature_config_path,
        train_artifact_suffix=train_artifact_suffix,
        skip_train=skip_train,
        train_max_train_rows=train_max_train_rows,
        train_max_valid_rows=train_max_valid_rows,
        evaluate_model_artifact_suffix=evaluate_model_artifact_suffix,
        evaluate_max_rows=evaluate_max_rows,
        evaluate_pre_feature_max_rows=evaluate_pre_feature_max_rows,
        evaluate_start_date=evaluate_start_date,
        evaluate_end_date=evaluate_end_date,
        evaluate_wf_mode=evaluate_wf_mode,
        evaluate_wf_scheme=evaluate_wf_scheme,
        wf_summary_output=wf_summary_output,
        promotion_min_feasible_folds=promotion_min_feasible_folds,
        promotion_min_formal_weighted_roi=promotion_min_formal_weighted_roi,
        promotion_output=promotion_output,
        manifest_output=manifest_output,
        executed_steps=executed_steps,
        promotion_report=None,
        challenger_equivalence=challenger_equivalence,
        recommended_action="wait_for_revision_gate_completion",
        current_phase=initial_phase,
        highlights=[
            f"revision {revision_slug} is running",
            f"current phase: {initial_phase}",
        ],
    )


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
    parser.add_argument("--evaluate-no-model-artifact-suffix", action="store_true")
    parser.add_argument("--evaluate-max-rows", type=int, default=120000)
    parser.add_argument("--evaluate-pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--evaluate-start-date", default=None)
    parser.add_argument("--evaluate-end-date", default=None)
    parser.add_argument("--evaluate-wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--evaluate-wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--wf-max-silent-seconds", type=float, default=None)
    parser.add_argument("--wf-max-fold-elapsed-seconds", type=float, default=None)
    parser.add_argument("--promotion-min-feasible-folds", type=int, default=1)
    parser.add_argument("--promotion-min-formal-weighted-roi", type=float, default=None)
    parser.add_argument("--promotion-output", default=None)
    parser.add_argument("--wf-summary-output", default=None)
    parser.add_argument("--manifest-output", default=None)
    parser.add_argument("--log-file", default=None)
    parser.add_argument("--challenger-anchor-evaluation-summary", default=None)
    parser.add_argument("--challenger-equivalence-tolerance", type=float, default=0.0)
    parser.add_argument("--stop-on-equivalent-challenger", action="store_true")
    parser.add_argument("--allow-wf-soft-block", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    lock_path: Path | None = None
    lock_acquired = False
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
        log_path = _resolve_path(args.log_file) if args.log_file else _default_log_path(revision_slug=revision_slug)
        _configure_live_log(log_path)
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
        lock_path = _build_lock_path(manifest_output, revision_slug=revision_slug)

        progress = ProgressBar(total=4, prefix="[revision-gate]", logger=log_progress, min_interval_sec=0.0)
        effective_evaluate_model_artifact_suffix = args.evaluate_model_artifact_suffix
        if args.evaluate_no_model_artifact_suffix:
            effective_evaluate_model_artifact_suffix = NO_MODEL_ARTIFACT_SUFFIX

        progress.start(
            message=(
                f"starting revision={revision_slug} profile={resolved_profile or 'custom'} config={config_path} "
                f"data_config={data_config_path} feature_config={feature_config_path} "
                f"train_max_train_rows={args.train_max_train_rows or 'config'} train_max_valid_rows={args.train_max_valid_rows or 'config'} "
                f"skip_train={args.skip_train} "
                f"evaluate_model_artifact_suffix={effective_evaluate_model_artifact_suffix or 'none'}"
            )
        )

        train_command = _python_command("scripts/run_train.py")
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
            *_python_command("scripts/run_evaluate.py"),
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
        if effective_evaluate_model_artifact_suffix is not None:
            evaluate_command.extend(["--model-artifact-suffix", str(effective_evaluate_model_artifact_suffix)])
        if args.evaluate_pre_feature_max_rows is not None:
            evaluate_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
        if args.evaluate_start_date:
            evaluate_command.extend(["--start-date", str(args.evaluate_start_date)])
        if args.evaluate_end_date:
            evaluate_command.extend(["--end-date", str(args.evaluate_end_date)])

        wf_command = [
            *_python_command("scripts/run_wf_feasibility_diag.py"),
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
        if effective_evaluate_model_artifact_suffix is not None:
            wf_command.extend(["--model-artifact-suffix", str(effective_evaluate_model_artifact_suffix)])
        if args.evaluate_pre_feature_max_rows is not None:
            wf_command.extend(["--pre-feature-max-rows", str(args.evaluate_pre_feature_max_rows)])
        if args.evaluate_start_date:
            wf_command.extend(["--start-date", str(args.evaluate_start_date)])
        if args.evaluate_end_date:
            wf_command.extend(["--end-date", str(args.evaluate_end_date)])
        if args.wf_max_silent_seconds is not None:
            wf_command.extend(["--max-silent-seconds", str(args.wf_max_silent_seconds)])
        if args.wf_max_fold_elapsed_seconds is not None:
            wf_command.extend(["--max-fold-elapsed-seconds", str(args.wf_max_fold_elapsed_seconds)])
        wf_command.extend(
            [
                "--summary-output",
                artifact_display_path(wf_summary_output, workspace_root=ROOT),
                "--detail-output",
                artifact_display_path(wf_summary_output.with_suffix(".csv"), workspace_root=ROOT),
            ]
        )

        promotion_command = [
            *_python_command("scripts/run_promotion_gate.py"),
            "--evaluation-manifest",
            "artifacts/reports/evaluation_manifest.json",
            "--wf-summary",
            artifact_display_path(wf_summary_output, workspace_root=ROOT),
            "--min-feasible-folds",
            str(args.promotion_min_feasible_folds),
            "--output",
            artifact_display_path(promotion_output, workspace_root=ROOT),
        ]
        if args.promotion_min_formal_weighted_roi is not None:
            promotion_command.extend(["--min-formal-weighted-roi", str(args.promotion_min_formal_weighted_roi)])

        started_at = utc_now_iso()
        executed_steps: list[dict[str, object]] = []
        status = "pass"
        decision = "promote"
        challenger_equivalence_report: dict[str, object] | None = None

        if args.dry_run:
            executed_steps = [
                _planned_step(
                    "train",
                    train_command,
                    outputs={"artifact_suffix": train_artifact_suffix},
                    note="train output artifact files are derived from the revision artifact suffix",
                ),
                _planned_step(
                    "evaluate",
                    evaluate_command,
                    outputs={
                        "evaluation_manifest": "artifacts/reports/evaluation_manifest.json",
                        "evaluation_summary": "artifacts/reports/evaluation_summary.json",
                    },
                ),
                _planned_step(
                    "wf_feasibility",
                    wf_command,
                    outputs={"wf_summary": artifact_display_path(wf_summary_output, workspace_root=ROOT)},
                ),
                _planned_step(
                    "promotion_gate",
                    promotion_command,
                    outputs={"promotion_report": artifact_display_path(promotion_output, workspace_root=ROOT)},
                ),
            ]
            status = "planned"
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
                promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=None,
                challenger_equivalence=None,
                dry_run=True,
            )
            manifest_payload["recommended_action"] = "run_revision_gate"
            manifest_payload["current_phase"] = "planned"
            manifest_payload["highlights"] = _planned_highlights(
                revision_slug=revision_slug,
                skip_train=bool(args.skip_train),
            )
            manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
            _write_manifest(manifest_output, manifest_payload, label="writing dry-run manifest")
            progress.complete(message="dry-run plan prepared")
            print(f"[revision-gate] dry-run manifest saved: {manifest_output}")
            print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
            print(f"[revision-gate] train command: {' '.join(train_command)}")
            print(f"[revision-gate] evaluate command: {' '.join(evaluate_command)}")
            print(f"[revision-gate] wf command: {' '.join(wf_command)}")
            print(f"[revision-gate] promotion command: {' '.join(promotion_command)}")
            return 0

        existing_lock = _acquire_run_lock(
            lock_path=lock_path,
            revision_slug=revision_slug,
            manifest_output=manifest_output,
        )
        if existing_lock:
            manifest_payload = _build_manifest_payload(
                revision_slug=revision_slug,
                started_at=started_at,
                status="blocked",
                decision="hold",
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
                promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=None,
                challenger_equivalence=None,
                error_code="duplicate_revision_gate_running",
                error_message="another revision gate process is already running for the same revision",
                recommended_action="wait_for_active_revision_gate_or_choose_new_revision",
                current_phase="duplicate_run_guard",
                highlights=_duplicate_run_highlights(
                    revision_slug=revision_slug,
                    lock_path=lock_path,
                    existing=existing_lock,
                ),
            )
            manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
            _write_manifest(manifest_output, manifest_payload, label="writing duplicate-run blocked manifest")
            print(f"[revision-gate] manifest saved: {manifest_output}")
            print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
            print("[revision-gate] decision: hold")
            return 1
        lock_acquired = True
        running_manifest_payload = _build_running_manifest_payload(
            revision_slug=revision_slug,
            started_at=started_at,
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
            promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
            promotion_output=promotion_output,
            manifest_output=manifest_output,
            executed_steps=executed_steps,
            challenger_equivalence=None,
        )
        running_manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
        _write_manifest(manifest_output, running_manifest_payload, label="writing running revision manifest")

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
                failure_details = _classify_step_failure(step_name="train", result=train_result)
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
                    promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                    promotion_output=promotion_output,
                    manifest_output=manifest_output,
                    executed_steps=executed_steps,
                    promotion_report=None,
                    challenger_equivalence=None,
                    error_code=str(failure_details.get("error_code") or "train_failed"),
                    error_message=str(failure_details.get("error_message") or "revision training failed"),
                    recommended_action=str(failure_details.get("recommended_action") or "inspect_revision_gate_failure"),
                    current_phase=str(failure_details.get("current_phase") or "train"),
                    highlights=_failure_highlights(failure_details),
                )
                manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
                _write_manifest(manifest_output, manifest_payload, label="writing failed revision manifest")
                print(f"[revision-gate] manifest saved: {manifest_output}")
                print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
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
            failure_details = _classify_step_failure(step_name="evaluate", result=evaluate_result)
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
                promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=None,
                challenger_equivalence=None,
                error_code=str(failure_details.get("error_code") or "evaluate_failed"),
                error_message=str(failure_details.get("error_message") or "revision evaluation failed"),
                recommended_action=str(failure_details.get("recommended_action") or "inspect_revision_gate_failure"),
                current_phase=str(failure_details.get("current_phase") or "evaluate"),
                highlights=_failure_highlights(failure_details),
            )
            manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
            _write_manifest(manifest_output, manifest_payload, label="writing failed revision manifest")
            print(f"[revision-gate] manifest saved: {manifest_output}")
            print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
            print("[revision-gate] decision: error")
            return int(evaluate_result.returncode) or 1
        progress.update(message="evaluation completed")

        if args.challenger_anchor_evaluation_summary:
            anchor_summary_path = _resolve_path(str(args.challenger_anchor_evaluation_summary))
            candidate_summary_path = _resolve_versioned_evaluation_summary_path()
            challenger_equivalence_report = _build_challenger_equivalence_report(
                anchor_summary_path=anchor_summary_path,
                candidate_summary_path=candidate_summary_path,
                tolerance=float(args.challenger_equivalence_tolerance),
            )
            equivalence_status = str(challenger_equivalence_report.get("status") or "unknown")
            log_progress(
                "challenger equivalence check "
                f"status={equivalence_status} anchor={artifact_display_path(anchor_summary_path, workspace_root=ROOT)} "
                f"candidate={artifact_display_path(candidate_summary_path, workspace_root=ROOT)}"
            )
            executed_steps.append(
                {
                    "name": "challenger_equivalence",
                    "status": equivalence_status,
                    "anchor_summary": artifact_display_path(anchor_summary_path, workspace_root=ROOT),
                    "candidate_summary": artifact_display_path(candidate_summary_path, workspace_root=ROOT),
                }
            )
            if equivalence_status == "equivalent":
                highlights = _challenger_equivalence_highlights(challenger_equivalence_report)
                for highlight in highlights:
                    log_progress(highlight)
                if args.stop_on_equivalent_challenger:
                    status = "block"
                    decision = "hold"
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
                        promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                        promotion_output=promotion_output,
                        manifest_output=manifest_output,
                        executed_steps=executed_steps,
                        promotion_report=None,
                        challenger_equivalence=challenger_equivalence_report,
                        error_code="equivalent_challenger_detected",
                        error_message="challenger evaluation summary matched the anchor across the configured decisive fields",
                        recommended_action="review_anchor_equivalence_before_running_wf",
                        current_phase="challenger_equivalence",
                        highlights=highlights,
                    )
                    manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
                    _write_manifest(manifest_output, manifest_payload, label="writing equivalent-challenger blocked manifest")
                    print(f"[revision-gate] manifest saved: {manifest_output}")
                    print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
                    print("[revision-gate] decision: hold")
                    return 1

        evaluation_summary_path = _resolve_versioned_evaluation_summary_path()
        promotion_alignment_report = _build_evaluation_promotion_alignment_report(
            evaluation_summary_path=evaluation_summary_path
        )
        if str(promotion_alignment_report.get("status") or "") == "triggered":
            highlights = _promotion_alignment_highlights(promotion_alignment_report)
            for highlight in highlights:
                log_progress(highlight)
            executed_steps.append(
                {
                    "name": "promotion_alignment_short_circuit",
                    "status": "triggered",
                    "evaluation_summary": artifact_display_path(evaluation_summary_path, workspace_root=ROOT),
                    "reason": promotion_alignment_report.get("reason"),
                }
            )
            promotion_report = {
                "status": "block",
                "decision": "hold",
                "current_phase": "evaluate",
                "recommended_action": "review_evaluation_nested_wf_before_formal_promotion",
                "summary": {
                    "status": "block",
                    "decision": "hold",
                    "revision": revision_slug,
                    "reason": promotion_alignment_report.get("reason"),
                },
                "formal_benchmark": None,
                "highlights": highlights,
                "alignment_short_circuit": promotion_alignment_report,
            }
            write_json(promotion_output, promotion_report)
            manifest_payload = _build_manifest_payload(
                revision_slug=revision_slug,
                started_at=started_at,
                status="block",
                decision="hold",
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
                promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                promotion_output=promotion_output,
                manifest_output=manifest_output,
                executed_steps=executed_steps,
                promotion_report=promotion_report,
                challenger_equivalence=challenger_equivalence_report,
                error_code="evaluation_nested_all_no_bet_short_circuit",
                error_message="evaluation nested walk-forward produced no feasible policy and zero bets across completed folds",
                recommended_action="review_evaluation_nested_wf_before_formal_promotion",
                current_phase="evaluate",
                highlights=highlights,
            )
            manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
            _write_manifest(manifest_output, manifest_payload, label="writing promotion-alignment blocked manifest")
            progress.complete(message="promotion alignment short-circuited status=block")
            print(f"[revision-gate] manifest saved: {manifest_output}")
            print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
            print("[revision-gate] decision: hold")
            return 1

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
            failure_details = _classify_step_failure(step_name="wf_feasibility", result=wf_result)
            if args.allow_wf_soft_block and str(failure_details.get("error_code") or "") in {"insufficient_wf_window_coverage", "missing_market_odds"}:
                _write_soft_block_wf_summary(
                    wf_summary_output=wf_summary_output,
                    config_path=config_path,
                    data_config_path=data_config_path,
                    feature_config_path=feature_config_path,
                    artifact_suffix=train_artifact_suffix,
                    model_artifact_suffix=args.evaluate_model_artifact_suffix,
                    start_date=args.evaluate_start_date,
                    end_date=args.evaluate_end_date,
                    wf_mode=args.evaluate_wf_mode,
                    wf_scheme=args.evaluate_wf_scheme,
                    failure_details=failure_details,
                )
                executed_steps[-1]["status"] = "soft_blocked"
                executed_steps[-1]["note"] = "wf summary was synthesized so promotion gate can record a formal block"
                status = "block"
                decision = "hold"
                progress.update(message="wf feasibility soft-blocked; synthesized probe-only summary")
            else:
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
                    promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
                    promotion_output=promotion_output,
                    manifest_output=manifest_output,
                    executed_steps=executed_steps,
                    promotion_report=None,
                    challenger_equivalence=challenger_equivalence_report,
                    error_code=str(failure_details.get("error_code") or "wf_feasibility_failed"),
                    error_message=str(failure_details.get("error_message") or "revision wf feasibility failed"),
                    recommended_action=str(failure_details.get("recommended_action") or "inspect_revision_gate_failure"),
                    current_phase=str(failure_details.get("current_phase") or "wf_feasibility"),
                    highlights=_failure_highlights(failure_details),
                )
                manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
                _write_manifest(manifest_output, manifest_payload, label="writing failed revision manifest")
                print(f"[revision-gate] manifest saved: {manifest_output}")
                print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
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
            promotion_min_formal_weighted_roi=args.promotion_min_formal_weighted_roi,
            promotion_output=promotion_output,
            manifest_output=manifest_output,
            executed_steps=executed_steps,
            promotion_report=promotion_report,
            challenger_equivalence=challenger_equivalence_report,
        )
        manifest_payload["artifacts"]["run_log"] = artifact_display_path(log_path, workspace_root=ROOT)
        if isinstance(promotion_report, dict):
            if promotion_report.get("recommended_action") is not None:
                manifest_payload["recommended_action"] = promotion_report.get("recommended_action")
            if promotion_report.get("current_phase") is not None:
                manifest_payload["current_phase"] = promotion_report.get("current_phase")
            if isinstance(promotion_report.get("highlights"), list):
                manifest_payload["highlights"] = promotion_report.get("highlights")
        _write_manifest(manifest_output, manifest_payload, label="writing revision manifest")
        print(f"[revision-gate] manifest saved: {manifest_output}")
        print(f"[revision-gate] run log: {artifact_display_path(log_path, workspace_root=ROOT)}")
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
    finally:
        if lock_acquired and lock_path is not None:
            _release_run_lock(lock_path)


if __name__ == "__main__":
    raise SystemExit(main())
