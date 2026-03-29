from __future__ import annotations

import argparse
from pathlib import Path
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


DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"


def log_progress(message: str) -> None:
    print(message, flush=True)


def _read_order(*, include_backfill_handoff: bool = False) -> list[str]:
    return [
        "local_public_snapshot",
        "local_revision_gate",
        *( ["local_backfill_then_benchmark", "backfill", "materialize"] if include_backfill_handoff else []),
        "promotion_gate",
        "revision_gate",
        "evaluation_pointer",
    ]


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _normalize_revision_slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("revision must not be empty")
    return normalized


def _read_required_payload(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {artifact_display_path(path, workspace_root=ROOT)}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {artifact_display_path(path, workspace_root=ROOT)}")
    return payload


def _read_optional_payload(path_text: object) -> dict[str, object] | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _dict_payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _extract_readiness(payload: dict[str, object]) -> dict[str, object]:
    readiness = payload.get("readiness")
    if isinstance(readiness, dict):
        return dict(readiness)
    benchmark_payload = payload.get("benchmark_gate_payload")
    if isinstance(benchmark_payload, dict):
        readiness = benchmark_payload.get("readiness")
        if isinstance(readiness, dict):
            return dict(readiness)
    return {}


def _extract_promotion_summary(payload: dict[str, object]) -> dict[str, object]:
    promotion_payload = payload.get("promotion_payload")
    revision_manifest_payload = payload.get("revision_manifest_payload")
    summary: dict[str, object] = {}
    if isinstance(promotion_payload, dict):
        for key in ("status", "decision", "recommended_action", "feasible_folds", "weighted_test_roi"):
            if key in promotion_payload:
                summary[key] = promotion_payload.get(key)
    if isinstance(revision_manifest_payload, dict):
        for key in ("status", "decision", "recommended_action"):
            if key not in summary and key in revision_manifest_payload:
                summary[key] = revision_manifest_payload.get(key)
    return summary


def _extract_backfill_handoff_summary(payload: dict[str, object]) -> dict[str, object]:
    handoff_payload = payload.get("backfill_handoff_payload") if isinstance(payload.get("backfill_handoff_payload"), dict) else None
    backfill_summary_payload = payload.get("backfill_summary_payload") if isinstance(payload.get("backfill_summary_payload"), dict) else None
    if handoff_payload is None and backfill_summary_payload is None:
        return {}

    artifacts = payload.get("artifacts") if isinstance(payload.get("artifacts"), dict) else {}
    summary: dict[str, object] = {}
    if isinstance(handoff_payload, dict):
        for key in ("status", "current_phase", "recommended_action", "highlights"):
            if key in handoff_payload:
                summary[key] = handoff_payload.get(key)
    if isinstance(backfill_summary_payload, dict):
        for key in ("status", "current_phase", "recommended_action", "stopped_reason", "highlights"):
            summary.setdefault(key, backfill_summary_payload.get(key))
    if isinstance(artifacts, dict):
        if artifacts.get("backfill_wrapper_manifest") is not None:
            summary["wrapper_manifest"] = artifacts.get("backfill_wrapper_manifest")
        if artifacts.get("backfill_manifest") is not None:
            summary["backfill_manifest"] = artifacts.get("backfill_manifest")
        if artifacts.get("primary_materialize_manifest") is not None:
            summary["materialize_manifest"] = artifacts.get("primary_materialize_manifest")
    return summary


def _planned_readiness(*, lineage_path: Path) -> dict[str, object]:
    return {
        "benchmark_rerun_ready": False,
        "recommended_action": "run_local_revision_gate",
        "reasons": [
            f"local public snapshot requires a lineage manifest at {artifact_display_path(lineage_path, workspace_root=ROOT)} before readiness and promotion state can be summarized.",
        ],
    }


def _planned_promotion_summary() -> dict[str, object]:
    return {
        "status": "planned",
        "decision": "not_run",
        "recommended_action": "run_local_revision_gate",
        "feasible_folds": None,
        "weighted_test_roi": None,
    }


def _planned_evaluation_summary() -> dict[str, object]:
    return {
        "status": "planned",
        "latest_manifest": "artifacts/reports/evaluation_manifest.json",
        "latest_summary": "artifacts/reports/evaluation_summary.json",
        "output_files": None,
    }


def _planned_benchmark_gate_summary() -> dict[str, object]:
    return {
        "status": "planned",
        "completed_step": "planned",
        "error_code": None,
        "recommended_action": "run_local_revision_gate",
    }


def _planned_backfill_handoff_summary() -> dict[str, object]:
    return {
        "status": "planned",
        "current_phase": "planned",
        "recommended_action": "run_local_revision_gate",
    }


def _current_phase(*, status: str, lineage_status: object = None, lineage_completed_step: object = None) -> str:
    if status == "planned":
        return "planned"
    if status == "failed":
        if isinstance(lineage_completed_step, str) and lineage_completed_step.strip():
            return f"lineage_{lineage_completed_step.strip()}"
        if isinstance(lineage_status, str) and lineage_status.strip():
            return f"lineage_{lineage_status.strip()}"
        return "snapshot_failed"
    if isinstance(lineage_completed_step, str) and lineage_completed_step.strip() and lineage_completed_step != "completed":
        return f"lineage_{lineage_completed_step.strip()}"
    if isinstance(lineage_status, str) and lineage_status.strip() and lineage_status != "completed":
        return f"lineage_{lineage_status.strip()}"
    return "completed"


def _recommended_action(*, status: str, lineage_status: object = None, lineage_completed_step: object = None) -> str:
    if status == "planned":
        return "run_local_revision_gate"
    if lineage_status == "missing" or lineage_completed_step == "missing":
        return "run_local_revision_gate"
    if lineage_status == "planned" or lineage_completed_step == "planned":
        return "run_local_revision_gate"
    if lineage_status in {"backfill_handoff_blocked", "backfill_handoff_failed", "benchmark_gate_blocked", "benchmark_gate_failed", "revision_gate_failed", "failed", "interrupted"}:
        return "inspect_local_revision_lineage"
    if isinstance(lineage_completed_step, str) and lineage_completed_step.strip() and lineage_completed_step != "completed":
        return "inspect_local_revision_lineage"
    if status == "failed":
        return "inspect_local_public_snapshot_inputs"
    return "review_public_snapshot"


def _lineage_failure_action(lineage_payload: dict[str, object]) -> str | None:
    action = lineage_payload.get("recommended_action")
    return str(action) if isinstance(action, str) and action.strip() else None


def _highlights(
    *,
    status: str,
    revision: str,
    lineage_manifest: str,
    recommended_action: str,
    lineage_status: str | None = None,
    lineage_completed_step: str | None = None,
    readiness: dict[str, object] | None = None,
    promotion_summary: dict[str, object] | None = None,
    backfill_handoff_summary: dict[str, object] | None = None,
    error_message: str | None = None,
) -> list[str]:
    readiness = readiness or {}
    promotion_summary = promotion_summary or {}
    backfill_handoff_summary = backfill_handoff_summary or {}
    highlights = [f"public snapshot revision={revision} status={status}"]
    if lineage_status:
        if lineage_completed_step:
            highlights.append(f"lineage status={lineage_status}, completed_step={lineage_completed_step}")
        else:
            highlights.append(f"lineage status={lineage_status}")
    benchmark_ready = readiness.get("benchmark_rerun_ready")
    if benchmark_ready is not None:
        highlights.append(f"benchmark_rerun_ready={benchmark_ready}")
    promotion_decision = promotion_summary.get("decision")
    promotion_status = promotion_summary.get("status")
    if promotion_decision is not None or promotion_status is not None:
        highlights.append(
            "promotion summary: "
            f"status={promotion_status if promotion_status is not None else 'unknown'}, "
            f"decision={promotion_decision if promotion_decision is not None else 'unknown'}"
        )
    handoff_status = backfill_handoff_summary.get("status")
    handoff_phase = backfill_handoff_summary.get("current_phase")
    if handoff_status is not None or handoff_phase is not None:
        highlights.append(
            f"backfill handoff: status={handoff_status if handoff_status is not None else 'unknown'}, "
            f"phase={handoff_phase if handoff_phase is not None else 'unknown'}"
        )
    highlights.append(f"lineage manifest: {lineage_manifest}")
    if error_message:
        highlights.append(error_message)
    highlights.append(f"next operator action: {recommended_action}")
    return highlights


def _build_failure_payload(
    *,
    revision: str,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    lineage_path: Path,
    output_path: Path,
    error_message: str,
    lineage_status: object = None,
    lineage_completed_step: object = None,
) -> dict[str, object]:
    status = "failed"
    normalized_lineage_status = lineage_status
    normalized_lineage_completed_step = lineage_completed_step
    if "local revision lineage not found" in error_message:
        normalized_lineage_status = "missing"
        normalized_lineage_completed_step = "missing"
    recommended_action = _recommended_action(
        status=status,
        lineage_status=normalized_lineage_status,
        lineage_completed_step=normalized_lineage_completed_step,
    )
    payload = {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": status,
        "snapshot_kind": "local_public_snapshot",
        "revision": revision,
        "universe": universe,
        "source_scope": source_scope,
        "baseline_reference": baseline_reference,
        "lineage_status": normalized_lineage_status,
        "lineage_completed_step": normalized_lineage_completed_step,
        "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        "error_message": error_message,
        "read_order": _read_order(),
        "current_phase": _current_phase(
            status=status,
            lineage_status=normalized_lineage_status,
            lineage_completed_step=normalized_lineage_completed_step,
        ),
        "recommended_action": recommended_action,
        "artifacts": {
            "public_snapshot": artifact_display_path(output_path, workspace_root=ROOT),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        },
        "highlights": _highlights(
            status=status,
            revision=revision,
            lineage_manifest=artifact_display_path(lineage_path, workspace_root=ROOT),
            recommended_action=recommended_action,
            lineage_status=str(normalized_lineage_status) if normalized_lineage_status is not None else None,
            lineage_completed_step=str(normalized_lineage_completed_step) if normalized_lineage_completed_step is not None else None,
            error_message=error_message,
        ),
    }
    return payload


def _planned_highlights(*, lineage_path: Path) -> list[str]:
    return [
        "public snapshot is the operator-facing entrypoint for local-only status and downstream mixed compare anchoring",
        f"the snapshot will summarize readiness, promotion, and evaluation after lineage becomes available at {artifact_display_path(lineage_path, workspace_root=ROOT)}",
        "compare_contract already points at the final public snapshot path so mixed compare can use the same contract once lineage exists",
    ]


def _build_planned_payload(
    *,
    revision_slug: str,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    lineage_path: Path,
    output_path: Path,
) -> dict[str, object]:
    recommended_action = _recommended_action(status="planned")
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
        "snapshot_kind": "local_public_snapshot",
        "revision": revision_slug,
        "universe": universe,
        "source_scope": source_scope,
        "baseline_reference": baseline_reference,
        "lineage_status": "planned",
        "lineage_completed_step": "planned",
        "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        "read_order": _read_order(),
        "current_phase": _current_phase(status="planned"),
        "recommended_action": recommended_action,
        "artifacts": {
            "public_snapshot": artifact_display_path(output_path, workspace_root=ROOT),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        },
        "readiness": _planned_readiness(lineage_path=lineage_path),
        "promotion_summary": _planned_promotion_summary(),
        "backfill_handoff_summary": _planned_backfill_handoff_summary(),
        "benchmark_gate_summary": _planned_benchmark_gate_summary(),
        "evaluation_summary": _planned_evaluation_summary(),
        "highlights": _planned_highlights(lineage_path=lineage_path)
        + [f"next operator action: {recommended_action}"],
        "compare_contract": _build_compare_contract(
            snapshot_path=output_path,
            universe=universe,
            revision=revision_slug,
        ),
    }


def _build_compare_contract(*, snapshot_path: Path, universe: str, revision: str) -> dict[str, str]:
    return {
        "local_only_public_snapshot": artifact_display_path(snapshot_path, workspace_root=ROOT),
        "mixed_compare_manifest": f"artifacts/reports/mixed_universe_compare_{universe}_vs_jra_{revision}.json",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--lineage-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--baseline-reference", default=DEFAULT_BASELINE_REFERENCE)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_revision_slug(revision_value)
    lineage_manifest = args.lineage_manifest or f"artifacts/reports/local_revision_gate_{revision_slug}.json"
    output = args.output or f"artifacts/reports/local_public_snapshot_{revision_slug}.json"

    lineage_path = _resolve_path(lineage_manifest)
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        progress = ProgressBar(total=3, prefix="[local-public-snapshot]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"starting revision={revision_slug} universe={args.universe} "
                f"dry_run={'yes' if args.dry_run else 'no'}"
            )
        )

        if args.dry_run and not lineage_path.exists():
            progress.update(message="preparing planned snapshot manifest")
            payload = _build_planned_payload(
                revision_slug=revision_slug,
                universe=args.universe,
                source_scope=args.source_scope,
                baseline_reference=args.baseline_reference,
                lineage_path=lineage_path,
                output_path=output_path,
            )
            with Heartbeat("[local-public-snapshot]", "writing planned snapshot manifest", logger=log_progress):
                write_json(output_path, payload)
            progress.complete(message=f"planned manifest saved path={artifact_display_path(output_path, workspace_root=ROOT)}")
            print(f"[local-public-snapshot] planned snapshot saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        with Heartbeat("[local-public-snapshot]", "loading lineage payload", logger=log_progress):
            lineage_payload = _read_required_payload(lineage_path, label="local revision lineage")
        progress.update(message=f"lineage payload loaded path={artifact_display_path(lineage_path, workspace_root=ROOT)}")
        artifacts = _dict_payload(lineage_payload.get("artifacts"))
        benchmark_payload = _dict_payload(lineage_payload.get("benchmark_gate_payload"))
        evaluation_pointer_payload = lineage_payload.get("evaluation_pointer_payload")
        if not isinstance(evaluation_pointer_payload, dict):
            evaluation_pointer_payload = _read_optional_payload(artifacts.get("evaluation_pointer"))
        resolved_revision = str(lineage_payload.get("revision") or revision_slug)
        resolved_universe = str(lineage_payload.get("universe") or args.universe)
        resolved_source_scope = str(lineage_payload.get("source_scope") or args.source_scope)
        resolved_baseline_reference = str(lineage_payload.get("baseline_reference") or args.baseline_reference)
        lineage_status = lineage_payload.get("status")
        lineage_completed_step = lineage_payload.get("completed_step")
        backfill_handoff_summary = _extract_backfill_handoff_summary(lineage_payload)
        recommended_action = _recommended_action(
            status="completed",
            lineage_status=lineage_status,
            lineage_completed_step=lineage_completed_step,
        )
        if str(lineage_status or "") in {"revision_gate_failed", "failed", "interrupted"}:
            recommended_action = _lineage_failure_action(lineage_payload) or recommended_action

        payload: dict[str, object] = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": "completed",
            "snapshot_kind": "local_public_snapshot",
            "revision": resolved_revision,
            "universe": resolved_universe,
            "source_scope": resolved_source_scope,
            "baseline_reference": resolved_baseline_reference,
            "lineage_status": str(lineage_status or "unknown"),
            "lineage_completed_step": str(lineage_completed_step or "unknown"),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
            "read_order": _read_order(include_backfill_handoff=bool(backfill_handoff_summary)),
            "current_phase": _current_phase(
                status="completed",
                lineage_status=lineage_status,
                lineage_completed_step=lineage_completed_step,
            ),
            "recommended_action": recommended_action,
            "artifacts": {
                "public_snapshot": artifact_display_path(output_path, workspace_root=ROOT),
                **artifacts,
            },
            "readiness": _extract_readiness(lineage_payload),
            "promotion_summary": _extract_promotion_summary(lineage_payload),
            "backfill_handoff_summary": backfill_handoff_summary,
            "compare_contract": _build_compare_contract(
                snapshot_path=output_path,
                universe=resolved_universe,
                revision=resolved_revision,
            ),
        }
        if benchmark_payload:
            payload["benchmark_gate_summary"] = {
                "status": benchmark_payload.get("status"),
                "completed_step": benchmark_payload.get("completed_step"),
                "error_code": benchmark_payload.get("error_code"),
                "recommended_action": benchmark_payload.get("recommended_action"),
            }
        if isinstance(evaluation_pointer_payload, dict):
            payload["evaluation_summary"] = {
                "status": evaluation_pointer_payload.get("status"),
                "latest_manifest": evaluation_pointer_payload.get("latest_manifest"),
                "latest_summary": evaluation_pointer_payload.get("latest_summary"),
                "output_files": evaluation_pointer_payload.get("output_files"),
            }
        if isinstance(lineage_payload.get("error_code"), str):
            payload["lineage_error_code"] = lineage_payload.get("error_code")
        if isinstance(lineage_payload.get("error_message"), str):
            payload["lineage_error_message"] = lineage_payload.get("error_message")

        payload["highlights"] = _highlights(
            status=str(payload["status"]),
            revision=resolved_revision,
            lineage_manifest=artifact_display_path(lineage_path, workspace_root=ROOT),
            recommended_action=recommended_action,
            lineage_status=str(lineage_status) if lineage_status is not None else None,
            lineage_completed_step=str(lineage_completed_step) if lineage_completed_step is not None else None,
            readiness=_dict_payload(payload.get("readiness")) or None,
            promotion_summary=_dict_payload(payload.get("promotion_summary")) or None,
            backfill_handoff_summary=_dict_payload(payload.get("backfill_handoff_summary")) or None,
        )

        progress.update(message=f"snapshot payload prepared status={payload['status']}")
        with Heartbeat("[local-public-snapshot]", "writing snapshot manifest", logger=log_progress):
            write_json(output_path, payload)
        progress.complete(message=f"saved path={artifact_display_path(output_path, workspace_root=ROOT)} status={payload['status']}")
        print(f"[local-public-snapshot] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[local-public-snapshot] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        failure_payload = _build_failure_payload(
            revision=revision_slug,
            universe=args.universe,
            source_scope=args.source_scope,
            baseline_reference=args.baseline_reference,
            lineage_path=lineage_path,
            output_path=output_path,
            error_message=str(error),
        )
        write_json(output_path, failure_payload)
        print(f"[local-public-snapshot] failed: {error}", flush=True)
        return 1
    except Exception as error:
        failure_payload = _build_failure_payload(
            revision=revision_slug,
            universe=args.universe,
            source_scope=args.source_scope,
            baseline_reference=args.baseline_reference,
            lineage_path=lineage_path,
            output_path=output_path,
            error_message=str(error),
        )
        write_json(output_path, failure_payload)
        print(f"[local-public-snapshot] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
