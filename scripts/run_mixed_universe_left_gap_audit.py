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
from racing_ml.common.mixed_artifacts import resolve_local_lineage_path
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_LEFT_UNIVERSE = "local_nankan"
DEFAULT_RIGHT_UNIVERSE = "jra"


def log_progress(message: str) -> None:
    print(message, flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _normalize_slug(value: str) -> str:
    normalized = "".join(character.lower() if character.isalnum() else "_" for character in str(value).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError("slug must not be empty")
    return normalized


def _read_optional_payload(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _read_required_payload(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {artifact_display_path(path, workspace_root=ROOT)}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {artifact_display_path(path, workspace_root=ROOT)}")
    return payload


def _dict_payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _list_payload(value: object) -> list[object]:
    return value if isinstance(value, list) else []


def _int_value(value: object, default: int = 0) -> int:
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str):
            return int(value)
        return default
    except (TypeError, ValueError):
        return default


def _artifact_exists(path_text: object) -> bool:
    if not isinstance(path_text, str) or not path_text.strip():
        return False
    return _resolve_path(path_text).exists()


def _backfill_handoff_blocker(lineage_payload: dict[str, object]) -> dict[str, object] | None:
    handoff_payload = _dict_payload(lineage_payload.get("backfill_handoff_payload"))
    backfill_summary_payload = _dict_payload(lineage_payload.get("backfill_summary_payload"))
    if not handoff_payload and not backfill_summary_payload:
        return None

    handoff_status = str(handoff_payload.get("status") or "")
    summary_status = str(backfill_summary_payload.get("status") or "")
    summary_phase = str(backfill_summary_payload.get("current_phase") or "")
    if handoff_status in {"", "completed"} and summary_status in {"", "completed", "partial"} and summary_phase in {"", "materialized_primary_raw"}:
        return None

    artifacts = lineage_payload.get("artifacts") if isinstance(lineage_payload.get("artifacts"), dict) else {}
    return {
        "source": "backfill_handoff",
        "status": handoff_status or summary_status or "unknown",
        "current_phase": handoff_payload.get("current_phase") or backfill_summary_payload.get("current_phase"),
        "error_code": handoff_payload.get("error_code") or backfill_summary_payload.get("stopped_reason"),
        "error_message": handoff_payload.get("error_message") or backfill_summary_payload.get("stopped_reason"),
        "recommended_action": handoff_payload.get("recommended_action") or backfill_summary_payload.get("recommended_action"),
        "artifact_path": (artifacts.get("backfill_wrapper_manifest") if isinstance(artifacts, dict) else None),
        "artifact_paths": [
            path
            for path in [
                artifacts.get("backfill_wrapper_manifest") if isinstance(artifacts, dict) else None,
                artifacts.get("backfill_manifest") if isinstance(artifacts, dict) else None,
                artifacts.get("primary_materialize_manifest") if isinstance(artifacts, dict) else None,
            ]
            if isinstance(path, str) and path.strip()
        ],
        "reasons": handoff_payload.get("highlights") or backfill_summary_payload.get("highlights"),
    }


def _lineage_blocker(lineage_payload: dict[str, object]) -> dict[str, object] | None:
    handoff_blocker = _backfill_handoff_blocker(lineage_payload)
    if handoff_blocker is not None:
        return handoff_blocker

    lineage_status = str(lineage_payload.get("status") or "")
    if lineage_status and lineage_status not in {"", "completed", "pass", "ready"}:
        artifacts = lineage_payload.get("artifacts") if isinstance(lineage_payload.get("artifacts"), dict) else {}
        return {
            "source": "local_revision_gate",
            "status": lineage_status,
            "current_phase": lineage_payload.get("current_phase"),
            "error_code": lineage_payload.get("error_code"),
            "error_message": lineage_payload.get("error_message"),
            "recommended_action": lineage_payload.get("recommended_action"),
            "artifact_path": artifacts.get("revision_manifest") if isinstance(artifacts, dict) else None,
            "artifact_paths": [
                path
                for path in [
                    artifacts.get("revision_manifest") if isinstance(artifacts, dict) else None,
                    artifacts.get("promotion_output") if isinstance(artifacts, dict) else None,
                    artifacts.get("evaluation_pointer") if isinstance(artifacts, dict) else None,
                    artifacts.get("wf_summary") if isinstance(artifacts, dict) else None,
                ]
                if isinstance(path, str) and path.strip()
            ],
            "reasons": lineage_payload.get("highlights"),
        }

    data_preflight = lineage_payload.get("data_preflight_payload") if isinstance(lineage_payload.get("data_preflight_payload"), dict) else None
    benchmark_gate = lineage_payload.get("benchmark_gate_payload") if isinstance(lineage_payload.get("benchmark_gate_payload"), dict) else None

    candidate = data_preflight if isinstance(data_preflight, dict) else benchmark_gate
    if not isinstance(candidate, dict):
        return None

    status = str(candidate.get("status") or "")
    if status in {"", "completed", "pass", "ready"}:
        return None

    artifacts = candidate.get("artifacts") if isinstance(candidate.get("artifacts"), dict) else {}
    return {
        "source": "benchmark_or_preflight",
        "status": status,
        "error_code": candidate.get("error_code"),
        "error_message": candidate.get("error_message"),
        "recommended_action": candidate.get("recommended_action"),
        "artifact_path": artifacts.get("preflight_manifest") if isinstance(artifacts, dict) else None,
        "artifact_paths": [artifacts.get("preflight_manifest")] if isinstance(artifacts, dict) and isinstance(artifacts.get("preflight_manifest"), str) else [],
        "reasons": ((candidate.get("readiness") or {}).get("reasons") if isinstance(candidate.get("readiness"), dict) else None),
    }


def _required_sources_for_row(row_name: str) -> list[dict[str, str]]:
    if row_name == "decision":
        return [
            {"artifact_key": "promotion_output", "label": "promotion manifest", "command_key": "revision_gate"},
            {"artifact_key": "revision_manifest", "label": "revision manifest", "command_key": "revision_gate"},
        ]
    if row_name in {"stability_assessment", "auc", "top1_roi", "ev_top1_roi", "nested_wf_bets_total"}:
        return [{"artifact_key": "evaluation_pointer", "label": "evaluation pointer", "command_key": "evaluation_pointer"}]
    if row_name in {"nested_wf_weighted_test_roi", "formal_benchmark_weighted_roi", "formal_benchmark_feasible_folds"}:
        return [
            {"artifact_key": "promotion_output", "label": "promotion manifest", "command_key": "revision_gate"},
            {"artifact_key": "wf_summary", "label": "wf summary", "command_key": "revision_gate"},
        ]
    return []


def _build_gap_rows(
    *,
    numeric_compare_payload: dict[str, object],
    lineage_payload: dict[str, object],
) -> list[dict[str, object]]:
    row_results = [row for row in _list_payload(numeric_compare_payload.get("row_results")) if isinstance(row, dict)]
    lineage_artifacts = _dict_payload(lineage_payload.get("artifacts"))
    lineage_blocker = _lineage_blocker(lineage_payload)

    gap_rows: list[dict[str, object]] = []
    for row in row_results:
        if str(row.get("comparison_status") or "") != "missing_left_value":
            continue
        row_name = str(row.get("name") or "")
        required_sources = _required_sources_for_row(row_name)
        source_checks = []
        for source in required_sources:
            command_payload = _dict_payload(lineage_payload.get(source["command_key"]))
            artifact_path = lineage_artifacts.get(source["artifact_key"]) if isinstance(lineage_artifacts, dict) else None
            source_checks.append(
                {
                    "artifact_key": source["artifact_key"],
                    "label": source["label"],
                    "artifact_path": artifact_path,
                    "exists": _artifact_exists(artifact_path),
                    "command_preview": command_payload.get("command"),
                    "command_status": command_payload.get("status"),
                    "blocking_action": (lineage_blocker or {}).get("recommended_action"),
                    "blocking_error_code": (lineage_blocker or {}).get("error_code"),
                    "blocking_error_message": (lineage_blocker or {}).get("error_message"),
                    "blocking_source": (lineage_blocker or {}).get("source"),
                    "blocking_phase": (lineage_blocker or {}).get("current_phase"),
                    "blocking_artifact_path": (lineage_blocker or {}).get("artifact_path"),
                    "blocking_artifact_paths": (lineage_blocker or {}).get("artifact_paths"),
                    "blocking_reasons": (lineage_blocker or {}).get("reasons"),
                }
            )
        gap_rows.append(
            {
                "name": row_name,
                "category": row.get("category"),
                "comparison_status": row.get("comparison_status"),
                "required_sources": source_checks,
                "lineage_blocker": lineage_blocker,
            }
        )
    return gap_rows


def _audit_summary(
    gap_rows: list[dict[str, object]],
    *,
    requested_revision: str | None,
    resolved_left_revision: str | None,
    resolved_left_source_kind: str | None,
    resolved_left_artifact: str | None,
) -> dict[str, object]:
    rows_missing_all_sources = []
    rows_with_planned_commands = []
    rows_with_blocking_action = []
    for row in gap_rows:
        required_sources = [source for source in _list_payload(row.get("required_sources")) if isinstance(source, dict)]
        if required_sources and all(not bool(source.get("exists")) for source in required_sources if isinstance(source, dict)):
            rows_missing_all_sources.append(row.get("name"))
        if any(str(source.get("command_status") or "") == "planned" for source in required_sources if isinstance(source, dict)):
            rows_with_planned_commands.append(row.get("name"))
        if any(str(source.get("blocking_action") or "").strip() for source in required_sources if isinstance(source, dict)):
            rows_with_blocking_action.append(row.get("name"))

    severity = "info"
    if rows_missing_all_sources:
        severity = "severe"
    elif gap_rows:
        severity = "moderate"

    notes = []
    if rows_missing_all_sources:
        notes.append(f"{len(rows_missing_all_sources)} rows still lack every required left-side source artifact")
    if rows_with_planned_commands:
        notes.append(f"{len(rows_with_planned_commands)} rows can be addressed by commands already recorded in local revision lineage")
    if rows_with_blocking_action:
        notes.append(f"{len(rows_with_blocking_action)} rows are currently blocked by an upstream local readiness action")
    if not notes:
        notes.append("no missing-left rows were found in the numeric compare manifest")

    return {
        "severity": severity,
        "requested_revision": requested_revision,
        "resolved_left_revision": resolved_left_revision,
        "resolved_left_source_kind": resolved_left_source_kind,
        "resolved_left_artifact": resolved_left_artifact,
        "row_count": len(gap_rows),
        "rows_missing_all_sources": rows_missing_all_sources,
        "rows_with_planned_commands": rows_with_planned_commands,
        "rows_with_blocking_action": rows_with_blocking_action,
        "notes": notes,
    }


def _planned_audit_summary(*, requested_revision: str, left_universe: str) -> dict[str, object]:
    return {
        "severity": "info",
        "requested_revision": requested_revision,
        "resolved_left_revision": None,
        "resolved_left_source_kind": None,
        "resolved_left_artifact": None,
        "row_count": 0,
        "rows_missing_all_sources": [],
        "rows_with_planned_commands": [],
        "rows_with_blocking_action": [],
        "notes": [
            f"numeric compare or lineage is not available yet for {left_universe}",
            "generate numeric compare and local lineage before auditing missing left-side rows",
        ],
    }


def _planned_lineage_blocker(*, left_universe: str) -> dict[str, object]:
    return {
        "status": "planned",
        "error_code": "lineage_manifest_missing",
        "error_message": f"local revision lineage is not available yet for {left_universe}",
        "recommended_action": "generate_numeric_compare_and_lineage_manifests",
        "artifact_path": None,
        "reasons": [
            "numeric compare manifest is missing",
            "local revision lineage manifest is missing",
        ],
    }


def _current_phase(*, status: str, lineage_blocker: dict[str, object] | None) -> str:
    if status == "planned":
        return "mixed_universe_numeric_compare"
    blocker_status = str((lineage_blocker or {}).get("status") or "")
    if blocker_status and blocker_status not in {"completed", "pass", "ready"}:
        blocker_source = str((lineage_blocker or {}).get("source") or "")
        if blocker_source == "backfill_handoff":
            return "local_backfill_then_benchmark"
        return "local_revision_gate"
    if status == "completed":
        return "mixed_universe_left_gap_audit_completed"
    return "mixed_universe_left_gap_audit_partial"


def _highlights(
    *,
    status: str,
    recommended_action: object,
    summary: dict[str, object],
    lineage_blocker: dict[str, object] | None,
) -> list[str]:
    highlights: list[str] = []
    blocker = lineage_blocker or {}
    blocker_error_code = str(blocker.get("error_code") or "")
    blocker_action = str(blocker.get("recommended_action") or recommended_action or "review_gap_rows")

    if status == "planned":
        highlights.append("gap audit is waiting for numeric compare and local revision lineage")
        highlights.append(f"next operator action: {blocker_action}")
        return highlights

    row_count = _int_value(summary.get("row_count"), default=0)
    missing_all_sources = _list_payload(summary.get("rows_missing_all_sources"))
    blocking_rows = _list_payload(summary.get("rows_with_blocking_action"))

    highlights.append(f"gap audit tracks {row_count} missing-left row(s)")
    if missing_all_sources:
        highlights.append(f"{len(missing_all_sources)} row(s) still lack every required left-side source artifact")
    if blocker_error_code:
        highlights.append(f"upstream local lineage is blocked with error_code={blocker_error_code}")
        blocker_source = str(blocker.get("source") or "")
        blocker_phase = str(blocker.get("current_phase") or "")
        if blocker_source == "backfill_handoff":
            highlights.append(
                f"upstream blocker is currently in local handoff phase={blocker_phase if blocker_phase else 'unknown'}"
            )
    elif blocking_rows:
        highlights.append(f"{len(blocking_rows)} row(s) are gated by an upstream local readiness action")
    if len(highlights) < 4:
        highlights.append(f"next operator action: {blocker_action}")
    return highlights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--numeric-compare-manifest", default=None)
    parser.add_argument("--left-lineage-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    numeric_compare_manifest = args.numeric_compare_manifest or f"artifacts/reports/mixed_universe_numeric_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    output = args.output or f"artifacts/reports/mixed_universe_left_gap_audit_{left_universe}_vs_{right_universe}_{revision_slug}.json"

    compare_path = _resolve_path(numeric_compare_manifest)
    lineage_path = resolve_local_lineage_path(
        workspace_root=ROOT,
        explicit_manifest=args.left_lineage_manifest,
        revision_slug=revision_slug,
        left_universe=left_universe,
        numeric_compare_path=compare_path,
    )
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        progress = ProgressBar(total=4, prefix="[mixed-universe-left-gap-audit]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"starting revision={revision_slug} left={left_universe} right={right_universe} "
                f"dry_run={'yes' if args.dry_run else 'no'}"
            )
        )

        if args.dry_run and not compare_path.exists() and not lineage_path.exists():
            recommended_action = "generate_numeric_compare_and_lineage_manifests"
            summary = _planned_audit_summary(requested_revision=revision_slug, left_universe=left_universe)
            lineage_blocker = _planned_lineage_blocker(left_universe=left_universe)
            status = "planned"
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": status,
                "audit_kind": "mixed_universe_left_gap_audit",
                "revision": revision_slug,
                "requested_revision": revision_slug,
                "resolved_left_revision": None,
                "resolved_left_source_kind": None,
                "resolved_left_artifact": None,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "current_phase": _current_phase(status=status, lineage_blocker=lineage_blocker),
                "recommended_action": recommended_action,
                "artifacts": {
                    "gap_audit_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "numeric_compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                    "left_lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
                },
                "read_order": [
                    "mixed_universe_numeric_compare",
                    "local_revision_gate",
                    "mixed_universe_left_gap_audit",
                ],
                "summary": summary,
                "lineage_blocker": lineage_blocker,
                "highlights": _highlights(
                    status=status,
                    recommended_action=recommended_action,
                    summary=summary,
                    lineage_blocker=lineage_blocker,
                ),
                "gap_rows": [],
            }
            progress.update(message="preparing planned gap audit manifest")
            with Heartbeat("[mixed-universe-left-gap-audit]", "writing planned gap audit manifest", logger=log_progress):
                write_json(output_path, payload)
            progress.complete(message=f"planned manifest saved path={artifact_display_path(output_path, workspace_root=ROOT)}")
            print(f"[mixed-universe-left-gap-audit] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        with Heartbeat("[mixed-universe-left-gap-audit]", "loading numeric compare and lineage manifests", logger=log_progress):
            numeric_compare_payload = _read_required_payload(compare_path, label="numeric compare manifest")
            lineage_payload = _read_required_payload(lineage_path, label="left lineage manifest")
        progress.update(message="upstream manifests loaded")
        with Heartbeat("[mixed-universe-left-gap-audit]", "building missing-left gap rows", logger=log_progress):
            gap_rows = _build_gap_rows(numeric_compare_payload=numeric_compare_payload, lineage_payload=lineage_payload)
        progress.update(message=f"gap rows built count={len(gap_rows)}")
        requested_revision = str(numeric_compare_payload.get("requested_revision") or numeric_compare_payload.get("revision") or revision_slug)
        resolved_left_revision = str(numeric_compare_payload.get("resolved_left_revision") or lineage_payload.get("revision") or "") or None
        resolved_left_source_kind = str(numeric_compare_payload.get("resolved_left_source_kind") or "local_revision_gate")
        resolved_left_artifact = str(numeric_compare_payload.get("resolved_left_artifact") or artifact_display_path(lineage_path, workspace_root=ROOT))
        audit_summary = _audit_summary(
            gap_rows,
            requested_revision=requested_revision,
            resolved_left_revision=resolved_left_revision,
            resolved_left_source_kind=resolved_left_source_kind,
            resolved_left_artifact=resolved_left_artifact,
        )
        status = "completed" if not gap_rows else "partial"
        lineage_blocker = _lineage_blocker(lineage_payload)
        recommended_action = (lineage_blocker or {}).get("recommended_action") or numeric_compare_payload.get("recommended_action")

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": status,
            "audit_kind": "mixed_universe_left_gap_audit",
            "revision": revision_slug,
            "requested_revision": requested_revision,
            "resolved_left_revision": resolved_left_revision,
            "resolved_left_source_kind": resolved_left_source_kind,
            "resolved_left_artifact": resolved_left_artifact,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "current_phase": _current_phase(status=status, lineage_blocker=lineage_blocker),
            "recommended_action": recommended_action,
            "artifacts": {
                "gap_audit_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "numeric_compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                "left_lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
            },
            "read_order": [
                "mixed_universe_numeric_compare",
                "local_revision_gate",
                "mixed_universe_left_gap_audit",
            ],
            "summary": audit_summary,
            "lineage_blocker": lineage_blocker,
            "highlights": _highlights(
                status=status,
                recommended_action=recommended_action,
                summary=audit_summary,
                lineage_blocker=lineage_blocker,
            ),
            "gap_rows": gap_rows,
        }
        progress.update(message=f"gap audit payload prepared status={status}")
        with Heartbeat("[mixed-universe-left-gap-audit]", "writing gap audit manifest", logger=log_progress):
            write_json(output_path, payload)
        progress.complete(message=f"saved path={artifact_display_path(output_path, workspace_root=ROOT)} status={status}")
        print(f"[mixed-universe-left-gap-audit] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if status == "completed" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-left-gap-audit] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-left-gap-audit] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-left-gap-audit] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
