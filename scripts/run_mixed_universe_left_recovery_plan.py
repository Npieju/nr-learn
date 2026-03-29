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


def _required_rows(step: dict[str, object]) -> list[str]:
    return [str(value) for value in _list_payload(step.get("required_for_rows")) if isinstance(value, str)]


def _artifact_paths(step: dict[str, object]) -> list[object]:
    return _list_payload(step.get("artifacts"))


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


def _command_key(command_preview: object) -> str | None:
    if not isinstance(command_preview, list) or not command_preview:
        return None
    return "\u241f".join(str(part) for part in command_preview)


def _step_name_from_command(command_preview: object) -> str:
    if not isinstance(command_preview, list) or len(command_preview) < 2:
        return "unknown_step"
    script_name = Path(str(command_preview[1])).name
    if script_name == "run_revision_gate.py":
        return "run_revision_gate"
    if script_name == "run_local_evaluate.py":
        return "run_local_evaluate"
    return script_name.replace(".py", "")


def _blocking_step_name(action: object) -> str:
    text = str(action or "").strip()
    return text if text else "resolve_upstream_blocker"


def _build_plan_steps(gap_audit_payload: dict[str, object]) -> list[dict[str, object]]:
    gap_rows = [row for row in _list_payload(gap_audit_payload.get("gap_rows")) if isinstance(row, dict)]
    deduped: dict[str, dict[str, object]] = {}

    lineage_blocker = _dict_payload(gap_audit_payload.get("lineage_blocker"))
    blocker_action = str(lineage_blocker.get("recommended_action") or "").strip()
    blocker_source = str(lineage_blocker.get("source") or "").strip()
    blocker_phase = str(lineage_blocker.get("current_phase") or "").strip()
    if blocker_action:
        blocker_artifacts = [
            artifact_path
            for artifact_path in _list_payload(lineage_blocker.get("artifact_paths"))
            if isinstance(artifact_path, str) and artifact_path.strip()
        ]
        deduped[f"blocker:{blocker_action}"] = {
            "step": _blocking_step_name(blocker_action),
            "action_kind": "manual",
            "recommended_action": blocker_action,
            "blocking_source": blocker_source or None,
            "blocking_phase": blocker_phase or None,
            "blocking_error_code": lineage_blocker.get("error_code"),
            "blocking_error_message": lineage_blocker.get("error_message"),
            "blocking_reasons": lineage_blocker.get("reasons"),
            "required_for_rows": [],
            "artifacts": blocker_artifacts,
        }

    for row in gap_rows:
        row_name = str(row.get("name") or "")
        required_sources = [source for source in _list_payload(row.get("required_sources")) if isinstance(source, dict)]
        for source in required_sources:
            if not isinstance(source, dict):
                continue
            command_preview = source.get("command_preview")
            key = _command_key(command_preview)
            if key is not None:
                current = deduped.get(key)
                if current is None:
                    deduped[key] = {
                        "step": _step_name_from_command(command_preview),
                        "command_preview": command_preview,
                        "command_status": source.get("command_status"),
                        "required_for_rows": [row_name],
                        "artifacts": [source.get("artifact_path")] if source.get("artifact_path") is not None else [],
                    }
                else:
                    current_rows = _required_rows(current)
                    if row_name not in current_rows:
                        current_rows.append(row_name)
                        current["required_for_rows"] = current_rows
                    artifact_path = source.get("artifact_path")
                    current_artifacts = _artifact_paths(current)
                    if artifact_path is not None and artifact_path not in current_artifacts:
                        current_artifacts.append(artifact_path)
                        current["artifacts"] = current_artifacts
                continue

            blocking_action = str(source.get("blocking_action") or "").strip()
            if not blocking_action:
                continue
            blocker_key = f"blocker:{blocking_action}"
            current = deduped.get(blocker_key)
            if current is None:
                artifacts = []
                if source.get("blocking_artifact_path") is not None:
                    artifacts.append(source.get("blocking_artifact_path"))
                for artifact_path in source.get("blocking_artifact_paths") or []:
                    if artifact_path is not None and artifact_path not in artifacts:
                        artifacts.append(artifact_path)
                deduped[blocker_key] = {
                    "step": _blocking_step_name(blocking_action),
                    "action_kind": "manual",
                    "recommended_action": blocking_action,
                    "blocking_source": source.get("blocking_source"),
                    "blocking_phase": source.get("blocking_phase"),
                    "blocking_error_code": source.get("blocking_error_code"),
                    "blocking_error_message": source.get("blocking_error_message"),
                    "blocking_reasons": source.get("blocking_reasons"),
                    "required_for_rows": [row_name],
                    "artifacts": artifacts,
                }
            else:
                current_rows = _required_rows(current)
                if row_name not in current_rows:
                    current_rows.append(row_name)
                    current["required_for_rows"] = current_rows
                artifact_path = source.get("blocking_artifact_path")
                current_artifacts = _artifact_paths(current)
                if artifact_path is not None and artifact_path not in current_artifacts:
                    current_artifacts.append(artifact_path)
                    current["artifacts"] = current_artifacts
                for blocking_artifact_path in source.get("blocking_artifact_paths") or []:
                    if blocking_artifact_path is not None and blocking_artifact_path not in current_artifacts:
                        current_artifacts.append(blocking_artifact_path)
                        current["artifacts"] = current_artifacts

    def _sort_key(item: dict[str, object]) -> tuple[int, str]:
        if str(item.get("action_kind") or "") == "manual":
            return (0, str(item.get("step") or ""))
        return (1, str(item.get("step") or ""))

    return sorted(deduped.values(), key=_sort_key)


def _build_summary(plan_steps: list[dict[str, object]], gap_audit_payload: dict[str, object]) -> dict[str, object]:
    gap_summary = _dict_payload(gap_audit_payload.get("summary"))
    statuses = {}
    for step in plan_steps:
        status = str(step.get("command_status") or step.get("action_kind") or "unknown")
        statuses[status] = int(statuses.get(status, 0)) + 1
    notes = []
    if plan_steps:
        notes.append(f"{len(plan_steps)} deduplicated recovery steps cover {sum(len(_required_rows(step)) for step in plan_steps)} missing-row dependencies")
    if gap_summary.get("rows_missing_all_sources"):
        notes.append("all missing-left rows currently depend on artifacts that do not yet exist")
    if gap_summary.get("rows_with_blocking_action"):
        notes.append("at least one upstream readiness blocker must be resolved before local metrics can be generated")
    if any(str(step.get("blocking_source") or "") == "backfill_handoff" for step in plan_steps):
        notes.append("the first unresolved blocker sits in the local backfill/materialize handoff rather than downstream mixed compare steps")
    return {
        "severity": gap_summary.get("severity"),
        "requested_revision": gap_audit_payload.get("requested_revision"),
        "resolved_left_revision": gap_audit_payload.get("resolved_left_revision"),
        "resolved_left_source_kind": gap_audit_payload.get("resolved_left_source_kind"),
        "resolved_left_artifact": gap_audit_payload.get("resolved_left_artifact"),
        "step_count": len(plan_steps),
        "command_status_counts": statuses,
        "rows_missing_all_sources": gap_summary.get("rows_missing_all_sources"),
        "rows_with_planned_commands": gap_summary.get("rows_with_planned_commands"),
        "rows_with_blocking_action": gap_summary.get("rows_with_blocking_action"),
        "notes": notes,
    }


def _recommended_action(plan_steps: list[dict[str, object]], gap_audit_payload: dict[str, object]) -> object:
    if plan_steps:
        first_step = plan_steps[0]
        manual_action = first_step.get("recommended_action")
        if isinstance(manual_action, str) and manual_action.strip():
            return manual_action
        step_name = first_step.get("step")
        if isinstance(step_name, str) and step_name.strip():
            return step_name
    return gap_audit_payload.get("recommended_action")


def _planned_summary(*, requested_revision: str, left_universe: str) -> dict[str, object]:
    return {
        "severity": "info",
        "requested_revision": requested_revision,
        "resolved_left_revision": None,
        "resolved_left_source_kind": None,
        "resolved_left_artifact": None,
        "step_count": 0,
        "command_status_counts": {},
        "rows_missing_all_sources": [],
        "rows_with_planned_commands": [],
        "rows_with_blocking_action": [],
        "notes": [
            f"gap audit is not available yet for {left_universe}",
            "generate the left gap audit manifest before building a recovery plan",
        ],
    }


def _current_phase(*, status: str, plan_steps: list[dict[str, object]]) -> str:
    if status == "planned":
        return "mixed_universe_left_gap_audit"
    if not plan_steps:
        return "mixed_universe_left_recovery_plan_completed"
    first_step = plan_steps[0]
    if str(first_step.get("action_kind") or "") == "manual":
        return "local_revision_gate"
    return "mixed_universe_left_recovery_plan_partial"


def _highlights(*, status: str, recommended_action: object, summary: dict[str, object], plan_steps: list[dict[str, object]]) -> list[str]:
    highlights: list[str] = []
    action = str(recommended_action or "review_recovery_plan")
    if status == "planned":
        highlights.append("recovery plan is waiting for left gap audit")
        highlights.append(f"next operator action: {action}")
        return highlights

    step_count = _int_value(summary.get("step_count"), default=0)
    blocking_rows = _list_payload(summary.get("rows_with_blocking_action"))
    first_step = plan_steps[0] if plan_steps else {}
    blocking_error_code = str(first_step.get("blocking_error_code") or "") if isinstance(first_step, dict) else ""

    highlights.append(f"recovery plan contains {step_count} deduplicated step(s)")
    if blocking_rows:
        highlights.append(f"{len(blocking_rows)} row(s) remain blocked by upstream local readiness")
    if blocking_error_code:
        highlights.append(f"first recovery step is blocked with error_code={blocking_error_code}")
    if isinstance(first_step, dict) and str(first_step.get("blocking_source") or "") == "backfill_handoff":
        highlights.append(
            f"first recovery step targets local handoff phase={first_step.get('blocking_phase') if first_step.get('blocking_phase') else 'unknown'}"
        )
    if len(highlights) < 4:
        highlights.append(f"next operator action: {action}")
    return highlights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--gap-audit-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    gap_audit_manifest = args.gap_audit_manifest or f"artifacts/reports/mixed_universe_left_gap_audit_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    output = args.output or f"artifacts/reports/mixed_universe_left_recovery_plan_{left_universe}_vs_{right_universe}_{revision_slug}.json"

    gap_audit_path = _resolve_path(gap_audit_manifest)
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        progress = ProgressBar(total=4, prefix="[mixed-universe-left-recovery-plan]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"starting revision={revision_slug} left={left_universe} right={right_universe} "
                f"dry_run={'yes' if args.dry_run else 'no'}"
            )
        )

        if args.dry_run and not gap_audit_path.exists():
            summary = _planned_summary(requested_revision=revision_slug, left_universe=left_universe)
            status = "planned"
            recommended_action = "generate_gap_audit_manifest"
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": status,
                "plan_kind": "mixed_universe_left_recovery_plan",
                "revision": revision_slug,
                "requested_revision": revision_slug,
                "resolved_left_revision": None,
                "resolved_left_source_kind": None,
                "resolved_left_artifact": None,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "current_phase": _current_phase(status=status, plan_steps=[]),
                "recommended_action": recommended_action,
                "artifacts": {
                    "recovery_plan_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "gap_audit_manifest": artifact_display_path(gap_audit_path, workspace_root=ROOT),
                },
                "read_order": [
                    "mixed_universe_left_gap_audit",
                    "mixed_universe_left_recovery_plan",
                ],
                "summary": summary,
                "highlights": _highlights(
                    status=status,
                    recommended_action=recommended_action,
                    summary=summary,
                    plan_steps=[],
                ),
                "plan_steps": [],
            }
            progress.update(message="preparing planned recovery plan manifest")
            with Heartbeat("[mixed-universe-left-recovery-plan]", "writing planned recovery plan manifest", logger=log_progress):
                write_json(output_path, payload)
            progress.complete(message=f"planned manifest saved path={artifact_display_path(output_path, workspace_root=ROOT)}")
            print(f"[mixed-universe-left-recovery-plan] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        with Heartbeat("[mixed-universe-left-recovery-plan]", "loading left gap audit manifest", logger=log_progress):
            gap_audit_payload = _read_required_payload(gap_audit_path, label="left gap audit manifest")
        progress.update(message=f"gap audit loaded path={artifact_display_path(gap_audit_path, workspace_root=ROOT)}")
        with Heartbeat("[mixed-universe-left-recovery-plan]", "building recovery steps", logger=log_progress):
            plan_steps = _build_plan_steps(gap_audit_payload)
        progress.update(message=f"recovery steps built count={len(plan_steps)}")
        status = "completed" if not plan_steps else "partial"
        summary = _build_summary(plan_steps, gap_audit_payload)
        recommended_action = _recommended_action(plan_steps, gap_audit_payload)

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": status,
            "plan_kind": "mixed_universe_left_recovery_plan",
            "revision": revision_slug,
            "requested_revision": gap_audit_payload.get("requested_revision") or gap_audit_payload.get("revision") or revision_slug,
            "resolved_left_revision": gap_audit_payload.get("resolved_left_revision"),
            "resolved_left_source_kind": gap_audit_payload.get("resolved_left_source_kind"),
            "resolved_left_artifact": gap_audit_payload.get("resolved_left_artifact"),
            "left_universe": left_universe,
            "right_universe": right_universe,
            "current_phase": _current_phase(status=status, plan_steps=plan_steps),
            "recommended_action": recommended_action,
            "artifacts": {
                "recovery_plan_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "gap_audit_manifest": artifact_display_path(gap_audit_path, workspace_root=ROOT),
            },
            "read_order": [
                "mixed_universe_left_gap_audit",
                "mixed_universe_left_recovery_plan",
            ],
            "summary": summary,
            "highlights": _highlights(
                status=status,
                recommended_action=recommended_action,
                summary=summary,
                plan_steps=plan_steps,
            ),
            "plan_steps": plan_steps,
        }
        progress.update(message=f"recovery plan payload prepared status={status}")
        with Heartbeat("[mixed-universe-left-recovery-plan]", "writing recovery plan manifest", logger=log_progress):
            write_json(output_path, payload)
        progress.complete(message=f"saved path={artifact_display_path(output_path, workspace_root=ROOT)} status={status}")
        print(f"[mixed-universe-left-recovery-plan] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if status == "completed" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-left-recovery-plan] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-left-recovery-plan] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-left-recovery-plan] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
