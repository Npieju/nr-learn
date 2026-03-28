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


DEFAULT_LEFT_UNIVERSE = "local_nankan"
DEFAULT_RIGHT_UNIVERSE = "jra"


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


def _build_plan_steps(gap_audit_payload: dict[str, object]) -> list[dict[str, object]]:
    gap_rows = [row for row in (gap_audit_payload.get("gap_rows") or []) if isinstance(row, dict)]
    deduped: dict[str, dict[str, object]] = {}

    for row in gap_rows:
        row_name = str(row.get("name") or "")
        required_sources = row.get("required_sources") if isinstance(row.get("required_sources"), list) else []
        for source in required_sources:
            if not isinstance(source, dict):
                continue
            command_preview = source.get("command_preview")
            key = _command_key(command_preview)
            if key is None:
                continue
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
                if row_name not in current["required_for_rows"]:
                    current["required_for_rows"].append(row_name)
                artifact_path = source.get("artifact_path")
                if artifact_path is not None and artifact_path not in current["artifacts"]:
                    current["artifacts"].append(artifact_path)

    return sorted(deduped.values(), key=lambda item: str(item.get("step") or ""))


def _build_summary(plan_steps: list[dict[str, object]], gap_audit_payload: dict[str, object]) -> dict[str, object]:
    gap_summary = gap_audit_payload.get("summary") if isinstance(gap_audit_payload.get("summary"), dict) else {}
    statuses = {}
    for step in plan_steps:
        status = str(step.get("command_status") or "unknown")
        statuses[status] = int(statuses.get(status, 0)) + 1
    notes = []
    if plan_steps:
        notes.append(f"{len(plan_steps)} deduplicated commands cover {sum(len(step.get('required_for_rows') or []) for step in plan_steps)} missing-row dependencies")
    if gap_summary.get("rows_missing_all_sources"):
        notes.append("all missing-left rows currently depend on artifacts that do not yet exist")
    return {
        "severity": gap_summary.get("severity"),
        "step_count": len(plan_steps),
        "command_status_counts": statuses,
        "rows_missing_all_sources": gap_summary.get("rows_missing_all_sources"),
        "rows_with_planned_commands": gap_summary.get("rows_with_planned_commands"),
        "notes": notes,
    }


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

        if args.dry_run and not gap_audit_path.exists():
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": "planned",
                "plan_kind": "mixed_universe_left_recovery_plan",
                "revision": revision_slug,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "recommended_action": "generate_gap_audit_manifest",
                "artifacts": {
                    "recovery_plan_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "gap_audit_manifest": artifact_display_path(gap_audit_path, workspace_root=ROOT),
                },
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-left-recovery-plan] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        gap_audit_payload = _read_required_payload(gap_audit_path, label="left gap audit manifest")
        plan_steps = _build_plan_steps(gap_audit_payload)
        status = "completed" if not plan_steps else "partial"
        summary = _build_summary(plan_steps, gap_audit_payload)

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": status,
            "plan_kind": "mixed_universe_left_recovery_plan",
            "revision": revision_slug,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "recommended_action": gap_audit_payload.get("recommended_action"),
            "artifacts": {
                "recovery_plan_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "gap_audit_manifest": artifact_display_path(gap_audit_path, workspace_root=ROOT),
            },
            "read_order": [
                "mixed_universe_left_gap_audit",
                "mixed_universe_left_recovery_plan",
            ],
            "summary": summary,
            "plan_steps": plan_steps,
        }
        write_json(output_path, payload)
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