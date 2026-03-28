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
from racing_ml.common.mixed_artifacts import prefer_existing_path
from racing_ml.common.mixed_artifacts import read_optional_json_path
from racing_ml.common.mixed_artifacts import resolve_local_snapshot_and_lineage_paths


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


def _read_optional_payload(path: Path) -> dict[str, object] | None:
    return read_optional_json_path(path, workspace_root=ROOT)


def _phase_status(payload: dict[str, object] | None, status_keys: list[str]) -> str:
    if not isinstance(payload, dict):
        return "missing"
    for key in status_keys:
        value = payload.get(key)
        if value is not None:
            return str(value)
    return "unknown"


def _artifact_summary(label: str, path: Path, payload: dict[str, object] | None, status_keys: list[str]) -> dict[str, object]:
    return {
        "label": label,
        "path": artifact_display_path(path, workspace_root=ROOT),
        "exists": path.exists(),
        "status": _phase_status(payload, status_keys),
        "recommended_action": payload.get("recommended_action") if isinstance(payload, dict) else None,
    }


def _derive_current_phase(summaries: list[dict[str, object]]) -> str:
    ordered_labels = [
        "public_snapshot",
        "readiness",
        "compare",
        "schema",
        "numeric_compare",
        "numeric_summary",
        "left_gap_audit",
        "left_recovery_plan",
    ]
    summary_by_label = {str(item.get("label")): item for item in summaries}
    current = "public_snapshot"
    for label in ordered_labels:
        item = summary_by_label.get(label)
        if not isinstance(item, dict):
            break
        status = str(item.get("status") or "unknown")
        current = label
        if status in {"missing", "planned", "partial", "schema_blocked", "not_ready"}:
            break
    return current


def _derive_next_action(payloads: dict[str, dict[str, object] | None]) -> tuple[str | None, str | None]:
    priority = [
        ("left_recovery_plan", "recommended_action"),
        ("left_gap_audit", "recommended_action"),
        ("numeric_summary", "recommended_action"),
        ("numeric_compare", "recommended_action"),
        ("schema", "recommended_action"),
        ("readiness", "recommended_action"),
        ("public_snapshot", None),
    ]
    for key, action_key in priority:
        payload = payloads.get(key)
        if not isinstance(payload, dict):
            continue
        action = payload.get(action_key) if action_key is not None else None
        if isinstance(action, str) and action.strip():
            return key, action
    return None, None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    output = args.output or f"artifacts/reports/mixed_universe_status_board_{left_universe}_vs_{right_universe}_{revision_slug}.json"

    public_snapshot_path, _ = resolve_local_snapshot_and_lineage_paths(
        workspace_root=ROOT,
        revision_slug=revision_slug,
        left_universe=left_universe,
    )
    paths = {
        "public_snapshot": public_snapshot_path,
        "readiness": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_*.json",
        ),
        "compare": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_*.json",
        ),
        "schema": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_schema_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_schema_{left_universe}_vs_{right_universe}_*.json",
        ),
        "numeric_compare": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_numeric_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_numeric_compare_{left_universe}_vs_{right_universe}_*.json",
        ),
        "numeric_summary": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_numeric_summary_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_numeric_summary_{left_universe}_vs_{right_universe}_*.json",
        ),
        "left_gap_audit": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_left_gap_audit_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_left_gap_audit_{left_universe}_vs_{right_universe}_*.json",
        ),
        "left_recovery_plan": prefer_existing_path(
            workspace_root=ROOT,
            expected_path=f"artifacts/reports/mixed_universe_left_recovery_plan_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            fallback_pattern=f"artifacts/reports/mixed_universe_left_recovery_plan_{left_universe}_vs_{right_universe}_*.json",
        ),
    }
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

        if args.dry_run and not any(path.exists() for path in paths.values()):
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": "planned",
                "board_kind": "mixed_universe_status_board",
                "revision": revision_slug,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "recommended_action": "generate_mixed_universe_manifests",
                "artifacts": {
                    "status_board_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                },
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-status-board] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        payloads = {key: _read_optional_payload(path) for key, path in paths.items()}
        summaries = [
            _artifact_summary("public_snapshot", paths["public_snapshot"], payloads["public_snapshot"], ["status", "lineage_status"]),
            _artifact_summary("readiness", paths["readiness"], payloads["readiness"], ["status"]),
            _artifact_summary("compare", paths["compare"], payloads["compare"], ["status", "compare_mode"]),
            _artifact_summary("schema", paths["schema"], payloads["schema"], ["status"]),
            _artifact_summary("numeric_compare", paths["numeric_compare"], payloads["numeric_compare"], ["status"]),
            _artifact_summary("numeric_summary", paths["numeric_summary"], payloads["numeric_summary"], ["status"]),
            _artifact_summary("left_gap_audit", paths["left_gap_audit"], payloads["left_gap_audit"], ["status"]),
            _artifact_summary("left_recovery_plan", paths["left_recovery_plan"], payloads["left_recovery_plan"], ["status"]),
        ]

        current_phase = _derive_current_phase(summaries)
        next_action_source, next_action = _derive_next_action(payloads)
        overall_status = "completed"
        if any(str(item.get("status") or "") in {"missing", "planned", "partial", "schema_blocked", "not_ready"} for item in summaries):
            overall_status = "partial"

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": overall_status,
            "board_kind": "mixed_universe_status_board",
            "revision": revision_slug,
            "requested_revision": revision_slug,
            "resolved_left_revision": (payloads.get("public_snapshot") or {}).get("revision") if isinstance(payloads.get("public_snapshot"), dict) else None,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "recommended_action": next_action,
            "artifacts": {
                "status_board_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                **{f"{key}_manifest": artifact_display_path(path, workspace_root=ROOT) for key, path in paths.items()},
            },
            "read_order": [
                "public_snapshot",
                "readiness",
                "compare",
                "schema",
                "numeric_compare",
                "numeric_summary",
                "left_gap_audit",
                "left_recovery_plan",
                "mixed_universe_status_board",
            ],
            "current_phase": current_phase,
            "next_action_source": next_action_source,
            "phase_summaries": summaries,
            "highlights": {
                "requested_revision": revision_slug,
                "resolved_left_revision": (payloads.get("public_snapshot") or {}).get("revision") if isinstance(payloads.get("public_snapshot"), dict) else None,
                "numeric_summary_verdict": ((payloads.get("numeric_summary") or {}).get("promote_safe_summary", {}) or {}).get("verdict") if isinstance(payloads.get("numeric_summary"), dict) else None,
                "numeric_summary_severity": ((payloads.get("numeric_summary") or {}).get("promote_safe_summary", {}) or {}).get("severity") if isinstance(payloads.get("numeric_summary"), dict) else None,
                "gap_audit_severity": ((payloads.get("left_gap_audit") or {}).get("summary", {}) or {}).get("severity") if isinstance(payloads.get("left_gap_audit"), dict) else None,
                "gap_audit_blocking_action": ((payloads.get("left_gap_audit") or {}).get("lineage_blocker", {}) or {}).get("recommended_action") if isinstance(payloads.get("left_gap_audit"), dict) else None,
                "gap_audit_blocking_error_code": ((payloads.get("left_gap_audit") or {}).get("lineage_blocker", {}) or {}).get("error_code") if isinstance(payloads.get("left_gap_audit"), dict) else None,
                "recovery_plan_step_count": ((payloads.get("left_recovery_plan") or {}).get("summary", {}) or {}).get("step_count") if isinstance(payloads.get("left_recovery_plan"), dict) else None,
            },
        }
        write_json(output_path, payload)
        print(f"[mixed-universe-status-board] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if overall_status == "completed" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-status-board] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-status-board] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-status-board] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())