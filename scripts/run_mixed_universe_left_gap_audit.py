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


def _artifact_exists(path_text: object) -> bool:
    if not isinstance(path_text, str) or not path_text.strip():
        return False
    return _resolve_path(path_text).exists()


def _lineage_blocker(lineage_payload: dict[str, object]) -> dict[str, object] | None:
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
        "status": status,
        "error_code": candidate.get("error_code"),
        "error_message": candidate.get("error_message"),
        "recommended_action": candidate.get("recommended_action"),
        "artifact_path": artifacts.get("preflight_manifest") if isinstance(artifacts, dict) else None,
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
    row_results = [row for row in (numeric_compare_payload.get("row_results") or []) if isinstance(row, dict)]
    lineage_artifacts = lineage_payload.get("artifacts") if isinstance(lineage_payload.get("artifacts"), dict) else {}
    lineage_blocker = _lineage_blocker(lineage_payload)

    gap_rows: list[dict[str, object]] = []
    for row in row_results:
        if str(row.get("comparison_status") or "") != "missing_left_value":
            continue
        row_name = str(row.get("name") or "")
        required_sources = _required_sources_for_row(row_name)
        source_checks = []
        for source in required_sources:
            artifact_path = lineage_artifacts.get(source["artifact_key"]) if isinstance(lineage_artifacts, dict) else None
            source_checks.append(
                {
                    "artifact_key": source["artifact_key"],
                    "label": source["label"],
                    "artifact_path": artifact_path,
                    "exists": _artifact_exists(artifact_path),
                    "command_preview": ((lineage_payload.get(source["command_key"]) or {}).get("command") if isinstance(lineage_payload.get(source["command_key"]), dict) else None),
                    "command_status": ((lineage_payload.get(source["command_key"]) or {}).get("status") if isinstance(lineage_payload.get(source["command_key"]), dict) else None),
                    "blocking_action": (lineage_blocker or {}).get("recommended_action"),
                    "blocking_error_code": (lineage_blocker or {}).get("error_code"),
                    "blocking_error_message": (lineage_blocker or {}).get("error_message"),
                    "blocking_artifact_path": (lineage_blocker or {}).get("artifact_path"),
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


def _audit_summary(gap_rows: list[dict[str, object]]) -> dict[str, object]:
    rows_missing_all_sources = []
    rows_with_planned_commands = []
    rows_with_blocking_action = []
    for row in gap_rows:
        required_sources = row.get("required_sources") if isinstance(row.get("required_sources"), list) else []
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
        "row_count": len(gap_rows),
        "rows_missing_all_sources": rows_missing_all_sources,
        "rows_with_planned_commands": rows_with_planned_commands,
        "rows_with_blocking_action": rows_with_blocking_action,
        "notes": notes,
    }


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

        if args.dry_run and not compare_path.exists() and not lineage_path.exists():
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": "planned",
                "audit_kind": "mixed_universe_left_gap_audit",
                "revision": revision_slug,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "recommended_action": "generate_numeric_compare_and_lineage_manifests",
                "artifacts": {
                    "gap_audit_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "numeric_compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                    "left_lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
                },
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-left-gap-audit] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        numeric_compare_payload = _read_required_payload(compare_path, label="numeric compare manifest")
        lineage_payload = _read_required_payload(lineage_path, label="left lineage manifest")
        gap_rows = _build_gap_rows(numeric_compare_payload=numeric_compare_payload, lineage_payload=lineage_payload)
        audit_summary = _audit_summary(gap_rows)
        status = "completed" if not gap_rows else "partial"

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": status,
            "audit_kind": "mixed_universe_left_gap_audit",
            "revision": revision_slug,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "recommended_action": numeric_compare_payload.get("recommended_action"),
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
            "lineage_blocker": _lineage_blocker(lineage_payload),
            "gap_rows": gap_rows,
        }
        write_json(output_path, payload)
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