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


def _row_names_with_status(rows: list[dict[str, object]], status: str) -> list[str]:
    return [str(row.get("name")) for row in rows if str(row.get("comparison_status") or "") == status and row.get("name") is not None]


def _numeric_row_summary(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_rows = [row for row in rows if str(row.get("comparison_status") or "") == "numeric_compared"]
    return {
        "row_count": len(numeric_rows),
        "positive_rows": [str(row.get("name")) for row in numeric_rows if row.get("delta_direction") == "positive" and row.get("name") is not None],
        "negative_rows": [str(row.get("name")) for row in numeric_rows if row.get("delta_direction") == "negative" and row.get("name") is not None],
        "zero_rows": [str(row.get("name")) for row in numeric_rows if row.get("delta_direction") == "zero" and row.get("name") is not None],
    }


def _build_notes(
    *,
    verdict: str,
    row_count: int,
    missing_left_rows: list[str],
    missing_right_rows: list[str],
    categorical_different_rows: list[str],
    numeric_summary: dict[str, object],
) -> tuple[str, list[str]]:
    notes: list[str] = []
    missing_total = len(missing_left_rows) + len(missing_right_rows)
    missing_ratio = (missing_total / row_count) if row_count else 0.0

    severity = "info"
    if verdict == "evidence_incomplete":
        severity = "severe" if missing_ratio >= 0.5 else "moderate"
        notes.append("numeric compare is incomplete and should not be used as promotion evidence")
    elif verdict == "categorical_difference_present":
        severity = "moderate"
        notes.append("categorical rows differ, so numeric deltas alone are insufficient for replacement decisions")
    else:
        severity = "info"
        notes.append("summary is intended for comparison triage and does not replace the underlying artifacts")

    if missing_left_rows:
        notes.append(f"left-side metrics are missing for {len(missing_left_rows)} rows")
    if missing_right_rows:
        notes.append(f"right-side reference is missing for {len(missing_right_rows)} rows")
    if categorical_different_rows:
        notes.append(f"categorical differences detected in {len(categorical_different_rows)} rows")

    negative_rows = numeric_summary.get("negative_rows") if isinstance(numeric_summary.get("negative_rows"), list) else []
    positive_rows = numeric_summary.get("positive_rows") if isinstance(numeric_summary.get("positive_rows"), list) else []
    if negative_rows:
        notes.append(f"negative numeric deltas detected in {len(negative_rows)} rows")
    if positive_rows:
        notes.append(f"positive numeric deltas detected in {len(positive_rows)} rows")

    return severity, notes


def _promote_safe_summary(compare_payload: dict[str, object]) -> dict[str, object]:
    rows = [row for row in (compare_payload.get("row_results") or []) if isinstance(row, dict)]
    summary = compare_payload.get("summary") if isinstance(compare_payload.get("summary"), dict) else {}
    blocking_context = compare_payload.get("blocking_context") if isinstance(compare_payload.get("blocking_context"), dict) else {}

    categorical_different_rows = _row_names_with_status(rows, "categorical_different")
    missing_left_rows = _row_names_with_status(rows, "missing_left_value")
    missing_right_rows = _row_names_with_status(rows, "missing_right_value")
    numeric_summary = _numeric_row_summary(rows)

    verdict = "review_ready"
    if missing_left_rows or missing_right_rows:
        verdict = "evidence_incomplete"
    elif categorical_different_rows:
        verdict = "categorical_difference_present"

    severity, notes = _build_notes(
        verdict=verdict,
        row_count=len(rows),
        missing_left_rows=missing_left_rows,
        missing_right_rows=missing_right_rows,
        categorical_different_rows=categorical_different_rows,
        numeric_summary=numeric_summary,
    )

    return {
        "verdict": verdict,
        "severity": severity,
        "readiness_status": blocking_context.get("readiness_status"),
        "schema_status": blocking_context.get("schema_status"),
        "numeric_rows": numeric_summary,
        "categorical_different_rows": categorical_different_rows,
        "missing_left_rows": missing_left_rows,
        "missing_right_rows": missing_right_rows,
        "numeric_compared_rows": summary.get("numeric_compared_rows"),
        "categorical_match_rows": summary.get("categorical_match_rows"),
        "categorical_different_rows_count": summary.get("categorical_different_rows"),
        "recommended_action": compare_payload.get("recommended_action"),
        "notes": notes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--numeric-compare-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    numeric_compare_manifest = args.numeric_compare_manifest or f"artifacts/reports/mixed_universe_numeric_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    output = args.output or f"artifacts/reports/mixed_universe_numeric_summary_{left_universe}_vs_{right_universe}_{revision_slug}.json"

    compare_path = _resolve_path(numeric_compare_manifest)
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

        if args.dry_run and not compare_path.exists():
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": "planned",
                "summary_kind": "mixed_universe_numeric_summary",
                "revision": revision_slug,
                "requested_revision": revision_slug,
                "resolved_left_revision": None,
                "resolved_left_source_kind": None,
                "resolved_left_artifact": None,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "recommended_action": "generate_numeric_compare_manifest",
                "artifacts": {
                    "summary_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "numeric_compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                },
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-numeric-summary] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        compare_payload = _read_required_payload(compare_path, label="numeric compare manifest")
        promote_safe_summary = _promote_safe_summary(compare_payload)
        compare_status = str(compare_payload.get("status") or "unknown")
        summary_status = "completed" if compare_status == "completed" else "partial"

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": summary_status,
            "summary_kind": "mixed_universe_numeric_summary",
            "revision": compare_payload.get("revision") or revision_slug,
            "requested_revision": compare_payload.get("requested_revision") or compare_payload.get("revision") or revision_slug,
            "resolved_left_revision": compare_payload.get("resolved_left_revision"),
            "resolved_left_source_kind": compare_payload.get("resolved_left_source_kind"),
            "resolved_left_artifact": compare_payload.get("resolved_left_artifact"),
            "left_universe": compare_payload.get("left_universe") or left_universe,
            "right_universe": compare_payload.get("right_universe") or right_universe,
            "right_reference": compare_payload.get("right_reference"),
            "recommended_action": compare_payload.get("recommended_action"),
            "artifacts": {
                "summary_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "numeric_compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                "numeric_compare_csv": ((compare_payload.get("artifacts") or {}).get("numeric_compare_csv") if isinstance(compare_payload.get("artifacts"), dict) else None),
            },
            "read_order": [
                "mixed_universe_numeric_compare",
                "mixed_universe_numeric_summary",
            ],
            "promote_safe_summary": promote_safe_summary,
        }
        write_json(output_path, payload)
        print(f"[mixed-universe-numeric-summary] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if summary_status == "completed" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-numeric-summary] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-numeric-summary] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-numeric-summary] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())