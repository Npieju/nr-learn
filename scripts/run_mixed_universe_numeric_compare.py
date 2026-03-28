from __future__ import annotations

import argparse
import csv
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
DEFAULT_RIGHT_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_RIGHT_REFERENCE_MANIFEST = "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json"


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


def _latest_matching(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _prefer_existing(path: Path, fallback_pattern: str) -> Path:
    if path.exists():
        return path
    fallback = _latest_matching(fallback_pattern)
    return fallback if fallback is not None else path


def _read_required_payload(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {artifact_display_path(path, workspace_root=ROOT)}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {artifact_display_path(path, workspace_root=ROOT)}")
    return payload


def _maybe_read_json(path_text: object) -> dict[str, object] | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _dig(payload: dict[str, object] | None, path_parts: list[str] | tuple[str, ...]) -> object:
    current: object = payload
    for part in path_parts:
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _extract_left_sources(compare_payload: dict[str, object]) -> dict[str, dict[str, object] | None]:
    left_summary = compare_payload.get("left_summary")
    artifacts = left_summary.get("artifacts") if isinstance(left_summary, dict) else None
    promotion_payload = _maybe_read_json(artifacts.get("promotion_output")) if isinstance(artifacts, dict) else None
    revision_payload = _maybe_read_json(artifacts.get("revision_manifest")) if isinstance(artifacts, dict) else None
    evaluation_pointer_payload = _maybe_read_json(artifacts.get("evaluation_pointer")) if isinstance(artifacts, dict) else None

    evaluation_manifest_payload = None
    if isinstance(evaluation_pointer_payload, dict):
        manifest_payload = evaluation_pointer_payload.get("latest_manifest_payload")
        if isinstance(manifest_payload, dict):
            evaluation_manifest_payload = manifest_payload

    return {
        "promotion_payload": promotion_payload,
        "revision_manifest_payload": revision_payload,
        "evaluation_manifest_payload": evaluation_manifest_payload,
        "evaluation_pointer_payload": evaluation_pointer_payload,
    }


def _extract_right_value(reference_payload: dict[str, object], row_name: str) -> tuple[object, str | None]:
    metrics = reference_payload.get("metrics")
    if isinstance(metrics, dict) and row_name in metrics:
        return metrics.get(row_name), f"metrics.{row_name}"

    promotion_summary = reference_payload.get("promotion_summary")
    if isinstance(promotion_summary, dict) and row_name in promotion_summary:
        return promotion_summary.get(row_name), f"promotion_summary.{row_name}"

    evaluation_summary = reference_payload.get("evaluation_summary")
    if isinstance(evaluation_summary, dict) and row_name in evaluation_summary:
        return evaluation_summary.get(row_name), f"evaluation_summary.{row_name}"

    revision_summary = reference_payload.get("revision_summary")
    if isinstance(revision_summary, dict) and row_name in revision_summary:
        return revision_summary.get(row_name), f"revision_summary.{row_name}"

    return None, None


def _to_number(value: object) -> float | int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _delta_direction(value: object) -> str:
    number = _to_number(value)
    if number is None:
        return "unknown"
    if number > 0:
        return "positive"
    if number < 0:
        return "negative"
    return "zero"


def _write_compare_csv(path: Path, rows: list[dict[str, object]]) -> None:
    output_path = artifact_ensure_output_file_path(path, label="csv output", workspace_root=ROOT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "name",
        "category",
        "comparison_role",
        "left_value",
        "right_value",
        "left_path",
        "right_path",
        "delta_kind",
        "delta_left_minus_right",
        "delta_direction",
        "comparison_status",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _build_row_result(
    row: dict[str, object],
    left_sources: dict[str, dict[str, object] | None],
    right_reference_payload: dict[str, object],
) -> dict[str, object]:
    left_path = row.get("left_path")
    left_value = row.get("left_value_preview")
    if isinstance(left_path, str) and left_path.strip():
        path_parts = left_path.split(".")
        left_value = _dig(left_sources.get(path_parts[0]), path_parts[1:])

    right_value, right_path = _extract_right_value(right_reference_payload, str(row.get("name") or ""))
    left_number = _to_number(left_value)
    right_number = _to_number(right_value)

    row_result: dict[str, object] = {
        "name": row.get("name"),
        "category": row.get("category"),
        "comparison_role": row.get("comparison_role"),
        "left_value": left_value,
        "right_value": right_value,
        "left_path": left_path,
        "right_path": right_path,
        "delta_left_minus_right": None,
        "delta_kind": None,
        "delta_direction": "unknown",
        "comparison_status": None,
    }

    if left_value is None:
        row_result["comparison_status"] = "missing_left_value"
        return row_result
    if right_value is None:
        row_result["comparison_status"] = "missing_right_value"
        return row_result

    if left_number is not None and right_number is not None:
        row_result["delta_left_minus_right"] = left_number - right_number
        row_result["delta_kind"] = "numeric"
        row_result["delta_direction"] = _delta_direction(row_result["delta_left_minus_right"])
        row_result["comparison_status"] = "numeric_compared"
        return row_result

    row_result["delta_kind"] = "categorical"
    row_result["comparison_status"] = "categorical_match" if left_value == right_value else "categorical_different"
    return row_result


def _summarize_rows(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_rows = [row for row in rows if row.get("comparison_status") == "numeric_compared"]
    categorical_matches = [row for row in rows if row.get("comparison_status") == "categorical_match"]
    categorical_differences = [row for row in rows if row.get("comparison_status") == "categorical_different"]
    missing_left_rows = [row.get("name") for row in rows if row.get("comparison_status") == "missing_left_value"]
    missing_right_rows = [row.get("name") for row in rows if row.get("comparison_status") == "missing_right_value"]

    return {
        "row_count": len(rows),
        "numeric_compared_rows": len(numeric_rows),
        "categorical_match_rows": len(categorical_matches),
        "categorical_different_rows": len(categorical_differences),
        "missing_left_rows": missing_left_rows,
        "missing_right_rows": missing_right_rows,
        "rows_with_positive_delta": [row.get("name") for row in numeric_rows if isinstance(row.get("delta_left_minus_right"), (int, float)) and row.get("delta_left_minus_right") > 0],
        "rows_with_negative_delta": [row.get("name") for row in numeric_rows if isinstance(row.get("delta_left_minus_right"), (int, float)) and row.get("delta_left_minus_right") < 0],
        "rows_with_zero_delta": [row.get("name") for row in numeric_rows if row.get("delta_left_minus_right") == 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--right-reference", default=DEFAULT_RIGHT_REFERENCE)
    parser.add_argument("--readiness-manifest", default=None)
    parser.add_argument("--compare-manifest", default=None)
    parser.add_argument("--schema-manifest", default=None)
    parser.add_argument("--right-reference-manifest", default=DEFAULT_RIGHT_REFERENCE_MANIFEST)
    parser.add_argument("--output", default=None)
    parser.add_argument("--csv-output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    readiness_manifest = args.readiness_manifest or f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    compare_manifest = args.compare_manifest or f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    schema_manifest = args.schema_manifest or f"artifacts/reports/mixed_universe_schema_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    output = args.output or f"artifacts/reports/mixed_universe_numeric_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    csv_output = args.csv_output or f"artifacts/reports/mixed_universe_numeric_compare_{left_universe}_vs_{right_universe}_{revision_slug}.csv"

    readiness_path = _prefer_existing(
        _resolve_path(readiness_manifest),
        f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_*.json",
    )
    compare_path = _prefer_existing(
        _resolve_path(compare_manifest),
        f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_*.json",
    )
    schema_path = _prefer_existing(
        _resolve_path(schema_manifest),
        f"artifacts/reports/mixed_universe_schema_{left_universe}_vs_{right_universe}_*.json",
    )
    right_reference_manifest_path = _resolve_path(args.right_reference_manifest)
    output_path = _resolve_path(output)
    csv_output_path = _resolve_path(csv_output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        artifact_ensure_output_file_path(csv_output_path, label="csv output", workspace_root=ROOT)

        if args.dry_run and not readiness_path.exists() and not compare_path.exists() and not schema_path.exists():
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": "planned",
                "compare_kind": "mixed_universe_numeric_compare",
                "compare_mode": "delta_rows",
                "revision": revision_slug,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "right_reference": args.right_reference,
                "recommended_action": "generate_readiness_compare_schema_manifests",
                "artifacts": {
                    "numeric_compare_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "numeric_compare_csv": artifact_display_path(csv_output_path, workspace_root=ROOT),
                    "readiness_manifest": artifact_display_path(readiness_path, workspace_root=ROOT),
                    "compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                    "schema_manifest": artifact_display_path(schema_path, workspace_root=ROOT),
                    "right_reference_manifest": artifact_display_path(right_reference_manifest_path, workspace_root=ROOT),
                },
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-numeric-compare] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        readiness_payload = _read_required_payload(readiness_path, label="mixed readiness manifest")
        compare_payload = _read_required_payload(compare_path, label="mixed compare manifest")
        schema_payload = _read_required_payload(schema_path, label="mixed schema manifest")
        right_reference_payload = _read_required_payload(right_reference_manifest_path, label="right reference manifest")

        left_sources = _extract_left_sources(compare_payload)
        metric_rows = schema_payload.get("metric_rows") if isinstance(schema_payload.get("metric_rows"), list) else []
        row_results = [
            _build_row_result(row, left_sources, right_reference_payload)
            for row in metric_rows
            if isinstance(row, dict)
        ]

        readiness_status = str(readiness_payload.get("status") or "unknown")
        schema_status = str(schema_payload.get("status") or "unknown")
        summary = _summarize_rows(row_results)
        missing_left_rows = summary.get("missing_left_rows") if isinstance(summary.get("missing_left_rows"), list) else []
        missing_right_rows = summary.get("missing_right_rows") if isinstance(summary.get("missing_right_rows"), list) else []
        compare_status = "completed"
        if missing_left_rows or missing_right_rows:
            compare_status = "partial"
        recommended_action = "review_delta_rows"
        if missing_left_rows:
            recommended_action = str(readiness_payload.get("recommended_action") or "complete_left_metrics")
        elif missing_right_rows:
            recommended_action = "complete_right_reference_metrics"

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": compare_status,
            "compare_kind": "mixed_universe_numeric_compare",
            "compare_mode": "delta_rows",
            "revision": revision_slug,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "right_reference": args.right_reference,
            "recommended_action": recommended_action,
            "artifacts": {
                "numeric_compare_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "numeric_compare_csv": artifact_display_path(csv_output_path, workspace_root=ROOT),
                "readiness_manifest": artifact_display_path(readiness_path, workspace_root=ROOT),
                "compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                "schema_manifest": artifact_display_path(schema_path, workspace_root=ROOT),
                "right_reference_manifest": artifact_display_path(right_reference_manifest_path, workspace_root=ROOT),
            },
            "read_order": [
                "mixed_universe_readiness",
                "mixed_universe_compare",
                "mixed_universe_schema",
                "mixed_universe_numeric_compare",
            ],
            "blocking_context": {
                "readiness_status": readiness_status,
                "schema_status": schema_status,
                "readiness_error_code": readiness_payload.get("error_code"),
                "schema_recommended_action": schema_payload.get("recommended_action"),
            },
            "summary": summary,
            "row_results": row_results,
        }
        _write_compare_csv(csv_output_path, row_results)
        write_json(output_path, payload)
        print(f"[mixed-universe-numeric-compare] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if compare_status == "completed" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-numeric-compare] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-numeric-compare] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-numeric-compare] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())