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


DEFAULT_LEFT_UNIVERSE = "local_nankan"
DEFAULT_RIGHT_UNIVERSE = "jra"
DEFAULT_RIGHT_REFERENCE = "current_recommended_serving_2025_latest"


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


def _maybe_read_json(path_text: object) -> dict[str, object] | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = _resolve_path(path_text)
    if not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _dig(payload: dict[str, object] | None, path: tuple[str, ...]) -> object:
    current: object = payload
    for key in path:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _extract_left_payloads(compare_payload: dict[str, object]) -> tuple[dict[str, object] | None, dict[str, object] | None, dict[str, object] | None]:
    left_summary = compare_payload.get("left_summary")
    artifacts = left_summary.get("artifacts") if isinstance(left_summary, dict) else None
    if not isinstance(artifacts, dict):
        return None, None, None
    promotion_payload = _maybe_read_json(artifacts.get("promotion_output"))
    revision_payload = _maybe_read_json(artifacts.get("revision_manifest"))
    evaluation_pointer_payload = _maybe_read_json(artifacts.get("evaluation_pointer"))
    return promotion_payload, revision_payload, evaluation_pointer_payload


def _left_metric_candidates() -> list[dict[str, object]]:
    return [
        {
            "name": "decision",
            "category": "promotion",
            "left_paths": [("promotion_payload", "decision"), ("revision_manifest_payload", "decision")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "stability_assessment",
            "category": "evaluation",
            "left_paths": [("evaluation_manifest_payload", "stability_assessment")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "auc",
            "category": "evaluation",
            "left_paths": [("evaluation_manifest_payload", "metrics", "auc")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "top1_roi",
            "category": "evaluation",
            "left_paths": [("evaluation_manifest_payload", "metrics", "top1_roi")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "ev_top1_roi",
            "category": "evaluation",
            "left_paths": [("evaluation_manifest_payload", "metrics", "ev_top1_roi")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "nested_wf_weighted_test_roi",
            "category": "evaluation",
            "left_paths": [("promotion_payload", "weighted_test_roi"), ("revision_manifest_payload", "weighted_test_roi")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "nested_wf_bets_total",
            "category": "evaluation",
            "left_paths": [("evaluation_manifest_payload", "metrics", "bets_total")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "formal_benchmark_weighted_roi",
            "category": "support",
            "left_paths": [("promotion_payload", "weighted_roi")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
        {
            "name": "formal_benchmark_feasible_folds",
            "category": "support",
            "left_paths": [("promotion_payload", "feasible_folds"), ("promotion_payload", "feasible_fold_count")],
            "right_source": "artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json",
        },
    ]


def _build_metric_rows(
    promotion_payload: dict[str, object] | None,
    revision_payload: dict[str, object] | None,
    evaluation_pointer_payload: dict[str, object] | None,
) -> list[dict[str, object]]:
    evaluation_manifest_payload = None
    if isinstance(evaluation_pointer_payload, dict):
        manifest_payload = evaluation_pointer_payload.get("latest_manifest_payload")
        if isinstance(manifest_payload, dict):
            evaluation_manifest_payload = manifest_payload

    sources = {
        "promotion_payload": promotion_payload,
        "revision_manifest_payload": revision_payload,
        "evaluation_manifest_payload": evaluation_manifest_payload,
    }
    rows: list[dict[str, object]] = []
    for definition in _left_metric_candidates():
        left_value = None
        left_path_used = None
        for candidate_path in definition["left_paths"]:
            source_name = candidate_path[0]
            value = _dig(sources.get(source_name), candidate_path[1:])
            if value is not None:
                left_value = value
                left_path_used = ".".join(candidate_path)
                break
        rows.append(
            {
                "name": definition["name"],
                "category": definition["category"],
                "left_value_preview": left_value,
                "left_path": left_path_used,
                "right_source": definition["right_source"],
                "comparison_role": "compare_if_available",
            }
        )
    return rows


def _comparison_axes() -> list[dict[str, str]]:
    return [
        {"name": "promotion", "description": "decision and promote eligibility are read separately from baseline replacement"},
        {"name": "evaluation", "description": "AUC, top1 ROI, EV top1 ROI, nested WF weighted test ROI, nested bets total"},
        {"name": "support", "description": "formal benchmark weighted ROI and feasible folds"},
    ]


def _requested_revision(*, readiness_payload: dict[str, object], compare_payload: dict[str, object], fallback: str) -> str:
    value = compare_payload.get("requested_revision")
    if value is None:
        value = readiness_payload.get("requested_revision")
    if value is None:
        value = compare_payload.get("revision")
    if value is None:
        value = readiness_payload.get("revision")
    return str(value or fallback)


def _resolved_left_revision(*, readiness_payload: dict[str, object], compare_payload: dict[str, object]) -> str | None:
    value = compare_payload.get("resolved_left_revision")
    if value is None:
        value = readiness_payload.get("resolved_left_revision")
    return str(value) if value is not None else None


def _resolved_left_source_kind(*, readiness_payload: dict[str, object], compare_payload: dict[str, object]) -> str | None:
    value = compare_payload.get("resolved_left_source_kind")
    if value is None:
        value = readiness_payload.get("resolved_left_source_kind")
    return str(value) if value is not None else None


def _resolved_left_artifact(*, readiness_payload: dict[str, object], compare_payload: dict[str, object]) -> str | None:
    value = compare_payload.get("resolved_left_artifact")
    if value is None:
        value = readiness_payload.get("resolved_left_artifact")
    return str(value) if value is not None else None


def _current_phase(*, status: str, readiness_status: str | None) -> str:
    if status == "planned":
        return "missing_readiness_compare"
    if readiness_status == "ready":
        return "future_numeric_compare"
    return "mixed_universe_readiness"


def _schema_highlights(
    *,
    status: str,
    readiness_status: str | None,
    recommended_action: str | None,
    blocking_context: dict[str, object],
) -> list[str]:
    highlights: list[str] = []
    if status == "planned":
        highlights.append("schema cannot be fixed until readiness and compare manifests exist")
        highlights.append("comparison axes and metric rows are already scaffolded so downstream numeric compare has a stable contract")
    elif readiness_status == "ready":
        highlights.append("schema is ready and has fixed the comparison axes for downstream numeric compare")
        highlights.append("metric rows now define where left-side values and right-side public reference values will be read")
    else:
        readiness_error_code = blocking_context.get("readiness_error_code")
        highlights.append("schema is blocked until mixed readiness becomes ready")
        if readiness_error_code:
            highlights.append(f"upstream readiness is currently failing with error_code={readiness_error_code}")
    if recommended_action:
        highlights.append(f"next operator action: {recommended_action}")
    return highlights


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--right-reference", default=DEFAULT_RIGHT_REFERENCE)
    parser.add_argument("--readiness-manifest", default=None)
    parser.add_argument("--compare-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    readiness_manifest = args.readiness_manifest or f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    compare_manifest = args.compare_manifest or f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    output = args.output or f"artifacts/reports/mixed_universe_schema_{left_universe}_vs_{right_universe}_{revision_slug}.json"

    readiness_path = prefer_existing_path(
        workspace_root=ROOT,
        expected_path=readiness_manifest,
        fallback_pattern=f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_*.json",
    )
    compare_path = prefer_existing_path(
        workspace_root=ROOT,
        expected_path=compare_manifest,
        fallback_pattern=f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_*.json",
    )
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

        if args.dry_run and not readiness_path.exists() and not compare_path.exists():
            blocking_context = {
                "readiness_status": "missing",
                "readiness_error_code": "readiness_manifest_missing",
                "readiness_checks": None,
            }
            payload = {
                "started_at": utc_now_iso(),
                "finished_at": utc_now_iso(),
                "status": "planned",
                "schema_kind": "mixed_universe_comparison_schema",
                "revision": revision_slug,
                "requested_revision": revision_slug,
                "resolved_left_revision": None,
                "resolved_left_source_kind": None,
                "resolved_left_artifact": None,
                "left_universe": left_universe,
                "right_universe": right_universe,
                "right_reference": args.right_reference,
                "current_phase": _current_phase(status="planned", readiness_status="missing"),
                "recommended_action": "generate_readiness_and_compare_manifests",
                "artifacts": {
                    "schema_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                    "readiness_manifest": artifact_display_path(readiness_path, workspace_root=ROOT),
                    "compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
                },
                "read_order": [
                    "mixed_universe_readiness",
                    "mixed_universe_compare",
                    "mixed_universe_schema",
                    "future_numeric_compare",
                ],
                "comparison_axes": _comparison_axes(),
                "metric_rows": _build_metric_rows(None, None, None),
                "blocking_context": blocking_context,
                "highlights": _schema_highlights(
                    status="planned",
                    readiness_status="missing",
                    recommended_action="generate_readiness_and_compare_manifests",
                    blocking_context=blocking_context,
                ),
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-schema] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        readiness_payload = _read_required_payload(readiness_path, label="mixed readiness manifest")
        compare_payload = _read_required_payload(compare_path, label="mixed compare manifest")
        promotion_payload, revision_payload, evaluation_pointer_payload = _extract_left_payloads(compare_payload)

        metric_rows = _build_metric_rows(promotion_payload, revision_payload, evaluation_pointer_payload)
        readiness_status = str(readiness_payload.get("status") or "unknown")
        schema_status = "schema_ready" if readiness_status == "ready" else "schema_blocked"
        blocking_context = {
            "readiness_status": readiness_status,
            "readiness_error_code": readiness_payload.get("error_code"),
            "readiness_checks": readiness_payload.get("checks"),
        }
        recommended_action = "populate_numeric_compare" if readiness_status == "ready" else str(
            readiness_payload.get("recommended_action") or "complete_mixed_readiness"
        )

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": schema_status,
            "schema_kind": "mixed_universe_comparison_schema",
            "revision": revision_slug,
            "requested_revision": _requested_revision(readiness_payload=readiness_payload, compare_payload=compare_payload, fallback=revision_slug),
            "resolved_left_revision": _resolved_left_revision(readiness_payload=readiness_payload, compare_payload=compare_payload),
            "resolved_left_source_kind": _resolved_left_source_kind(readiness_payload=readiness_payload, compare_payload=compare_payload),
            "resolved_left_artifact": _resolved_left_artifact(readiness_payload=readiness_payload, compare_payload=compare_payload),
            "left_universe": left_universe,
            "right_universe": right_universe,
            "right_reference": args.right_reference,
            "current_phase": _current_phase(status=schema_status, readiness_status=readiness_status),
            "recommended_action": recommended_action,
            "artifacts": {
                "schema_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "readiness_manifest": artifact_display_path(readiness_path, workspace_root=ROOT),
                "compare_manifest": artifact_display_path(compare_path, workspace_root=ROOT),
            },
            "read_order": [
                "mixed_universe_readiness",
                "mixed_universe_compare",
                "mixed_universe_schema",
                "future_numeric_compare",
            ],
            "comparison_axes": _comparison_axes(),
            "metric_rows": metric_rows,
            "blocking_context": blocking_context,
            "highlights": _schema_highlights(
                status=schema_status,
                readiness_status=readiness_status,
                recommended_action=recommended_action,
                blocking_context=blocking_context,
            ),
        }
        write_json(output_path, payload)
        print(f"[mixed-universe-schema] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if schema_status == "schema_ready" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-schema] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-schema] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-schema] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())