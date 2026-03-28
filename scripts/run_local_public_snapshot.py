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


DEFAULT_UNIVERSE = "local_nankan"
DEFAULT_SOURCE_SCOPE = "local_only"
DEFAULT_BASELINE_REFERENCE = "current_recommended_serving_2025_latest"


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


def _build_planned_payload(
    *,
    revision_slug: str,
    universe: str,
    source_scope: str,
    baseline_reference: str,
    lineage_path: Path,
    output_path: Path,
) -> dict[str, object]:
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
        "read_order": [
            "local_public_snapshot",
            "local_revision_gate",
            "promotion_gate",
            "revision_gate",
            "evaluation_pointer",
        ],
        "artifacts": {
            "public_snapshot": artifact_display_path(output_path, workspace_root=ROOT),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
        },
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

        if args.dry_run and not lineage_path.exists():
            payload = _build_planned_payload(
                revision_slug=revision_slug,
                universe=args.universe,
                source_scope=args.source_scope,
                baseline_reference=args.baseline_reference,
                lineage_path=lineage_path,
                output_path=output_path,
            )
            write_json(output_path, payload)
            print(f"[local-public-snapshot] planned snapshot saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        lineage_payload = _read_required_payload(lineage_path, label="local revision lineage")
        artifacts = dict(lineage_payload.get("artifacts", {})) if isinstance(lineage_payload.get("artifacts"), dict) else {}
        benchmark_payload = lineage_payload.get("benchmark_gate_payload") if isinstance(lineage_payload.get("benchmark_gate_payload"), dict) else None
        evaluation_pointer_payload = lineage_payload.get("evaluation_pointer_payload")
        if not isinstance(evaluation_pointer_payload, dict):
            evaluation_pointer_payload = _read_optional_payload(artifacts.get("evaluation_pointer"))
        resolved_revision = str(lineage_payload.get("revision") or revision_slug)
        resolved_universe = str(lineage_payload.get("universe") or args.universe)
        resolved_source_scope = str(lineage_payload.get("source_scope") or args.source_scope)
        resolved_baseline_reference = str(lineage_payload.get("baseline_reference") or args.baseline_reference)

        payload: dict[str, object] = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": "completed",
            "snapshot_kind": "local_public_snapshot",
            "revision": resolved_revision,
            "universe": resolved_universe,
            "source_scope": resolved_source_scope,
            "baseline_reference": resolved_baseline_reference,
            "lineage_status": str(lineage_payload.get("status") or "unknown"),
            "lineage_completed_step": str(lineage_payload.get("completed_step") or "unknown"),
            "lineage_manifest": artifact_display_path(lineage_path, workspace_root=ROOT),
            "read_order": [
                "local_public_snapshot",
                "local_revision_gate",
                "promotion_gate",
                "revision_gate",
                "evaluation_pointer",
            ],
            "artifacts": {
                "public_snapshot": artifact_display_path(output_path, workspace_root=ROOT),
                **artifacts,
            },
            "readiness": _extract_readiness(lineage_payload),
            "promotion_summary": _extract_promotion_summary(lineage_payload),
            "compare_contract": _build_compare_contract(
                snapshot_path=output_path,
                universe=resolved_universe,
                revision=resolved_revision,
            ),
        }
        if benchmark_payload is not None:
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

        write_json(output_path, payload)
        print(f"[local-public-snapshot] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[local-public-snapshot] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[local-public-snapshot] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[local-public-snapshot] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())