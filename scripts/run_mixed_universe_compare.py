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


def _read_optional_payload(path_text: object) -> dict[str, object] | None:
    if not isinstance(path_text, str) or not path_text.strip():
        return None
    path = _resolve_path(path_text)
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


def _summary_from_local_public(payload: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": payload.get("status"),
        "revision": payload.get("revision"),
        "universe": payload.get("universe"),
        "source_scope": payload.get("source_scope"),
        "readiness": payload.get("readiness"),
        "promotion_summary": payload.get("promotion_summary"),
        "evaluation_summary": payload.get("evaluation_summary"),
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        summary["artifacts"] = artifacts
    return summary


def _summary_from_local_lineage(payload: dict[str, object]) -> dict[str, object]:
    summary: dict[str, object] = {
        "status": payload.get("status"),
        "completed_step": payload.get("completed_step"),
        "revision": payload.get("revision"),
        "universe": payload.get("universe"),
        "source_scope": payload.get("source_scope"),
        "promotion_payload": payload.get("promotion_payload"),
        "evaluation_pointer_payload": payload.get("evaluation_pointer_payload"),
    }
    artifacts = payload.get("artifacts")
    if isinstance(artifacts, dict):
        summary["artifacts"] = artifacts
    return summary


def _build_planned_payload(
    *,
    revision_slug: str,
    left_universe: str,
    right_universe: str,
    right_reference: str,
    output_path: Path,
) -> dict[str, object]:
    return {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "planned",
        "compare_kind": "mixed_universe_compare",
        "compare_mode": "pointer_only",
        "revision": revision_slug,
        "left_universe": left_universe,
        "right_universe": right_universe,
        "right_reference": right_reference,
        "decision": "separate_lineage_required",
        "recommended_action": "populate_left_snapshot_or_lineage",
        "read_order": [
            "mixed_universe_compare",
            "left_public_snapshot_or_lineage",
            "right_public_reference",
        ],
        "artifacts": {
            "compare_manifest": artifact_display_path(output_path, workspace_root=ROOT),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--left-public-snapshot", default=None)
    parser.add_argument("--left-lineage-manifest", default=None)
    parser.add_argument("--right-reference", default=DEFAULT_RIGHT_REFERENCE)
    parser.add_argument("--right-public-doc", default="docs/public_benchmark_snapshot.md")
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_value = args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}"
    revision_slug = _normalize_slug(revision_value)
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    output = args.output or f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    output_path = _resolve_path(output)

    default_left_snapshot = f"artifacts/reports/local_public_snapshot_{revision_slug}.json"
    default_left_lineage = f"artifacts/reports/local_revision_gate_{revision_slug}.json"
    left_snapshot_path = _resolve_path(args.left_public_snapshot or default_left_snapshot)
    left_lineage_path = _resolve_path(args.left_lineage_manifest or default_left_lineage)
    right_public_doc_path = _resolve_path(args.right_public_doc)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

        if args.dry_run and not left_snapshot_path.exists() and not left_lineage_path.exists():
            payload = _build_planned_payload(
                revision_slug=revision_slug,
                left_universe=left_universe,
                right_universe=right_universe,
                right_reference=args.right_reference,
                output_path=output_path,
            )
            payload["left_inputs"] = {
                "public_snapshot": artifact_display_path(left_snapshot_path, workspace_root=ROOT),
                "lineage_manifest": artifact_display_path(left_lineage_path, workspace_root=ROOT),
            }
            payload["right_inputs"] = {
                "public_doc": artifact_display_path(right_public_doc_path, workspace_root=ROOT),
            }
            write_json(output_path, payload)
            print(f"[mixed-universe-compare] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        left_payload: dict[str, object] | None = None
        left_source_kind = None
        if left_snapshot_path.exists():
            left_payload = _read_required_payload(left_snapshot_path, label="left public snapshot")
            left_source_kind = "local_public_snapshot"
        elif left_lineage_path.exists():
            left_payload = _read_required_payload(left_lineage_path, label="left lineage manifest")
            left_source_kind = "local_revision_gate"
        else:
            raise FileNotFoundError(
                "left input not found: provide --left-public-snapshot or --left-lineage-manifest, "
                f"or generate {artifact_display_path(left_snapshot_path, workspace_root=ROOT)} first"
            )

        if not right_public_doc_path.exists():
            raise FileNotFoundError(f"right public doc not found: {artifact_display_path(right_public_doc_path, workspace_root=ROOT)}")

        payload: dict[str, object] = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": "completed",
            "compare_kind": "mixed_universe_compare",
            "compare_mode": "pointer_only",
            "revision": revision_slug,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "decision": "separate_lineage_required",
            "recommended_action": "read_left_then_right_public_reference",
            "read_order": [
                "mixed_universe_compare",
                "left_public_snapshot_or_lineage",
                "right_public_reference",
            ],
            "artifacts": {
                "compare_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "left_public_snapshot": artifact_display_path(left_snapshot_path, workspace_root=ROOT) if left_snapshot_path.exists() else None,
                "left_lineage_manifest": artifact_display_path(left_lineage_path, workspace_root=ROOT) if left_lineage_path.exists() else None,
            },
            "left_summary": _summary_from_local_public(left_payload) if left_source_kind == "local_public_snapshot" else _summary_from_local_lineage(left_payload),
            "right_summary": {
                "universe": right_universe,
                "reference": args.right_reference,
                "public_doc": artifact_display_path(right_public_doc_path, workspace_root=ROOT),
                "reading_role": "jra_latest_public_snapshot",
            },
            "comparison_contract": {
                "left_source_kind": left_source_kind,
                "right_source_kind": "jra_public_reference",
                "notes": [
                    "this manifest is a pointer-only bridge, not a numeric promote gate",
                    "mixed-universe compare does not overwrite jra-only public snapshot",
                ],
            },
        }

        compare_contract = None
        if isinstance(left_payload, dict):
            compare_contract = left_payload.get("compare_contract")
        if isinstance(compare_contract, dict):
            payload["left_compare_contract"] = compare_contract
        write_json(output_path, payload)
        print(f"[mixed-universe-compare] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[mixed-universe-compare] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-compare] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-compare] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())