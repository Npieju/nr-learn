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


def _extract_evaluation_pointer(left_payload: dict[str, object]) -> dict[str, object] | None:
    artifacts = left_payload.get("artifacts")
    if isinstance(artifacts, dict):
        pointer_payload = _read_optional_payload(artifacts.get("evaluation_pointer"))
        if isinstance(pointer_payload, dict):
            return pointer_payload
    pointer_payload = left_payload.get("evaluation_pointer_payload")
    return pointer_payload if isinstance(pointer_payload, dict) else None


def _extract_readiness(left_payload: dict[str, object]) -> dict[str, object]:
    readiness = left_payload.get("readiness")
    if isinstance(readiness, dict):
        return dict(readiness)
    benchmark_payload = left_payload.get("benchmark_gate_payload")
    if isinstance(benchmark_payload, dict):
        readiness = benchmark_payload.get("readiness")
        if isinstance(readiness, dict):
            return dict(readiness)
    return {}


def _extract_promotion_decision(left_payload: dict[str, object]) -> str | None:
    promotion_summary = left_payload.get("promotion_summary")
    if isinstance(promotion_summary, dict) and promotion_summary.get("decision") is not None:
        return str(promotion_summary.get("decision"))
    promotion_payload = left_payload.get("promotion_payload")
    if isinstance(promotion_payload, dict) and promotion_payload.get("decision") is not None:
        return str(promotion_payload.get("decision"))
    revision_payload = left_payload.get("revision_manifest_payload")
    if isinstance(revision_payload, dict) and revision_payload.get("decision") is not None:
        return str(revision_payload.get("decision"))
    return None


def _extract_stability_assessment(pointer_payload: dict[str, object] | None) -> str | None:
    if not isinstance(pointer_payload, dict):
        return None
    manifest_payload = pointer_payload.get("latest_manifest_payload")
    if isinstance(manifest_payload, dict) and manifest_payload.get("stability_assessment") is not None:
        return str(manifest_payload.get("stability_assessment"))
    return None


def _extract_left_payload(left_snapshot_path: Path, left_lineage_path: Path) -> tuple[dict[str, object], str]:
    if left_snapshot_path.exists():
        return _read_required_payload(left_snapshot_path, label="left public snapshot"), "local_public_snapshot"
    if left_lineage_path.exists():
        return _read_required_payload(left_lineage_path, label="left lineage manifest"), "local_revision_gate"
    raise FileNotFoundError(
        "left input not found: provide --left-public-snapshot or --left-lineage-manifest, "
        f"or generate {artifact_display_path(left_snapshot_path, workspace_root=ROOT)} first"
    )


def _evaluate_requirements(
    *,
    left_payload: dict[str, object],
    left_source_kind: str,
    left_universe: str,
    right_public_doc_path: Path,
    right_reference: str,
) -> tuple[list[dict[str, object]], str, str, str]:
    pointer_payload = _extract_evaluation_pointer(left_payload)
    readiness = _extract_readiness(left_payload)
    stability_assessment = _extract_stability_assessment(pointer_payload)
    promotion_decision = _extract_promotion_decision(left_payload)

    checks: list[dict[str, object]] = []
    checks.append(
        {
            "name": "left_input_exists",
            "status": "passed",
            "details": left_source_kind,
        }
    )
    checks.append(
        {
            "name": "left_universe_matches",
            "status": "passed" if str(left_payload.get("universe") or left_universe) == left_universe else "failed",
            "details": str(left_payload.get("universe") or ""),
        }
    )
    checks.append(
        {
            "name": "left_readiness_ready",
            "status": "passed" if bool(readiness.get("benchmark_rerun_ready")) else "failed",
            "details": readiness,
        }
    )
    checks.append(
        {
            "name": "left_evaluation_pointer_present",
            "status": "passed" if isinstance(pointer_payload, dict) else "failed",
            "details": "evaluation_pointer" if isinstance(pointer_payload, dict) else None,
        }
    )
    checks.append(
        {
            "name": "left_representative_evaluation",
            "status": "passed" if stability_assessment == "representative" else "failed",
            "details": stability_assessment,
        }
    )
    checks.append(
        {
            "name": "right_public_reference_present",
            "status": "passed" if right_public_doc_path.exists() and bool(str(right_reference).strip()) else "failed",
            "details": {
                "public_doc": artifact_display_path(right_public_doc_path, workspace_root=ROOT),
                "reference": right_reference,
            },
        }
    )

    failed_checks = [check["name"] for check in checks if check["status"] != "passed"]
    if not failed_checks:
        return checks, "ready", "mixed_compare_ready", "run_mixed_universe_compare"
    if "left_readiness_ready" in failed_checks:
        return checks, "not_ready", "left_readiness_blocked", "complete_local_readiness"
    if "left_representative_evaluation" in failed_checks:
        return checks, "not_ready", "left_evaluation_not_representative", "rerun_local_evaluation_representative"
    if "left_evaluation_pointer_present" in failed_checks:
        return checks, "not_ready", "left_evaluation_missing", "generate_local_evaluation_pointer"
    return checks, "not_ready", "mixed_compare_prerequisites_missing", "inspect_mixed_compare_inputs"


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
    default_snapshot = f"artifacts/reports/local_public_snapshot_{revision_slug}.json"
    default_lineage = f"artifacts/reports/local_revision_gate_{revision_slug}.json"
    output = args.output or f"artifacts/reports/mixed_universe_readiness_{left_universe}_vs_{right_universe}_{revision_slug}.json"

    left_snapshot_path = _resolve_path(args.left_public_snapshot or default_snapshot)
    left_lineage_path = _resolve_path(args.left_lineage_manifest or default_lineage)
    right_public_doc_path = _resolve_path(args.right_public_doc)
    output_path = _resolve_path(output)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

        payload: dict[str, object] = {
            "started_at": utc_now_iso(),
            "finished_at": None,
            "status": "running",
            "completed_step": "init",
            "readiness_kind": "mixed_universe_compare_precheck",
            "revision": revision_slug,
            "left_universe": left_universe,
            "right_universe": right_universe,
            "right_reference": args.right_reference,
            "error_code": None,
            "recommended_action": None,
            "artifacts": {
                "readiness_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "left_public_snapshot": artifact_display_path(left_snapshot_path, workspace_root=ROOT),
                "left_lineage_manifest": artifact_display_path(left_lineage_path, workspace_root=ROOT),
                "right_public_doc": artifact_display_path(right_public_doc_path, workspace_root=ROOT),
                "mixed_compare_manifest": f"artifacts/reports/mixed_universe_compare_{left_universe}_vs_{right_universe}_{revision_slug}.json",
            },
            "read_order": [
                "mixed_universe_readiness",
                "mixed_universe_compare",
                "left_public_snapshot_or_lineage",
                "right_public_reference",
            ],
        }

        if args.dry_run and not left_snapshot_path.exists() and not left_lineage_path.exists():
            payload.update(
                {
                    "status": "planned",
                    "completed_step": "planned",
                    "finished_at": utc_now_iso(),
                    "error_code": "left_input_missing",
                    "recommended_action": "generate_local_public_snapshot_or_lineage",
                }
            )
            write_json(output_path, payload)
            print(f"[mixed-universe-readiness] planned manifest saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
            return 0

        payload["completed_step"] = "inspect_left_inputs"
        left_payload, left_source_kind = _extract_left_payload(left_snapshot_path, left_lineage_path)
        payload["left_source_kind"] = left_source_kind

        payload["completed_step"] = "evaluate_requirements"
        checks, status, error_code, recommended_action = _evaluate_requirements(
            left_payload=left_payload,
            left_source_kind=left_source_kind,
            left_universe=left_universe,
            right_public_doc_path=right_public_doc_path,
            right_reference=args.right_reference,
        )
        pointer_payload = _extract_evaluation_pointer(left_payload)
        payload["checks"] = checks
        payload["left_summary"] = {
            "status": left_payload.get("status"),
            "revision": left_payload.get("revision"),
            "universe": left_payload.get("universe"),
            "readiness": _extract_readiness(left_payload),
            "promotion_decision": _extract_promotion_decision(left_payload),
            "evaluation_stability_assessment": _extract_stability_assessment(pointer_payload),
        }
        payload["compare_command_preview"] = [
            sys.executable,
            str(ROOT / "scripts/run_mixed_universe_compare.py"),
            "--revision",
            revision_slug,
            "--left-universe",
            left_universe,
            "--right-universe",
            right_universe,
            "--right-reference",
            args.right_reference,
        ]
        if left_source_kind == "local_public_snapshot":
            payload["compare_command_preview"].extend(["--left-public-snapshot", artifact_display_path(left_snapshot_path, workspace_root=ROOT)])
        else:
            payload["compare_command_preview"].extend(["--left-lineage-manifest", artifact_display_path(left_lineage_path, workspace_root=ROOT)])

        payload["status"] = status
        payload["error_code"] = None if status == "ready" else error_code
        payload["recommended_action"] = recommended_action
        payload["completed_step"] = "completed"
        payload["finished_at"] = utc_now_iso()
        write_json(output_path, payload)
        print(f"[mixed-universe-readiness] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0 if status == "ready" else 2
    except KeyboardInterrupt:
        print("[mixed-universe-readiness] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-readiness] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-readiness] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())