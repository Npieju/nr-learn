from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
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


def _latest_matching(pattern: str) -> Path | None:
    matches = sorted(ROOT.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


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


def _path_arg(path: Path) -> str:
    return artifact_display_path(path, workspace_root=ROOT)


def _status_board_path(*, revision_slug: str, left_universe: str, right_universe: str, explicit: str | None) -> Path:
    if explicit:
        return _resolve_path(explicit)
    expected = _resolve_path(
        f"artifacts/reports/mixed_universe_status_board_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    )
    if expected.exists():
        return expected
    fallback = _latest_matching(f"artifacts/reports/mixed_universe_status_board_{left_universe}_vs_{right_universe}_*.json")
    return fallback if fallback is not None else expected


def _output_path(*, revision_slug: str, left_universe: str, right_universe: str, explicit: str | None) -> Path:
    if explicit:
        return _resolve_path(explicit)
    return _resolve_path(
        f"artifacts/reports/mixed_universe_recovery_{left_universe}_vs_{right_universe}_{revision_slug}.json"
    )


def _extract_right_reference(
    readiness_payload: dict[str, object] | None,
    compare_payload: dict[str, object] | None,
    numeric_compare_payload: dict[str, object] | None,
) -> tuple[str, Path, Path]:
    right_reference = None
    right_reference_manifest = None
    right_public_doc = None

    if isinstance(compare_payload, dict):
        right_reference = compare_payload.get("right_reference")
        right_summary = compare_payload.get("right_summary") if isinstance(compare_payload.get("right_summary"), dict) else None
        if isinstance(right_summary, dict):
            right_reference_manifest = right_summary.get("reference_manifest")
            right_public_doc = right_summary.get("public_doc")

    if isinstance(readiness_payload, dict):
        if right_reference is None:
            right_reference = readiness_payload.get("right_reference")
        artifacts = readiness_payload.get("artifacts") if isinstance(readiness_payload.get("artifacts"), dict) else None
        if isinstance(artifacts, dict):
            if right_reference_manifest is None:
                right_reference_manifest = artifacts.get("right_reference_manifest")
            if right_public_doc is None:
                right_public_doc = artifacts.get("right_public_doc")

    if isinstance(numeric_compare_payload, dict):
        if right_reference is None:
            right_reference = numeric_compare_payload.get("right_reference")
        artifacts = numeric_compare_payload.get("artifacts") if isinstance(numeric_compare_payload.get("artifacts"), dict) else None
        if isinstance(artifacts, dict) and right_reference_manifest is None:
            right_reference_manifest = artifacts.get("right_reference_manifest")

    if not isinstance(right_reference, str) or not right_reference.strip():
        raise ValueError("right reference is missing from readiness/compare payloads")
    if not isinstance(right_reference_manifest, str) or not right_reference_manifest.strip():
        raise ValueError("right reference manifest is missing from readiness/compare payloads")
    if not isinstance(right_public_doc, str) or not right_public_doc.strip():
        right_public_doc = "docs/public_benchmark_snapshot.md"
    return right_reference, _resolve_path(right_reference_manifest), _resolve_path(right_public_doc)


def _build_local_revision_command(
    *,
    lineage_payload: dict[str, object],
    lineage_path: Path,
) -> list[str]:
    run_context = lineage_payload.get("run_context") if isinstance(lineage_payload.get("run_context"), dict) else {}
    artifacts = lineage_payload.get("artifacts") if isinstance(lineage_payload.get("artifacts"), dict) else {}
    revision_slug = str(lineage_payload.get("revision") or "").strip()
    universe = str(lineage_payload.get("universe") or DEFAULT_LEFT_UNIVERSE)
    source_scope = str(lineage_payload.get("source_scope") or "local_only")
    baseline_reference = str(lineage_payload.get("baseline_reference") or "current_recommended_serving_2025_latest")

    if not revision_slug:
        raise ValueError("left lineage payload is missing revision")

    command = [
        sys.executable,
        str(ROOT / "scripts/run_local_revision_gate.py"),
        "--revision",
        revision_slug,
        "--data-config",
        str(run_context.get("data_config") or "configs/data_local_nankan.yaml"),
        "--model-config",
        str(run_context.get("model_config") or "configs/model_local_baseline.yaml"),
        "--feature-config",
        str(run_context.get("feature_config") or "configs/features_local_baseline.yaml"),
        "--evaluate-max-rows",
        str(run_context.get("evaluate_max_rows") or 120000),
        "--evaluate-wf-mode",
        str(run_context.get("evaluate_wf_mode") or "fast"),
        "--evaluate-wf-scheme",
        str(run_context.get("evaluate_wf_scheme") or "nested"),
        "--promotion-min-feasible-folds",
        str(run_context.get("promotion_min_feasible_folds") or 1),
        "--universe",
        universe,
        "--source-scope",
        source_scope,
        "--baseline-reference",
        baseline_reference,
        "--lineage-output",
        _path_arg(lineage_path),
        "--snapshot-output",
        str(artifacts.get("snapshot") or f"artifacts/reports/coverage_snapshot_{revision_slug}.json"),
        "--benchmark-manifest-output",
        str(artifacts.get("benchmark_manifest") or f"artifacts/reports/benchmark_gate_{revision_slug}.json"),
        "--evaluation-pointer-output",
        str(artifacts.get("evaluation_pointer") or f"artifacts/reports/evaluation_{revision_slug}_pointer.json"),
        "--promotion-output",
        str(artifacts.get("promotion_output") or f"artifacts/reports/promotion_gate_{revision_slug}.json"),
        "--revision-manifest-output",
        str(artifacts.get("revision_manifest") or f"artifacts/reports/revision_gate_{revision_slug}.json"),
        "--wf-summary-output",
        str(artifacts.get("wf_summary") or f"artifacts/reports/wf_feasibility_diag_{revision_slug}.json"),
    ]

    if run_context.get("tail_rows") is not None:
        command.extend(["--tail-rows", str(run_context.get("tail_rows"))])
    if run_context.get("evaluate_pre_feature_max_rows") is not None:
        command.extend(["--evaluate-pre-feature-max-rows", str(run_context.get("evaluate_pre_feature_max_rows"))])
    if run_context.get("evaluate_start_date"):
        command.extend(["--evaluate-start-date", str(run_context.get("evaluate_start_date"))])
    if run_context.get("evaluate_end_date"):
        command.extend(["--evaluate-end-date", str(run_context.get("evaluate_end_date"))])
    return command


def _step(name: str, command: list[str], *, output_path: Path | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"name": name, "command": command}
    if output_path is not None:
        payload["output_path"] = _path_arg(output_path)
    return payload


def _build_recovery_steps(
    *,
    board_payload: dict[str, object],
    board_path: Path,
) -> tuple[list[dict[str, object]], dict[str, str]]:
    artifacts = board_payload.get("artifacts") if isinstance(board_payload.get("artifacts"), dict) else {}
    if not isinstance(artifacts, dict):
        raise ValueError("status board artifacts are missing")

    public_snapshot_path = _resolve_path(str(artifacts.get("public_snapshot_manifest") or ""))
    readiness_path = _resolve_path(str(artifacts.get("readiness_manifest") or ""))
    compare_path = _resolve_path(str(artifacts.get("compare_manifest") or ""))
    schema_path = _resolve_path(str(artifacts.get("schema_manifest") or ""))
    numeric_compare_path = _resolve_path(str(artifacts.get("numeric_compare_manifest") or ""))
    numeric_summary_path = _resolve_path(str(artifacts.get("numeric_summary_manifest") or ""))
    gap_audit_path = _resolve_path(str(artifacts.get("left_gap_audit_manifest") or ""))
    recovery_plan_path = _resolve_path(str(artifacts.get("left_recovery_plan_manifest") or ""))

    public_snapshot_payload = _read_required_payload(public_snapshot_path, label="local public snapshot")
    readiness_payload = _read_optional_payload(_path_arg(readiness_path))
    compare_payload = _read_optional_payload(_path_arg(compare_path))
    numeric_compare_payload = _read_optional_payload(_path_arg(numeric_compare_path))

    lineage_manifest_text = public_snapshot_payload.get("lineage_manifest")
    if not isinstance(lineage_manifest_text, str) or not lineage_manifest_text.strip():
        raise ValueError("local public snapshot is missing lineage manifest")
    lineage_path = _resolve_path(lineage_manifest_text)
    lineage_payload = _read_required_payload(lineage_path, label="local revision lineage")

    right_reference, right_reference_manifest_path, right_public_doc_path = _extract_right_reference(
        readiness_payload,
        compare_payload,
        numeric_compare_payload,
    )

    csv_text = None
    if isinstance(numeric_compare_payload, dict):
        numeric_artifacts = numeric_compare_payload.get("artifacts") if isinstance(numeric_compare_payload.get("artifacts"), dict) else {}
        if isinstance(numeric_artifacts, dict):
            csv_text = numeric_artifacts.get("numeric_compare_csv")
    numeric_compare_csv_path = _resolve_path(csv_text) if isinstance(csv_text, str) and csv_text.strip() else numeric_compare_path.with_suffix(".csv")

    left_universe = str(board_payload.get("left_universe") or public_snapshot_payload.get("universe") or DEFAULT_LEFT_UNIVERSE)
    right_universe = str(board_payload.get("right_universe") or DEFAULT_RIGHT_UNIVERSE)
    revision_slug = str(board_payload.get("revision") or public_snapshot_payload.get("revision") or "")
    baseline_reference = str(public_snapshot_payload.get("baseline_reference") or lineage_payload.get("baseline_reference") or right_reference)
    source_scope = str(public_snapshot_payload.get("source_scope") or lineage_payload.get("source_scope") or "local_only")

    local_revision_command = _build_local_revision_command(lineage_payload=lineage_payload, lineage_path=lineage_path)
    public_snapshot_command = [
        sys.executable,
        str(ROOT / "scripts/run_local_public_snapshot.py"),
        "--revision",
        str(public_snapshot_payload.get("revision") or revision_slug),
        "--lineage-manifest",
        _path_arg(lineage_path),
        "--output",
        _path_arg(public_snapshot_path),
        "--universe",
        left_universe,
        "--source-scope",
        source_scope,
        "--baseline-reference",
        baseline_reference,
    ]
    readiness_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_readiness.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--left-public-snapshot",
        _path_arg(public_snapshot_path),
        "--right-reference",
        right_reference,
        "--right-reference-manifest",
        _path_arg(right_reference_manifest_path),
        "--right-public-doc",
        _path_arg(right_public_doc_path),
        "--output",
        _path_arg(readiness_path),
    ]
    compare_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_compare.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--left-public-snapshot",
        _path_arg(public_snapshot_path),
        "--right-reference",
        right_reference,
        "--right-reference-manifest",
        _path_arg(right_reference_manifest_path),
        "--right-public-doc",
        _path_arg(right_public_doc_path),
        "--output",
        _path_arg(compare_path),
    ]
    schema_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_schema.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--right-reference",
        right_reference,
        "--readiness-manifest",
        _path_arg(readiness_path),
        "--compare-manifest",
        _path_arg(compare_path),
        "--output",
        _path_arg(schema_path),
    ]
    numeric_compare_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_numeric_compare.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--right-reference",
        right_reference,
        "--readiness-manifest",
        _path_arg(readiness_path),
        "--compare-manifest",
        _path_arg(compare_path),
        "--schema-manifest",
        _path_arg(schema_path),
        "--right-reference-manifest",
        _path_arg(right_reference_manifest_path),
        "--output",
        _path_arg(numeric_compare_path),
        "--csv-output",
        _path_arg(numeric_compare_csv_path),
    ]
    numeric_summary_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_numeric_summary.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--numeric-compare-manifest",
        _path_arg(numeric_compare_path),
        "--output",
        _path_arg(numeric_summary_path),
    ]
    gap_audit_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_left_gap_audit.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--numeric-compare-manifest",
        _path_arg(numeric_compare_path),
        "--left-lineage-manifest",
        _path_arg(lineage_path),
        "--output",
        _path_arg(gap_audit_path),
    ]
    recovery_plan_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_left_recovery_plan.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--gap-audit-manifest",
        _path_arg(gap_audit_path),
        "--output",
        _path_arg(recovery_plan_path),
    ]
    status_board_command = [
        sys.executable,
        str(ROOT / "scripts/run_mixed_universe_status_board.py"),
        "--revision",
        revision_slug,
        "--left-universe",
        left_universe,
        "--right-universe",
        right_universe,
        "--output",
        _path_arg(board_path),
    ]

    steps = [
        _step("local_revision_gate", local_revision_command, output_path=lineage_path),
        _step("local_public_snapshot", public_snapshot_command, output_path=public_snapshot_path),
        _step("mixed_universe_readiness", readiness_command, output_path=readiness_path),
        _step("mixed_universe_compare", compare_command, output_path=compare_path),
        _step("mixed_universe_schema", schema_command, output_path=schema_path),
        _step("mixed_universe_numeric_compare", numeric_compare_command, output_path=numeric_compare_path),
        _step("mixed_universe_numeric_summary", numeric_summary_command, output_path=numeric_summary_path),
        _step("mixed_universe_left_gap_audit", gap_audit_command, output_path=gap_audit_path),
        _step("mixed_universe_left_recovery_plan", recovery_plan_command, output_path=recovery_plan_path),
        _step("mixed_universe_status_board", status_board_command, output_path=board_path),
    ]
    artifact_map = {
        "status_board_manifest": _path_arg(board_path),
        "local_lineage_manifest": _path_arg(lineage_path),
        "public_snapshot_manifest": _path_arg(public_snapshot_path),
        "readiness_manifest": _path_arg(readiness_path),
        "compare_manifest": _path_arg(compare_path),
        "schema_manifest": _path_arg(schema_path),
        "numeric_compare_manifest": _path_arg(numeric_compare_path),
        "numeric_compare_csv": _path_arg(numeric_compare_csv_path),
        "numeric_summary_manifest": _path_arg(numeric_summary_path),
        "left_gap_audit_manifest": _path_arg(gap_audit_path),
        "left_recovery_plan_manifest": _path_arg(recovery_plan_path),
    }
    return steps, artifact_map


def _run_command(step: dict[str, object]) -> dict[str, object]:
    command = [str(part) for part in (step.get("command") or [])]
    started_at = utc_now_iso()
    print(f"[mixed-universe-recovery] running {step.get('name')}: {' '.join(command)}", flush=True)
    result = subprocess.run(command, cwd=ROOT, check=False)
    finished_at = utc_now_iso()
    return {
        "name": step.get("name"),
        "command": command,
        "output_path": step.get("output_path"),
        "status": "completed" if result.returncode == 0 else ("partial" if result.returncode == 2 else "failed"),
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--revision", default=None)
    parser.add_argument("--left-universe", default=DEFAULT_LEFT_UNIVERSE)
    parser.add_argument("--right-universe", default=DEFAULT_RIGHT_UNIVERSE)
    parser.add_argument("--status-board-manifest", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    revision_slug = _normalize_slug(args.revision or f"{args.left_universe}_vs_{args.right_universe}_{time.strftime('%Y%m%d_%H%M%S')}")
    left_universe = _normalize_slug(args.left_universe)
    right_universe = _normalize_slug(args.right_universe)
    board_path = _status_board_path(
        revision_slug=revision_slug,
        left_universe=left_universe,
        right_universe=right_universe,
        explicit=args.status_board_manifest,
    )
    output_path = _output_path(
        revision_slug=revision_slug,
        left_universe=left_universe,
        right_universe=right_universe,
        explicit=args.output,
    )

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

        board_payload = _read_required_payload(board_path, label="mixed status board")
        steps, artifact_map = _build_recovery_steps(board_payload=board_payload, board_path=board_path)
        payload: dict[str, object] = {
            "started_at": utc_now_iso(),
            "finished_at": None,
            "status": "planned" if args.dry_run else "running",
            "run_kind": "mixed_universe_recovery",
            "revision": str(board_payload.get("revision") or revision_slug),
            "left_universe": str(board_payload.get("left_universe") or left_universe),
            "right_universe": str(board_payload.get("right_universe") or right_universe),
            "source_status_board": _path_arg(board_path),
            "completed_step": "plan_built",
            "recommended_action": None,
            "artifacts": {
                "recovery_run_manifest": _path_arg(output_path),
                **artifact_map,
            },
            "steps": steps if args.dry_run else [],
        }
        write_json(output_path, payload)

        if args.dry_run:
            payload["finished_at"] = utc_now_iso()
            write_json(output_path, payload)
            print(f"[mixed-universe-recovery] planned manifest saved: {_path_arg(output_path)}", flush=True)
            return 0

        executed_steps: list[dict[str, object]] = []
        overall_exit_code = 0
        for step in steps:
            result = _run_command(step)
            executed_steps.append(result)
            payload["steps"] = executed_steps
            payload["completed_step"] = str(step.get("name") or "unknown")
            if int(result["exit_code"]) not in {0, 2} and overall_exit_code == 0:
                overall_exit_code = int(result["exit_code"])
            elif int(result["exit_code"]) == 2 and overall_exit_code == 0:
                overall_exit_code = 2
            write_json(output_path, payload)

        payload["finished_at"] = utc_now_iso()
        payload["status"] = "completed" if overall_exit_code == 0 else ("partial" if overall_exit_code == 2 else "failed")

        refreshed_board_payload = _read_optional_payload(payload["artifacts"].get("status_board_manifest"))
        if isinstance(refreshed_board_payload, dict):
            payload["recommended_action"] = refreshed_board_payload.get("recommended_action")
            payload["refreshed_board_status"] = refreshed_board_payload.get("status")
            payload["refreshed_current_phase"] = refreshed_board_payload.get("current_phase")
            payload["refreshed_next_action_source"] = refreshed_board_payload.get("next_action_source")
            payload["refreshed_highlights"] = refreshed_board_payload.get("highlights")
        write_json(output_path, payload)
        print(f"[mixed-universe-recovery] manifest saved: {_path_arg(output_path)}", flush=True)
        return overall_exit_code
    except KeyboardInterrupt:
        print("[mixed-universe-recovery] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[mixed-universe-recovery] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[mixed-universe-recovery] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())