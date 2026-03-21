from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from racing_ml.common.artifacts import read_json


DEFAULT_EVALUATION_MANIFEST = "artifacts/reports/evaluation_manifest.json"


def resolve_artifact_path(root: Path, path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (root / path)


def display_artifact_path(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "ok": bool(ok),
    }
    if details:
        payload.update(details)
    return payload


def _append_result(
    checks: list[dict[str, Any]],
    errors: list[str],
    check: dict[str, Any],
    error_message: str | None = None,
) -> None:
    checks.append(check)
    if not check.get("ok") and error_message:
        errors.append(error_message)


def collect_evaluation_manifest_paths(root: Path) -> list[Path]:
    report_dir = root / "artifacts" / "reports"
    candidates: list[Path] = [report_dir / "evaluation_manifest.json"]
    candidates.extend(
        sorted(
            path
            for path in report_dir.glob("evaluation_manifest_*.json")
            if "_validation" not in path.name
        )
    )
    unique_paths: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        if path in seen:
            continue
        seen.add(path)
        unique_paths.append(path)
    return unique_paths


def validate_evaluation_manifest(manifest_path: Path, *, root: Path) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError(f"Evaluation manifest must be a JSON object: {manifest_path}")

    files_payload = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
    latest_summary_path = resolve_artifact_path(root, files_payload.get("latest_summary"))
    latest_by_date_path = resolve_artifact_path(root, files_payload.get("latest_by_date"))
    latest_manifest_path = resolve_artifact_path(root, files_payload.get("latest_manifest"))
    versioned_summary_path = resolve_artifact_path(root, files_payload.get("versioned_summary"))
    versioned_by_date_path = resolve_artifact_path(root, files_payload.get("versioned_by_date"))
    versioned_manifest_path = resolve_artifact_path(root, files_payload.get("versioned_manifest"))

    checks: list[dict[str, Any]] = []
    errors: list[str] = []
    validating_latest_manifest = manifest_path == latest_manifest_path if latest_manifest_path is not None else False
    validating_versioned_manifest = manifest_path == versioned_manifest_path if versioned_manifest_path is not None else False

    for key, path in [
        ("latest_summary", latest_summary_path),
        ("latest_manifest", latest_manifest_path),
        ("versioned_summary", versioned_summary_path),
        ("versioned_manifest", versioned_manifest_path),
    ]:
        exists = path is not None and path.exists()
        _append_result(
            checks,
            errors,
            _build_check(f"file_exists:{key}", exists, {"path": str(path) if path else None}),
            f"Missing required file for {key}: {path}",
        )

    by_date_present = bool(manifest.get("by_date_present"))
    if by_date_present:
        for key, path in [
            ("latest_by_date", latest_by_date_path),
            ("versioned_by_date", versioned_by_date_path),
        ]:
            exists = path is not None and path.exists()
            _append_result(
                checks,
                errors,
                _build_check(f"file_exists:{key}", exists, {"path": str(path) if path else None}),
                f"Missing by-date file for {key}: {path}",
            )
    else:
        _append_result(
            checks,
            errors,
            _build_check(
                "by_date_absent_paths_null",
                latest_by_date_path is None and versioned_by_date_path is None,
                {
                    "latest_by_date": str(latest_by_date_path) if latest_by_date_path else None,
                    "versioned_by_date": str(versioned_by_date_path) if versioned_by_date_path else None,
                },
            ),
            "Manifest marks by_date absent but still points to by-date files",
        )

    summary_sha256 = manifest.get("checksums", {}).get("summary_sha256") if isinstance(manifest.get("checksums"), dict) else None
    summary_targets: list[tuple[str, Path | None]] = []
    if validating_latest_manifest:
        summary_targets.append(("latest_summary", latest_summary_path))
    if validating_latest_manifest or validating_versioned_manifest:
        summary_targets.append(("versioned_summary", versioned_summary_path))
    for key, path in summary_targets:
        if path is not None and path.exists() and summary_sha256:
            actual_summary_sha256 = _sha256_file(path)
            _append_result(
                checks,
                errors,
                _build_check(
                    f"checksum_match:{key}",
                    actual_summary_sha256 == summary_sha256,
                    {
                        "expected": summary_sha256,
                        "actual": actual_summary_sha256,
                    },
                ),
                f"Checksum mismatch for {key}",
            )

    by_date_sha256 = manifest.get("checksums", {}).get("by_date_sha256") if isinstance(manifest.get("checksums"), dict) else None
    current_by_date_path = latest_by_date_path if validating_latest_manifest else versioned_by_date_path
    if by_date_present and current_by_date_path is not None and current_by_date_path.exists():
        by_date_targets: list[tuple[str, Path | None]] = []
        if validating_latest_manifest:
            by_date_targets.append(("latest_by_date", latest_by_date_path))
        if validating_latest_manifest or validating_versioned_manifest:
            by_date_targets.append(("versioned_by_date", versioned_by_date_path))
        for key, path in by_date_targets:
            if path is None or not path.exists():
                continue
            actual_by_date_sha256 = _sha256_file(path)
            _append_result(
                checks,
                errors,
                _build_check(
                    f"checksum_match:{key}",
                    actual_by_date_sha256 == by_date_sha256,
                    {
                        "expected": by_date_sha256,
                        "actual": actual_by_date_sha256,
                    },
                ),
                f"Checksum mismatch for {key}",
            )

        by_date_frame = pd.read_csv(current_by_date_path)
        actual_by_date_rows = int(len(by_date_frame))
        expected_by_date_rows = manifest.get("by_date_rows")
        _append_result(
            checks,
            errors,
            _build_check(
                "by_date_row_count_match",
                expected_by_date_rows == actual_by_date_rows,
                {
                    "expected": expected_by_date_rows,
                    "actual": actual_by_date_rows,
                },
            ),
            "By-date row count mismatch",
        )

        consistency_payload = manifest.get("consistency") if isinstance(manifest.get("consistency"), dict) else {}
        if "date" in by_date_frame.columns and not by_date_frame.empty:
            actual_start = str(by_date_frame["date"].iloc[0])
            actual_end = str(by_date_frame["date"].iloc[-1])
        else:
            actual_start = None
            actual_end = None
        _append_result(
            checks,
            errors,
            _build_check(
                "by_date_start_match",
                consistency_payload.get("by_date_start") == actual_start,
                {"expected": consistency_payload.get("by_date_start"), "actual": actual_start},
            ),
            "By-date start mismatch",
        )
        _append_result(
            checks,
            errors,
            _build_check(
                "by_date_end_match",
                consistency_payload.get("by_date_end") == actual_end,
                {"expected": consistency_payload.get("by_date_end"), "actual": actual_end},
            ),
            "By-date end mismatch",
        )
    else:
        _append_result(
            checks,
            errors,
            _build_check(
                "by_date_checksum_absent_when_no_by_date",
                (not by_date_present and by_date_sha256 is None),
                {"by_date_sha256": by_date_sha256},
            ),
            "Manifest marks by_date absent but checksum is still populated",
        )

    manifest_targets: list[tuple[str, Path | None]] = []
    if validating_latest_manifest:
        manifest_targets.append(("latest_manifest", latest_manifest_path))
    if validating_latest_manifest or validating_versioned_manifest:
        manifest_targets.append(("versioned_manifest", versioned_manifest_path))
    for key, path in manifest_targets:
        if path is None or not path.exists():
            continue
        other_manifest = read_json(path)
        _append_result(
            checks,
            errors,
            _build_check(
                f"manifest_payload_match:{key}",
                other_manifest == manifest,
                {"path": str(path)},
            ),
            f"Manifest payload mismatch for {key}",
        )

    current_manifest_registered = manifest_path in {
        path
        for path in [latest_manifest_path, versioned_manifest_path]
        if path is not None
    }
    _append_result(
        checks,
        errors,
        _build_check(
            "current_manifest_registered",
            current_manifest_registered,
            {"manifest_path": str(manifest_path)},
        ),
        "Current manifest path is not listed in manifest files",
    )

    status = "ok" if not errors else "failed"
    return {
        "status": status,
        "manifest": display_artifact_path(root, manifest_path),
        "profile": manifest.get("profile"),
        "config": manifest.get("config"),
        "checks": checks,
        "errors": errors,
    }


def validate_evaluation_manifest_safe(manifest_path: Path, *, root: Path) -> dict[str, Any]:
    try:
        return validate_evaluation_manifest(manifest_path, root=root)
    except Exception as error:
        return {
            "status": "failed",
            "manifest": display_artifact_path(root, manifest_path),
            "profile": None,
            "config": None,
            "checks": [],
            "errors": [str(error)],
        }