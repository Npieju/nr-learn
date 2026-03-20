import argparse
import hashlib
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[eval-manifest {now}] {message}", flush=True)


def _resolve_path(path_value: str | None) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value)
    return path if path.is_absolute() else (ROOT / path)


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


def _append_result(checks: list[dict[str, Any]], errors: list[str], check: dict[str, Any], error_message: str | None = None) -> None:
    checks.append(check)
    if not check.get("ok") and error_message:
        errors.append(error_message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/reports/evaluation_manifest.json")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        manifest_path = _resolve_path(args.manifest)
        if manifest_path is None:
            raise ValueError("Manifest path is required")

        progress = ProgressBar(total=4, prefix="[eval-manifest]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"loading manifest={manifest_path.name}")

        with Heartbeat("[eval-manifest]", "reading manifest payload", logger=log_progress):
            manifest = read_json(manifest_path)
        if not isinstance(manifest, dict):
            raise RuntimeError(f"Evaluation manifest must be a JSON object: {manifest_path}")
        progress.update(message="manifest loaded")

        files_payload = manifest.get("files") if isinstance(manifest.get("files"), dict) else {}
        latest_summary_path = _resolve_path(files_payload.get("latest_summary"))
        latest_by_date_path = _resolve_path(files_payload.get("latest_by_date"))
        latest_manifest_path = _resolve_path(files_payload.get("latest_manifest"))
        versioned_summary_path = _resolve_path(files_payload.get("versioned_summary"))
        versioned_by_date_path = _resolve_path(files_payload.get("versioned_by_date"))
        versioned_manifest_path = _resolve_path(files_payload.get("versioned_manifest"))

        checks: list[dict[str, Any]] = []
        errors: list[str] = []

        with Heartbeat("[eval-manifest]", "verifying referenced files", logger=log_progress):
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
        progress.update(message="referenced files verified")

        with Heartbeat("[eval-manifest]", "verifying checksums and consistency", logger=log_progress):
            summary_sha256 = manifest.get("checksums", {}).get("summary_sha256") if isinstance(manifest.get("checksums"), dict) else None
            for key, path in [
                ("latest_summary", latest_summary_path),
                ("versioned_summary", versioned_summary_path),
            ]:
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
            if by_date_present and latest_by_date_path is not None and latest_by_date_path.exists():
                for key, path in [
                    ("latest_by_date", latest_by_date_path),
                    ("versioned_by_date", versioned_by_date_path),
                ]:
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

                by_date_frame = pd.read_csv(latest_by_date_path)
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
                    _build_check("by_date_start_match", consistency_payload.get("by_date_start") == actual_start, {"expected": consistency_payload.get("by_date_start"), "actual": actual_start}),
                    "By-date start mismatch",
                )
                _append_result(
                    checks,
                    errors,
                    _build_check("by_date_end_match", consistency_payload.get("by_date_end") == actual_end, {"expected": consistency_payload.get("by_date_end"), "actual": actual_end}),
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

            for key, path in [
                ("latest_manifest", latest_manifest_path),
                ("versioned_manifest", versioned_manifest_path),
            ]:
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
        progress.update(message="checksums and consistency verified")

        status = "ok" if not errors else "failed"
        report = {
            "status": status,
            "manifest": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
            "profile": manifest.get("profile"),
            "config": manifest.get("config"),
            "checks": checks,
            "errors": errors,
        }

        if args.output:
            output_path = _resolve_path(args.output)
            if output_path is None:
                raise ValueError("Output path could not be resolved")
            with Heartbeat("[eval-manifest]", "writing validation report", logger=log_progress):
                write_json(output_path, report)
            progress.complete(message=f"validation report written status={status}")
            print(f"[eval-manifest] report saved: {output_path}")
        else:
            progress.complete(message=f"validation completed status={status}")

        print(f"[eval-manifest] manifest: {manifest_path}")
        print(f"[eval-manifest] status: {status}")
        print(f"[eval-manifest] checks={sum(1 for check in checks if check.get('ok'))}/{len(checks)}")
        if errors:
            print(f"[eval-manifest] errors: {errors}")
            return 1
        return 0
    except KeyboardInterrupt:
        print("[eval-manifest] interrupted by user")
        return 130
    except Exception as error:
        print(f"[eval-manifest] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())