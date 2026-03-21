import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.evaluation.manifest_validation import (
    DEFAULT_EVALUATION_MANIFEST,
    collect_evaluation_manifest_paths,
    resolve_artifact_path,
    validate_evaluation_manifest_safe,
)


DEFAULT_MANIFEST = DEFAULT_EVALUATION_MANIFEST


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[eval-manifest {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--all-manifests",
        action="store_true",
        help="Validate evaluation_manifest.json and all versioned evaluation_manifest_*.json files in artifacts/reports.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    try:
        if args.all_manifests and args.manifest != DEFAULT_MANIFEST:
            raise ValueError("--all-manifests cannot be combined with an explicit --manifest")

        if args.all_manifests:
            manifest_paths = collect_evaluation_manifest_paths(ROOT)
            progress = ProgressBar(total=max(1, len(manifest_paths)), prefix="[eval-manifest batch]", logger=log_progress, min_interval_sec=0.0)
            progress.start(message=f"validating manifests count={len(manifest_paths)}")
            reports: list[dict[str, Any]] = []
            for manifest_path in manifest_paths:
                with Heartbeat("[eval-manifest batch]", f"validating {manifest_path.name}", logger=log_progress):
                    report = validate_evaluation_manifest_safe(manifest_path, root=ROOT)
                reports.append(report)
                progress.update(message=f"{manifest_path.name} status={report['status']}")

            failed_reports = [report for report in reports if report.get("status") != "ok"]
            aggregate_report = {
                "status": "ok" if not failed_reports else "failed",
                "manifest_count": len(reports),
                "ok_count": len(reports) - len(failed_reports),
                "failed_count": len(failed_reports),
                "reports": reports,
            }

            if args.output:
                output_path = resolve_artifact_path(ROOT, args.output)
                if output_path is None:
                    raise ValueError("Output path could not be resolved")
                with Heartbeat("[eval-manifest batch]", "writing validation report", logger=log_progress):
                    write_json(output_path, aggregate_report)
                print(f"[eval-manifest batch] report saved: {output_path}")

            progress.complete(message=f"batch validation completed status={aggregate_report['status']}")
            print(f"[eval-manifest batch] manifests={len(reports)} ok={aggregate_report['ok_count']} failed={aggregate_report['failed_count']}")
            if failed_reports:
                print("[eval-manifest batch] failed manifests:")
                for report in failed_reports:
                    print(f"  - {report['manifest']}: {report.get('errors')}" )
                return 1
            return 0

        manifest_path = resolve_artifact_path(ROOT, args.manifest)
        if manifest_path is None:
            raise ValueError("Manifest path is required")

        progress = ProgressBar(total=1, prefix="[eval-manifest]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"validating manifest={manifest_path.name}")
        with Heartbeat("[eval-manifest]", f"validating {manifest_path.name}", logger=log_progress):
            report = validate_evaluation_manifest_safe(manifest_path, root=ROOT)

        if args.output:
            output_path = resolve_artifact_path(ROOT, args.output)
            if output_path is None:
                raise ValueError("Output path could not be resolved")
            with Heartbeat("[eval-manifest]", "writing validation report", logger=log_progress):
                write_json(output_path, report)
            print(f"[eval-manifest] report saved: {output_path}")

        progress.complete(message=f"validation completed status={report['status']}")

        print(f"[eval-manifest] manifest: {manifest_path}")
        print(f"[eval-manifest] status: {report['status']}")
        print(f"[eval-manifest] checks={sum(1 for check in report.get('checks', []) if check.get('ok'))}/{len(report.get('checks', []))}")
        if report.get("errors"):
            print(f"[eval-manifest] errors: {report['errors']}")
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