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
from racing_ml.common.artifacts import write_json
from racing_ml.common.progress import ProgressBar
from racing_ml.evaluation.summary_equivalence import DEFAULT_IGNORED_PATHS, compare_summary_files


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[summary-equivalence {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-summary", required=True)
    parser.add_argument("--right-summary", required=True)
    parser.add_argument("--manifest-file", default="artifacts/reports/summary_equivalence.json")
    parser.add_argument("--ignore-path", action="append", default=[])
    parser.add_argument("--fail-on-diff", action="store_true")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[summary-equivalence]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="resolving inputs")
        left_summary = ROOT / args.left_summary
        right_summary = ROOT / args.right_summary
        manifest_path = ROOT / args.manifest_file
        artifact_ensure_output_file_path(manifest_path, label="summary equivalence manifest", workspace_root=ROOT)
        progress.update(
            message=(
                f"inputs ready left={artifact_display_path(left_summary, workspace_root=ROOT)} "
                f"right={artifact_display_path(right_summary, workspace_root=ROOT)}"
            )
        )
        ignored_paths = set(DEFAULT_IGNORED_PATHS)
        ignored_paths.update(str(path) for path in args.ignore_path if str(path).strip())
        comparison = compare_summary_files(
            left_summary=left_summary,
            right_summary=right_summary,
            ignored_paths=ignored_paths,
        )
        progress.update(
            message=(
                f"comparison complete exact_equal={comparison['exact_equal']} "
                f"difference_count={comparison['difference_count']}"
            )
        )
        comparison["manifest_file"] = artifact_display_path(manifest_path, workspace_root=ROOT)
        write_json(manifest_path, comparison)
        progress.update(message=f"manifest written path={artifact_display_path(manifest_path, workspace_root=ROOT)}")
        print(
            "[summary-equivalence] "
            f"left={args.left_summary} right={args.right_summary} "
            f"exact_equal={comparison['exact_equal']} difference_count={comparison['difference_count']}",
            flush=True,
        )
        progress.complete(message="summary equivalence completed")
        if args.fail_on_diff and not comparison["exact_equal"]:
            return 2
        return 0
    except KeyboardInterrupt:
        print("[summary-equivalence] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[summary-equivalence] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[summary-equivalence] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
