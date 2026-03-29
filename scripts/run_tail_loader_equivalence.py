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
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import _pick_dataset
from racing_ml.data.tail_equivalence import TAIL_READER_CANDIDATES, compare_tail_readers, comparison_passes_gate


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[tail-equivalence {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-path", default=None)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--tail-rows", type=int, default=10000)
    parser.add_argument("--left-reader", choices=sorted(TAIL_READER_CANDIDATES.keys()), default="current")
    parser.add_argument("--right-reader", choices=sorted(TAIL_READER_CANDIDATES.keys()), default="deque_trim")
    parser.add_argument("--manifest-file", default="artifacts/reports/tail_loader_equivalence.json")
    parser.add_argument("--fail-on-diff", action="store_true")
    parser.add_argument("--fail-gate", choices=["exact", "canonical", "value"], default="exact")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[tail-equivalence]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="resolving input csv")
        csv_path = Path(args.csv_path) if args.csv_path is not None else _pick_dataset(ROOT / args.raw_dir)
        manifest_path = ROOT / args.manifest_file
        artifact_ensure_output_file_path(manifest_path, label="tail equivalence manifest", workspace_root=ROOT)
        progress.update(message=f"input ready csv={artifact_display_path(csv_path, workspace_root=ROOT)}")
        with Heartbeat("[tail-equivalence]", "comparing tail readers", logger=log_progress):
            comparison = compare_tail_readers(
                left_name=args.left_reader,
                left_reader=TAIL_READER_CANDIDATES[args.left_reader],
                right_name=args.right_reader,
                right_reader=TAIL_READER_CANDIDATES[args.right_reader],
                csv_path=csv_path,
                tail_rows=int(args.tail_rows),
            )
        raw_exact_equal = comparison["comparison"]["raw"]["exact_equal"]
        normalized_exact_equal = comparison["comparison"]["normalized"]["exact_equal"]
        raw_canonical_dtype_only = comparison["comparison"]["raw"]["canonical_dtype_only_difference"]
        normalized_canonical_dtype_only = comparison["comparison"]["normalized"]["canonical_dtype_only_difference"]
        gate_passed = comparison_passes_gate(comparison, args.fail_gate)
        progress.update(
            message=(
                "comparison complete "
                f"raw_exact_equal={raw_exact_equal} "
                f"normalized_exact_equal={normalized_exact_equal} "
                f"raw_canonical_dtype_only={raw_canonical_dtype_only} "
                f"normalized_canonical_dtype_only={normalized_canonical_dtype_only} "
                f"gate={args.fail_gate} gate_passed={gate_passed}"
            )
        )
        comparison["manifest_file"] = artifact_display_path(manifest_path, workspace_root=ROOT)
        comparison["gate"] = {
            "mode": args.fail_gate,
            "passed": bool(gate_passed),
        }
        write_json(manifest_path, comparison)
        progress.update(message=f"manifest written path={artifact_display_path(manifest_path, workspace_root=ROOT)}")
        print(
            "[tail-equivalence] "
            f"left={args.left_reader} right={args.right_reader} "
            f"raw_exact_equal={raw_exact_equal} "
            f"normalized_exact_equal={normalized_exact_equal} "
            f"raw_canonical_dtype_only={raw_canonical_dtype_only} "
            f"normalized_canonical_dtype_only={normalized_canonical_dtype_only} "
            f"gate={args.fail_gate} gate_passed={gate_passed}",
            flush=True,
        )
        progress.complete(message="tail equivalence completed")
        if args.fail_on_diff and not gate_passed:
            return 2
        return 0
    except KeyboardInterrupt:
        print("[tail-equivalence] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[tail-equivalence] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[tail-equivalence] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
