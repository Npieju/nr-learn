from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[local-nankan-pre-race-loop {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[local-nankan-pre-race-loop] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[local-nankan-pre-race-loop]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _read_json_dict(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    payload = read_json(path)
    return payload if isinstance(payload, dict) else {}


def _build_pass_paths(snapshot_dir: Path, pass_index: int) -> tuple[Path, Path]:
    summary_path = snapshot_dir / f"pass_{pass_index:03d}_coverage_summary.json"
    date_coverage_path = snapshot_dir / f"pass_{pass_index:03d}_date_coverage.csv"
    return summary_path, date_coverage_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-config", default="configs/crawl_local_nankan_template.yaml")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--include-completed", action="store_true")
    parser.add_argument("--target", choices=["race_card", "all"], default="race_card")
    parser.add_argument("--max-passes", type=int, default=1)
    parser.add_argument("--poll-interval-seconds", type=int, default=300)
    parser.add_argument("--snapshot-dir", default="artifacts/reports/local_nankan_pre_race_capture_snapshots")
    parser.add_argument("--wrapper-manifest-output", default="artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json")
    parser.add_argument("--baseline-summary-input", default="artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json")
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=True)
    args = parser.parse_args()

    try:
        max_passes = max(1, int(args.max_passes))
        poll_interval_seconds = max(1, int(args.poll_interval_seconds))
        progress = ProgressBar(total=max_passes * 3, prefix="[local-nankan-pre-race-loop]", logger=log_progress, min_interval_sec=0.0)

        snapshot_dir = _resolve_path(args.snapshot_dir)
        wrapper_manifest_path = _resolve_path(args.wrapper_manifest_output)
        baseline_summary_path = _resolve_path(args.baseline_summary_input) if args.baseline_summary_input else None

        progress.start(message=f"starting bounded pre-race capture loop max_passes={max_passes}")
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        pass_records: list[dict[str, Any]] = []
        current_baseline_path = baseline_summary_path
        latest_summary: dict[str, Any] = {}

        for pass_index in range(1, max_passes + 1):
            pass_summary_path, pass_date_coverage_path = _build_pass_paths(snapshot_dir, pass_index)
            prepare_command = [
                sys.executable,
                str(ROOT / "scripts/run_prepare_local_nankan_ids.py"),
                "--crawl-config",
                args.crawl_config,
                "--race-id-source",
                "race_list",
                "--target",
                args.target,
                "--date-order",
                args.date_order,
            ]
            if args.start_date:
                prepare_command.extend(["--start-date", args.start_date])
            if args.end_date:
                prepare_command.extend(["--end-date", args.end_date])
            if args.include_completed:
                prepare_command.append("--include-completed")

            collect_command = [
                sys.executable,
                str(ROOT / "scripts/run_collect_local_nankan.py"),
                "--config",
                args.crawl_config,
                "--target",
                args.target,
            ]
            collect_command.append("--overwrite" if args.overwrite else "--no-overwrite")

            coverage_command = [
                sys.executable,
                str(ROOT / "scripts/run_local_nankan_pre_race_capture_coverage.py"),
                "--summary-output",
                str(pass_summary_path),
                "--date-coverage-output",
                str(pass_date_coverage_path),
            ]
            if current_baseline_path is not None:
                coverage_command.extend(["--baseline-summary-input", str(current_baseline_path)])

            prepare_exit = _run_command(label=f"prepare_ids pass={pass_index}/{max_passes}", command=prepare_command)
            progress.update(current=((pass_index - 1) * 3) + 1, message=f"pass={pass_index}/{max_passes} prepare exit_code={prepare_exit}")
            if prepare_exit != 0:
                latest_summary = {}
                pass_records.append(
                    {
                        "pass_index": pass_index,
                        "prepare_exit_code": prepare_exit,
                        "collect_exit_code": None,
                        "coverage_exit_code": None,
                        "summary_output": str(pass_summary_path),
                        "date_coverage_output": str(pass_date_coverage_path),
                        "status": "failed_prepare",
                    }
                )
                break

            collect_exit = _run_command(label=f"collect pass={pass_index}/{max_passes}", command=collect_command)
            progress.update(current=((pass_index - 1) * 3) + 2, message=f"pass={pass_index}/{max_passes} collect exit_code={collect_exit}")
            if collect_exit not in {0, 2}:
                latest_summary = {}
                pass_records.append(
                    {
                        "pass_index": pass_index,
                        "prepare_exit_code": prepare_exit,
                        "collect_exit_code": collect_exit,
                        "coverage_exit_code": None,
                        "summary_output": str(pass_summary_path),
                        "date_coverage_output": str(pass_date_coverage_path),
                        "status": "failed_collect",
                    }
                )
                break

            coverage_exit = _run_command(label=f"coverage pass={pass_index}/{max_passes}", command=coverage_command)
            latest_summary = _read_json_dict(pass_summary_path)
            progress.update(
                current=((pass_index - 1) * 3) + 3,
                message=(
                    f"pass={pass_index}/{max_passes} coverage exit_code={coverage_exit} "
                    f"phase={latest_summary.get('current_phase')}"
                ),
            )

            pass_records.append(
                {
                    "pass_index": pass_index,
                    "prepare_exit_code": prepare_exit,
                    "collect_exit_code": collect_exit,
                    "coverage_exit_code": coverage_exit,
                    "summary_output": str(pass_summary_path),
                    "date_coverage_output": str(pass_date_coverage_path),
                    "status": "completed" if coverage_exit == 0 else "failed_coverage",
                    "pre_race_only_rows": latest_summary.get("pre_race_only_rows"),
                    "pre_race_only_races": latest_summary.get("pre_race_only_races"),
                    "result_ready_races": latest_summary.get("result_ready_races"),
                    "pending_result_races": latest_summary.get("pending_result_races"),
                    "delta_pre_race_only_rows": latest_summary.get("baseline_comparison", {}).get("delta_pre_race_only_rows"),
                    "delta_pre_race_only_races": latest_summary.get("baseline_comparison", {}).get("delta_pre_race_only_races"),
                }
            )
            if coverage_exit != 0:
                break

            current_baseline_path = pass_summary_path
            if pass_index < max_passes:
                print(
                    "[local-nankan-pre-race-loop] "
                    f"sleeping before next pass seconds={poll_interval_seconds}",
                    flush=True,
                )
                time.sleep(poll_interval_seconds)

        completed_passes = len(pass_records)
        latest_status = str(latest_summary.get("status") or ("failed" if completed_passes < max_passes else "completed"))
        latest_phase = str(latest_summary.get("current_phase") or "capture_loop")
        latest_action = str(latest_summary.get("recommended_action") or "inspect_capture_loop_manifest")

        wrapper_manifest = {
            "status": latest_status,
            "current_phase": latest_phase,
            "recommended_action": latest_action,
            "max_passes": max_passes,
            "completed_passes": completed_passes,
            "poll_interval_seconds": poll_interval_seconds,
            "snapshot_dir": str(snapshot_dir),
            "latest_summary": latest_summary,
            "pass_snapshots": pass_records,
        }
        write_json(wrapper_manifest_path, wrapper_manifest)
        progress.complete(message=f"loop manifest ready output={wrapper_manifest_path}")
        return 0 if all(record.get("status") == "completed" for record in pass_records) else 1
    except KeyboardInterrupt:
        print("[local-nankan-pre-race-loop] interrupted by user")
        return 130
    except Exception as error:
        print(f"[local-nankan-pre-race-loop] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
