from __future__ import annotations

import argparse
import subprocess
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]


def _run(command: list[str]) -> int:
    return subprocess.run(command, cwd=ROOT, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-config", default="configs/crawl_local_nankan_template.yaml")
    parser.add_argument("--data-config", default="configs/data_local_nankan.yaml")
    parser.add_argument("--target", default="race_card")
    parser.add_argument("--race-id-source", default="race_list")
    parser.add_argument("--start-date", default="2006-03-29")
    parser.add_argument("--end-date", default="2026-03-28")
    parser.add_argument("--date-order", default="desc")
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--manifest-file", default="artifacts/reports/local_nankan_backfill_race_card_20y.json")
    parser.add_argument("--coverage-output", default="artifacts/reports/coverage_snapshot_local_nankan_current.json")
    parser.add_argument("--status-board-output", default="artifacts/reports/local_nankan_data_status_board.json")
    parser.add_argument("--backfill-aggregate", default="artifacts/reports/local_nankan_backfill_race_card_20y.json")
    args = parser.parse_args()

    backfill_command = [
        sys.executable,
        str(ROOT / "scripts/run_backfill_local_nankan.py"),
        "--crawl-config",
        args.crawl_config,
        "--data-config",
        args.data_config,
        "--target",
        args.target,
        "--race-id-source",
        args.race_id_source,
        "--start-date",
        args.start_date,
        "--end-date",
        args.end_date,
        "--date-order",
        args.date_order,
        "--chunk-months",
        str(args.chunk_months),
        "--max-date-windows",
        "1",
        "--max-cycles",
        str(args.max_cycles),
        "--manifest-file",
        args.manifest_file,
    ]
    coverage_command = [
        sys.executable,
        str(ROOT / "scripts/run_local_coverage_snapshot.py"),
        "--data-config",
        args.data_config,
        "--output",
        args.coverage_output,
    ]
    status_command = [
        sys.executable,
        str(ROOT / "scripts/run_local_nankan_status_board.py"),
        "--coverage-snapshot",
        args.coverage_output,
        "--backfill-aggregate",
        args.backfill_aggregate,
        "--output",
        args.status_board_output,
    ]

    print(
        f"[local-nankan-next-window] starting target={args.target} manifest={args.manifest_file} date_order={args.date_order}",
        flush=True,
    )
    backfill_exit_code = _run(backfill_command)
    print(f"[local-nankan-next-window] backfill_exit_code={backfill_exit_code}", flush=True)
    if backfill_exit_code != 0:
        return backfill_exit_code

    coverage_exit_code = _run(coverage_command)
    print(f"[local-nankan-next-window] coverage_exit_code={coverage_exit_code}", flush=True)
    if coverage_exit_code != 0:
        return coverage_exit_code

    status_exit_code = _run(status_command)
    print(f"[local-nankan-next-window] status_board_exit_code={status_exit_code}", flush=True)
    return status_exit_code


if __name__ == "__main__":
    raise SystemExit(main())