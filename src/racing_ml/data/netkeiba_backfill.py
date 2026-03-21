from __future__ import annotations

import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

from racing_ml.common.artifacts import write_json
from racing_ml.common.progress import Heartbeat
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.data.netkeiba_crawler import crawl_netkeiba_from_config, netkeiba_crawl_lock
from racing_ml.data.netkeiba_id_prep import prepare_netkeiba_ids_from_config


def _log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-backfill {now}] {message}", flush=True)


def _extract_report(summary: dict[str, Any], kind: str) -> dict[str, Any]:
    for report in summary.get("reports", []):
        if report.get("kind") == kind:
            return report
    return {}


def _extract_target(summary: dict[str, Any], target_name: str) -> dict[str, Any]:
    for report in summary.get("targets", []):
        if report.get("target") == target_name:
            return report
    return {}


def _write_json(path: Path, data: dict[str, Any]) -> None:
    write_json(path, data)


def _run_post_cycle_command(
    *,
    command: str,
    base_dir: Path,
    cycle: int,
    manifest_path: Path,
) -> dict[str, Any]:
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    argv = shlex.split(command)
    env = dict(os.environ)
    env["NETKEIBA_BACKFILL_CYCLE"] = str(cycle)
    env["NETKEIBA_BACKFILL_MANIFEST"] = str(manifest_path)

    _log_progress(f"cycle={cycle} running post-cycle command: {command}")
    result = subprocess.run(argv, cwd=base_dir, env=env, check=False)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    status = "completed" if result.returncode == 0 else "failed"
    _log_progress(f"cycle={cycle} post-cycle command exit_code={result.returncode}")
    return {
        "command": command,
        "argv": argv,
        "status": status,
        "exit_code": int(result.returncode),
        "started_at": started_at,
        "finished_at": finished_at,
    }


def _build_summary(
    *,
    started_at: str,
    finished_at: str | None,
    start_date: str | None,
    end_date: str | None,
    race_id_source: str,
    race_batch_size: int,
    pedigree_batch_size: int,
    include_race_card: bool,
    include_pedigree: bool,
    max_cycles: int | None,
    date_order: str,
    post_cycle_command: str | None,
    stop_on_post_cycle_failure: bool,
    stopped_reason: str,
    cycle_reports: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "date_window": {"start": start_date, "end": end_date},
        "date_order": str(date_order),
        "race_id_source": str(race_id_source),
        "race_batch_size": int(race_batch_size),
        "pedigree_batch_size": int(pedigree_batch_size),
        "include_race_card": bool(include_race_card),
        "include_pedigree": bool(include_pedigree),
        "max_cycles": int(max_cycles) if max_cycles is not None else None,
        "post_cycle_command": str(post_cycle_command).strip() if post_cycle_command else None,
        "stop_on_post_cycle_failure": bool(stop_on_post_cycle_failure),
        "stopped_reason": stopped_reason,
        "completed_cycles": int(len(cycle_reports)),
        "cycles": cycle_reports,
    }


def run_netkeiba_backfill_from_config(
    data_config: dict[str, Any],
    crawl_config: dict[str, Any],
    *,
    base_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
    date_order: str = "desc",
    race_id_source: str = "training_table",
    race_batch_size: int = 100,
    pedigree_batch_size: int = 500,
    include_race_card: bool = True,
    include_pedigree: bool = True,
    max_cycles: int | None = None,
    post_cycle_command: str | None = None,
    stop_on_post_cycle_failure: bool = False,
    refresh: bool = False,
    parse_only: bool = False,
    manifest_file: str | Path = "artifacts/reports/netkeiba_backfill_manifest.json",
) -> dict[str, Any]:
    with netkeiba_crawl_lock(crawl_config, base_dir=base_dir):
        dataset_cfg = data_config.get("dataset", data_config)
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        manifest_path = Path(base_dir) / manifest_file

        with Heartbeat("[netkeiba-backfill]", "loading training table", logger=_log_progress):
            training_frame = load_training_table(raw_dir, dataset_config=data_config, base_dir=base_dir)

        started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        cycle_reports: list[dict[str, Any]] = []
        cycle = 0
        stopped_reason = "running"
        _write_json(
            manifest_path,
            _build_summary(
                started_at=started_at,
                finished_at=None,
                start_date=start_date,
                end_date=end_date,
                date_order=date_order,
                race_id_source=race_id_source,
                race_batch_size=race_batch_size,
                pedigree_batch_size=pedigree_batch_size,
                include_race_card=include_race_card,
                include_pedigree=include_pedigree,
                max_cycles=max_cycles,
                post_cycle_command=post_cycle_command,
                stop_on_post_cycle_failure=stop_on_post_cycle_failure,
                stopped_reason=stopped_reason,
                cycle_reports=cycle_reports,
            ),
        )

        while True:
            if max_cycles is not None and max_cycles > 0 and cycle >= max_cycles:
                stopped_reason = "max_cycles_reached"
                break

            cycle += 1
            cycle_started_at = time.strftime("%Y-%m-%dT%H:%M:%S")
            _log_progress(f"cycle={cycle} preparing ids")

            race_targets = ["race_result"]
            if include_race_card:
                race_targets.append("race_card")

            race_prep_summary = prepare_netkeiba_ids_from_config(
                data_config,
                crawl_config,
                base_dir=base_dir,
                target_names=race_targets,
                start_date=start_date,
                end_date=end_date,
                limit=race_batch_size,
                include_completed=False,
                training_frame=training_frame,
                date_order=date_order,
                race_id_source=race_id_source,
                refresh=refresh,
                parse_only=parse_only,
            )
            race_prep_report = _extract_report(race_prep_summary, "race_ids")
            pending_race_ids = int(race_prep_report.get("row_count") or 0)

            cycle_report: dict[str, Any] = {
                "cycle": cycle,
                "started_at": cycle_started_at,
                "race_id_prep": race_prep_report,
            }

            if pending_race_ids > 0:
                _log_progress(f"cycle={cycle} crawling race_result batch={pending_race_ids}")
                race_result_summary = crawl_netkeiba_from_config(
                    crawl_config,
                    base_dir=base_dir,
                    target_filter="race_result",
                    override_limit=race_batch_size,
                    refresh=refresh,
                    parse_only=parse_only,
                    use_lock=False,
                )
                cycle_report["race_result"] = _extract_target(race_result_summary, "race_result")

                if include_race_card:
                    _log_progress(f"cycle={cycle} crawling race_card batch={pending_race_ids}")
                    race_card_summary = crawl_netkeiba_from_config(
                        crawl_config,
                        base_dir=base_dir,
                        target_filter="race_card",
                        override_limit=race_batch_size,
                        refresh=refresh,
                        parse_only=parse_only,
                        use_lock=False,
                    )
                    cycle_report["race_card"] = _extract_target(race_card_summary, "race_card")

            pedigree_prep_report: dict[str, Any] = {}
            pending_pedigree_ids = 0
            if include_pedigree:
                pedigree_prep_summary = prepare_netkeiba_ids_from_config(
                    data_config,
                    crawl_config,
                    base_dir=base_dir,
                    target_names=["pedigree"],
                    start_date=start_date,
                    end_date=end_date,
                    limit=pedigree_batch_size,
                    include_completed=False,
                    training_frame=training_frame,
                )
                pedigree_prep_report = _extract_report(pedigree_prep_summary, "horse_keys")
                pending_pedigree_ids = int(pedigree_prep_report.get("row_count") or 0)
                cycle_report["horse_key_prep"] = pedigree_prep_report

                if pending_pedigree_ids > 0:
                    _log_progress(f"cycle={cycle} crawling pedigree batch={pending_pedigree_ids}")
                    pedigree_summary = crawl_netkeiba_from_config(
                        crawl_config,
                        base_dir=base_dir,
                        target_filter="pedigree",
                        override_limit=pedigree_batch_size,
                        refresh=refresh,
                        parse_only=parse_only,
                        use_lock=False,
                    )
                    cycle_report["pedigree"] = _extract_target(pedigree_summary, "pedigree")

            post_cycle_failed = False
            if post_cycle_command:
                post_cycle_report = _run_post_cycle_command(
                    command=post_cycle_command,
                    base_dir=base_dir,
                    cycle=cycle,
                    manifest_path=manifest_path,
                )
                cycle_report["post_cycle_command"] = post_cycle_report
                post_cycle_failed = int(post_cycle_report.get("exit_code", 1)) != 0

            cycle_report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            cycle_reports.append(cycle_report)
            _write_json(
                manifest_path,
                _build_summary(
                    started_at=started_at,
                    finished_at=None,
                    start_date=start_date,
                    end_date=end_date,
                    date_order=date_order,
                    race_id_source=race_id_source,
                    race_batch_size=race_batch_size,
                    pedigree_batch_size=pedigree_batch_size,
                    include_race_card=include_race_card,
                    include_pedigree=include_pedigree,
                    max_cycles=max_cycles,
                    post_cycle_command=post_cycle_command,
                    stop_on_post_cycle_failure=stop_on_post_cycle_failure,
                    stopped_reason=stopped_reason,
                    cycle_reports=cycle_reports,
                ),
            )

            _log_progress(
                f"cycle={cycle} done pending_race_ids={pending_race_ids} pending_pedigree_ids={pending_pedigree_ids}"
            )

            if post_cycle_failed and stop_on_post_cycle_failure:
                stopped_reason = "post_cycle_command_failed"
                break

            if pending_race_ids == 0 and pending_pedigree_ids == 0:
                stopped_reason = "completed"
                break

        finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        if stopped_reason == "running":
            stopped_reason = "completed"
        summary = _build_summary(
            started_at=started_at,
            finished_at=finished_at,
            start_date=start_date,
            end_date=end_date,
            race_id_source=race_id_source,
            race_batch_size=race_batch_size,
            pedigree_batch_size=pedigree_batch_size,
            include_race_card=include_race_card,
            include_pedigree=include_pedigree,
            max_cycles=max_cycles,
            post_cycle_command=post_cycle_command,
            stop_on_post_cycle_failure=stop_on_post_cycle_failure,
            date_order=date_order,
            stopped_reason=stopped_reason,
            cycle_reports=cycle_reports,
        )
        _write_json(manifest_path, summary)
        return summary