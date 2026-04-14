from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from racing_ml.common.artifacts import display_path, utc_now_iso, write_json
from racing_ml.data.local_nankan_collect import collect_local_nankan_from_config
from racing_ml.data.local_nankan_id_prep import prepare_local_nankan_ids_from_config
from racing_ml.data.local_nankan_primary import materialize_local_nankan_primary_from_config


def _with_cycle_suffix(path: Path, *, cycle: int, total_cycles: int) -> Path:
    if total_cycles <= 1:
        return path
    return path.with_name(f"{path.stem}_cycle{cycle}{path.suffix}")


def _dict_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _prepare_pending_count(summary: dict[str, Any]) -> int:
    reports = summary.get("reports")
    if not isinstance(reports, list):
        return 0
    pending = 0
    for report in reports:
        if not isinstance(report, dict):
            continue
        pending += int(report.get("row_count") or 0)
    return pending


def _prepare_pending_counts_by_kind(summary: dict[str, Any]) -> dict[str, int]:
    reports = summary.get("reports")
    if not isinstance(reports, list):
        return {}
    counts: dict[str, int] = {}
    for report in reports:
        if not isinstance(report, dict):
            continue
        kind = str(report.get("kind") or "unknown")
        counts[kind] = counts.get(kind, 0) + int(report.get("row_count") or 0)
    return counts


def _collect_requested_total(summary: dict[str, Any]) -> int:
    targets = summary.get("targets")
    if not isinstance(targets, list):
        return 0
    requested = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        requested += int(target.get("requested_ids") or 0)
    return requested


def _collect_failure_total(summary: dict[str, Any]) -> int:
    targets = summary.get("targets")
    if not isinstance(targets, list):
        return 0
    failures = 0
    for target in targets:
        if not isinstance(target, dict):
            continue
        failures += int(target.get("failure_count") or 0)
    return failures


def _format_counts_by_kind(counts: dict[str, int]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{kind}={int(count)}" for kind, count in sorted(counts.items()))


def _build_running_summary(
    *,
    started_at: str,
    cycle_reports: list[dict[str, Any]],
    base_dir: Path,
    manifest_path: Path,
    start_date: str | None,
    end_date: str | None,
    date_order: str,
    limit: int | None,
    max_cycles: int | None,
    materialize_after_collect: bool,
    target_filter: str,
    active_cycle: int,
    active_phase: str,
) -> dict[str, Any]:
    latest_cycle = cycle_reports[-1] if cycle_reports else {}
    pending_id_count = int(latest_cycle.get("pending_id_count") or 0)
    raw_pending_counts_by_kind = latest_cycle.get("pending_counts_by_kind")
    pending_counts_by_kind: dict[str, int] = raw_pending_counts_by_kind if isinstance(raw_pending_counts_by_kind, dict) else {}
    collect_summary = _dict_payload(latest_cycle.get("collect_summary"))
    materialize_summary = _dict_payload(latest_cycle.get("materialize_summary"))
    collect_requested_total = _collect_requested_total(collect_summary)
    collect_failure_total = _collect_failure_total(collect_summary)
    highlights = [
        f"active_cycle={active_cycle}",
        f"completed_cycles={sum(1 for report in cycle_reports if report.get('finished_at'))}",
        f"active_phase={active_phase}",
        f"pending_work_item_count={pending_id_count}",
    ]
    for kind, count in sorted((str(key), int(value)) for key, value in pending_counts_by_kind.items()):
        highlights.append(f"pending_{kind}={count}")
    if collect_summary:
        highlights.append(f"collect_requested_total={collect_requested_total}")
        highlights.append(f"collect_failure_total={collect_failure_total}")
    if materialize_after_collect and materialize_summary:
        highlights.append(f"materialize_status={materialize_summary.get('status') or 'unknown'}")

    return {
        "started_at": started_at,
        "finished_at": None,
        "status": "running",
        "stopped_reason": None,
        "manifest_file": display_path(manifest_path, workspace_root=base_dir),
        "date_window": {"start": start_date, "end": end_date},
        "date_order": date_order,
        "limit": limit,
        "max_cycles": max_cycles,
        "target_filter": target_filter,
        "current_phase": active_phase,
        "recommended_action": None,
        "materialize_after_collect": bool(materialize_after_collect),
        "active_cycle": int(active_cycle),
        "highlights": highlights,
        "cycle_reports": cycle_reports,
        "last_updated_at": utc_now_iso(),
    }


def _build_summary(
    *,
    started_at: str,
    finished_at: str | None,
    stopped_reason: str,
    cycle_reports: list[dict[str, Any]],
    base_dir: Path,
    manifest_path: Path,
    start_date: str | None,
    end_date: str | None,
    date_order: str,
    limit: int | None,
    max_cycles: int | None,
    dry_run: bool,
    materialize_after_collect: bool,
    target_filter: str,
) -> dict[str, Any]:
    final_cycle = cycle_reports[-1] if cycle_reports else {}
    collect_summary: dict[str, Any] = {}
    materialize_summary: dict[str, Any] = {}
    for cycle_report in reversed(cycle_reports):
        candidate_collect = _dict_payload(cycle_report.get("collect_summary"))
        candidate_materialize = _dict_payload(cycle_report.get("materialize_summary"))
        if not collect_summary and candidate_collect:
            collect_summary = candidate_collect
        if not materialize_summary and candidate_materialize:
            materialize_summary = candidate_materialize
        if collect_summary and (materialize_summary or not materialize_after_collect):
            break
    collect_status = str(collect_summary.get("status") or "")
    materialize_status = str(materialize_summary.get("status") or "")
    collect_failures = _collect_failure_total(collect_summary)
    fully_exhausted = stopped_reason in {"pending_ids_exhausted", "requested_window_collected"}
    collected_anything = any(_dict_payload(report.get("collect_summary")) for report in cycle_reports)
    final_pending_id_count = int(final_cycle.get("pending_id_count") or 0)
    raw_final_pending_counts_by_kind = final_cycle.get("pending_counts_by_kind")
    final_pending_counts_by_kind: dict[str, int] = raw_final_pending_counts_by_kind if isinstance(raw_final_pending_counts_by_kind, dict) else {}

    if dry_run:
        status = "planned"
        current_phase = "planned"
        recommended_action = "review_backfill_plan"
    elif stopped_reason == "pending_ids_exhausted" and not collected_anything and final_pending_id_count == 0:
        status = "completed"
        current_phase = "no_pending_ids"
        recommended_action = "continue_windowed_backfill_if_needed"
    elif materialize_after_collect and materialize_status == "completed":
        status = "completed" if fully_exhausted and collect_failures == 0 else "partial"
        current_phase = "materialized_primary_raw"
        recommended_action = "run_local_preflight" if fully_exhausted and collect_failures == 0 else "rerun_local_backfill"
    elif collect_status == "completed":
        status = "completed" if fully_exhausted and collect_failures == 0 else "partial"
        current_phase = "crawl_completed"
        recommended_action = "run_local_materialize" if fully_exhausted and collect_failures == 0 else "rerun_local_backfill"
    elif collect_status == "partial":
        status = "partial"
        current_phase = "crawl_completed_with_failures"
        recommended_action = "inspect_local_crawl_failures"
    else:
        fallback_action = materialize_summary.get("recommended_action") if materialize_after_collect else collect_summary.get("recommended_action")
        status = "failed"
        current_phase = "crawl_failed"
        recommended_action = str(fallback_action or "inspect_local_crawl_failures")

    highlights = [
        f"completed_cycles={len(cycle_reports)}",
        f"dry_run={dry_run}",
        f"stopped_reason={stopped_reason}",
        f"final_pending_work_item_count={final_pending_id_count}",
    ]
    for kind, count in sorted((str(key), int(value)) for key, value in final_pending_counts_by_kind.items()):
        highlights.append(f"final_pending_{kind}={count}")
    if materialize_after_collect:
        highlights.append(f"materialize_status={materialize_status or 'not_run'}")
    else:
        highlights.append(f"collect_status={collect_status or 'unknown'}")
    highlights.append(f"collect_failure_total={collect_failures}")

    return {
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "stopped_reason": stopped_reason,
        "manifest_file": display_path(manifest_path, workspace_root=base_dir),
        "date_window": {"start": start_date, "end": end_date},
        "date_order": date_order,
        "limit": limit,
        "max_cycles": max_cycles,
        "target_filter": target_filter,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "materialize_after_collect": bool(materialize_after_collect),
        "highlights": highlights,
        "cycle_reports": cycle_reports,
    }


def run_local_nankan_backfill_from_config(
    crawl_config: dict[str, Any],
    *,
    base_dir: Path,
    seed_file: str | Path | None = None,
    race_id_source: str | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    date_order: str = "desc",
    limit: int | None = None,
    max_cycles: int | None = None,
    target_filter: str = "all",
    manifest_file: str | Path = "artifacts/reports/local_nankan_backfill_manifest.json",
    data_config: dict[str, Any] | None = None,
    materialize_after_collect: bool = False,
    race_result_path: str | Path | None = None,
    race_card_path: str | Path | None = None,
    pedigree_path: str | Path | None = None,
    materialize_output_file: str | Path | None = None,
    materialize_manifest_file: str | Path = "artifacts/reports/local_nankan_primary_materialize_manifest.json",
    dry_run: bool = False,
    progress_logger: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(manifest_file)
    if not manifest_path.is_absolute():
        manifest_path = base_dir / manifest_path

    started_at = utc_now_iso()
    cycle_reports: list[dict[str, Any]] = []
    cycles_to_run = max_cycles if max_cycles and max_cycles > 0 else 1
    stopped_reason = "cycle_limit_reached"
    materialize_manifest_path = Path(materialize_manifest_file)
    if not materialize_manifest_path.is_absolute():
        materialize_manifest_path = base_dir / materialize_manifest_path
    materialize_output_path = Path(materialize_output_file) if materialize_output_file is not None else None
    if materialize_output_path is not None and not materialize_output_path.is_absolute():
        materialize_output_path = base_dir / materialize_output_path

    write_json(
        manifest_path,
        _build_running_summary(
            started_at=started_at,
            cycle_reports=cycle_reports,
            base_dir=base_dir,
            manifest_path=manifest_path,
            start_date=start_date,
            end_date=end_date,
            date_order=date_order,
            limit=limit,
            max_cycles=max_cycles,
            materialize_after_collect=materialize_after_collect,
            target_filter=target_filter,
            active_cycle=1,
            active_phase="prepare_ids",
        ),
    )

    for cycle in range(1, cycles_to_run + 1):
        prepare_summary = prepare_local_nankan_ids_from_config(
            crawl_config,
            base_dir=base_dir,
            seed_file=seed_file,
            target_filter=target_filter,
            start_date=start_date,
            end_date=end_date,
            date_order=date_order,
            limit=limit,
            include_completed=False,
            race_id_source=race_id_source,
        )
        pending_id_count = _prepare_pending_count(prepare_summary)
        pending_counts_by_kind = _prepare_pending_counts_by_kind(prepare_summary)
        cycle_report: dict[str, Any] = {
            "cycle": cycle,
            "started_at": utc_now_iso(),
            "prepare_summary": prepare_summary,
            "pending_id_count": int(pending_id_count),
            "pending_counts_by_kind": pending_counts_by_kind,
        }
        cycle_reports.append(cycle_report)
        if progress_logger is not None:
            progress_logger(
                f"cycle={cycle}/{cycles_to_run} phase=prepare_ids_completed pending_work_item_count={pending_id_count} pending_breakdown={_format_counts_by_kind(pending_counts_by_kind)}"
            )
        write_json(
            manifest_path,
            _build_running_summary(
                started_at=started_at,
                cycle_reports=cycle_reports,
                base_dir=base_dir,
                manifest_path=manifest_path,
                start_date=start_date,
                end_date=end_date,
                date_order=date_order,
                limit=limit,
                max_cycles=max_cycles,
                materialize_after_collect=materialize_after_collect,
                target_filter=target_filter,
                active_cycle=cycle,
                active_phase="prepare_ids_completed",
            ),
        )
        if pending_id_count <= 0:
            cycle_report["finished_at"] = utc_now_iso()
            if progress_logger is not None:
                progress_logger(f"cycle={cycle}/{cycles_to_run} phase=prepare_ids_completed pending_work_item_count=0 stop=pending_ids_exhausted")
            stopped_reason = "pending_ids_exhausted"
            break

        collect_summary = collect_local_nankan_from_config(
            crawl_config,
            base_dir=base_dir,
            target_filter=target_filter,
            override_limit=limit,
            dry_run=dry_run,
        )
        cycle_report["collect_summary"] = collect_summary
        if progress_logger is not None:
            progress_logger(
                f"cycle={cycle}/{cycles_to_run} phase=collect_completed requested_total={_collect_requested_total(collect_summary)} failure_total={_collect_failure_total(collect_summary)} status={collect_summary.get('status')}"
            )
        write_json(
            manifest_path,
            _build_running_summary(
                started_at=started_at,
                cycle_reports=cycle_reports,
                base_dir=base_dir,
                manifest_path=manifest_path,
                start_date=start_date,
                end_date=end_date,
                date_order=date_order,
                limit=limit,
                max_cycles=max_cycles,
                materialize_after_collect=materialize_after_collect,
                target_filter=target_filter,
                active_cycle=cycle,
                active_phase="collect_completed",
            ),
        )
        if materialize_after_collect:
            if data_config is None:
                raise ValueError("data_config is required when materialize_after_collect is enabled")
            cycle_materialize_manifest = _with_cycle_suffix(materialize_manifest_path, cycle=cycle, total_cycles=cycles_to_run)
            materialize_summary = materialize_local_nankan_primary_from_config(
                data_config,
                base_dir=base_dir,
                race_result_path=race_result_path,
                race_card_path=race_card_path,
                pedigree_path=pedigree_path,
                output_file=materialize_output_path,
                manifest_file=cycle_materialize_manifest,
                dry_run=dry_run,
            )
            cycle_report["materialize_summary"] = materialize_summary
            if progress_logger is not None:
                progress_logger(
                    f"cycle={cycle}/{cycles_to_run} phase=materialize_completed status={materialize_summary.get('status')} row_count={materialize_summary.get('row_count')}"
                )
            write_json(
                manifest_path,
                _build_running_summary(
                    started_at=started_at,
                    cycle_reports=cycle_reports,
                    base_dir=base_dir,
                    manifest_path=manifest_path,
                    start_date=start_date,
                    end_date=end_date,
                    date_order=date_order,
                    limit=limit,
                    max_cycles=max_cycles,
                    materialize_after_collect=materialize_after_collect,
                    target_filter=target_filter,
                    active_cycle=cycle,
                    active_phase="materialize_completed",
                ),
            )
        cycle_report["finished_at"] = utc_now_iso()
        collect_failures = _collect_failure_total(collect_summary)
        if dry_run:
            stopped_reason = "completed_plan"
            break

        if collect_failures == 0:
            if limit is None:
                stopped_reason = "requested_window_collected"
                if progress_logger is not None:
                    progress_logger(f"cycle={cycle}/{cycles_to_run} phase=completed stop=requested_window_collected")
                break
            if pending_id_count < int(limit):
                stopped_reason = "requested_window_collected"
                if progress_logger is not None:
                    progress_logger(f"cycle={cycle}/{cycles_to_run} phase=completed stop=requested_window_collected")
                break

    summary = _build_summary(
        started_at=started_at,
        finished_at=utc_now_iso(),
        stopped_reason=stopped_reason,
        cycle_reports=cycle_reports,
        base_dir=base_dir,
        manifest_path=manifest_path,
        start_date=start_date,
        end_date=end_date,
        date_order=date_order,
        limit=limit,
        max_cycles=max_cycles,
        dry_run=dry_run,
        materialize_after_collect=materialize_after_collect,
        target_filter=target_filter,
    )
    write_json(manifest_path, summary)
    return summary