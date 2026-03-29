from __future__ import annotations

import argparse
from datetime import date, datetime, timedelta
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.data.local_nankan_backfill import run_local_nankan_backfill_from_config


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[backfill-local-nankan {now}] {message}", flush=True)


def _apply_crawl_overrides(
    config: dict[str, object],
    *,
    delay_sec: float | None,
    timeout_sec: float | None,
    retry_count: int | None,
    retry_backoff_sec: float | None,
    overwrite: bool | None,
) -> dict[str, object]:
    output = dict(config)
    crawl_cfg = output.get("crawl")
    crawl_section = dict(crawl_cfg) if isinstance(crawl_cfg, dict) else output
    if delay_sec is not None:
        crawl_section["delay_sec"] = float(delay_sec)
    if timeout_sec is not None:
        crawl_section["timeout_sec"] = float(timeout_sec)
    if retry_count is not None:
        crawl_section["retry_count"] = int(retry_count)
    if retry_backoff_sec is not None:
        crawl_section["retry_backoff_sec"] = float(retry_backoff_sec)
    if overwrite is not None:
        crawl_section["overwrite"] = bool(overwrite)
    if isinstance(crawl_cfg, dict):
        output["crawl"] = crawl_section
    else:
        output = crawl_section
    return output


def _parse_iso_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _format_iso_date(value: date) -> str:
    return value.isoformat()


def _last_day_of_month(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    return (next_month - timedelta(days=1)).day


def _add_months(value: date, months: int) -> date:
    month_index = value.month - 1 + months
    year = value.year + month_index // 12
    month = month_index % 12 + 1
    day = min(value.day, _last_day_of_month(year, month))
    return date(year, month, day)


def _build_date_windows(start_date: str, end_date: str, chunk_months: int, date_order: str) -> list[tuple[str, str]]:
    if chunk_months <= 0:
        raise ValueError("chunk_months must be > 0")
    start_value = _parse_iso_date(start_date)
    end_value = _parse_iso_date(end_date)
    if end_value < start_value:
        raise ValueError(f"end_date must be >= start_date: {start_date} .. {end_date}")

    windows: list[tuple[str, str]] = []
    current_start = start_value
    while current_start <= end_value:
        next_start = _add_months(current_start, chunk_months)
        current_end = min(end_value, next_start - timedelta(days=1))
        windows.append((_format_iso_date(current_start), _format_iso_date(current_end)))
        current_start = current_end + timedelta(days=1)

    if str(date_order).strip().lower() == "desc":
        windows.reverse()
    return windows


def _with_window_suffix(path: Path, *, window_index: int, start_date: str, end_date: str) -> Path:
    suffix = f"_window{window_index:03d}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"
    return path.with_name(f"{path.stem}{suffix}{path.suffix}")


def _window_key(*, window_index: int, start_date: str, end_date: str) -> str:
    return f"{window_index}:{start_date}:{end_date}"


def _window_key_from_report(report: dict[str, Any]) -> str | None:
    window_index = report.get("window_index")
    date_window = report.get("date_window")
    if not isinstance(window_index, int) or not isinstance(date_window, dict):
        return None
    start_date = str(date_window.get("start") or "").strip()
    end_date = str(date_window.get("end") or "").strip()
    if not start_date or not end_date:
        return None
    return _window_key(window_index=window_index, start_date=start_date, end_date=end_date)


def _load_existing_window_reports(
    manifest_path: Path,
    *,
    start_date: str,
    end_date: str,
    date_order: str,
    chunk_months: int,
) -> list[dict[str, Any]]:
    if not manifest_path.exists():
        return []
    try:
        payload = read_json(manifest_path)
    except Exception:
        return []
    if not isinstance(payload, dict):
        return []
    payload_date_window = payload.get("date_window")
    if not isinstance(payload_date_window, dict):
        return []
    if str(payload_date_window.get("start") or "") != start_date:
        return []
    if str(payload_date_window.get("end") or "") != end_date:
        return []
    if str(payload.get("date_order") or "") != date_order:
        return []
    if int(payload.get("chunk_months") or 0) != int(chunk_months):
        return []
    window_reports = payload.get("window_reports")
    if not isinstance(window_reports, list):
        return []
    latest_by_key: dict[str, dict[str, Any]] = {}
    for report in window_reports:
        if not isinstance(report, dict):
            continue
        key = _window_key_from_report(report)
        if key is None:
            continue
        latest_by_key[key] = report
    return sorted(latest_by_key.values(), key=lambda report: int(report.get("window_index") or 0))


def _completed_window_keys(window_reports: list[dict[str, Any]]) -> set[str]:
    completed: set[str] = set()
    for report in window_reports:
        key = _window_key_from_report(report)
        if key is None:
            continue
        if str(report.get("status") or "") == "completed":
            completed.add(key)
    return completed


def _upsert_window_report(window_reports: list[dict[str, Any]], report: dict[str, Any]) -> list[dict[str, Any]]:
    next_reports = [item for item in window_reports if _window_key_from_report(item) != _window_key_from_report(report)]
    next_reports.append(report)
    return sorted(next_reports, key=lambda item: int(item.get("window_index") or 0))


def _aggregate_window_status(
    window_reports: list[dict[str, Any]],
    *,
    dry_run: bool,
    requested_window_count: int,
) -> tuple[str, str, str]:
    if dry_run:
        return "planned", "planned", "review_backfill_plan"
    statuses = [str(report.get("status") or "") for report in window_reports]
    completed_window_count = len(_completed_window_keys(window_reports))
    if requested_window_count > 0 and completed_window_count >= requested_window_count and statuses and all(status == "completed" for status in statuses):
        return "completed", "window_batch_completed", "continue_windowed_backfill_if_needed"
    if any(status in {"failed", "blocked", "not_ready"} for status in statuses):
        return "partial", "window_batch_partial", "inspect_failed_window"
    return "partial", "window_batch_partial", "continue_windowed_backfill"


def _aggregate_window_stop_reason(
    *,
    window_reports: list[dict[str, Any]],
    requested_window_count: int,
    dry_run: bool,
    default_stop_reason: str,
) -> str:
    if dry_run:
        return default_stop_reason
    if not window_reports:
        return "all_windows_already_completed" if requested_window_count > 0 else default_stop_reason
    completed_window_count = len(_completed_window_keys(window_reports))
    last_report = window_reports[-1] if window_reports else {}
    last_status = str(last_report.get("status") or "")
    last_stop_reason = str(last_report.get("stopped_reason") or "").strip()
    if requested_window_count > 0 and completed_window_count >= requested_window_count and last_status == "completed":
        return "completed_window_batch"
    if last_stop_reason:
        return f"window_{last_stop_reason}"
    return default_stop_reason


def _write_windowed_running_manifest(manifest_path: Path, aggregate: dict[str, Any], *, active_window: int | None, active_window_date_window: dict[str, str] | None, active_phase: str, stop_reason: str | None = None) -> None:
    payload = dict(aggregate)
    payload["finished_at"] = None
    payload["status"] = "planned" if str(payload.get("status") or "") == "planned" else "running"
    payload["current_phase"] = active_phase
    payload["active_window"] = active_window
    payload["active_window_date_window"] = active_window_date_window
    payload["completed_window_count"] = len(payload.get("window_reports", []))
    payload["stop_reason"] = stop_reason
    payload["last_updated_at"] = utc_now_iso()
    write_json(manifest_path, payload)


def _run_windowed_backfill(
    *,
    crawl_config: dict[str, Any],
    data_config: dict[str, Any] | None,
    seed_file: str | None,
    race_id_source: str,
    target_filter: str,
    start_date: str,
    end_date: str,
    date_order: str,
    limit: int | None,
    max_cycles: int,
    manifest_file: str,
    materialize_after_collect: bool,
    race_result_path: str | None,
    race_card_path: str | None,
    pedigree_path: str | None,
    materialize_output_file: str | None,
    materialize_manifest_file: str,
    dry_run: bool,
    chunk_months: int,
    max_date_windows: int | None,
    sleep_sec_between_windows: float,
) -> dict[str, Any]:
    manifest_path = ROOT / manifest_file if not Path(manifest_file).is_absolute() else Path(manifest_file)
    materialize_manifest_path = ROOT / materialize_manifest_file if not Path(materialize_manifest_file).is_absolute() else Path(materialize_manifest_file)
    materialize_output_path = None
    if materialize_output_file:
        materialize_output_path = ROOT / materialize_output_file if not Path(materialize_output_file).is_absolute() else Path(materialize_output_file)

    all_windows = _build_date_windows(start_date, end_date, chunk_months, date_order)
    existing_window_reports = _load_existing_window_reports(
        manifest_path,
        start_date=start_date,
        end_date=end_date,
        date_order=date_order,
        chunk_months=chunk_months,
    )
    completed_window_keys = _completed_window_keys(existing_window_reports)
    pending_windows = [
        (index, window_start, window_end)
        for index, (window_start, window_end) in enumerate(all_windows, start=1)
        if _window_key(window_index=index, start_date=window_start, end_date=window_end) not in completed_window_keys
    ]
    windows = pending_windows[: max_date_windows if max_date_windows is not None and max_date_windows > 0 else None]

    aggregate: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "running" if not dry_run else "planned",
        "current_phase": "window_batch_running" if not dry_run else "planned",
        "recommended_action": None,
        "manifest_file": str(manifest_file),
        "date_window": {"start": start_date, "end": end_date},
        "date_order": date_order,
        "chunk_months": int(chunk_months),
        "requested_window_count": int(len(all_windows)),
        "executed_window_count": int(len(windows)),
        "resume_completed_window_count": int(len(completed_window_keys)),
        "remaining_window_count": int(len(pending_windows)),
        "limit": limit,
        "max_cycles": max_cycles,
        "sleep_sec_between_windows": float(sleep_sec_between_windows),
        "materialize_after_collect": bool(materialize_after_collect),
        "window_reports": list(existing_window_reports),
    }
    _write_windowed_running_manifest(
        manifest_path,
        aggregate,
        active_window=windows[0][0] if windows else None,
        active_window_date_window={"start": windows[0][1], "end": windows[0][2]} if windows else None,
        active_phase="window_batch_running" if not dry_run else "planned",
    )

    progress = ProgressBar(total=max(len(windows), 1), prefix="[backfill-local-nankan-windowed]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message=f"windows={len(windows)} chunk_months={chunk_months} resumed={len(completed_window_keys)}")

    stop_reason = "completed_window_batch"
    if not windows and not dry_run and completed_window_keys:
        log_progress(f"all_requested_windows_already_completed count={len(completed_window_keys)}")
    for run_window_count, (window_index, window_start, window_end) in enumerate(windows, start=1):
        _write_windowed_running_manifest(
            manifest_path,
            aggregate,
            active_window=window_index,
            active_window_date_window={"start": window_start, "end": window_end},
            active_phase="running_window",
        )
        log_progress(f"resumed_window_progress completed={len(completed_window_keys)}/{len(all_windows)} running_window={window_index}/{len(all_windows)}")
        window_manifest_path = _with_window_suffix(manifest_path, window_index=window_index, start_date=window_start, end_date=window_end)
        window_materialize_manifest_path = _with_window_suffix(materialize_manifest_path, window_index=window_index, start_date=window_start, end_date=window_end)
        window_materialize_output = None
        if materialize_output_path is not None:
            window_materialize_output = _with_window_suffix(materialize_output_path, window_index=window_index, start_date=window_start, end_date=window_end)

        with Heartbeat("[backfill-local-nankan-windowed]", f"running window {window_index}/{len(all_windows)} {window_start}..{window_end}", logger=log_progress):
            summary = run_local_nankan_backfill_from_config(
                crawl_config,
                base_dir=ROOT,
                seed_file=seed_file,
                race_id_source=race_id_source,
                target_filter=target_filter,
                start_date=window_start,
                end_date=window_end,
                date_order=date_order,
                limit=limit,
                max_cycles=max_cycles,
                manifest_file=str(window_manifest_path),
                data_config=data_config,
                materialize_after_collect=materialize_after_collect,
                race_result_path=race_result_path,
                race_card_path=race_card_path,
                pedigree_path=pedigree_path,
                materialize_output_file=str(window_materialize_output) if window_materialize_output is not None else materialize_output_file,
                materialize_manifest_file=str(window_materialize_manifest_path),
                dry_run=dry_run,
                progress_logger=log_progress,
            )

        window_report = {
            "window_index": int(window_index),
            "date_window": {"start": window_start, "end": window_end},
            "manifest_file": str(window_manifest_path.relative_to(ROOT)) if window_manifest_path.is_relative_to(ROOT) else str(window_manifest_path),
            "status": summary.get("status"),
            "current_phase": summary.get("current_phase"),
            "recommended_action": summary.get("recommended_action"),
            "stopped_reason": summary.get("stopped_reason"),
            "summary": summary,
        }
        aggregate["window_reports"] = _upsert_window_report(aggregate["window_reports"], window_report)
        completed_window_keys = _completed_window_keys(aggregate["window_reports"])
        aggregate["resume_completed_window_count"] = int(len(completed_window_keys))
        aggregate["remaining_window_count"] = max(int(len(all_windows)) - int(len(completed_window_keys)), 0)
        progress.update(current=run_window_count, message=f"window={window_index}/{len(all_windows)} status={summary.get('status')} reason={summary.get('stopped_reason')}")
        _write_windowed_running_manifest(
            manifest_path,
            aggregate,
            active_window=window_index,
            active_window_date_window={"start": window_start, "end": window_end},
            active_phase="window_completed",
        )

        if not dry_run and str(summary.get("status") or "") in {"failed", "blocked", "not_ready"}:
            stop_reason = "window_failed"
            break
        next_pending_windows = [
            (idx, start, end)
            for idx, start, end in pending_windows
            if _window_key(window_index=idx, start_date=start, end_date=end) not in completed_window_keys and idx != window_index
        ]
        if not dry_run and sleep_sec_between_windows > 0 and next_pending_windows:
            next_window_index, next_window_start, next_window_end = next_pending_windows[0]
            log_progress(f"sleeping_between_windows seconds={sleep_sec_between_windows} next_window={next_window_index}/{len(all_windows)}")
            _write_windowed_running_manifest(
                manifest_path,
                aggregate,
                active_window=next_window_index,
                active_window_date_window={"start": next_window_start, "end": next_window_end},
                active_phase="sleeping_between_windows",
            )
            time.sleep(sleep_sec_between_windows)

    status, current_phase, recommended_action = _aggregate_window_status(
        aggregate["window_reports"],
        dry_run=dry_run,
        requested_window_count=len(all_windows),
    )
    stop_reason = _aggregate_window_stop_reason(
        window_reports=aggregate["window_reports"],
        requested_window_count=len(all_windows),
        dry_run=dry_run,
        default_stop_reason=stop_reason,
    )
    aggregate["finished_at"] = utc_now_iso()
    aggregate["status"] = status
    aggregate["current_phase"] = current_phase
    aggregate["recommended_action"] = recommended_action
    aggregate["stop_reason"] = stop_reason
    aggregate["highlights"] = [
        f"requested_window_count={aggregate['requested_window_count']}",
        f"executed_window_count={aggregate['executed_window_count']}",
        f"resume_completed_window_count={aggregate['resume_completed_window_count']}",
        f"remaining_window_count={aggregate['remaining_window_count']}",
        f"chunk_months={chunk_months}",
        f"sleep_sec_between_windows={sleep_sec_between_windows}",
    ]
    write_json(manifest_path, aggregate)
    progress.complete(message=f"windowed backfill status={status}")
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-config", default="configs/crawl_local_nankan_template.yaml")
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--race-id-source", choices=["seed_file", "race_list"], default="seed_file")
    parser.add_argument("--target", choices=["all", "race_result", "race_card", "pedigree"], default="all")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="desc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--delay-sec", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--retry-count", type=int, default=None)
    parser.add_argument("--retry-backoff-sec", type=float, default=None)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=None)
    parser.add_argument("--chunk-months", type=int, default=None)
    parser.add_argument("--max-date-windows", type=int, default=None)
    parser.add_argument("--sleep-sec-between-windows", type=float, default=0.0)
    parser.add_argument("--manifest-file", default="artifacts/reports/local_nankan_backfill_manifest.json")
    parser.add_argument("--materialize-after-collect", action="store_true")
    parser.add_argument("--race-result-path", default=None)
    parser.add_argument("--race-card-path", default=None)
    parser.add_argument("--pedigree-path", default=None)
    parser.add_argument("--materialize-output-file", default=None)
    parser.add_argument("--materialize-manifest-file", default="artifacts/reports/local_nankan_primary_materialize_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=3, prefix="[backfill-local-nankan]", logger=log_progress, min_interval_sec=0.0)
        crawl_config = _apply_crawl_overrides(
            load_yaml(ROOT / args.crawl_config),
            delay_sec=args.delay_sec,
            timeout_sec=args.timeout_sec,
            retry_count=args.retry_count,
            retry_backoff_sec=args.retry_backoff_sec,
            overwrite=args.overwrite,
        )
        data_config = load_yaml(ROOT / args.data_config) if args.data_config else None
        progress.start(message="config loaded")
        if args.chunk_months is not None:
            if not args.start_date or not args.end_date:
                raise ValueError("--chunk-months requires both --start-date and --end-date")
            if args.chunk_months <= 0:
                raise ValueError("--chunk-months must be > 0")
            if args.max_date_windows is not None and args.max_date_windows <= 0:
                raise ValueError("--max-date-windows must be > 0 when provided")
            with Heartbeat("[backfill-local-nankan]", "running windowed backfill plan", logger=log_progress):
                summary = _run_windowed_backfill(
                    crawl_config=crawl_config,
                    data_config=data_config,
                    seed_file=args.seed_file,
                    race_id_source=args.race_id_source,
                    target_filter=args.target,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    date_order=args.date_order,
                    limit=args.limit,
                    max_cycles=args.max_cycles,
                    manifest_file=args.manifest_file,
                    materialize_after_collect=args.materialize_after_collect,
                    race_result_path=args.race_result_path,
                    race_card_path=args.race_card_path,
                    pedigree_path=args.pedigree_path,
                    materialize_output_file=args.materialize_output_file,
                    materialize_manifest_file=args.materialize_manifest_file,
                    dry_run=args.dry_run,
                    chunk_months=args.chunk_months,
                    max_date_windows=args.max_date_windows,
                    sleep_sec_between_windows=args.sleep_sec_between_windows,
                )
        else:
            with Heartbeat("[backfill-local-nankan]", "running backfill plan", logger=log_progress):
                summary = run_local_nankan_backfill_from_config(
                    crawl_config,
                    base_dir=ROOT,
                    seed_file=args.seed_file,
                    race_id_source=args.race_id_source,
                    target_filter=args.target,
                    start_date=args.start_date,
                    end_date=args.end_date,
                    date_order=args.date_order,
                    limit=args.limit,
                    max_cycles=args.max_cycles,
                    manifest_file=args.manifest_file,
                    data_config=data_config,
                    materialize_after_collect=args.materialize_after_collect,
                    race_result_path=args.race_result_path,
                    race_card_path=args.race_card_path,
                    pedigree_path=args.pedigree_path,
                    materialize_output_file=args.materialize_output_file,
                    materialize_manifest_file=args.materialize_manifest_file,
                    dry_run=args.dry_run,
                    progress_logger=log_progress,
                )
        cycle_count = len(summary.get("cycle_reports", []))
        if cycle_count <= 0:
            cycle_count = len(summary.get("window_reports", []))
        progress.update(message=f"backfill manifest ready cycles={cycle_count}")
        print(
            "[backfill-local-nankan] "
            f"status={summary.get('status')} phase={summary.get('current_phase')} stopped_reason={summary.get('stopped_reason') or summary.get('stop_reason')} manifest={summary.get('manifest_file')}"
        )
        progress.complete(message="backfill planning completed")
        if str(summary.get("status")) in {"planned", "completed", "partial"}:
            return 0
        return 2
    except KeyboardInterrupt:
        print("[backfill-local-nankan] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[backfill-local-nankan] failed: {error}")
        return 1
    except Exception as error:
        print(f"[backfill-local-nankan] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())