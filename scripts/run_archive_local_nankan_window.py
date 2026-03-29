from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import traceback
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, utc_now_iso, write_csv_file, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_CRAWL_CONFIG = "configs/crawl_local_nankan_template.yaml"
DEFAULT_DATA_CONFIG = "configs/data_local_nankan.yaml"
DEFAULT_AGGREGATE_MANIFEST = "artifacts/reports/local_nankan_backfill_20y_windowed.json"
DEFAULT_ARCHIVE_ROOT = "artifacts/archives/local_nankan"
DEFAULT_ARCHIVE_INDEX = "artifacts/archives/local_nankan/archive_index.json"
DEFAULT_RUN_MANIFEST = "artifacts/reports/local_nankan_archive_run.json"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[archive-local-nankan {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _dict_payload(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _list_payload(value: object) -> list[Any]:
    return value if isinstance(value, list) else []


def _normalize_key(report: dict[str, Any]) -> str | None:
    window_index = report.get("window_index")
    date_window = report.get("date_window")
    if not isinstance(window_index, int) or not isinstance(date_window, dict):
        return None
    start_date = str(date_window.get("start") or "").strip()
    end_date = str(date_window.get("end") or "").strip()
    if not start_date or not end_date:
        return None
    return f"{window_index}:{start_date}:{end_date}"


def _fingerprint(report: dict[str, Any]) -> str:
    return json.dumps(report, ensure_ascii=False, sort_keys=True)


def _window_slug(*, window_index: int, start_date: str, end_date: str) -> str:
    return f"window{window_index:03d}_{start_date.replace('-', '')}_{end_date.replace('-', '')}"


def _load_window_map(manifest_path: Path) -> dict[str, dict[str, Any]]:
    if not manifest_path.exists():
        return {}
    payload = read_json(manifest_path)
    if not isinstance(payload, dict):
        return {}
    output: dict[str, dict[str, Any]] = {}
    for report in _list_payload(payload.get("window_reports")):
        if not isinstance(report, dict):
            continue
        key = _normalize_key(report)
        if key is None:
            continue
        output[key] = report
    return output


def _changed_window_reports(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    changed: list[dict[str, Any]] = []
    for key, report in after.items():
        previous = before.get(key)
        if previous is None or _fingerprint(previous) != _fingerprint(report):
            changed.append(report)
    return sorted(changed, key=lambda item: int(item.get("window_index") or 0))


def _target_output_paths(crawl_config_path: Path) -> dict[str, Path]:
    crawl_config = load_yaml(crawl_config_path)
    crawl_cfg = crawl_config.get("crawl") if isinstance(crawl_config.get("crawl"), dict) else crawl_config
    targets = crawl_cfg.get("targets") if isinstance(crawl_cfg, dict) else {}
    if not isinstance(targets, dict):
        return {}
    output: dict[str, Path] = {}
    for target_name in ("race_result", "race_card", "pedigree"):
        target_cfg = targets.get(target_name)
        if not isinstance(target_cfg, dict):
            continue
        output_file = target_cfg.get("output_file")
        if not output_file:
            continue
        output[target_name] = _resolve_path(str(output_file))
    return output


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame()


def _slice_csv_by_date(source_path: Path, *, start_date: str, end_date: str) -> pd.DataFrame:
    if not source_path.exists():
        return _empty_frame()
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    chunks: list[pd.DataFrame] = []
    try:
        iterator = pd.read_csv(source_path, low_memory=False, chunksize=100000)
    except pd.errors.EmptyDataError:
        return _empty_frame()

    for chunk in iterator:
        if "date" not in chunk.columns:
            return _empty_frame()
        date_series = pd.to_datetime(chunk["date"], errors="coerce")
        mask = date_series.ge(start_ts) & date_series.le(end_ts)
        filtered = chunk.loc[mask].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return _empty_frame()
    return pd.concat(chunks, ignore_index=True, sort=False)


def _slice_csv_by_keys(source_path: Path, *, key_column: str, keys: set[str]) -> pd.DataFrame:
    if not source_path.exists() or not keys:
        return _empty_frame()
    chunks: list[pd.DataFrame] = []
    try:
        iterator = pd.read_csv(source_path, low_memory=False, chunksize=100000)
    except pd.errors.EmptyDataError:
        return _empty_frame()

    for chunk in iterator:
        if key_column not in chunk.columns:
            return _empty_frame()
        normalized = chunk[key_column].astype("string").fillna("").str.strip()
        filtered = chunk.loc[normalized.isin(keys)].copy()
        if not filtered.empty:
            chunks.append(filtered)
    if not chunks:
        return _empty_frame()
    return pd.concat(chunks, ignore_index=True, sort=False)


def _horse_keys_from_frames(*frames: pd.DataFrame) -> set[str]:
    keys: set[str] = set()
    for frame in frames:
        if frame.empty or "horse_key" not in frame.columns:
            continue
        for value in frame["horse_key"].astype("string").fillna("").str.strip().tolist():
            if value:
                keys.add(str(value))
    return keys


def _write_window_archive(
    *,
    report: dict[str, Any],
    crawl_config_path: Path,
    archive_root: Path,
    archive_index_path: Path,
    dry_run: bool,
) -> dict[str, Any]:
    window_index = int(report.get("window_index") or 0)
    date_window = _dict_payload(report.get("date_window"))
    start_date = str(date_window.get("start") or "").strip()
    end_date = str(date_window.get("end") or "").strip()
    if window_index <= 0 or not start_date or not end_date:
        raise ValueError("window report is missing window_index or date_window")

    slug = _window_slug(window_index=window_index, start_date=start_date, end_date=end_date)
    window_dir = archive_root / slug
    data_dir = window_dir / "data"
    manifest_dir = window_dir / "manifests"
    tarball_path = archive_root / "tarballs" / f"{slug}.tar.gz"
    backfill_manifest_path = _resolve_path(str(report.get("manifest_file") or ""))
    if not backfill_manifest_path.exists():
        raise FileNotFoundError(f"window manifest not found: {backfill_manifest_path}")
    backfill_payload = read_json(backfill_manifest_path)
    if not isinstance(backfill_payload, dict):
        raise ValueError(f"window manifest is not a JSON object: {backfill_manifest_path}")

    materialize_manifest_path: Path | None = None
    for cycle_report in _list_payload(backfill_payload.get("cycle_reports")):
        if not isinstance(cycle_report, dict):
            continue
        materialize_summary = _dict_payload(cycle_report.get("materialize_summary"))
        manifest_file = materialize_summary.get("manifest_file")
        if isinstance(manifest_file, str) and manifest_file.strip():
            candidate = _resolve_path(manifest_file)
            if candidate.exists():
                materialize_manifest_path = candidate
                break

    target_output_paths = _target_output_paths(crawl_config_path)
    result_slice = _slice_csv_by_date(target_output_paths.get("race_result", ROOT / "missing.csv"), start_date=start_date, end_date=end_date)
    card_slice = _slice_csv_by_date(target_output_paths.get("race_card", ROOT / "missing.csv"), start_date=start_date, end_date=end_date)
    horse_keys = _horse_keys_from_frames(result_slice, card_slice)
    pedigree_slice = _slice_csv_by_keys(target_output_paths.get("pedigree", ROOT / "missing.csv"), key_column="horse_key", keys=horse_keys)

    if not dry_run:
        data_dir.mkdir(parents=True, exist_ok=True)
        manifest_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_dir / "backfill_manifest.json", backfill_payload)
        if materialize_manifest_path is not None:
            write_json(manifest_dir / "materialize_manifest.json", read_json(materialize_manifest_path))
        if not result_slice.empty:
            write_csv_file(data_dir / "race_result_window.csv", result_slice, index=False, label="archive race_result slice")
        if not card_slice.empty:
            write_csv_file(data_dir / "race_card_window.csv", card_slice, index=False, label="archive race_card slice")
        if not pedigree_slice.empty:
            write_csv_file(data_dir / "pedigree_window.csv", pedigree_slice, index=False, label="archive pedigree slice")

    archive_manifest: dict[str, Any] = {
        "created_at": utc_now_iso(),
        "status": "planned" if dry_run else "completed",
        "window_index": window_index,
        "window_slug": slug,
        "date_window": {"start": start_date, "end": end_date},
        "backfill_status": report.get("status"),
        "backfill_phase": report.get("current_phase"),
        "backfill_recommended_action": report.get("recommended_action"),
        "source_manifests": {
            "aggregate_window_manifest": artifact_display_path(backfill_manifest_path, workspace_root=ROOT),
            "materialize_manifest": artifact_display_path(materialize_manifest_path, workspace_root=ROOT) if materialize_manifest_path is not None else None,
        },
        "archive_paths": {
            "archive_dir": artifact_display_path(window_dir, workspace_root=ROOT),
            "tarball": artifact_display_path(tarball_path, workspace_root=ROOT),
            "archive_index": artifact_display_path(archive_index_path, workspace_root=ROOT),
        },
        "slice_counts": {
            "race_result_rows": int(len(result_slice)),
            "race_card_rows": int(len(card_slice)),
            "pedigree_rows": int(len(pedigree_slice)),
            "horse_key_count": int(len(horse_keys)),
        },
        "git_commit_targets": [
            artifact_display_path(window_dir, workspace_root=ROOT),
            artifact_display_path(tarball_path, workspace_root=ROOT),
            artifact_display_path(archive_index_path, workspace_root=ROOT),
        ],
    }

    if not dry_run:
        write_json(window_dir / "window_archive_manifest.json", archive_manifest)
        tarball_path.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(window_dir, arcname=window_dir.name)
        archive_manifest["tarball_size_bytes"] = int(tarball_path.stat().st_size)
        write_json(window_dir / "window_archive_manifest.json", archive_manifest)

    return archive_manifest


def _load_archive_index(index_path: Path) -> dict[str, Any]:
    if not index_path.exists():
        return {
            "created_at": utc_now_iso(),
            "updated_at": None,
            "status": "running",
            "archives": [],
        }
    payload = read_json(index_path)
    return payload if isinstance(payload, dict) else {
        "created_at": utc_now_iso(),
        "updated_at": None,
        "status": "running",
        "archives": [],
    }


def _upsert_archive_index(index_payload: dict[str, Any], archive_reports: list[dict[str, Any]]) -> dict[str, Any]:
    existing = {}
    for report in _list_payload(index_payload.get("archives")):
        if not isinstance(report, dict):
            continue
        slug = str(report.get("window_slug") or "").strip()
        if slug:
            existing[slug] = report
    for report in archive_reports:
        slug = str(report.get("window_slug") or "").strip()
        if slug:
            existing[slug] = report
    updated = sorted(existing.values(), key=lambda item: int(item.get("window_index") or 0))
    return {
        "created_at": index_payload.get("created_at") or utc_now_iso(),
        "updated_at": utc_now_iso(),
        "status": "completed",
        "archive_count": int(len(updated)),
        "archives": updated,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--crawl-config", default=DEFAULT_CRAWL_CONFIG)
    parser.add_argument("--data-config", default=DEFAULT_DATA_CONFIG)
    parser.add_argument("--seed-file", default=None)
    parser.add_argument("--race-id-source", choices=["seed_file", "race_list"], default="race_list")
    parser.add_argument("--target", choices=["all", "race_result", "race_card", "pedigree"], default="race_result")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--date-order", choices=["asc", "desc"], default="asc")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-cycles", type=int, default=1)
    parser.add_argument("--chunk-months", type=int, default=6)
    parser.add_argument("--max-date-windows", type=int, default=1)
    parser.add_argument("--sleep-sec-between-windows", type=float, default=0.0)
    parser.add_argument("--delay-sec", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument("--retry-count", type=int, default=None)
    parser.add_argument("--retry-backoff-sec", type=float, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-overwrite", action="store_true")
    parser.add_argument("--manifest-file", default=DEFAULT_AGGREGATE_MANIFEST)
    parser.add_argument("--materialize-after-collect", action="store_true")
    parser.add_argument("--archive-root", default=DEFAULT_ARCHIVE_ROOT)
    parser.add_argument("--archive-index", default=DEFAULT_ARCHIVE_INDEX)
    parser.add_argument("--run-manifest-output", default=DEFAULT_RUN_MANIFEST)
    parser.add_argument("--skip-backfill-run", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    crawl_config_path = _resolve_path(args.crawl_config)
    aggregate_manifest_path = _resolve_path(args.manifest_file)
    archive_root = _resolve_path(args.archive_root)
    archive_index_path = _resolve_path(args.archive_index)
    run_manifest_path = _resolve_path(args.run_manifest_output)

    run_payload: dict[str, Any] = {
        "started_at": utc_now_iso(),
        "finished_at": None,
        "status": "planned" if args.dry_run else "running",
        "current_phase": "planned" if args.dry_run else "running_backfill",
        "recommended_action": "run_archive_local_nankan_window" if args.dry_run else None,
        "configs": {
            "crawl_config": args.crawl_config,
            "data_config": args.data_config,
            "seed_file": args.seed_file,
            "race_id_source": args.race_id_source,
            "target": args.target,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "date_order": args.date_order,
            "limit": args.limit,
            "max_cycles": int(args.max_cycles),
            "chunk_months": int(args.chunk_months),
            "max_date_windows": int(args.max_date_windows),
            "sleep_sec_between_windows": float(args.sleep_sec_between_windows),
            "materialize_after_collect": bool(args.materialize_after_collect),
            "skip_backfill_run": bool(args.skip_backfill_run),
            "dry_run": bool(args.dry_run),
        },
        "artifacts": {
            "aggregate_manifest": artifact_display_path(aggregate_manifest_path, workspace_root=ROOT),
            "archive_root": artifact_display_path(archive_root, workspace_root=ROOT),
            "archive_index": artifact_display_path(archive_index_path, workspace_root=ROOT),
            "run_manifest": artifact_display_path(run_manifest_path, workspace_root=ROOT),
        },
        "archive_reports": [],
    }

    try:
        artifact_ensure_output_file_path(run_manifest_path, label="run manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(aggregate_manifest_path, label="aggregate manifest output", workspace_root=ROOT)
        artifact_ensure_output_file_path(archive_index_path, label="archive index output", workspace_root=ROOT)
        write_json(run_manifest_path, run_payload)

        progress = ProgressBar(total=4, prefix="[archive-local-nankan]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"target={args.target} chunk_months={args.chunk_months} max_date_windows={args.max_date_windows}")

        before_window_map = _load_window_map(aggregate_manifest_path)
        if not args.skip_backfill_run:
            backfill_command = [
                sys.executable,
                str(ROOT / "scripts/run_backfill_local_nankan.py"),
                "--crawl-config",
                args.crawl_config,
                "--data-config",
                args.data_config,
                "--race-id-source",
                args.race_id_source,
                "--target",
                args.target,
                "--start-date",
                args.start_date,
                "--end-date",
                args.end_date,
                "--date-order",
                args.date_order,
                "--max-cycles",
                str(args.max_cycles),
                "--chunk-months",
                str(args.chunk_months),
                "--max-date-windows",
                str(args.max_date_windows),
                "--sleep-sec-between-windows",
                str(args.sleep_sec_between_windows),
                "--manifest-file",
                args.manifest_file,
            ]
            if args.seed_file:
                backfill_command.extend(["--seed-file", args.seed_file])
            if args.limit is not None:
                backfill_command.extend(["--limit", str(args.limit)])
            if args.delay_sec is not None:
                backfill_command.extend(["--delay-sec", str(args.delay_sec)])
            if args.timeout_sec is not None:
                backfill_command.extend(["--timeout-sec", str(args.timeout_sec)])
            if args.retry_count is not None:
                backfill_command.extend(["--retry-count", str(args.retry_count)])
            if args.retry_backoff_sec is not None:
                backfill_command.extend(["--retry-backoff-sec", str(args.retry_backoff_sec)])
            if args.materialize_after_collect:
                backfill_command.append("--materialize-after-collect")
            if args.overwrite and not args.no_overwrite:
                backfill_command.append("--overwrite")
            if args.no_overwrite:
                backfill_command.append("--no-overwrite")
            if args.dry_run:
                backfill_command.append("--dry-run")

            run_payload["backfill_command"] = backfill_command
            write_json(run_manifest_path, run_payload)
            with Heartbeat("[archive-local-nankan]", "running backfill before archive", logger=log_progress):
                result = subprocess.run(backfill_command, cwd=ROOT, check=False)
            run_payload["backfill_exit_code"] = int(result.returncode)
            if int(result.returncode) != 0:
                run_payload["status"] = "backfill_failed"
                run_payload["current_phase"] = "running_backfill"
                run_payload["recommended_action"] = "inspect_local_nankan_backfill"
                run_payload["finished_at"] = utc_now_iso()
                write_json(run_manifest_path, run_payload)
                return int(result.returncode) or 1
            progress.update(current=1, message="backfill completed")
        else:
            progress.update(current=1, message="skipped backfill run")

        after_window_map = _load_window_map(aggregate_manifest_path)
        changed_reports = _changed_window_reports(before_window_map, after_window_map)
        if not changed_reports:
            run_payload["status"] = "completed"
            run_payload["current_phase"] = "no_new_windows"
            run_payload["recommended_action"] = "rerun_archive_after_next_window"
            run_payload["finished_at"] = utc_now_iso()
            write_json(run_manifest_path, run_payload)
            progress.complete(message="no new or updated windows to archive")
            return 0
        progress.update(current=2, message=f"changed_windows={len(changed_reports)}")

        archive_reports: list[dict[str, Any]] = []
        archive_progress = ProgressBar(total=max(len(changed_reports), 1), prefix="[archive-local-nankan windows]", logger=log_progress, min_interval_sec=0.0)
        archive_progress.start(message="building window archives")
        for index, report in enumerate(changed_reports, start=1):
            archive_report = _write_window_archive(
                report=report,
                crawl_config_path=crawl_config_path,
                archive_root=archive_root,
                archive_index_path=archive_index_path,
                dry_run=args.dry_run,
            )
            archive_reports.append(archive_report)
            archive_progress.update(current=index, message=f"archived={archive_report.get('window_slug')} status={archive_report.get('status')}")
        archive_progress.complete(message="window archives ready")
        run_payload["archive_reports"] = archive_reports
        progress.update(current=3, message=f"archive_reports={len(archive_reports)}")

        if not args.dry_run:
            index_payload = _load_archive_index(archive_index_path)
            updated_index = _upsert_archive_index(index_payload, archive_reports)
            write_json(archive_index_path, updated_index)
        else:
            updated_index = _upsert_archive_index(_load_archive_index(archive_index_path), archive_reports)
        run_payload["archive_index_preview"] = {
            "archive_count": updated_index.get("archive_count"),
            "updated_at": updated_index.get("updated_at"),
        }
        run_payload["status"] = "planned" if args.dry_run else "completed"
        run_payload["current_phase"] = "planned" if args.dry_run else "archived_windows"
        run_payload["recommended_action"] = "review_archive_plan" if args.dry_run else "commit_archive_to_github"
        run_payload["finished_at"] = utc_now_iso()
        write_json(run_manifest_path, run_payload)
        progress.complete(message=f"archive run status={run_payload.get('status')}")
        print(f"[archive-local-nankan] run manifest: {artifact_display_path(run_manifest_path, workspace_root=ROOT)}", flush=True)
        return 0
    except KeyboardInterrupt:
        run_payload["status"] = "interrupted"
        run_payload["current_phase"] = run_payload.get("current_phase") or "interrupted"
        run_payload["recommended_action"] = "rerun_archive_local_nankan_window"
        run_payload["finished_at"] = utc_now_iso()
        write_json(run_manifest_path, run_payload)
        print("[archive-local-nankan] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        run_payload["status"] = "failed"
        run_payload["current_phase"] = run_payload.get("current_phase") or "failed"
        run_payload["recommended_action"] = "inspect_archive_local_nankan_inputs"
        run_payload["error_message"] = str(error)
        run_payload["finished_at"] = utc_now_iso()
        write_json(run_manifest_path, run_payload)
        print(f"[archive-local-nankan] failed: {error}", flush=True)
        return 1
    except Exception as error:
        run_payload["status"] = "failed"
        run_payload["current_phase"] = run_payload.get("current_phase") or "failed"
        run_payload["recommended_action"] = "inspect_archive_local_nankan_traceback"
        run_payload["error_message"] = str(error)
        run_payload["finished_at"] = utc_now_iso()
        write_json(run_manifest_path, run_payload)
        print(f"[archive-local-nankan] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())