from __future__ import annotations

import argparse
from bs4 import BeautifulSoup
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import json
import os
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

from racing_ml.common.artifacts import read_json, write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.local_nankan_collect import _classify_snapshot_timing, _extract_race_metadata


PROVENANCE_COLUMNS = [
    "post_time",
    "scheduled_post_at",
    "card_source_url",
    "card_fetch_mode",
    "card_snapshot_at",
    "card_snapshot_relation",
    "odds_source_url",
    "odds_fetch_mode",
    "odds_snapshot_at",
    "odds_snapshot_relation",
]


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[repair-local-racecard-prov {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _load_cached_provenance(*, output_path: Path, source_url: str) -> dict[str, Any]:
    metadata_path = output_path.with_name(f"{output_path.name}.meta.json")
    payload: dict[str, Any] = {}
    if metadata_path.exists():
        loaded = read_json(metadata_path)
        if isinstance(loaded, dict):
            payload = dict(loaded)
    snapshot_at = payload.get("fetched_at")
    if not snapshot_at and output_path.exists():
        snapshot_at = datetime.fromtimestamp(output_path.stat().st_mtime, tz=timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    return {
        "source_url": str(payload.get("source_url") or source_url),
        "fetch_mode": "cache_manifest" if payload.get("fetched_at") else "cache_legacy",
        "snapshot_at": snapshot_at,
        "metadata_path": metadata_path.as_posix(),
    }


def _repair_single_race(args: tuple[str, str, str]) -> tuple[str, dict[str, Any] | None, str | None]:
    race_id, race_card_dir_text, race_card_odds_dir_text = args
    race_card_dir = Path(race_card_dir_text)
    race_card_odds_dir = Path(race_card_odds_dir_text)
    html_path = race_card_dir / f"{race_id}.html"
    odds_path = race_card_odds_dir / f"{race_id}.js"
    if not html_path.exists():
        return race_id, None, "race_card_html_missing"
    try:
        html = html_path.read_text(encoding="utf-8")
        card_url = f"https://www.nankankeiba.com/syousai/{race_id}.do"
        odds_url = f"https://www.nankankeiba.com/oddsJS/{race_id}.do"
        card_provenance = _load_cached_provenance(output_path=html_path, source_url=card_url)
        odds_provenance = _load_cached_provenance(output_path=odds_path, source_url=odds_url) if odds_path.exists() else {"source_url": odds_url, "fetch_mode": "missing", "snapshot_at": None}
        metadata = _extract_race_metadata(BeautifulSoup(html, "html.parser"), race_id)
        scheduled_post_at = metadata.get("scheduled_post_at")
        return race_id, {
            "race_id": race_id,
            "post_time": metadata.get("post_time"),
            "scheduled_post_at": scheduled_post_at,
            "card_source_url": card_provenance.get("source_url"),
            "card_fetch_mode": card_provenance.get("fetch_mode"),
            "card_snapshot_at": card_provenance.get("snapshot_at"),
            "card_snapshot_relation": _classify_snapshot_timing(card_provenance.get("snapshot_at"), scheduled_post_at),
            "odds_source_url": odds_provenance.get("source_url"),
            "odds_fetch_mode": odds_provenance.get("fetch_mode"),
            "odds_snapshot_at": odds_provenance.get("snapshot_at"),
            "odds_snapshot_relation": _classify_snapshot_timing(odds_provenance.get("snapshot_at"), scheduled_post_at),
        }, None
    except Exception as error:
        return race_id, None, str(error)


def _repair_missing_provenance(
    *,
    frame: pd.DataFrame,
    race_card_dir: Path,
    race_card_odds_dir: Path,
    max_workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = frame.copy()
    if "race_id" in output.columns:
        output["race_id"] = output["race_id"].astype("string")
    missing_mask = pd.Series(False, index=output.index)
    for column in PROVENANCE_COLUMNS:
        if column not in output.columns:
            output[column] = pd.NA
        missing_mask = missing_mask | output[column].isna()

    missing_race_ids = sorted(output.loc[missing_mask, "race_id"].dropna().astype(str).unique().tolist())
    progress = ProgressBar(total=max(len(missing_race_ids), 1), prefix="[repair-local-racecard-prov]", logger=log_progress, min_interval_sec=0.0)
    progress.start(message=f"repairing provenance race_ids={len(missing_race_ids)}")

    repaired_rows: list[dict[str, Any]] = []
    repaired_race_ids = 0
    failed_race_ids = 0
    failures: list[dict[str, str]] = []

    worker_count = max(1, int(max_workers))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_map = {
            executor.submit(_repair_single_race, (race_id, race_card_dir.as_posix(), race_card_odds_dir.as_posix())): race_id
            for race_id in missing_race_ids
        }
        for index, future in enumerate(as_completed(future_map), start=1):
            race_id = future_map[future]
            try:
                _, repaired_row, error_text = future.result()
            except Exception as error:
                repaired_row = None
                error_text = str(error)
            if repaired_row is not None:
                repaired_rows.append(repaired_row)
                repaired_race_ids += 1
            else:
                failed_race_ids += 1
                failures.append({"race_id": race_id, "error": error_text or "unknown_error"})
            progress.update(current=index, message=f"repaired={repaired_race_ids} failed={failed_race_ids} workers={worker_count}")

    if repaired_rows:
        repair_frame = pd.DataFrame.from_records(repaired_rows)
        repair_frame["race_id"] = repair_frame["race_id"].astype("string")
        repair_frame = repair_frame.drop_duplicates(subset=["race_id"], keep="last")
        merged = output.merge(repair_frame, on=["race_id"], how="left", suffixes=("", "__repair"))
        for column in PROVENANCE_COLUMNS:
            repair_column = f"{column}__repair"
            if repair_column not in merged.columns:
                continue
            merged[column] = merged[column].where(merged[column].notna(), merged[repair_column])
            merged = merged.drop(columns=[repair_column])
        output = merged

    summary = {
        "requested_race_ids": int(len(missing_race_ids)),
        "repaired_race_ids": int(repaired_race_ids),
        "failed_race_ids": int(failed_race_ids),
        "failure_samples": failures[:50],
    }
    progress.complete(message=f"repair complete repaired={repaired_race_ids} failed={failed_race_ids}")
    return output, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", default="data/external/local_nankan/racecard/local_racecard.csv")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--summary-output", default="artifacts/reports/local_nankan_racecard_provenance_repair_summary.json")
    parser.add_argument("--race-card-dir", default="data/external/local_nankan/raw_html/race_card")
    parser.add_argument("--race-card-odds-dir", default="data/external/local_nankan/raw_html/race_card_odds")
    parser.add_argument("--max-workers", type=int, default=max(1, min(8, (os.cpu_count() or 4))))
    args = parser.parse_args()

    try:
        input_path = _resolve_path(args.input_file)
        output_path = _resolve_path(args.output_file or args.input_file)
        summary_path = _resolve_path(args.summary_output)
        race_card_dir = _resolve_path(args.race_card_dir)
        race_card_odds_dir = _resolve_path(args.race_card_odds_dir)

        progress = ProgressBar(total=3, prefix="[repair-local-racecard-prov]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"loading input={input_path}")
        frame = pd.read_csv(input_path, low_memory=False)
        progress.update(current=1, message=f"loaded rows={len(frame)}")

        with Heartbeat("[repair-local-racecard-prov]", "repairing missing provenance from cache", logger=log_progress):
            repaired_frame, repair_summary = _repair_missing_provenance(
                frame=frame,
                race_card_dir=race_card_dir,
                race_card_odds_dir=race_card_odds_dir,
                max_workers=args.max_workers,
            )
        progress.update(current=2, message=f"repair summary repaired={repair_summary['repaired_race_ids']} failed={repair_summary['failed_race_ids']}")

        write_csv_file(output_path, repaired_frame, index=False, label="local_nankan race_card provenance repaired")
        summary = {
            "input_file": str(input_path),
            "output_file": str(output_path),
            "race_card_dir": str(race_card_dir),
            "race_card_odds_dir": str(race_card_odds_dir),
            "repair_summary": repair_summary,
            "row_count": int(len(repaired_frame)),
            "provenance_non_null": {
                column: int(repaired_frame[column].notna().sum()) if column in repaired_frame.columns else 0
                for column in PROVENANCE_COLUMNS
            },
        }
        write_json(summary_path, summary)
        progress.complete(message=f"repair summary saved output={summary_path}")
        return 0
    except KeyboardInterrupt:
        print("[repair-local-racecard-prov] interrupted by user")
        return 130
    except Exception as error:
        print(f"[repair-local-racecard-prov] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())