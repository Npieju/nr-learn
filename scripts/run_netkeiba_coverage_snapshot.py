import argparse
import json
import os
from pathlib import Path
import sys
import time
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import utc_now_iso
from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table, load_training_table_tail


DEFAULT_COLUMNS = [
    "horse_key",
    "breeder_name",
    "sire_name",
    "dam_name",
    "damsire_name",
    "owner_name",
]

TARGET_MANIFEST_PATHS = {
    "race_result": ROOT / "artifacts/reports/netkeiba_crawl_manifest_race_result.json",
    "race_card": ROOT / "artifacts/reports/netkeiba_crawl_manifest_race_card.json",
    "pedigree": ROOT / "artifacts/reports/netkeiba_crawl_manifest_pedigree.json",
}
DEFAULT_CRAWL_LOCK_PATH = ROOT / "artifacts/reports/netkeiba_crawl_manifest.json.lock"
DEFAULT_RACE_RESULT_PATH = "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
DEFAULT_RACE_CARD_PATH = "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"
DEFAULT_PEDIGREE_PATH = "data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv"
DEFAULT_UNIVERSE = "jra"
DEFAULT_SOURCE_SCOPE = "netkeiba"
DEFAULT_SCHEMA_VERSION = "netkeiba.coverage_snapshot.v1"


def _safe_write_snapshot(path: Path | None, payload: dict[str, object]) -> None:
    if path is None:
        return
    try:
        write_json(path, payload)
    except Exception:
        return


def _set_step(payload: dict[str, object], step_name: str) -> None:
    payload["completed_step"] = step_name


def _require_text(value: str, *, field_name: str, error_code: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError(f"{error_code}:{field_name} must not be empty")
    return normalized


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-snapshot {now}] {message}", flush=True)


def _safe_ratio(series: pd.Series | None) -> float | None:
    if series is None or len(series) == 0:
        return None
    return round(float(series.notna().mean()), 6)


def _safe_nunique(series: pd.Series | None) -> int | None:
    if series is None:
        return None
    return int(series.nunique(dropna=True))


def _build_coverage(frame: pd.DataFrame, columns: list[str]) -> dict[str, dict[str, object]]:
    coverage: dict[str, dict[str, object]] = {}
    for column in columns:
        if column not in frame.columns:
            coverage[column] = {"present": False, "non_null_ratio": None, "nunique": None}
            continue
        series = frame[column]
        coverage[column] = {
            "present": True,
            "non_null_ratio": _safe_ratio(series),
            "nunique": _safe_nunique(series),
        }
    return coverage


def _build_result_integrity_summary(frame: pd.DataFrame) -> dict[str, object]:
    if frame.empty or "race_id" not in frame.columns:
        return {
            "rows": int(len(frame)),
            "races": 0,
            "odds_present": False,
            "rank_present": False,
            "all_odds_missing_races": 0,
            "all_rank_missing_races": 0,
        }

    race_count = int(frame["race_id"].nunique(dropna=True))
    summary: dict[str, object] = {
        "rows": int(len(frame)),
        "races": race_count,
        "odds_present": "odds" in frame.columns,
        "rank_present": "rank" in frame.columns,
    }

    if "odds" in frame.columns:
        odds_series = frame["odds"]
        odds_missing = odds_series.isna()
        if odds_series.dtype == object:
            normalized_odds = odds_series.astype(str).str.strip()
            odds_missing = odds_missing | normalized_odds.isin(["", "nan", "None", "---", "---.--"])
        all_odds_missing_races = int(odds_missing.groupby(frame["race_id"]).all().sum())
        summary["all_odds_missing_races"] = all_odds_missing_races
        summary["all_odds_missing_ratio"] = round(all_odds_missing_races / race_count, 6) if race_count else None
    else:
        summary["all_odds_missing_races"] = None
        summary["all_odds_missing_ratio"] = None

    if "rank" in frame.columns:
        rank_series = frame["rank"]
        rank_missing = rank_series.isna()
        if rank_series.dtype == object:
            normalized_rank = rank_series.astype(str).str.strip()
            rank_missing = rank_missing | normalized_rank.isin(["", "nan", "None"])
        all_rank_missing_races = int(rank_missing.groupby(frame["race_id"]).all().sum())
        summary["all_rank_missing_races"] = all_rank_missing_races
        summary["all_rank_missing_ratio"] = round(all_rank_missing_races / race_count, 6) if race_count else None
    else:
        summary["all_rank_missing_races"] = None
        summary["all_rank_missing_ratio"] = None

    return summary


def _read_external(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _read_json(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _dict_payload(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(str(value))
    except (TypeError, ValueError):
        return None


def _safe_int(value: object, default: int = 0) -> int:
    parsed = _optional_int(value)
    return default if parsed is None else parsed


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _build_crawl_lock_state(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    pid = _optional_int(payload.get("pid")) if isinstance(payload, dict) else None
    return {
        "present": path.exists(),
        "path": str(path),
        "pid": pid,
        "pid_running": _pid_is_running(pid) if pid is not None else None,
        "started_at": payload.get("started_at") if isinstance(payload, dict) else None,
    }


def _build_target_state(path: Path, *, output_path: Path, output_frame: pd.DataFrame, key_column: str) -> dict[str, object]:
    payload = _read_json(path)
    output_exists = output_path.exists()
    output_rows = int(len(output_frame))
    unique_key_count = int(output_frame[key_column].nunique(dropna=True)) if not output_frame.empty and key_column in output_frame.columns else 0
    if not payload:
        inferred_status = "missing"
        if output_rows > 0:
            inferred_status = "completed_untracked"
        elif output_exists:
            inferred_status = "empty_output"
        return {
            "present": False,
            "status": inferred_status,
            "requested_ids": None,
            "processed_ids": None,
            "parsed_ids": None,
            "failure_count": None,
            "rows_written": None,
            "started_at": None,
            "finished_at": None,
            "pid": None,
            "lock_file": None,
            "stale_reason": None,
            "manifest_path": str(path),
            "output_path": str(output_path),
            "output_exists": output_exists,
            "output_rows": output_rows,
            f"unique_{key_column}": unique_key_count,
            "inferred_from_output": output_rows > 0 or output_exists,
        }
    status = str(payload.get("status", "unknown"))
    pid = _optional_int(payload.get("pid"))
    lock_file_text = payload.get("lock_file") if isinstance(payload, dict) else None
    lock_path = Path(str(lock_file_text)) if lock_file_text else DEFAULT_CRAWL_LOCK_PATH
    stale_reason = None
    if status == "running":
        if pid is not None and not _pid_is_running(pid):
            status = "stale"
            stale_reason = f"pid_not_running:{pid}"
        elif not lock_path.exists():
            status = "stale"
            stale_reason = "lock_missing"
    return {
        "present": True,
        "status": status,
        "requested_ids": _optional_int(payload.get("requested_ids")),
        "processed_ids": _optional_int(payload.get("processed_ids")),
        "parsed_ids": _optional_int(payload.get("parsed_ids")),
        "failure_count": _optional_int(payload.get("failure_count")),
        "rows_written": _optional_int(payload.get("rows_written")),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
        "pid": pid,
        "lock_file": str(lock_path),
        "stale_reason": stale_reason,
        "manifest_path": str(path),
        "output_path": str(output_path),
        "output_exists": output_exists,
        "output_rows": output_rows,
        f"unique_{key_column}": unique_key_count,
        "inferred_from_output": False,
    }


def _summarize_external(frame: pd.DataFrame, key_column: str) -> dict[str, object]:
    if frame.empty or key_column not in frame.columns:
        return {"rows": 0, f"unique_{key_column}": 0}
    return {
        "rows": int(len(frame)),
        f"unique_{key_column}": int(frame[key_column].nunique(dropna=True)),
    }


def _is_completed_like(status: str) -> bool:
    return status in {"completed", "completed_untracked"}


def _build_progress_summary(
    *,
    target_states: dict[str, dict[str, object]],
    external_outputs: dict[str, dict[str, object]],
    alignment: dict[str, object],
    readiness: dict[str, object],
) -> dict[str, object]:
    status_counts = {
        "completed": 0,
        "planned": 0,
        "running": 0,
        "partial": 0,
        "failed": 0,
        "missing": 0,
        "stale": 0,
        "empty_output": 0,
        "other": 0,
    }
    completed_targets: list[str] = []
    incomplete_targets: list[str] = []
    failed_targets: list[str] = []

    for target_name, target_state in target_states.items():
        status = str(target_state.get("status") or "missing")
        if _is_completed_like(status):
            status_counts["completed"] += 1
            completed_targets.append(target_name)
            continue
        if status == "running":
            status_counts["running"] += 1
        elif status == "planned":
            status_counts["planned"] += 1
        elif status == "partial":
            status_counts["partial"] += 1
        elif status == "failed":
            status_counts["failed"] += 1
            failed_targets.append(target_name)
        elif status == "missing":
            status_counts["missing"] += 1
        elif status == "stale":
            status_counts["stale"] += 1
            failed_targets.append(target_name)
        elif status == "empty_output":
            status_counts["empty_output"] += 1
        else:
            status_counts["other"] += 1
        incomplete_targets.append(target_name)

    race_result_races = _safe_int(alignment.get("race_result_races"))
    race_card_races = _safe_int(alignment.get("race_card_races"))
    pedigree_rows = _safe_int(_dict_payload(external_outputs.get("pedigree")).get("rows"))
    race_card_coverage_ratio = round(race_card_races / race_result_races, 6) if race_result_races > 0 else None

    if not _is_completed_like(str(target_states.get("race_result", {}).get("status") or "")):
        current_stage = "race_result_collection"
    elif race_card_races < race_result_races or not _is_completed_like(str(target_states.get("race_card", {}).get("status") or "")):
        current_stage = "race_card_collection"
    elif not bool(readiness.get("snapshot_consistent")):
        current_stage = "alignment_validation"
    elif not bool(readiness.get("benchmark_rerun_ready")):
        current_stage = "benchmark_readiness_validation"
    else:
        current_stage = "ready_for_benchmark"

    return {
        "current_stage": current_stage,
        "target_status_counts": status_counts,
        "completed_targets": completed_targets,
        "incomplete_targets": incomplete_targets,
        "failed_targets": failed_targets,
        "target_completion_ratio": round(len(completed_targets) / max(len(target_states), 1), 6),
        "race_card_race_coverage_ratio_vs_result": race_card_coverage_ratio,
        "pedigree_rows": pedigree_rows,
        "recommended_action": readiness.get("recommended_action"),
    }


def _summarize_alignment_metrics(merged: pd.DataFrame) -> dict[str, int]:
    if merged.empty:
        return {
            "mismatch_races": 0,
            "positive_diff_races": 0,
            "negative_diff_races": 0,
            "max_abs_diff": 0,
        }

    diff = merged["race_card_rows"] - merged["race_result_rows"]
    return {
        "mismatch_races": int((diff != 0).sum()),
        "positive_diff_races": int((diff > 0).sum()),
        "negative_diff_races": int((diff < 0).sum()),
        "max_abs_diff": int(diff.abs().max()),
    }


def _build_alignment_summary(race_result: pd.DataFrame, race_card: pd.DataFrame) -> dict[str, object]:
    if race_result.empty and race_card.empty:
        return {
            "race_result_races": 0,
            "race_card_races": 0,
            "intersection_races": 0,
            "race_result_only_races": 0,
            "race_card_only_races": 0,
            "mismatch_races": 0,
            "positive_diff_races": 0,
            "negative_diff_races": 0,
            "max_abs_diff": 0,
            "paired_mismatch_races": 0,
            "paired_positive_diff_races": 0,
            "paired_negative_diff_races": 0,
            "paired_max_abs_diff": 0,
        }

    rr_counts = (
        race_result.groupby("race_id").size().rename("race_result_rows")
        if not race_result.empty and "race_id" in race_result.columns
        else pd.Series(dtype=int, name="race_result_rows")
    )
    rc_counts = (
        race_card.groupby("race_id").size().rename("race_card_rows")
        if not race_card.empty and "race_id" in race_card.columns
        else pd.Series(dtype=int, name="race_card_rows")
    )

    merged = pd.concat([rr_counts, rc_counts], axis=1).fillna(0).astype(int)
    if merged.empty:
        return {
            "race_result_races": int(len(rr_counts)),
            "race_card_races": int(len(rc_counts)),
            "intersection_races": 0,
            "race_result_only_races": int(len(rr_counts)),
            "race_card_only_races": int(len(rc_counts)),
            "mismatch_races": 0,
            "positive_diff_races": 0,
            "negative_diff_races": 0,
            "max_abs_diff": 0,
            "paired_mismatch_races": 0,
            "paired_positive_diff_races": 0,
            "paired_negative_diff_races": 0,
            "paired_max_abs_diff": 0,
        }

    overall_metrics = _summarize_alignment_metrics(merged)
    paired = merged[(merged["race_result_rows"] > 0) & (merged["race_card_rows"] > 0)]
    paired_metrics = _summarize_alignment_metrics(paired)
    return {
        "race_result_races": int(len(rr_counts)),
        "race_card_races": int(len(rc_counts)),
        "intersection_races": int(len(set(rr_counts.index).intersection(set(rc_counts.index)))),
        "race_result_only_races": int(((merged["race_result_rows"] > 0) & (merged["race_card_rows"] == 0)).sum()),
        "race_card_only_races": int(((merged["race_card_rows"] > 0) & (merged["race_result_rows"] == 0)).sum()),
        **overall_metrics,
        "paired_mismatch_races": paired_metrics["mismatch_races"],
        "paired_positive_diff_races": paired_metrics["positive_diff_races"],
        "paired_negative_diff_races": paired_metrics["negative_diff_races"],
        "paired_max_abs_diff": paired_metrics["max_abs_diff"],
    }


def _build_readiness(
    target_states: dict[str, dict[str, object]],
    alignment: dict[str, object],
    result_integrity: dict[str, object],
) -> dict[str, object]:
    stale_targets = [
        target_name
        for target_name, target_state in target_states.items()
        if target_state.get("status") == "stale"
    ]
    race_targets_complete = all(
        target_states.get(name, {}).get("status") == "completed"
        for name in ("race_result", "race_card")
    )
    no_unpaired_races = (
        _safe_int(alignment.get("race_result_only_races")) == 0
        and _safe_int(alignment.get("race_card_only_races")) == 0
    )
    paired_result_coverage_ok = _safe_int(alignment.get("paired_negative_diff_races")) == 0
    paired_result_odds_ok = _safe_int(result_integrity.get("all_odds_missing_races")) == 0
    pedigree_stable = target_states.get("pedigree", {}).get("status") not in {"running", "stale"}

    snapshot_consistent = race_targets_complete and paired_result_coverage_ok and no_unpaired_races and paired_result_odds_ok
    benchmark_rerun_ready = snapshot_consistent and pedigree_stable

    reasons: list[str] = []
    if stale_targets:
        reasons.append(f"stale crawl manifests detected: {', '.join(stale_targets)}")
    if not race_targets_complete:
        reasons.append("race_result and race_card must both be completed")
    if not no_unpaired_races:
        reasons.append("race_result and race_card still have unpaired races")
    if not paired_result_coverage_ok:
        reasons.append("paired races still have race_result coverage gaps against race_card")
    if not paired_result_odds_ok:
        reasons.append("paired races still include all-odds-missing race results")
    if not pedigree_stable:
        pedigree_status = target_states.get("pedigree", {}).get("status")
        if pedigree_status == "stale":
            reasons.append("pedigree manifest is stale; inspect crawl state")
        elif pedigree_status == "running":
            reasons.append("pedigree crawl is still running")

    if benchmark_rerun_ready:
        recommended_action = "rerun_enriched_benchmark"
    elif stale_targets:
        recommended_action = "inspect_manifests"
    elif not race_targets_complete:
        recommended_action = "wait_for_race_targets"
    elif not no_unpaired_races or not paired_result_coverage_ok or not paired_result_odds_ok:
        recommended_action = "inspect_race_alignment"
    elif not pedigree_stable:
        recommended_action = "wait_for_pedigree"
    else:
        recommended_action = "inspect_manifests"

    return {
        "snapshot_consistent": snapshot_consistent,
        "benchmark_rerun_ready": benchmark_rerun_ready,
        "recommended_action": recommended_action,
        "reasons": reasons,
        "paired_result_odds_ok": paired_result_odds_ok,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/reports/netkeiba_coverage_snapshot.json")
    parser.add_argument("--universe", default=DEFAULT_UNIVERSE)
    parser.add_argument("--source-scope", default=DEFAULT_SOURCE_SCOPE)
    parser.add_argument("--schema-version", default=DEFAULT_SCHEMA_VERSION)
    parser.add_argument("--baseline-reference", default=None)
    parser.add_argument("--race-result-path", default=DEFAULT_RACE_RESULT_PATH)
    parser.add_argument("--race-card-path", default=DEFAULT_RACE_CARD_PATH)
    parser.add_argument("--pedigree-path", default=DEFAULT_PEDIGREE_PATH)
    parser.add_argument("--race-result-manifest", default=None)
    parser.add_argument("--race-card-manifest", default=None)
    parser.add_argument("--pedigree-manifest", default=None)
    parser.add_argument("--crawl-lock-path", default=None)
    parser.add_argument(
        "--columns",
        nargs="*",
        default=DEFAULT_COLUMNS,
    )
    args = parser.parse_args()

    output_path = ROOT / args.output
    payload: dict[str, object] = {
        "schema_version": DEFAULT_SCHEMA_VERSION,
        "artifact_type": "coverage_snapshot",
        "status": "running",
        "completed_step": "init",
        "started_at": utc_now_iso(),
        "finished_at": None,
        "universe": DEFAULT_UNIVERSE,
        "source_scope": DEFAULT_SOURCE_SCOPE,
        "baseline_reference": None,
    }

    try:
        universe = _require_text(args.universe, field_name="universe", error_code="missing_universe")
        source_scope = _require_text(args.source_scope, field_name="source_scope", error_code="missing_source_scope")
        schema_version = _require_text(args.schema_version, field_name="schema_version", error_code="missing_schema_version")
        baseline_reference = str(args.baseline_reference).strip() or None
        payload.update(
            {
                "schema_version": schema_version,
                "universe": universe,
                "source_scope": source_scope,
                "baseline_reference": baseline_reference,
            }
        )
        _set_step(payload, "load_config")
        progress = ProgressBar(total=4, prefix="[netkeiba-snapshot]", logger=log_progress, min_interval_sec=0.0)
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        data_cfg = load_yaml(ROOT / args.config)
        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        target_manifest_paths = {
            "race_result": Path(args.race_result_manifest) if args.race_result_manifest else TARGET_MANIFEST_PATHS["race_result"],
            "race_card": Path(args.race_card_manifest) if args.race_card_manifest else TARGET_MANIFEST_PATHS["race_card"],
            "pedigree": Path(args.pedigree_manifest) if args.pedigree_manifest else TARGET_MANIFEST_PATHS["pedigree"],
        }
        target_manifest_paths = {
            name: path if path.is_absolute() else (ROOT / path)
            for name, path in target_manifest_paths.items()
        }
        crawl_lock_path = Path(args.crawl_lock_path) if args.crawl_lock_path else DEFAULT_CRAWL_LOCK_PATH
        if not crawl_lock_path.is_absolute():
            crawl_lock_path = ROOT / crawl_lock_path
        race_result_path = Path(args.race_result_path)
        if not race_result_path.is_absolute():
            race_result_path = ROOT / race_result_path
        race_card_path = Path(args.race_card_path)
        if not race_card_path.is_absolute():
            race_card_path = ROOT / race_card_path
        pedigree_path = Path(args.pedigree_path)
        if not pedigree_path.is_absolute():
            pedigree_path = ROOT / pedigree_path
        tail_rows = max(int(args.tail_rows), 0)
        progress.start(message=f"config loaded tail_rows={tail_rows}")
        _set_step(payload, "load_source_tables")
        if tail_rows > 0:
            with Heartbeat("[netkeiba-snapshot]", "loading tail training table", logger=log_progress):
                tail_frame, primary_source_rows_total = load_training_table_tail(
                    raw_dir,
                    tail_rows=tail_rows,
                    dataset_config=dataset_cfg,
                    base_dir=ROOT,
                )
            frame = tail_frame.copy()
        else:
            with Heartbeat("[netkeiba-snapshot]", "loading full training table", logger=log_progress):
                frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
            primary_source_rows_total = int(len(frame))
            tail_frame = frame.copy()
        progress.update(message=f"training data loaded rows={len(frame)}")
        with Heartbeat("[netkeiba-snapshot]", "loading external outputs", logger=log_progress):
            race_result = _read_external(race_result_path)
            race_card = _read_external(race_card_path)
            pedigree = _read_external(pedigree_path)
        target_states = {
            "race_result": _build_target_state(
                target_manifest_paths["race_result"],
                output_path=race_result_path,
                output_frame=race_result,
                key_column="race_id",
            ),
            "race_card": _build_target_state(
                target_manifest_paths["race_card"],
                output_path=race_card_path,
                output_frame=race_card,
                key_column="race_id",
            ),
            "pedigree": _build_target_state(
                target_manifest_paths["pedigree"],
                output_path=pedigree_path,
                output_frame=pedigree,
                key_column="horse_key",
            ),
        }
        progress.update(message="external outputs loaded")

        _set_step(payload, "compute_alignment")
        paired_race_ids: set[object] = set()
        if "race_id" in race_result.columns and "race_id" in race_card.columns:
            paired_race_ids = set(race_result["race_id"].dropna().tolist()).intersection(set(race_card["race_id"].dropna().tolist()))

        collected_subset = (
            frame[frame["race_id"].isin(paired_race_ids)].copy()
            if paired_race_ids and "race_id" in frame.columns
            else pd.DataFrame(columns=frame.columns)
        )

        alignment = _build_alignment_summary(race_result, race_card)
        result_integrity = {
            "latest_tail": _build_result_integrity_summary(tail_frame),
            "paired_race_subset": _build_result_integrity_summary(collected_subset),
            "external_race_result": _build_result_integrity_summary(race_result),
        }
        _set_step(payload, "compute_coverage")
        coverage = {
            "latest_tail": _build_coverage(tail_frame, list(args.columns)),
            "paired_race_subset": _build_coverage(collected_subset, list(args.columns)),
        }
        external_outputs = {
            "race_result": _summarize_external(race_result, "race_id"),
            "race_card": _summarize_external(race_card, "race_id"),
            "pedigree": _summarize_external(pedigree, "horse_key"),
        }
        readiness = _build_readiness(target_states, alignment, result_integrity["external_race_result"])
        progress_summary = _build_progress_summary(
            target_states=target_states,
            external_outputs=external_outputs,
            alignment=alignment,
            readiness=readiness,
        )
        payload.update({
            "run_context": {
                "config": args.config,
                "tail_rows": int(args.tail_rows),
                "universe": universe,
                "source_scope": source_scope,
                "schema_version": schema_version,
                "baseline_reference": baseline_reference,
                "primary_source_rows_total": int(primary_source_rows_total),
                "rows_tail": int(len(tail_frame)),
                "external_output_paths": {
                    "race_result": artifact_display_path(race_result_path, workspace_root=ROOT),
                    "race_card": artifact_display_path(race_card_path, workspace_root=ROOT),
                    "pedigree": artifact_display_path(pedigree_path, workspace_root=ROOT),
                },
                "target_manifests": {
                    name: artifact_display_path(path, workspace_root=ROOT)
                    for name, path in target_manifest_paths.items()
                },
                "crawl_lock_path": artifact_display_path(crawl_lock_path, workspace_root=ROOT),
            },
            "crawl_lock": _build_crawl_lock_state(crawl_lock_path),
            "external_outputs": external_outputs,
            "target_states": target_states,
            "alignment": alignment,
            "result_integrity": result_integrity,
            "coverage": coverage,
            "paired_race_subset": {
                "rows": int(len(collected_subset)),
                "races": int(len(paired_race_ids)),
            },
            "readiness": readiness,
            "progress": progress_summary,
            "coverage_summary": {
                "latest_tail_horse_key_ratio": coverage["latest_tail"].get("horse_key", {}).get("non_null_ratio"),
                "latest_tail_breeder_ratio": coverage["latest_tail"].get("breeder_name", {}).get("non_null_ratio"),
                "latest_tail_sire_ratio": coverage["latest_tail"].get("sire_name", {}).get("non_null_ratio"),
                "paired_subset_horse_key_ratio": coverage["paired_race_subset"].get("horse_key", {}).get("non_null_ratio"),
            },
            "integrity_summary": {
                "race_result_only_races": alignment.get("race_result_only_races"),
                "race_card_only_races": alignment.get("race_card_only_races"),
                "paired_negative_diff_races": alignment.get("paired_negative_diff_races"),
                "external_all_odds_missing_races": result_integrity["external_race_result"].get("all_odds_missing_races"),
            },
        })
        _set_step(payload, "write_snapshot")
        with Heartbeat("[netkeiba-snapshot]", "writing snapshot output", logger=log_progress):
            write_json(output_path, payload)
        payload["status"] = "completed"
        payload["finished_at"] = utc_now_iso()
        _set_step(payload, "completed")
        _safe_write_snapshot(output_path, payload)

        print(f"[netkeiba-snapshot] output={output_path}")
        print(f"[netkeiba-snapshot] alignment={payload['alignment']}")
        readiness_payload = _dict_payload(payload.get("readiness"))
        readiness_reasons_raw = readiness_payload.get("reasons")
        readiness_reasons: list[str] = []
        if isinstance(readiness_reasons_raw, list):
            readiness_reasons = [str(reason) for reason in readiness_reasons_raw]
        reason_text = "; ".join(readiness_reasons) if readiness_reasons else "none"
        print(
            "[netkeiba-snapshot] "
            f"readiness action={readiness_payload.get('recommended_action')} "
            f"snapshot_consistent={readiness_payload.get('snapshot_consistent')} "
            f"benchmark_rerun_ready={readiness_payload.get('benchmark_rerun_ready')} "
            f"reasons={reason_text}"
        )
        coverage_payload = _dict_payload(payload.get("coverage"))
        for scope_name, scope_payload in coverage_payload.items():
            if not isinstance(scope_payload, dict):
                continue
            summary = ", ".join(
                f"{column}={metrics['non_null_ratio']}"
                for column, metrics in scope_payload.items()
                if isinstance(metrics, dict)
            )
            print(f"[netkeiba-snapshot] {scope_name}: {summary}")
        progress.complete(message="snapshot completed")
        return 0
    except KeyboardInterrupt:
        payload["status"] = "interrupted"
        payload["finished_at"] = utc_now_iso()
        payload["error_code"] = "interrupted"
        payload["error_message"] = "interrupted by user"
        payload["recommended_action"] = "rerun_snapshot"
        _safe_write_snapshot(output_path, payload)
        print("[netkeiba-snapshot] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        error_text = str(error)
        error_code, _, message = error_text.partition(":")
        if error_code not in {
            "missing_universe",
            "missing_source_scope",
            "missing_schema_version",
            "invalid_output_path",
        }:
            error_code = "invalid_output_path" if isinstance(error, IsADirectoryError) else "snapshot_failed"
            message = error_text
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error_code"] = error_code
        payload["error_message"] = message or error_text
        payload["recommended_action"] = "inspect_snapshot_inputs"
        _safe_write_snapshot(output_path, payload)
        print(f"[netkeiba-snapshot] failed: {error}")
        return 1
    except Exception as error:
        payload["status"] = "failed"
        payload["finished_at"] = utc_now_iso()
        payload["error_code"] = "snapshot_failed"
        payload["error_message"] = str(error)
        payload["recommended_action"] = "inspect_snapshot_traceback"
        _safe_write_snapshot(output_path, payload)
        print(f"[netkeiba-snapshot] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())