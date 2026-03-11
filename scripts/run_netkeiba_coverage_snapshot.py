import argparse
import json
from pathlib import Path
import sys
import traceback

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml
from racing_ml.data.dataset_loader import load_training_table


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


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _build_target_state(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    if not payload:
        return {
            "present": False,
            "status": "missing",
            "requested_ids": None,
            "processed_ids": None,
            "parsed_ids": None,
            "failure_count": None,
            "rows_written": None,
            "started_at": None,
            "finished_at": None,
        }
    return {
        "present": True,
        "status": payload.get("status", "unknown"),
        "requested_ids": _optional_int(payload.get("requested_ids")),
        "processed_ids": _optional_int(payload.get("processed_ids")),
        "parsed_ids": _optional_int(payload.get("parsed_ids")),
        "failure_count": _optional_int(payload.get("failure_count")),
        "rows_written": _optional_int(payload.get("rows_written")),
        "started_at": payload.get("started_at"),
        "finished_at": payload.get("finished_at"),
    }


def _summarize_external(frame: pd.DataFrame, key_column: str) -> dict[str, object]:
    if frame.empty or key_column not in frame.columns:
        return {"rows": 0, f"unique_{key_column}": 0}
    return {
        "rows": int(len(frame)),
        f"unique_{key_column}": int(frame[key_column].nunique(dropna=True)),
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


def _build_readiness(target_states: dict[str, dict[str, object]], alignment: dict[str, object]) -> dict[str, object]:
    race_targets_complete = all(
        target_states.get(name, {}).get("status") == "completed"
        for name in ("race_result", "race_card")
    )
    no_unpaired_races = (
        int(alignment.get("race_result_only_races", 0)) == 0
        and int(alignment.get("race_card_only_races", 0)) == 0
    )
    paired_alignment_ok = int(alignment.get("paired_mismatch_races", 0)) == 0
    pedigree_stable = target_states.get("pedigree", {}).get("status") != "running"

    snapshot_consistent = race_targets_complete and paired_alignment_ok and no_unpaired_races
    benchmark_rerun_ready = snapshot_consistent and pedigree_stable

    reasons: list[str] = []
    if not race_targets_complete:
        reasons.append("race_result and race_card must both be completed")
    if not no_unpaired_races:
        reasons.append("race_result and race_card still have unpaired races")
    if not paired_alignment_ok:
        reasons.append("paired races still have row-count mismatches")
    if not pedigree_stable:
        reasons.append("pedigree crawl is still running")

    if benchmark_rerun_ready:
        recommended_action = "rerun_enriched_benchmark"
    elif not race_targets_complete:
        recommended_action = "wait_for_race_targets"
    elif not no_unpaired_races or not paired_alignment_ok:
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
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--tail-rows", type=int, default=5000)
    parser.add_argument("--output", default="artifacts/reports/netkeiba_coverage_snapshot.json")
    parser.add_argument(
        "--columns",
        nargs="*",
        default=DEFAULT_COLUMNS,
    )
    args = parser.parse_args()

    try:
        data_cfg = load_yaml(ROOT / args.config)
        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)

        tail_rows = max(int(args.tail_rows), 0)
        tail_frame = frame.tail(tail_rows).reset_index(drop=True) if tail_rows > 0 and len(frame) > tail_rows else frame.copy()

        result_path = ROOT / "data/external/netkeiba/results/netkeiba_race_result_crawled.csv"
        race_card_path = ROOT / "data/external/netkeiba/racecard/netkeiba_racecard_crawled.csv"
        pedigree_path = ROOT / "data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv"

        race_result = _read_external(result_path)
        race_card = _read_external(race_card_path)
        pedigree = _read_external(pedigree_path)
        target_states = {
            target_name: _build_target_state(path)
            for target_name, path in TARGET_MANIFEST_PATHS.items()
        }

        paired_race_ids: set[object] = set()
        if "race_id" in race_result.columns and "race_id" in race_card.columns:
            paired_race_ids = set(race_result["race_id"].dropna().tolist()).intersection(set(race_card["race_id"].dropna().tolist()))

        collected_subset = (
            frame[frame["race_id"].isin(paired_race_ids)].copy()
            if paired_race_ids and "race_id" in frame.columns
            else pd.DataFrame(columns=frame.columns)
        )

        alignment = _build_alignment_summary(race_result, race_card)
        payload = {
            "run_context": {
                "config": args.config,
                "tail_rows": int(args.tail_rows),
                "rows_total": int(len(frame)),
                "rows_tail": int(len(tail_frame)),
            },
            "external_outputs": {
                "race_result": _summarize_external(race_result, "race_id"),
                "race_card": _summarize_external(race_card, "race_id"),
                "pedigree": _summarize_external(pedigree, "horse_key"),
            },
            "target_states": target_states,
            "alignment": alignment,
            "coverage": {
                "latest_tail": _build_coverage(tail_frame, list(args.columns)),
                "paired_race_subset": _build_coverage(collected_subset, list(args.columns)),
            },
            "paired_race_subset": {
                "rows": int(len(collected_subset)),
                "races": int(len(paired_race_ids)),
            },
            "readiness": _build_readiness(target_states, alignment),
        }
        write_json(ROOT / args.output, payload)

        print(f"[netkeiba-snapshot] output={ROOT / args.output}")
        print(f"[netkeiba-snapshot] alignment={payload['alignment']}")
        readiness = payload["readiness"]
        reason_text = "; ".join(readiness["reasons"]) if readiness["reasons"] else "none"
        print(
            "[netkeiba-snapshot] "
            f"readiness action={readiness['recommended_action']} "
            f"snapshot_consistent={readiness['snapshot_consistent']} "
            f"benchmark_rerun_ready={readiness['benchmark_rerun_ready']} "
            f"reasons={reason_text}"
        )
        for scope_name, scope_payload in payload["coverage"].items():
            summary = ", ".join(
                f"{column}={metrics['non_null_ratio']}"
                for column, metrics in scope_payload.items()
            )
            print(f"[netkeiba-snapshot] {scope_name}: {summary}")
        return 0
    except KeyboardInterrupt:
        print("[netkeiba-snapshot] interrupted by user")
        return 130
    except Exception as error:
        print(f"[netkeiba-snapshot] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())