from __future__ import annotations

from typing import Any

import pandas as pd


RELATION_PRE_RACE = "pre_race"
RELATION_POST_RACE = "post_race"
RELATION_UNKNOWN = "unknown"
MARKET_TIMING_BUCKET_COLUMN = "market_timing_bucket"


def normalize_snapshot_relation(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == RELATION_PRE_RACE:
        return RELATION_PRE_RACE
    if text == RELATION_POST_RACE:
        return RELATION_POST_RACE
    return RELATION_UNKNOWN


def classify_market_timing_bucket(
    *,
    card_snapshot_relation: object,
    odds_snapshot_relation: object,
) -> str:
    card_relation = normalize_snapshot_relation(card_snapshot_relation)
    odds_relation = normalize_snapshot_relation(odds_snapshot_relation)
    if RELATION_POST_RACE in {card_relation, odds_relation}:
        return RELATION_POST_RACE
    if card_relation == RELATION_PRE_RACE and odds_relation == RELATION_PRE_RACE:
        return RELATION_PRE_RACE
    return RELATION_UNKNOWN


def annotate_market_timing_bucket(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    card_relations = output.get("card_snapshot_relation")
    odds_relations = output.get("odds_snapshot_relation")
    if card_relations is None:
        card_relations = pd.Series([RELATION_UNKNOWN] * len(output), index=output.index)
    if odds_relations is None:
        odds_relations = pd.Series([RELATION_UNKNOWN] * len(output), index=output.index)
    output[MARKET_TIMING_BUCKET_COLUMN] = [
        classify_market_timing_bucket(
            card_snapshot_relation=card_relation,
            odds_snapshot_relation=odds_relation,
        )
        for card_relation, odds_relation in zip(card_relations.tolist(), odds_relations.tolist())
    ]
    return output


def filter_pre_race_only(frame: pd.DataFrame) -> pd.DataFrame:
    annotated = annotate_market_timing_bucket(frame)
    return annotated.loc[annotated[MARKET_TIMING_BUCKET_COLUMN] == RELATION_PRE_RACE].reset_index(drop=True)


def filter_result_ready_pre_race_only(
    frame: pd.DataFrame,
    result_frame: pd.DataFrame | None,
) -> pd.DataFrame:
    pre_race_only = filter_pre_race_only(frame)
    if result_frame is None or "race_id" not in result_frame.columns or "race_id" not in pre_race_only.columns:
        return pre_race_only.iloc[0:0].copy()
    ready_race_ids = {str(value) for value in result_frame["race_id"].dropna().tolist()}
    race_ids = pre_race_only["race_id"].astype(str)
    return pre_race_only.loc[race_ids.isin(ready_race_ids)].reset_index(drop=True)


def build_provenance_summary(frame: pd.DataFrame) -> dict[str, Any]:
    annotated = annotate_market_timing_bucket(frame)
    bucket_counts = annotated[MARKET_TIMING_BUCKET_COLUMN].fillna(RELATION_UNKNOWN).value_counts().to_dict()
    card_counts = (
        annotated.get("card_snapshot_relation", pd.Series([], dtype="object"))
        .map(normalize_snapshot_relation)
        .fillna(RELATION_UNKNOWN)
        .value_counts()
        .to_dict()
    )
    odds_counts = (
        annotated.get("odds_snapshot_relation", pd.Series([], dtype="object"))
        .map(normalize_snapshot_relation)
        .fillna(RELATION_UNKNOWN)
        .value_counts()
        .to_dict()
    )
    summary = {
        "row_count": int(len(annotated)),
        "race_count": int(annotated["race_id"].nunique()) if "race_id" in annotated.columns else None,
        "bucket_counts": {
            RELATION_PRE_RACE: int(bucket_counts.get(RELATION_PRE_RACE, 0)),
            RELATION_UNKNOWN: int(bucket_counts.get(RELATION_UNKNOWN, 0)),
            RELATION_POST_RACE: int(bucket_counts.get(RELATION_POST_RACE, 0)),
        },
        "card_snapshot_relation_counts": {
            RELATION_PRE_RACE: int(card_counts.get(RELATION_PRE_RACE, 0)),
            RELATION_UNKNOWN: int(card_counts.get(RELATION_UNKNOWN, 0)),
            RELATION_POST_RACE: int(card_counts.get(RELATION_POST_RACE, 0)),
        },
        "odds_snapshot_relation_counts": {
            RELATION_PRE_RACE: int(odds_counts.get(RELATION_PRE_RACE, 0)),
            RELATION_UNKNOWN: int(odds_counts.get(RELATION_UNKNOWN, 0)),
            RELATION_POST_RACE: int(odds_counts.get(RELATION_POST_RACE, 0)),
        },
        "pre_race_only_rows": int(bucket_counts.get(RELATION_PRE_RACE, 0)),
        "post_race_rows": int(bucket_counts.get(RELATION_POST_RACE, 0)),
        "unknown_rows": int(bucket_counts.get(RELATION_UNKNOWN, 0)),
    }
    return summary


def build_pre_race_only_materialization_summary(
    frame: pd.DataFrame,
    *,
    result_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    pre_race_only = filter_pre_race_only(frame)
    race_ids = sorted({str(value) for value in pre_race_only.get("race_id", pd.Series([], dtype="object")).dropna().tolist()})
    dates = sorted({str(value) for value in pre_race_only.get("date", pd.Series([], dtype="object")).dropna().tolist()})
    result_race_ids: set[str] = set()
    if result_frame is not None and "race_id" in result_frame.columns:
        result_race_ids = {str(value) for value in result_frame["race_id"].dropna().tolist()}
    ready_race_ids = sorted(set(race_ids) & result_race_ids)
    pending_race_ids = sorted(set(race_ids) - result_race_ids)
    row_race_ids = pre_race_only.get("race_id", pd.Series([], dtype="object")).astype(str)
    ready_row_count = int(row_race_ids.isin(ready_race_ids).sum()) if len(pre_race_only) else 0
    pending_row_count = int(row_race_ids.isin(pending_race_ids).sum()) if len(pre_race_only) else 0
    return {
        "pre_race_only_rows": int(len(pre_race_only)),
        "pre_race_only_races": int(len(race_ids)),
        "pre_race_only_dates": dates,
        "result_ready_races": int(len(ready_race_ids)),
        "pending_result_races": int(len(pending_race_ids)),
        "result_ready_rows": ready_row_count,
        "pending_result_rows": pending_row_count,
        "ready_for_benchmark_rerun": bool(len(pre_race_only) > 0 and len(pending_race_ids) == 0),
    }


def build_pre_race_readiness_probe_summary(
    frame: pd.DataFrame,
    *,
    result_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    materialization_summary = build_pre_race_only_materialization_summary(frame, result_frame=result_frame)
    ready_races = int(materialization_summary["result_ready_races"])
    pending_races = int(materialization_summary["pending_result_races"])
    status = "ready" if ready_races > 0 else "not_ready"
    current_phase = "ready_for_benchmark_handoff" if ready_races > 0 else "await_result_arrival"
    recommended_action = "run_pre_race_benchmark_handoff" if ready_races > 0 else "wait_for_result_ready_pre_race_races"
    return {
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "materialization_summary": materialization_summary,
    }


def _extract_summary_value(summary: dict[str, Any] | None, key: str) -> Any:
    if not isinstance(summary, dict):
        return None
    if key in summary:
        return summary.get(key)
    materialization = summary.get("materialization_summary")
    if isinstance(materialization, dict) and key in materialization:
        return materialization.get(key)
    return None


def build_pre_race_capture_date_coverage(
    frame: pd.DataFrame,
    *,
    result_frame: pd.DataFrame | None = None,
) -> pd.DataFrame:
    pre_race_only = filter_pre_race_only(frame)
    coverage_columns = [
        "date",
        "pre_race_rows",
        "pre_race_races",
        "result_ready_races",
        "pending_result_races",
        "result_ready_rows",
        "pending_result_rows",
    ]
    if len(pre_race_only) == 0:
        return pd.DataFrame(columns=coverage_columns)

    working = pre_race_only.copy()
    if "date" not in working.columns:
        working["date"] = RELATION_UNKNOWN
    working["date"] = working["date"].fillna(RELATION_UNKNOWN).astype(str)
    if "race_id" not in working.columns:
        working["race_id"] = ""
    working["race_id"] = working["race_id"].fillna("").astype(str)

    ready_race_ids: set[str] = set()
    if result_frame is not None and "race_id" in result_frame.columns:
        ready_race_ids = {str(value) for value in result_frame["race_id"].dropna().tolist()}

    records: list[dict[str, Any]] = []
    for date_value, date_frame in working.groupby("date", dropna=False):
        race_ids = {str(value) for value in date_frame["race_id"].dropna().tolist() if str(value)}
        ready_ids = sorted(race_ids & ready_race_ids)
        pending_ids = sorted(race_ids - ready_race_ids)
        records.append(
            {
                "date": str(date_value),
                "pre_race_rows": int(len(date_frame)),
                "pre_race_races": int(len(race_ids)),
                "result_ready_races": int(len(ready_ids)),
                "pending_result_races": int(len(pending_ids)),
                "result_ready_rows": int(date_frame["race_id"].isin(ready_ids).sum()),
                "pending_result_rows": int(date_frame["race_id"].isin(pending_ids).sum()),
            }
        )

    return pd.DataFrame.from_records(records, columns=coverage_columns).sort_values("date").reset_index(drop=True)


def build_pre_race_capture_coverage_summary(
    frame: pd.DataFrame,
    *,
    result_frame: pd.DataFrame | None = None,
    previous_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    materialization_summary = build_pre_race_only_materialization_summary(frame, result_frame=result_frame)
    date_coverage = build_pre_race_capture_date_coverage(frame, result_frame=result_frame)

    if materialization_summary["pre_race_only_rows"] <= 0:
        status = "empty"
        current_phase = "capture_empty"
        recommended_action = "expand_capture_window_or_wait_for_upcoming_cards"
    elif materialization_summary["ready_for_benchmark_rerun"]:
        status = "ready"
        current_phase = "ready_for_benchmark_rerun"
        recommended_action = "run_pre_race_benchmark_handoff"
    else:
        status = "capturing"
        current_phase = "capturing_pre_race_pool"
        recommended_action = "continue_recrawl_cadence_and_wait_for_results"

    current_dates = materialization_summary["pre_race_only_dates"]
    previous_rows = _extract_summary_value(previous_summary, "pre_race_only_rows")
    previous_races = _extract_summary_value(previous_summary, "pre_race_only_races")
    previous_ready_races = _extract_summary_value(previous_summary, "result_ready_races")
    previous_pending_races = _extract_summary_value(previous_summary, "pending_result_races")
    previous_dates_raw = _extract_summary_value(previous_summary, "pre_race_only_dates")
    previous_dates = [str(value) for value in previous_dates_raw] if isinstance(previous_dates_raw, list) else []

    return {
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "pre_race_only_rows": int(materialization_summary["pre_race_only_rows"]),
        "pre_race_only_races": int(materialization_summary["pre_race_only_races"]),
        "pre_race_only_dates": list(current_dates),
        "date_count": int(len(current_dates)),
        "result_ready_races": int(materialization_summary["result_ready_races"]),
        "pending_result_races": int(materialization_summary["pending_result_races"]),
        "result_ready_rows": int(materialization_summary["result_ready_rows"]),
        "pending_result_rows": int(materialization_summary["pending_result_rows"]),
        "ready_for_benchmark_rerun": bool(materialization_summary["ready_for_benchmark_rerun"]),
        "date_coverage": date_coverage.to_dict(orient="records"),
        "baseline_comparison": {
            "available": previous_summary is not None,
            "previous_pre_race_only_rows": int(previous_rows) if previous_rows is not None else None,
            "previous_pre_race_only_races": int(previous_races) if previous_races is not None else None,
            "previous_result_ready_races": int(previous_ready_races) if previous_ready_races is not None else None,
            "previous_pending_result_races": int(previous_pending_races) if previous_pending_races is not None else None,
            "delta_pre_race_only_rows": (
                int(materialization_summary["pre_race_only_rows"]) - int(previous_rows)
                if previous_rows is not None
                else None
            ),
            "delta_pre_race_only_races": (
                int(materialization_summary["pre_race_only_races"]) - int(previous_races)
                if previous_races is not None
                else None
            ),
            "delta_result_ready_races": (
                int(materialization_summary["result_ready_races"]) - int(previous_ready_races)
                if previous_ready_races is not None
                else None
            ),
            "delta_pending_result_races": (
                int(materialization_summary["pending_result_races"]) - int(previous_pending_races)
                if previous_pending_races is not None
                else None
            ),
            "added_dates": sorted(set(current_dates) - set(previous_dates)),
            "removed_dates": sorted(set(previous_dates) - set(current_dates)),
        },
    }


def _normalize_string_series(frame: pd.DataFrame, column: str, *, default: str = RELATION_UNKNOWN) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([default] * len(frame), index=frame.index, dtype="object")
    return frame[column].fillna(default).astype(str)


def _build_source_timing_breakdown(
    frame: pd.DataFrame,
    *,
    group_column: str,
    result_ready_race_ids: set[str],
) -> pd.DataFrame:
    breakdown_columns = [
        group_column,
        "row_count",
        "race_count",
        "result_ready_rows",
        "result_ready_races",
        "pre_race_rows",
        "unknown_rows",
        "post_race_rows",
        "result_ready_pre_race_rows",
        "result_ready_unknown_rows",
        "result_ready_post_race_rows",
    ]
    if len(frame) == 0:
        return pd.DataFrame(columns=breakdown_columns)

    working = frame.copy()
    working[group_column] = _normalize_string_series(working, group_column)
    working["race_id"] = _normalize_string_series(working, "race_id", default="")
    working["market_timing_bucket"] = _normalize_string_series(working, MARKET_TIMING_BUCKET_COLUMN)
    working["is_result_ready"] = working["race_id"].isin(result_ready_race_ids)

    records: list[dict[str, Any]] = []
    for group_value, group_frame in working.groupby(group_column, dropna=False):
        row_count = int(len(group_frame))
        race_count = int(group_frame["race_id"].loc[group_frame["race_id"] != ""].nunique())
        result_ready_frame = group_frame.loc[group_frame["is_result_ready"]]
        bucket_counts = group_frame["market_timing_bucket"].value_counts().to_dict()
        result_ready_bucket_counts = result_ready_frame["market_timing_bucket"].value_counts().to_dict()
        records.append(
            {
                group_column: str(group_value),
                "row_count": row_count,
                "race_count": race_count,
                "result_ready_rows": int(len(result_ready_frame)),
                "result_ready_races": int(result_ready_frame["race_id"].loc[result_ready_frame["race_id"] != ""].nunique()),
                "pre_race_rows": int(bucket_counts.get(RELATION_PRE_RACE, 0)),
                "unknown_rows": int(bucket_counts.get(RELATION_UNKNOWN, 0)),
                "post_race_rows": int(bucket_counts.get(RELATION_POST_RACE, 0)),
                "result_ready_pre_race_rows": int(result_ready_bucket_counts.get(RELATION_PRE_RACE, 0)),
                "result_ready_unknown_rows": int(result_ready_bucket_counts.get(RELATION_UNKNOWN, 0)),
                "result_ready_post_race_rows": int(result_ready_bucket_counts.get(RELATION_POST_RACE, 0)),
            }
        )

    return pd.DataFrame.from_records(records, columns=breakdown_columns).sort_values(group_column).reset_index(drop=True)


def build_source_timing_audit_summary(
    frame: pd.DataFrame,
    *,
    result_frame: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    annotated = annotate_market_timing_bucket(frame)
    working = annotated.copy()
    working["race_id"] = _normalize_string_series(working, "race_id", default="")
    working["date"] = _normalize_string_series(working, "date")
    working["card_fetch_mode"] = _normalize_string_series(working, "card_fetch_mode")
    working["odds_fetch_mode"] = _normalize_string_series(working, "odds_fetch_mode")
    working["fetch_mode_pair"] = working["card_fetch_mode"] + "|" + working["odds_fetch_mode"]
    working["year"] = working["date"].str.slice(0, 4)

    result_ready_race_ids: set[str] = set()
    if result_frame is not None and "race_id" in result_frame.columns:
        result_ready_race_ids = {str(value) for value in result_frame["race_id"].dropna().tolist()}
    working["is_result_ready"] = working["race_id"].isin(result_ready_race_ids)

    result_ready_frame = working.loc[working["is_result_ready"]].copy()
    overall_summary = build_provenance_summary(working)
    result_ready_summary = build_provenance_summary(result_ready_frame)

    by_date = _build_source_timing_breakdown(working, group_column="date", result_ready_race_ids=result_ready_race_ids)
    by_year = _build_source_timing_breakdown(working, group_column="year", result_ready_race_ids=result_ready_race_ids)

    fetch_mode_records: list[dict[str, Any]] = []
    for fetch_mode_pair, pair_frame in working.groupby("fetch_mode_pair", dropna=False):
        result_ready_pair = pair_frame.loc[pair_frame["is_result_ready"]]
        bucket_counts = pair_frame[MARKET_TIMING_BUCKET_COLUMN].value_counts().to_dict()
        result_ready_bucket_counts = result_ready_pair[MARKET_TIMING_BUCKET_COLUMN].value_counts().to_dict()
        card_mode, odds_mode = str(fetch_mode_pair).split("|", 1)
        fetch_mode_records.append(
            {
                "card_fetch_mode": card_mode,
                "odds_fetch_mode": odds_mode,
                "row_count": int(len(pair_frame)),
                "race_count": int(pair_frame["race_id"].loc[pair_frame["race_id"] != ""].nunique()),
                "result_ready_rows": int(len(result_ready_pair)),
                "result_ready_races": int(result_ready_pair["race_id"].loc[result_ready_pair["race_id"] != ""].nunique()),
                "pre_race_rows": int(bucket_counts.get(RELATION_PRE_RACE, 0)),
                "unknown_rows": int(bucket_counts.get(RELATION_UNKNOWN, 0)),
                "post_race_rows": int(bucket_counts.get(RELATION_POST_RACE, 0)),
                "result_ready_pre_race_rows": int(result_ready_bucket_counts.get(RELATION_PRE_RACE, 0)),
                "result_ready_unknown_rows": int(result_ready_bucket_counts.get(RELATION_UNKNOWN, 0)),
                "result_ready_post_race_rows": int(result_ready_bucket_counts.get(RELATION_POST_RACE, 0)),
            }
        )

    fetch_mode_summary = sorted(fetch_mode_records, key=lambda record: (record["card_fetch_mode"], record["odds_fetch_mode"]))
    pre_race_dates = sorted({str(value) for value in working.loc[working[MARKET_TIMING_BUCKET_COLUMN] == RELATION_PRE_RACE, "date"].tolist() if str(value)})
    result_ready_pre_race_dates = sorted(
        {
            str(value)
            for value in result_ready_frame.loc[result_ready_frame[MARKET_TIMING_BUCKET_COLUMN] == RELATION_PRE_RACE, "date"].tolist()
            if str(value)
        }
    )
    result_ready_unknown_dates = sorted(
        {
            str(value)
            for value in result_ready_frame.loc[result_ready_frame[MARKET_TIMING_BUCKET_COLUMN] == RELATION_UNKNOWN, "date"].tolist()
            if str(value)
        }
    )

    result_ready_pre_race_rows = int(result_ready_summary.get("pre_race_only_rows") or 0)
    future_only_pre_race_rows = int(overall_summary.get("pre_race_only_rows") or 0) - result_ready_pre_race_rows
    if result_ready_pre_race_rows > 0:
        historical_status = "historical_pre_race_subset_available"
        recommended_action = "materialize_result_ready_pre_race_subset"
    elif future_only_pre_race_rows > 0:
        historical_status = "future_only_pre_race_capture_available"
        recommended_action = "downgrade_historical_benchmark_to_diagnostic_only"
    else:
        historical_status = "no_pre_race_capture_in_current_cache"
        recommended_action = "rebuild_local_backfill_with_true_pre_race_capture_source"

    summary = {
        "status": "completed",
        "current_phase": historical_status,
        "recommended_action": recommended_action,
        "overall_summary": overall_summary,
        "result_ready_summary": result_ready_summary,
        "result_ready_race_count": int(len(result_ready_race_ids)),
        "pre_race_dates": pre_race_dates,
        "result_ready_pre_race_dates": result_ready_pre_race_dates,
        "result_ready_unknown_dates": result_ready_unknown_dates,
        "fetch_mode_summary": fetch_mode_summary,
        "historical_pre_race_recoverability": {
            "status": historical_status,
            "result_ready_pre_race_rows": result_ready_pre_race_rows,
            "future_only_pre_race_rows": int(max(future_only_pre_race_rows, 0)),
            "result_ready_post_race_rows": int(result_ready_summary.get("post_race_rows") or 0),
            "result_ready_unknown_rows": int(result_ready_summary.get("unknown_rows") or 0),
        },
    }
    return summary, by_date, by_year


def build_historical_source_timing_context(source_timing_summary: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(source_timing_summary, dict):
        return None
    recoverability = source_timing_summary.get("historical_pre_race_recoverability")
    if not isinstance(recoverability, dict):
        return None

    status = str(recoverability.get("status") or source_timing_summary.get("current_phase") or "")
    result_ready_pre_race_rows = int(recoverability.get("result_ready_pre_race_rows") or 0)
    future_only_pre_race_rows = int(recoverability.get("future_only_pre_race_rows") or 0)

    if result_ready_pre_race_rows > 0:
        return {
            "status": "historical_pre_race_subset_available",
            "current_phase": "historical_pre_race_subset_available",
            "recommended_action": "run_pre_race_benchmark_handoff",
            "result_ready_pre_race_rows": result_ready_pre_race_rows,
            "future_only_pre_race_rows": future_only_pre_race_rows,
        }

    if status == "future_only_pre_race_capture_available":
        return {
            "status": status,
            "current_phase": "future_only_readiness_track",
            "recommended_action": "capture_future_pre_race_rows_and_wait_for_results",
            "result_ready_pre_race_rows": result_ready_pre_race_rows,
            "future_only_pre_race_rows": future_only_pre_race_rows,
        }

    if status == "no_pre_race_capture_in_current_cache":
        return {
            "status": status,
            "current_phase": "historical_source_timing_blocked",
            "recommended_action": str(source_timing_summary.get("recommended_action") or "rebuild_local_backfill_with_true_pre_race_capture_source"),
            "result_ready_pre_race_rows": result_ready_pre_race_rows,
            "future_only_pre_race_rows": future_only_pre_race_rows,
        }

    return {
        "status": status or "unknown",
        "current_phase": str(source_timing_summary.get("current_phase") or "historical_source_timing_review"),
        "recommended_action": str(source_timing_summary.get("recommended_action") or "inspect_source_timing_audit"),
        "result_ready_pre_race_rows": result_ready_pre_race_rows,
        "future_only_pre_race_rows": future_only_pre_race_rows,
    }


def apply_source_timing_context_to_readiness_summary(
    readiness_summary: dict[str, Any],
    *,
    source_timing_summary: dict[str, Any] | None,
) -> dict[str, Any]:
    output = dict(readiness_summary)
    context = build_historical_source_timing_context(source_timing_summary)
    if context is None:
        return output
    output["historical_source_timing"] = context
    if str(output.get("status") or "") == "ready":
        return output
    output["current_phase"] = context["current_phase"]
    output["recommended_action"] = context["recommended_action"]
    return output
