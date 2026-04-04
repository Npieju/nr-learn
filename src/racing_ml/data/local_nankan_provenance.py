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
