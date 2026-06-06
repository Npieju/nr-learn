from __future__ import annotations

from typing import Any

import pandas as pd


DEFAULT_JOIN_KEY_CANDIDATES: tuple[tuple[str, ...], ...] = (
    ("race_id", "gate_no"),
    ("race_id", "horse_no"),
    ("race_id", "horse_id"),
    ("race_id", "horse_key"),
)

DEFAULT_MARKET_COLUMNS: tuple[str, ...] = (
    "odds",
    "単勝",
    "popularity",
    "odds_official_datetime",
    "odds_updated_at",
)

_NUMERIC_JOIN_KEYS = {"gate_no", "horse_no", "horse_number", "frame_no", "枠番", "馬番"}


def _normalize_join_value(series: pd.Series, key: str) -> pd.Series:
    if key in _NUMERIC_JOIN_KEYS:
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.round().astype("Int64").astype("string")
    return series.astype("string").str.strip()


def resolve_market_join_keys(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    explicit_join_keys: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    if explicit_join_keys is not None:
        keys = [str(key).strip() for key in explicit_join_keys if str(key).strip()]
        if not keys:
            raise ValueError("explicit join keys must not be empty")
        missing = [key for key in keys if key not in predictions.columns or key not in market.columns]
        if missing:
            raise ValueError(f"join keys missing from predictions or market frame: {', '.join(missing)}")
        return keys

    for candidate in DEFAULT_JOIN_KEY_CANDIDATES:
        if all(key in predictions.columns and key in market.columns for key in candidate):
            return list(candidate)
    raise ValueError(
        "could not resolve join keys; expected one of: "
        + ", ".join("+".join(candidate) for candidate in DEFAULT_JOIN_KEY_CANDIDATES)
    )


def refresh_prediction_market_data(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    *,
    join_keys: list[str] | tuple[str, ...] | None = None,
    market_columns: list[str] | tuple[str, ...] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    resolved_join_keys = resolve_market_join_keys(predictions, market, explicit_join_keys=join_keys)
    resolved_market_columns = [
        column
        for column in (list(market_columns) if market_columns is not None else list(DEFAULT_MARKET_COLUMNS))
        if column in market.columns
    ]
    if not resolved_market_columns:
        raise ValueError("market frame does not include any refreshable market columns")

    prediction_rows = int(len(predictions))
    market_subset = market.loc[:, list(dict.fromkeys([*resolved_join_keys, *resolved_market_columns]))].copy()
    duplicate_mask = market_subset.duplicated(subset=resolved_join_keys, keep=False)
    duplicate_rows = int(duplicate_mask.sum())
    if duplicate_rows:
        market_subset = market_subset.drop_duplicates(subset=resolved_join_keys, keep="last").copy()

    pred_working = predictions.copy()
    market_working = market_subset.copy()
    temp_keys: list[str] = []
    for key in resolved_join_keys:
        temp_key = f"__join_{key}"
        pred_working[temp_key] = _normalize_join_value(pred_working[key], key)
        market_working[temp_key] = _normalize_join_value(market_working[key], key)
        temp_keys.append(temp_key)

    pred_working = pred_working.drop(columns=resolved_market_columns, errors="ignore")
    market_payload = market_working.drop(columns=resolved_join_keys, errors="ignore")
    merged = pred_working.merge(market_payload, on=temp_keys, how="left", validate="m:1")

    refreshed_rows = int(pd.to_numeric(merged.get("odds"), errors="coerce").notna().sum()) if "odds" in merged.columns else 0
    missing_market_rows = prediction_rows - refreshed_rows if "odds" in merged.columns else prediction_rows

    merged = merged.drop(columns=temp_keys, errors="ignore")
    summary: dict[str, Any] = {
        "prediction_rows": prediction_rows,
        "market_rows": int(len(market_subset)),
        "join_keys": resolved_join_keys,
        "market_columns": resolved_market_columns,
        "market_duplicate_rows_dropped": duplicate_rows,
        "rows_with_odds_after_refresh": refreshed_rows,
        "rows_missing_odds_after_refresh": missing_market_rows,
    }
    return merged, summary