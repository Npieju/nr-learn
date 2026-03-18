from __future__ import annotations

from typing import Any

import pandas as pd


def _resolve_when_value(when_cfg: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        value = when_cfg.get(key)
        if value is not None:
            return value
    return None


def extract_date_window(
    frame: pd.DataFrame,
    *,
    date_col: str = "date",
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    if date_col not in frame.columns:
        return None

    date_series = pd.to_datetime(frame[date_col], errors="coerce")
    valid_dates = pd.Series(date_series.dropna().unique()).sort_values().reset_index(drop=True)
    if valid_dates.empty:
        return None

    window_start = pd.Timestamp(valid_dates.iloc[0])
    window_end = pd.Timestamp(valid_dates.iloc[-1])
    return window_start, window_end


def regime_matches_window(
    when_cfg: dict[str, Any] | None,
    *,
    window_start: pd.Timestamp,
    window_end: pd.Timestamp,
) -> bool:
    if not isinstance(when_cfg, dict):
        return False

    start_after = _resolve_when_value(when_cfg, "valid_start_on_or_after", "start_on_or_after", "date_on_or_after")
    start_month_on_or_after = _resolve_when_value(
        when_cfg,
        "valid_start_month_on_or_after",
        "start_month_on_or_after",
        "date_month_on_or_after",
    )
    start_month_on_or_before = _resolve_when_value(
        when_cfg,
        "valid_start_month_on_or_before",
        "start_month_on_or_before",
        "date_month_on_or_before",
    )
    end_before = _resolve_when_value(when_cfg, "valid_end_before", "end_before", "date_before")
    end_on_or_before = _resolve_when_value(when_cfg, "valid_end_on_or_before", "end_on_or_before", "date_on_or_before")
    end_month_on_or_after = _resolve_when_value(
        when_cfg,
        "valid_end_month_on_or_after",
        "end_month_on_or_after",
        "date_month_on_or_after",
    )
    end_month_on_or_before = _resolve_when_value(
        when_cfg,
        "valid_end_month_on_or_before",
        "end_month_on_or_before",
        "date_month_on_or_before",
    )
    end_month_in = _resolve_when_value(when_cfg, "valid_end_month_in", "end_month_in", "date_month_in")

    if start_after is not None and window_start < pd.Timestamp(start_after):
        return False
    if start_month_on_or_after is not None and int(window_start.month) < int(start_month_on_or_after):
        return False
    if start_month_on_or_before is not None and int(window_start.month) > int(start_month_on_or_before):
        return False
    if end_before is not None and window_end >= pd.Timestamp(end_before):
        return False
    if end_on_or_before is not None and window_end > pd.Timestamp(end_on_or_before):
        return False
    if end_month_on_or_after is not None and int(window_end.month) < int(end_month_on_or_after):
        return False
    if end_month_on_or_before is not None and int(window_end.month) > int(end_month_on_or_before):
        return False
    if isinstance(end_month_in, (list, tuple)) and end_month_in:
        allowed_months = {int(value) for value in end_month_in}
        if int(window_end.month) not in allowed_months:
            return False

    return True


def resolve_regime_override(
    overrides: list[dict[str, Any]] | None,
    *,
    frame: pd.DataFrame | None = None,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
    date_col: str = "date",
) -> dict[str, Any] | None:
    if not isinstance(overrides, list) or not overrides:
        return None

    if frame is not None:
        window = extract_date_window(frame, date_col=date_col)
        if window is None:
            return None
        resolved_start, resolved_end = window
    else:
        if window_start is None or window_end is None:
            return None
        resolved_start = pd.Timestamp(window_start)
        resolved_end = pd.Timestamp(window_end)

    for override in overrides:
        if not isinstance(override, dict):
            continue
        if regime_matches_window(
            override.get("when"),
            window_start=resolved_start,
            window_end=resolved_end,
        ):
            return override

    return None


def resolve_regime_name(
    overrides: list[dict[str, Any]] | None,
    *,
    frame: pd.DataFrame | None = None,
    window_start: pd.Timestamp | str | None = None,
    window_end: pd.Timestamp | str | None = None,
    date_col: str = "date",
    default_name: str = "default",
) -> str:
    override = resolve_regime_override(
        overrides,
        frame=frame,
        window_start=window_start,
        window_end=window_end,
        date_col=date_col,
    )
    if not isinstance(override, dict):
        return default_name

    override_name = str(override.get("name", "")).strip()
    return override_name or default_name