from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pandas as pd


DEFAULT_STABILITY_THRESHOLDS: dict[str, int] = {
    "min_races": 500,
    "min_dates": 20,
    "min_window_days": 90,
    "min_ev_threshold_1_0_bets": 200,
}


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_date(value: Any) -> pd.Timestamp | None:
    if value is None:
        return None
    timestamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(timestamp):
        return None
    return pd.Timestamp(timestamp)


def _window_days_from_dates(start_value: Any, end_value: Any) -> int | None:
    start_ts = _coerce_date(start_value)
    end_ts = _coerce_date(end_value)
    if start_ts is None or end_ts is None:
        return None
    return int((end_ts - start_ts).days + 1)


def _safe_series_stats(series: pd.Series) -> tuple[float | None, float | None, float | None]:
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return None, None, None
    mean_value = float(numeric.mean())
    std_value = float(numeric.std(ddof=0)) if len(numeric) > 1 else 0.0
    positive_ratio = float((numeric > 1.0).mean())
    return mean_value, std_value, positive_ratio


def _resolve_summary_window_days(summary: Mapping[str, Any] | None) -> int | None:
    if not isinstance(summary, Mapping):
        return None

    date_window = summary.get("date_window")
    if isinstance(date_window, Mapping):
        window_days = _window_days_from_dates(
            date_window.get("start", date_window.get("start_date")),
            date_window.get("end", date_window.get("end_date")),
        )
        if window_days is not None:
            return window_days

    run_context = summary.get("run_context")
    if isinstance(run_context, Mapping):
        window_days = _window_days_from_dates(
            run_context.get("start_date"),
            run_context.get("end_date"),
        )
        if window_days is not None:
            return window_days

    return _window_days_from_dates(summary.get("start_date"), summary.get("end_date"))


def build_stability_guardrail(
    *,
    frame: pd.DataFrame | None = None,
    by_date: pd.DataFrame | None = None,
    summary: Mapping[str, Any] | None = None,
    thresholds: Mapping[str, int] | None = None,
    date_col: str = "date",
    race_id_col: str = "race_id",
) -> dict[str, Any]:
    resolved_thresholds = dict(DEFAULT_STABILITY_THRESHOLDS)
    if thresholds:
        resolved_thresholds.update({key: int(value) for key, value in thresholds.items()})

    frame_date_series = pd.Series(dtype="datetime64[ns]")
    if frame is not None and date_col in frame.columns:
        frame_date_series = pd.to_datetime(frame[date_col], errors="coerce").dropna().sort_values().reset_index(drop=True)

    by_date_series = pd.Series(dtype="datetime64[ns]")
    if by_date is not None and not by_date.empty and "date" in by_date.columns:
        by_date_series = pd.to_datetime(by_date["date"], errors="coerce").dropna().sort_values().reset_index(drop=True)

    n_races = None
    if frame is not None and race_id_col in frame.columns:
        n_races = int(frame[race_id_col].nunique())
    elif isinstance(summary, Mapping):
        n_races = _coerce_int(summary.get("n_races", summary.get("races")))

    n_dates = None
    if not by_date_series.empty:
        n_dates = int(by_date_series.dt.normalize().nunique())
    elif not frame_date_series.empty:
        n_dates = int(frame_date_series.dt.normalize().nunique())
    elif isinstance(summary, Mapping):
        n_dates = _coerce_int(summary.get("n_dates"))

    window_days = None
    active_date_series = by_date_series if not by_date_series.empty else frame_date_series
    if not active_date_series.empty:
        window_days = int((pd.Timestamp(active_date_series.iloc[-1]) - pd.Timestamp(active_date_series.iloc[0])).days + 1)
    else:
        window_days = _resolve_summary_window_days(summary)

    ev_threshold_1_0_bets = None
    ev_threshold_1_2_bets = None
    if isinstance(summary, Mapping):
        ev_threshold_1_0_bets = _coerce_int(summary.get("ev_threshold_1_0_bets"))
        ev_threshold_1_2_bets = _coerce_int(summary.get("ev_threshold_1_2_bets"))

    observed: dict[str, Any] = {
        "n_races": n_races,
        "n_dates": n_dates,
        "window_days": window_days,
        "top1_bets": n_races,
        "ev_threshold_1_0_bets": ev_threshold_1_0_bets,
        "ev_threshold_1_2_bets": ev_threshold_1_2_bets,
    }
    if n_races is not None and n_dates not in (None, 0):
        observed["mean_races_per_date"] = float(n_races / n_dates)
    else:
        observed["mean_races_per_date"] = None
    if n_dates is not None and window_days not in (None, 0):
        observed["date_coverage_ratio"] = float(n_dates / window_days)
    else:
        observed["date_coverage_ratio"] = None

    if by_date is not None and not by_date.empty:
        for column in ("top1_roi", "ev_threshold_1_0_roi", "ev_threshold_1_2_roi"):
            if column not in by_date.columns:
                continue
            mean_value, std_value, positive_ratio = _safe_series_stats(by_date[column])
            prefix = column.replace("_roi", "")
            observed[f"{prefix}_roi_by_date_mean"] = mean_value
            observed[f"{prefix}_roi_by_date_std"] = std_value
            observed[f"{prefix}_profitable_date_ratio"] = positive_ratio

    checks = [
        ("n_races", "min_races", "race sample is too small"),
        ("n_dates", "min_dates", "covered dates are too few"),
        ("window_days", "min_window_days", "calendar span is too short"),
        (
            "ev_threshold_1_0_bets",
            "min_ev_threshold_1_0_bets",
            "EV>=1.0 bet count is too small",
        ),
    ]

    failed_checks: list[dict[str, Any]] = []
    skipped_checks: list[dict[str, Any]] = []
    warnings: list[str] = []
    severe_failures = 0

    for metric_name, threshold_name, message in checks:
        observed_value = observed.get(metric_name)
        threshold_value = int(resolved_thresholds[threshold_name])
        if observed_value is None:
            skipped_checks.append(
                {
                    "metric": metric_name,
                    "threshold": threshold_value,
                    "message": message,
                    "reason": "metric_unavailable",
                }
            )
            continue
        if observed_value >= threshold_value:
            continue
        shortfall_ratio = float(observed_value) / float(threshold_value) if threshold_value else 0.0
        severity = "severe" if shortfall_ratio < 0.5 else "moderate"
        if severity == "severe":
            severe_failures += 1
        failed_checks.append(
            {
                "metric": metric_name,
                "observed": observed_value,
                "threshold": threshold_value,
                "severity": severity,
                "message": message,
            }
        )
        warnings.append(f"{message}: observed={observed_value}, threshold={threshold_value}")

    available_check_count = len(checks) - len(skipped_checks)
    if available_check_count == 0:
        assessment = "caution"
        warnings.append("no stability checks were available for this artifact")
    elif not failed_checks:
        assessment = "representative"
    elif severe_failures >= 1 or len(failed_checks) >= 2:
        assessment = "probe_only"
    else:
        assessment = "caution"

    notes = [
        "assessment is support-based and intentionally ignores achieved ROI so small-sample wins are not over-weighted",
        "probe_only windows are suitable for smoke checks and regression detection, not for model promotion decisions",
    ]
    if skipped_checks:
        notes.append("some checks may be skipped when the artifact does not expose the required metrics")

    return {
        "assessment": assessment,
        "is_representative": bool(assessment == "representative"),
        "available_check_count": int(available_check_count),
        "failed_check_count": int(len(failed_checks)),
        "failed_checks": failed_checks,
        "skipped_checks": skipped_checks,
        "thresholds": resolved_thresholds,
        "observed": observed,
        "warnings": warnings,
        "notes": notes,
    }
