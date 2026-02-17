from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def run_leakage_audit(
    frame: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    corr_threshold: float = 0.98,
    exact_match_threshold: float = 0.999,
) -> dict[str, Any]:
    available_features = [column for column in feature_columns if column in frame.columns]
    report: dict[str, Any] = {
        "enabled": True,
        "label_column": label_column,
        "n_features_checked": int(len(available_features)),
        "label_missing": bool(label_column not in frame.columns),
        "keyword_flags": [],
        "high_corr_flags": [],
        "exact_match_flags": [],
        "suspicious_feature_count": 0,
        "is_suspicious": False,
    }

    if label_column not in frame.columns or not available_features:
        return report

    label_series = pd.to_numeric(frame[label_column], errors="coerce")
    if label_series.isna().all():
        return report

    risky_keywords = {
        "rank",
        "is_win",
        "着順",
        "result",
        "payout",
        "払戻",
        "配当",
        "確定",
    }

    keyword_flags: list[str] = []
    high_corr_flags: list[dict[str, Any]] = []
    exact_match_flags: list[dict[str, Any]] = []

    for feature in available_features:
        lower_name = feature.lower()
        looks_like_history = ("last_" in lower_name) or lower_name.startswith("prev_")
        if (not looks_like_history) and any(keyword in lower_name for keyword in risky_keywords):
            keyword_flags.append(feature)

        series = pd.to_numeric(frame[feature], errors="coerce")
        valid_mask = series.notna() & label_series.notna()
        if int(valid_mask.sum()) < 50:
            continue

        feature_vals = series[valid_mask].to_numpy(dtype=float)
        label_vals = label_series[valid_mask].to_numpy(dtype=float)
        if np.std(feature_vals) <= 1e-12 or np.std(label_vals) <= 1e-12:
            continue

        corr = float(np.corrcoef(feature_vals, label_vals)[0, 1])
        if np.isfinite(corr) and abs(corr) >= corr_threshold:
            high_corr_flags.append({"feature": feature, "abs_corr": float(abs(corr))})

        rounded = np.rint(feature_vals)
        if np.all(np.isfinite(rounded)):
            match_ratio = float(np.mean((rounded == label_vals).astype(float)))
            if match_ratio >= exact_match_threshold:
                exact_match_flags.append({"feature": feature, "match_ratio": match_ratio})

    suspicious = sorted(set(keyword_flags) | {item["feature"] for item in high_corr_flags} | {item["feature"] for item in exact_match_flags})

    report["keyword_flags"] = sorted(keyword_flags)
    report["high_corr_flags"] = sorted(high_corr_flags, key=lambda x: x["abs_corr"], reverse=True)
    report["exact_match_flags"] = sorted(exact_match_flags, key=lambda x: x["match_ratio"], reverse=True)
    report["suspicious_features"] = suspicious
    report["suspicious_feature_count"] = int(len(suspicious))
    report["is_suspicious"] = bool(len(suspicious) > 0)
    return report
