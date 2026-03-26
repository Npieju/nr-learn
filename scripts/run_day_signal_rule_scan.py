from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DEFAULT_FEATURES = [
    "selected_race_count",
    "max_expected_value_median",
    "max_expected_value_max",
    "max_prob_median",
    "max_prob_max",
    "max_edge_median",
    "max_edge_max",
]


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[day-signal-scan {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _ensure_output_path(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _load_summary(path: Path, stage_index: int, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "stage_index" not in frame.columns:
        raise ValueError(f"missing stage_index column: {path}")
    filtered = frame.loc[frame["stage_index"] == stage_index].copy()
    if filtered.empty:
        raise ValueError(f"no rows for stage_index={stage_index}: {path}")
    filtered["window_label"] = label
    filtered["date"] = pd.to_datetime(filtered["date"]).dt.strftime("%Y-%m-%d")
    return filtered


def _normalize_feature_list(raw_values: list[str] | None) -> list[str]:
    if not raw_values:
        return list(DEFAULT_FEATURES)
    values: list[str] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            text = part.strip()
            if text:
                values.append(text)
    if not values:
        raise ValueError("feature list is empty")
    return values


def _normalize_net_mode(text: str) -> str:
    value = text.strip().lower()
    choices = {"positive", "non_positive", "negative", "non_negative"}
    if value not in choices:
        raise ValueError(f"unsupported net mode: {text}")
    return value


def _net_mask(frame: pd.DataFrame, mode: str) -> pd.Series:
    values = frame["total_selected_net_units"].fillna(0.0)
    if mode == "positive":
        return values > 0.0
    if mode == "non_positive":
        return values <= 0.0
    if mode == "negative":
        return values < 0.0
    if mode == "non_negative":
        return values >= 0.0
    raise ValueError(f"unsupported net mode: {mode}")


def _sample_thresholds(series: pd.Series, limit: int) -> list[float]:
    values = sorted(set(float(v) for v in series.dropna().round(6).tolist()))
    if len(values) <= limit:
        return values
    step = max(1, len(values) // limit)
    sampled = values[::step]
    if sampled[-1] != values[-1]:
        sampled.append(values[-1])
    return sampled


def _evaluate_rule(
    frame: pd.DataFrame,
    *,
    mask: pd.Series,
    target_index: pd.Index,
    avoid_index: pd.Index,
) -> dict[str, Any]:
    target_mask = frame.index.isin(target_index) & mask.fillna(False)
    avoid_mask = frame.index.isin(avoid_index) & mask.fillna(False)
    target_hits = frame.loc[target_mask].copy()
    avoid_hits = frame.loc[avoid_mask].copy()
    target_size = int(len(target_index))
    avoid_size = int(len(avoid_index))
    target_covered = int(len(target_hits))
    avoid_hit = int(len(avoid_hits))
    target_missed = int(target_size - target_covered)
    return {
        "target_size": target_size,
        "avoid_size": avoid_size,
        "target_covered": target_covered,
        "target_missed": target_missed,
        "avoid_hit": avoid_hit,
        "target_cover_rate": float(target_covered / target_size) if target_size else 0.0,
        "avoid_hit_rate": float(avoid_hit / avoid_size) if avoid_size else 0.0,
        "target_dates": sorted(target_hits["date"].astype(str).tolist()),
        "avoid_dates": sorted(avoid_hits["date"].astype(str).tolist()),
    }


def _single_feature_rules(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target_index: pd.Index,
    avoid_index: pd.Index,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for feature in features:
        thresholds = _sample_thresholds(frame[feature], limit=64)
        for threshold in thresholds:
            for operator in ("<=", ">="):
                if operator == "<=":
                    mask = frame[feature] <= threshold
                else:
                    mask = frame[feature] >= threshold
                metrics = _evaluate_rule(frame, mask=mask, target_index=target_index, avoid_index=avoid_index)
                rows.append(
                    {
                        "rule_kind": "single",
                        "feature": feature,
                        "operator": operator,
                        "threshold": float(threshold),
                        "rule_text": f"{feature} {operator} {threshold:.6f}",
                        **metrics,
                    }
                )
    return rows


def _pair_rules(
    frame: pd.DataFrame,
    *,
    features: list[str],
    target_index: pd.Index,
    avoid_index: pd.Index,
    threshold_limit: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    threshold_map = {feature: _sample_thresholds(frame[feature], limit=threshold_limit) for feature in features}
    operators = ("<=", ">=")
    for left_feature in features:
        for right_feature in features:
            for left_threshold in threshold_map[left_feature]:
                for right_threshold in threshold_map[right_feature]:
                    for left_operator in operators:
                        for right_operator in operators:
                            if left_operator == "<=":
                                left_mask = frame[left_feature] <= left_threshold
                            else:
                                left_mask = frame[left_feature] >= left_threshold
                            if right_operator == "<=":
                                right_mask = frame[right_feature] <= right_threshold
                            else:
                                right_mask = frame[right_feature] >= right_threshold
                            mask = left_mask & right_mask
                            metrics = _evaluate_rule(frame, mask=mask, target_index=target_index, avoid_index=avoid_index)
                            rows.append(
                                {
                                    "rule_kind": "pair",
                                    "feature": left_feature,
                                    "operator": left_operator,
                                    "threshold": float(left_threshold),
                                    "feature_2": right_feature,
                                    "operator_2": right_operator,
                                    "threshold_2": float(right_threshold),
                                    "rule_text": (
                                        f"{left_feature} {left_operator} {left_threshold:.6f} AND "
                                        f"{right_feature} {right_operator} {right_threshold:.6f}"
                                    ),
                                    **metrics,
                                }
                            )
    return rows


def _sort_rule_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            int(row["avoid_hit"]),
            int(row["target_missed"]),
            -float(row["target_cover_rate"]),
            row["rule_text"],
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-summary", required=True)
    parser.add_argument("--avoid-summary", required=True)
    parser.add_argument("--target-label", default="target")
    parser.add_argument("--avoid-label", default="avoid")
    parser.add_argument("--stage-index", type=int, default=1)
    parser.add_argument("--target-net-mode", default="non_positive")
    parser.add_argument("--avoid-net-mode", default="positive")
    parser.add_argument("--feature", action="append", default=None)
    parser.add_argument("--include-pair-rules", action="store_true")
    parser.add_argument("--pair-threshold-limit", type=int, default=12)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    try:
        target_summary = _resolve_path(args.target_summary)
        avoid_summary = _resolve_path(args.avoid_summary)
        features = _normalize_feature_list(args.feature)
        target_net_mode = _normalize_net_mode(args.target_net_mode)
        avoid_net_mode = _normalize_net_mode(args.avoid_net_mode)

        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / "day_signal_rule_scan.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / "day_signal_rule_scan.csv"
        _ensure_output_path(output_json)
        _ensure_output_path(output_csv)

        log_progress("loading target/avoid summaries")
        target_frame = _load_summary(target_summary, stage_index=args.stage_index, label=args.target_label)
        avoid_frame = _load_summary(avoid_summary, stage_index=args.stage_index, label=args.avoid_label)

        missing_features = [feature for feature in features if feature not in target_frame.columns or feature not in avoid_frame.columns]
        if missing_features:
            raise ValueError(f"missing features in summaries: {missing_features}")

        frame = pd.concat([target_frame, avoid_frame], ignore_index=True)
        target_mask = (frame["window_label"] == args.target_label) & _net_mask(frame, target_net_mode)
        avoid_mask = (frame["window_label"] == args.avoid_label) & _net_mask(frame, avoid_net_mode)
        target_index = frame.index[target_mask]
        avoid_index = frame.index[avoid_mask]
        if len(target_index) == 0:
            raise ValueError("target selection is empty")
        if len(avoid_index) == 0:
            raise ValueError("avoid selection is empty")

        log_progress(f"scanning single-feature rules target={len(target_index)} avoid={len(avoid_index)}")
        rule_rows = _single_feature_rules(
            frame,
            features=features,
            target_index=target_index,
            avoid_index=avoid_index,
        )

        if args.include_pair_rules:
            log_progress("scanning pair rules")
            rule_rows.extend(
                _pair_rules(
                    frame,
                    features=features,
                    target_index=target_index,
                    avoid_index=avoid_index,
                    threshold_limit=max(2, int(args.pair_threshold_limit)),
                )
            )

        ranked_rows = _sort_rule_rows(rule_rows)
        top_k = max(1, int(args.top_k))
        top_rows = ranked_rows[:top_k]
        perfect_rows = [row for row in ranked_rows if int(row["avoid_hit"]) == 0 and int(row["target_missed"]) == 0]

        result = {
            "target_summary": str(target_summary.relative_to(ROOT)) if target_summary.is_relative_to(ROOT) else str(target_summary),
            "avoid_summary": str(avoid_summary.relative_to(ROOT)) if avoid_summary.is_relative_to(ROOT) else str(avoid_summary),
            "stage_index": int(args.stage_index),
            "target_label": args.target_label,
            "avoid_label": args.avoid_label,
            "target_net_mode": target_net_mode,
            "avoid_net_mode": avoid_net_mode,
            "features": features,
            "target_dates": sorted(frame.loc[target_index, "date"].astype(str).tolist()),
            "avoid_dates": sorted(frame.loc[avoid_index, "date"].astype(str).tolist()),
            "perfect_rule_count": int(len(perfect_rows)),
            "top_rules": top_rows,
        }

        with output_json.open("w", encoding="utf-8") as file:
            json.dump(result, file, ensure_ascii=False, indent=2)
            file.write("\n")
        pd.DataFrame(ranked_rows).to_csv(output_csv, index=False)

        log_progress(
            f"wrote top_rules={len(top_rows)} perfect_rules={len(perfect_rows)} "
            f"json={output_json} csv={output_csv}"
        )
        return 0
    except Exception as exc:  # pragma: no cover - operator-facing CLI
        log_progress(f"failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())