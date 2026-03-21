from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json
from racing_ml.evaluation.policy import PolicyConstraints, apply_selection_mode, evaluate_candidate_gate


SIGNATURE_FIELDS = [
    "strategy_kind",
    "blend_weight",
    "min_edge",
    "min_prob",
    "fractional_kelly",
    "max_fraction",
    "odds_min",
    "odds_max",
    "top_k",
    "min_expected_value",
]

VARIANT_FIELDS = [
    "strategy_kind",
    "blend_weight",
    "min_prob",
    "fractional_kelly",
    "max_fraction",
    "top_k",
    "min_expected_value",
]


def _normalize_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _value_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return [_value_or_none(item) for item in value]
    if isinstance(value, dict):
        return {key: _value_or_none(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _summarize_numeric(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "min": None, "median": None, "max": None}
    series = pd.Series(values, dtype=float)
    return {
        "count": int(len(values)),
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def _candidate_signature(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in SIGNATURE_FIELDS:
        parts.append(f"{field}={_value_or_none(row.get(field))}")
    return "|".join(parts)


def _variant_label(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in VARIANT_FIELDS:
        parts.append(f"{field}={_value_or_none(row.get(field))}")
    return "|".join(parts)


def _primary_block_reason(row: dict[str, Any]) -> str:
    if bool(row.get("is_feasible")):
        return "feasible"
    failures = row.get("gate_failures")
    if isinstance(failures, list) and failures:
        for reason in ("min_bets", "min_final_bankroll", "max_drawdown"):
            if reason in failures:
                return reason
        return str(failures[0])
    return "other"


def _pick_best(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: (
            float(row.get("selection_score")) if row.get("selection_score") is not None else float("-inf"),
            float(row.get("base_score")) if row.get("base_score") is not None else float("-inf"),
            float(row.get("final_bankroll")) if row.get("final_bankroll") is not None else float("-inf"),
        ),
    )


def _load_compare_report(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"threshold compare report is not a JSON object: {path}")
    return payload


def _load_threshold_report(path: Path, cache: dict[str, dict[str, Any]]) -> dict[str, Any]:
    cache_key = str(path)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"threshold sweep report is not a JSON object: {path}")
    cache[cache_key] = payload
    return payload


def _load_source_inputs(
    threshold_report_path: Path,
    threshold_report: dict[str, Any],
    cache: dict[str, tuple[dict[str, Any], pd.DataFrame]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    cache_key = str(threshold_report_path)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    run_context = threshold_report.get("run_context") if isinstance(threshold_report.get("run_context"), dict) else {}
    raw_summary_path = run_context.get("wf_summary")
    if not raw_summary_path:
        raise ValueError(f"threshold report missing run_context.wf_summary: {threshold_report_path}")
    summary_path = _normalize_path(str(raw_summary_path))
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"WF summary is not a JSON object: {summary_path}")
    detail_df = pd.read_csv(summary_path.with_suffix(".csv"))
    if "fold" not in detail_df.columns:
        raise ValueError(f"WF detail CSV does not contain 'fold': {summary_path.with_suffix('.csv')}")
    detail_df["fold"] = pd.to_numeric(detail_df["fold"], errors="coerce").fillna(0).astype(int)
    cache[cache_key] = (summary, detail_df)
    return summary, detail_df


def _build_constraints(summary: dict[str, Any], threshold: int) -> PolicyConstraints:
    policy_constraints = summary.get("policy_constraints") if isinstance(summary.get("policy_constraints"), dict) else {}
    return PolicyConstraints(
        min_bet_ratio=float(policy_constraints.get("min_bet_ratio") or 0.05),
        min_bets_abs=int(threshold),
        max_drawdown=float(policy_constraints.get("max_drawdown") or 0.40),
        min_final_bankroll=float(policy_constraints.get("min_final_bankroll") or 0.85),
        selection_mode=str(policy_constraints.get("selection_mode") or "gate_then_roi"),
    )


def _score_fold_candidates(fold_df: pd.DataFrame, *, valid_races: int, constraints: PolicyConstraints) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_row in fold_df.to_dict(orient="records"):
        row = {key: _value_or_none(value) for key, value in raw_row.items()}
        bets = int(row.get("bets") or 0)
        max_drawdown = float(row.get("max_drawdown")) if row.get("max_drawdown") is not None else None
        final_bankroll = float(row.get("final_bankroll")) if row.get("final_bankroll") is not None else None
        base_score = float(row.get("base_score")) if row.get("base_score") is not None else None
        gate_result = evaluate_candidate_gate(
            bets=bets,
            max_drawdown=max_drawdown,
            final_bankroll=final_bankroll,
            constraints=constraints,
            n_races=valid_races,
        )
        selection_score = (
            apply_selection_mode(
                base_score,
                gate_result,
                constraints,
                bets=bets,
                max_drawdown=max_drawdown,
                final_bankroll=final_bankroll,
            )
            if base_score is not None
            else None
        )
        row["selection_score"] = selection_score
        row["is_feasible"] = bool(gate_result.get("is_feasible"))
        row["gate_failures"] = list(gate_result.get("gate_failures") or [])
        row["signature"] = _candidate_signature(row)
        row["variant_label"] = _variant_label(row)
        rows.append(row)
    return rows


def _resolve_signature(compare_payload: dict[str, Any], requested: str | None) -> str:
    if requested:
        return requested
    fold_rows = compare_payload.get("fold_snapshots") if isinstance(compare_payload.get("fold_snapshots"), list) else []
    counts: Counter[str] = Counter(
        str(row.get("best_over_bet_floor_signature") or "")
        for row in fold_rows
        if isinstance(row, dict)
        and str(row.get("status") or "") in {"min_bets", "min_final_bankroll", "max_drawdown", "other"}
        and str(row.get("best_over_bet_floor_signature") or "")
    )
    if not counts:
        raise ValueError("no blocked signature found in compare report")
    return counts.most_common(1)[0][0]


def _same_family(candidate: dict[str, Any], anchor: dict[str, Any]) -> bool:
    return (
        _value_or_none(candidate.get("strategy_kind")) == _value_or_none(anchor.get("strategy_kind"))
        and _value_or_none(candidate.get("odds_min")) == _value_or_none(anchor.get("odds_min"))
        and _value_or_none(candidate.get("odds_max")) == _value_or_none(anchor.get("odds_max"))
        and _value_or_none(candidate.get("top_k")) == _value_or_none(anchor.get("top_k"))
    )


def _build_family_compare(compare_payload: dict[str, Any], signature: str) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_rows = compare_payload.get("fold_snapshots") if isinstance(compare_payload.get("fold_snapshots"), list) else []
    target_rows = [
        row for row in fold_rows
        if isinstance(row, dict)
        and str(row.get("best_over_bet_floor_signature") or "") == signature
        and str(row.get("status") or "") in {"min_bets", "min_final_bankroll", "max_drawdown", "other"}
    ]
    if not target_rows:
        raise ValueError("requested signature does not appear in blocked fold snapshots")

    threshold_report_cache: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    grouped: dict[str, dict[str, Any]] = {}

    target_label = signature
    target_occurrence_count = 0

    for occurrence in target_rows:
        threshold_report_path = _normalize_path(str(occurrence.get("report")))
        threshold_report = _load_threshold_report(threshold_report_path, threshold_report_cache)
        summary, detail_df = _load_source_inputs(threshold_report_path, threshold_report, source_cache)
        threshold = int(occurrence.get("min_bets_abs") or 0)
        fold = int(occurrence.get("fold") or 0)
        valid_races = int(occurrence.get("valid_races") or 0)
        constraints = _build_constraints(summary, threshold)
        scored_rows = _score_fold_candidates(
            detail_df.loc[detail_df["fold"] == fold].copy(),
            valid_races=valid_races,
            constraints=constraints,
        )
        target_candidates = [row for row in scored_rows if str(row.get("signature") or "") == signature]
        target_candidate = _pick_best(target_candidates)
        if target_candidate is None:
            continue
        target_occurrence_count += 1
        target_variant = str(target_candidate.get("variant_label") or target_label)
        family_rows = [row for row in scored_rows if _same_family(row, target_candidate)]
        family_rows_by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in family_rows:
            family_rows_by_variant[str(row.get("variant_label") or _variant_label(row))].append(row)
        family_rows = [_pick_best(rows) for rows in family_rows_by_variant.values()]
        family_rows = [row for row in family_rows if row is not None]
        best_family_candidate = _pick_best(family_rows)

        target_bets = float(target_candidate.get("bets") or 0.0)
        target_bankroll = float(target_candidate.get("final_bankroll")) if target_candidate.get("final_bankroll") is not None else None
        target_base_score = float(target_candidate.get("base_score")) if target_candidate.get("base_score") is not None else None
        occurrence_key = f"{occurrence.get('label')}|{threshold}|{fold}"

        for row in family_rows:
            variant_label = str(row.get("variant_label") or _variant_label(row))
            entry = grouped.setdefault(
                variant_label,
                {
                    "variant_label": variant_label,
                    "signature_example": str(row.get("signature") or ""),
                    "strategy_kind": _value_or_none(row.get("strategy_kind")),
                    "blend_weight": _value_or_none(row.get("blend_weight")),
                    "min_prob": _value_or_none(row.get("min_prob")),
                    "fractional_kelly": _value_or_none(row.get("fractional_kelly")),
                    "max_fraction": _value_or_none(row.get("max_fraction")),
                    "top_k": _value_or_none(row.get("top_k")),
                    "min_expected_value": _value_or_none(row.get("min_expected_value")),
                    "occurrence_keys": set(),
                    "status_counts": Counter(),
                    "label_counts": Counter(),
                    "threshold_counts": Counter(),
                    "bets_values": [],
                    "final_bankroll_values": [],
                    "base_score_values": [],
                    "delta_bets_vs_target_values": [],
                    "delta_final_bankroll_vs_target_values": [],
                    "delta_base_score_vs_target_values": [],
                    "best_family_candidate_count": 0,
                    "higher_bankroll_lower_bets_count": 0,
                    "higher_bankroll_count": 0,
                    "lower_bets_count": 0,
                    "is_target_variant": variant_label == target_variant,
                },
            )
            entry["occurrence_keys"].add(occurrence_key)
            status = _primary_block_reason(row)
            entry["status_counts"][status] += 1
            entry["label_counts"][str(occurrence.get("label") or "unknown")] += 1
            entry["threshold_counts"][threshold] += 1
            bets = float(row.get("bets") or 0.0)
            entry["bets_values"].append(bets)
            if row.get("final_bankroll") is not None:
                final_bankroll = float(row.get("final_bankroll"))
                entry["final_bankroll_values"].append(final_bankroll)
                if target_bankroll is not None:
                    delta_bankroll = final_bankroll - target_bankroll
                    entry["delta_final_bankroll_vs_target_values"].append(delta_bankroll)
                    if delta_bankroll > 0:
                        entry["higher_bankroll_count"] += 1
                        if bets < target_bets:
                            entry["higher_bankroll_lower_bets_count"] += 1
            if row.get("base_score") is not None:
                base_score = float(row.get("base_score"))
                entry["base_score_values"].append(base_score)
                if target_base_score is not None:
                    entry["delta_base_score_vs_target_values"].append(base_score - target_base_score)
            entry["delta_bets_vs_target_values"].append(bets - target_bets)
            if bets < target_bets:
                entry["lower_bets_count"] += 1
            if best_family_candidate is not None and str(best_family_candidate.get("variant_label") or "") == variant_label:
                entry["best_family_candidate_count"] += 1

    overview_rows: list[dict[str, Any]] = []
    for entry in grouped.values():
        occurrence_count = int(len(entry["occurrence_keys"]))
        status_counts = dict(entry["status_counts"])
        overview_rows.append(
            {
                "variant_label": entry["variant_label"],
                "signature_example": entry["signature_example"],
                "strategy_kind": entry["strategy_kind"],
                "blend_weight": entry["blend_weight"],
                "min_prob": entry["min_prob"],
                "fractional_kelly": entry["fractional_kelly"],
                "max_fraction": entry["max_fraction"],
                "top_k": entry["top_k"],
                "min_expected_value": entry["min_expected_value"],
                "is_target_variant": bool(entry["is_target_variant"]),
                "occurrence_count": occurrence_count,
                "status_feasible_count": int(status_counts.get("feasible", 0)),
                "status_min_bets_count": int(status_counts.get("min_bets", 0)),
                "status_min_final_bankroll_count": int(status_counts.get("min_final_bankroll", 0)),
                "status_max_drawdown_count": int(status_counts.get("max_drawdown", 0)),
                "status_other_count": int(status_counts.get("other", 0)),
                "best_family_candidate_count": int(entry["best_family_candidate_count"]),
                "higher_bankroll_count": int(entry["higher_bankroll_count"]),
                "lower_bets_count": int(entry["lower_bets_count"]),
                "higher_bankroll_lower_bets_count": int(entry["higher_bankroll_lower_bets_count"]),
                "labels": dict(entry["label_counts"]),
                "thresholds": {str(key): int(value) for key, value in entry["threshold_counts"].items()},
                "bets_summary": _summarize_numeric(entry["bets_values"]),
                "final_bankroll_summary": _summarize_numeric(entry["final_bankroll_values"]),
                "base_score_summary": _summarize_numeric(entry["base_score_values"]),
                "delta_bets_vs_target_summary": _summarize_numeric(entry["delta_bets_vs_target_values"]),
                "delta_final_bankroll_vs_target_summary": _summarize_numeric(entry["delta_final_bankroll_vs_target_values"]),
                "delta_base_score_vs_target_summary": _summarize_numeric(entry["delta_base_score_vs_target_values"]),
            }
        )

    overview_rows.sort(
        key=lambda row: (
            not bool(row.get("is_target_variant")),
            -int(row.get("occurrence_count") or 0),
            -int(row.get("higher_bankroll_lower_bets_count") or 0),
            -int(row.get("status_feasible_count") or 0),
        )
    )

    report = {
        "target_signature": signature,
        "blocked_target_occurrence_count": int(target_occurrence_count),
        "family_variant_count": int(len(overview_rows)),
        "family_definition": {
            "shared_fields": ["strategy_kind", "odds_min", "odds_max", "top_k"],
            "variant_fields": VARIANT_FIELDS,
        },
        "variant_overview": overview_rows,
    }
    summary_df = pd.DataFrame(
        [
            {
                "variant_label": row["variant_label"],
                "strategy_kind": row["strategy_kind"],
                "blend_weight": row["blend_weight"],
                "min_prob": row["min_prob"],
                "fractional_kelly": row["fractional_kelly"],
                "max_fraction": row["max_fraction"],
                "top_k": row["top_k"],
                "min_expected_value": row["min_expected_value"],
                "is_target_variant": row["is_target_variant"],
                "occurrence_count": row["occurrence_count"],
                "status_feasible_count": row["status_feasible_count"],
                "status_min_bets_count": row["status_min_bets_count"],
                "status_min_final_bankroll_count": row["status_min_final_bankroll_count"],
                "status_max_drawdown_count": row["status_max_drawdown_count"],
                "best_family_candidate_count": row["best_family_candidate_count"],
                "higher_bankroll_count": row["higher_bankroll_count"],
                "lower_bets_count": row["lower_bets_count"],
                "higher_bankroll_lower_bets_count": row["higher_bankroll_lower_bets_count"],
                "bets_median": row["bets_summary"].get("median"),
                "final_bankroll_median": row["final_bankroll_summary"].get("median"),
                "base_score_median": row["base_score_summary"].get("median"),
                "delta_bets_vs_target_median": row["delta_bets_vs_target_summary"].get("median"),
                "delta_final_bankroll_vs_target_median": row["delta_final_bankroll_vs_target_summary"].get("median"),
                "delta_base_score_vs_target_median": row["delta_base_score_vs_target_summary"].get("median"),
            }
            for row in overview_rows
        ]
    )
    return report, summary_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-report", default="artifacts/reports/wf_threshold_compare_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--signature", default=None)
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_signature_family_compare.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_signature_family_compare.csv")
    args = parser.parse_args()

    compare_report_path = _normalize_path(args.compare_report)
    output_path = _normalize_path(args.output)
    summary_csv_path = _normalize_path(args.summary_csv)

    compare_payload = _load_compare_report(compare_report_path)
    signature = _resolve_signature(compare_payload, args.signature)
    report, summary_df = _build_family_compare(compare_payload, signature)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"saved threshold signature family compare to {output_path.relative_to(ROOT)}")
    print(f"saved threshold signature family table to {summary_csv_path.relative_to(ROOT)}")
    print(f"target_signature={signature}")
    print(f"blocked_target_occurrence_count={report['blocked_target_occurrence_count']}")
    print(f"family_variant_count={report['family_variant_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())