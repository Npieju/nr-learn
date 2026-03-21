from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json


def _normalize_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_signature(signature: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not signature:
        return result
    for part in str(signature).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if value == "None":
            result[key] = None
            continue
        try:
            result[key] = float(value)
            continue
        except ValueError:
            result[key] = value
    return result


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


def _recovery_map(drilldown_payload: dict[str, Any]) -> dict[tuple[str, int, int], dict[str, Any]]:
    rows = drilldown_payload.get("occurrences") if isinstance(drilldown_payload.get("occurrences"), list) else []
    mapping: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("label") or ""), int(row.get("min_bets_abs") or 0), int(row.get("fold") or 0))
        mapping[key] = row
    return mapping


def _select_stage(row: dict[str, Any], recovery_row: dict[str, Any] | None) -> tuple[str, dict[str, Any], str | None]:
    recommendation = str(row.get("recommendation") or "")
    if recommendation == "candidate_b_higher_bankroll":
        selected = {
            "signature": row.get("candidate_b_signature"),
            "strategy_kind": row.get("candidate_b_strategy_kind"),
            "blend_weight": row.get("candidate_b_blend_weight"),
            "min_prob": row.get("candidate_b_min_prob"),
            "fractional_kelly": row.get("candidate_b_fractional_kelly"),
            "max_fraction": row.get("candidate_b_max_fraction"),
            "top_k": row.get("candidate_b_top_k"),
            "min_expected_value": row.get("candidate_b_min_expected_value"),
            "bets": row.get("candidate_b_bets"),
            "roi": row.get("candidate_b_roi"),
            "final_bankroll": row.get("candidate_b_final_bankroll"),
            "base_score": row.get("candidate_b_base_score"),
        }
        return "portfolio_ev_only", selected, None

    selected = {
        "signature": row.get("candidate_a_signature"),
        "strategy_kind": row.get("candidate_a_strategy_kind"),
        "blend_weight": row.get("candidate_a_blend_weight"),
        "min_prob": row.get("candidate_a_min_prob"),
        "fractional_kelly": row.get("candidate_a_fractional_kelly"),
        "max_fraction": row.get("candidate_a_max_fraction"),
        "top_k": row.get("candidate_a_top_k"),
        "min_expected_value": row.get("candidate_a_min_expected_value"),
        "bets": row.get("candidate_a_bets"),
        "roi": row.get("candidate_a_roi"),
        "final_bankroll": row.get("candidate_a_final_bankroll"),
        "base_score": row.get("candidate_a_base_score"),
    }
    if recovery_row is not None and str(recovery_row.get("recovery_strategy_kind") or "") == "kelly":
        kelly = {
            "signature": recovery_row.get("recovery_signature"),
            "strategy_kind": recovery_row.get("recovery_strategy_kind"),
            "blend_weight": recovery_row.get("recovery_blend_weight"),
            "min_prob": recovery_row.get("recovery_min_prob"),
            "fractional_kelly": recovery_row.get("recovery_fractional_kelly"),
            "max_fraction": recovery_row.get("recovery_max_fraction"),
            "top_k": recovery_row.get("recovery_top_k"),
            "min_expected_value": recovery_row.get("recovery_min_expected_value"),
            "bets": recovery_row.get("recovery_bets"),
            "roi": recovery_row.get("recovery_roi"),
            "final_bankroll": recovery_row.get("recovery_final_bankroll"),
            "base_score": recovery_row.get("recovery_base_score"),
        }
        return "kelly_fallback", kelly, str(recovery_row.get("recovery_signature") or None)
    return "portfolio_lower_blend", selected, None


def _flatten(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in payload.items()}


def _build_probe(focus_payload: dict[str, Any], drilldown_payload: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    focus_rows = focus_payload.get("occurrences") if isinstance(focus_payload.get("occurrences"), list) else []
    recovery_lookup = _recovery_map(drilldown_payload)
    staged_rows: list[dict[str, Any]] = []
    stage_counts: Counter[str] = Counter()
    kelly_signature_counts: Counter[str] = Counter()
    delta_bets_values: list[float] = []
    delta_bankroll_values: list[float] = []

    for row in focus_rows:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("label") or ""), int(row.get("min_bets_abs") or 0), int(row.get("fold") or 0))
        recovery_row = recovery_lookup.get(key)
        stage, selected, kelly_signature = _select_stage(row, recovery_row)
        stage_counts[stage] += 1
        if kelly_signature:
            kelly_signature_counts[kelly_signature] += 1

        target_bets = row.get("target_bets")
        target_bankroll = row.get("target_final_bankroll")
        selected_bets = selected.get("bets")
        selected_bankroll = selected.get("final_bankroll")
        delta_bets = None
        delta_bankroll = None
        if target_bets is not None and selected_bets is not None:
            delta_bets = float(selected_bets) - float(target_bets)
            delta_bets_values.append(delta_bets)
        if target_bankroll is not None and selected_bankroll is not None:
            delta_bankroll = float(selected_bankroll) - float(target_bankroll)
            delta_bankroll_values.append(delta_bankroll)

        staged_rows.append(
            {
                "label": row.get("label"),
                "report": row.get("report"),
                "min_bets_abs": row.get("min_bets_abs"),
                "fold": row.get("fold"),
                "status": row.get("status"),
                "recommendation": row.get("recommendation"),
                "stage": stage,
                **_flatten(
                    "target",
                    {
                        "signature": row.get("target_signature"),
                        "strategy_kind": row.get("target_strategy_kind"),
                        "blend_weight": row.get("target_blend_weight"),
                        "min_prob": row.get("target_min_prob"),
                        "fractional_kelly": row.get("target_fractional_kelly"),
                        "max_fraction": row.get("target_max_fraction"),
                        "top_k": row.get("target_top_k"),
                        "min_expected_value": row.get("target_min_expected_value"),
                        "bets": row.get("target_bets"),
                        "roi": row.get("target_roi"),
                        "final_bankroll": row.get("target_final_bankroll"),
                        "base_score": row.get("target_base_score"),
                    },
                ),
                **_flatten("selected", selected),
                "delta_bets_target_to_selected": delta_bets,
                "delta_bankroll_target_to_selected": delta_bankroll,
                "recovery_threshold": recovery_row.get("recovery_threshold") if recovery_row is not None else None,
            }
        )

    report = {
        "target_signature": focus_payload.get("target_signature"),
        "candidate_a_signature": focus_payload.get("candidate_a_signature"),
        "candidate_b_signature": focus_payload.get("candidate_b_signature"),
        "occurrence_count": int(len(staged_rows)),
        "stage_counts": dict(stage_counts),
        "kelly_signature_counts": dict(kelly_signature_counts),
        "delta_bets_target_to_selected_summary": _summarize_numeric(delta_bets_values),
        "delta_bankroll_target_to_selected_summary": _summarize_numeric(delta_bankroll_values),
        "occurrences": staged_rows,
    }
    return report, pd.DataFrame(staged_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--focus-report", default="artifacts/reports/wf_threshold_mitigation_focus_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--drilldown-report", default="artifacts/reports/wf_threshold_signature_drilldown_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_mitigation_policy_probe.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_mitigation_policy_probe.csv")
    args = parser.parse_args()

    focus_payload = _load_json(_normalize_path(args.focus_report))
    drilldown_payload = _load_json(_normalize_path(args.drilldown_report))
    report, summary_df = _build_probe(focus_payload, drilldown_payload)

    output_path = _normalize_path(args.output)
    summary_csv_path = _normalize_path(args.summary_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"saved mitigation policy probe to {output_path.relative_to(ROOT)}")
    print(f"saved mitigation policy probe table to {summary_csv_path.relative_to(ROOT)}")
    print(f"stage_counts={report['stage_counts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())