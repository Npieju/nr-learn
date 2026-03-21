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
from racing_ml.evaluation.policy import PolicyConstraints, apply_selection_mode, evaluate_candidate_gate


def _parse_path_list(raw_values: list[str]) -> list[Path]:
    paths: list[Path] = []
    for raw in raw_values:
        for part in raw.split(","):
            token = part.strip()
            if not token:
                continue
            path = Path(token)
            if not path.is_absolute():
                path = (ROOT / token).resolve()
            paths.append(path)
    if not paths:
        raise ValueError("at least one threshold sweep report is required")
    return paths


def _parse_thresholds(raw: str) -> list[int]:
    values: list[int] = []
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        values = [100, 60, 55, 45, 34]
    return sorted(set(values), reverse=True)


def _load_report(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"threshold sweep report is not a JSON object: {path}")
    return payload


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
        return {
            "count": 0,
            "min": None,
            "median": None,
            "max": None,
        }
    series = pd.Series(values, dtype=float)
    return {
        "count": int(len(values)),
        "min": float(series.min()),
        "median": float(series.median()),
        "max": float(series.max()),
    }


def _resolve_source_summary_path(path: Path, payload: dict[str, Any]) -> Path:
    run_context = payload.get("run_context") if isinstance(payload.get("run_context"), dict) else {}
    raw = run_context.get("wf_summary")
    if not raw:
        raise ValueError(f"report does not contain run_context.wf_summary: {path}")
    summary_path = Path(str(raw))
    if not summary_path.is_absolute():
        summary_path = (ROOT / summary_path).resolve()
    return summary_path


def _load_source_inputs(
    path: Path,
    payload: dict[str, Any],
    cache: dict[str, tuple[dict[str, Any], pd.DataFrame]],
) -> tuple[dict[str, Any], pd.DataFrame]:
    cache_key = str(path)
    cached = cache.get(cache_key)
    if cached is not None:
        return cached

    summary_path = _resolve_source_summary_path(path, payload)
    summary = read_json(summary_path)
    if not isinstance(summary, dict):
        raise ValueError(f"WF summary is not a JSON object: {summary_path}")
    detail_csv_path = summary_path.with_suffix(".csv")
    detail_df = pd.read_csv(detail_csv_path)
    if "fold" not in detail_df.columns:
        raise ValueError(f"WF detail CSV does not contain 'fold': {detail_csv_path}")
    detail_df["fold"] = pd.to_numeric(detail_df["fold"], errors="coerce").fillna(0).astype(int)
    cache[cache_key] = (summary, detail_df)
    return summary, detail_df


def _primary_block_reason(gate_failures: list[str], *, has_bets_gap: bool) -> str:
    if has_bets_gap:
        return "min_bets"
    for reason in ["min_final_bankroll", "max_drawdown", "min_bets"]:
        if reason in gate_failures:
            return reason
    return "other"


def _build_threshold_fold_diagnostics(
    path: Path,
    payload: dict[str, Any],
    threshold: int,
    cache: dict[str, tuple[dict[str, Any], pd.DataFrame]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    summary, detail_df = _load_source_inputs(path, payload, cache)
    policy_constraints = summary.get("policy_constraints") if isinstance(summary.get("policy_constraints"), dict) else {}
    constraints = PolicyConstraints(
        min_bet_ratio=float(policy_constraints.get("min_bet_ratio") or 0.05),
        min_bets_abs=int(threshold),
        max_drawdown=float(policy_constraints.get("max_drawdown") or 0.40),
        min_final_bankroll=float(policy_constraints.get("min_final_bankroll") or 0.85),
        selection_mode=str(policy_constraints.get("selection_mode") or "gate_then_roi"),
    )

    fold_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    feasible_bankroll_values: list[float] = []
    bankroll_gap_values: list[float] = []
    min_bets_gap_values: list[float] = []
    max_bets_values: list[float] = []

    folds = summary.get("folds") if isinstance(summary.get("folds"), list) else []
    for fold in folds:
        if not isinstance(fold, dict):
            continue
        fold_index = int(fold.get("fold") or 0)
        if fold_index <= 0:
            continue
        valid_races = int(fold.get("valid_races") or 0)
        min_bets_required = constraints.min_bets_required(valid_races)
        fold_df = detail_df.loc[detail_df["fold"] == fold_index].copy()
        if fold_df.empty:
            continue

        candidate_rows: list[dict[str, Any]] = []
        best_feasible: dict[str, Any] | None = None
        best_feasible_score = float("-inf")
        best_over_bet_floor: dict[str, Any] | None = None
        best_over_bet_floor_score = float("-inf")
        max_bets_candidate: dict[str, Any] | None = None

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
            row["min_bets_required"] = int(gate_result.get("min_bets_required") or min_bets_required)
            candidate_rows.append(row)

            if max_bets_candidate is None or int(row.get("bets") or 0) > int(max_bets_candidate.get("bets") or 0):
                max_bets_candidate = row
            if bets >= min_bets_required and base_score is not None and base_score > best_over_bet_floor_score:
                best_over_bet_floor_score = float(base_score)
                best_over_bet_floor = row
            if selection_score is not None and selection_score > best_feasible_score:
                best_feasible_score = float(selection_score)
                best_feasible = row

        if isinstance(best_feasible, dict):
            status = "feasible"
            feasible_bankroll = float(best_feasible.get("final_bankroll") or 0.0)
            feasible_bankroll_values.append(feasible_bankroll)
            status_counts[status] += 1
            fold_rows.append(
                {
                    "fold": fold_index,
                    "valid_races": valid_races,
                    "min_bets_required": min_bets_required,
                    "status": status,
                    "best_feasible_strategy": best_feasible.get("strategy_kind"),
                    "best_feasible_bets": int(best_feasible.get("bets") or 0),
                    "best_feasible_roi": best_feasible.get("roi"),
                    "best_feasible_final_bankroll": best_feasible.get("final_bankroll"),
                    "best_feasible_max_drawdown": best_feasible.get("max_drawdown"),
                    "bankroll_gap_to_min": 0.0,
                    "min_bets_gap": max(0, min_bets_required - int(best_feasible.get("bets") or 0)),
                    "best_over_bet_floor_strategy": best_feasible.get("strategy_kind"),
                    "best_over_bet_floor_bets": int(best_feasible.get("bets") or 0),
                    "best_over_bet_floor_final_bankroll": best_feasible.get("final_bankroll"),
                    "max_bets_any_candidate": int(best_feasible.get("bets") or 0),
                }
            )
            continue

        if isinstance(best_over_bet_floor, dict):
            gate_failures = list(best_over_bet_floor.get("gate_failures") or [])
            status = _primary_block_reason(gate_failures, has_bets_gap=False)
            bankroll_gap = max(0.0, constraints.min_final_bankroll - float(best_over_bet_floor.get("final_bankroll") or 0.0))
            bankroll_gap_values.append(bankroll_gap)
            status_counts[status] += 1
            fold_rows.append(
                {
                    "fold": fold_index,
                    "valid_races": valid_races,
                    "min_bets_required": min_bets_required,
                    "status": status,
                    "best_feasible_strategy": None,
                    "best_feasible_bets": None,
                    "best_feasible_roi": None,
                    "best_feasible_final_bankroll": None,
                    "best_feasible_max_drawdown": None,
                    "bankroll_gap_to_min": bankroll_gap,
                    "min_bets_gap": 0,
                    "best_over_bet_floor_strategy": best_over_bet_floor.get("strategy_kind"),
                    "best_over_bet_floor_bets": int(best_over_bet_floor.get("bets") or 0),
                    "best_over_bet_floor_final_bankroll": best_over_bet_floor.get("final_bankroll"),
                    "max_bets_any_candidate": int(max_bets_candidate.get("bets") or 0) if isinstance(max_bets_candidate, dict) else None,
                }
            )
            continue

        best_support_candidate = max(
            candidate_rows,
            key=lambda row: (
                int(row.get("bets") or 0),
                float(row.get("final_bankroll") or 0.0),
                float(row.get("base_score") or float("-inf")),
            ),
        )
        min_bets_gap = max(0, min_bets_required - int(best_support_candidate.get("bets") or 0))
        min_bets_gap_values.append(float(min_bets_gap))
        max_bets_values.append(float(best_support_candidate.get("bets") or 0))
        status = "min_bets"
        status_counts[status] += 1
        fold_rows.append(
            {
                "fold": fold_index,
                "valid_races": valid_races,
                "min_bets_required": min_bets_required,
                "status": status,
                "best_feasible_strategy": None,
                "best_feasible_bets": None,
                "best_feasible_roi": None,
                "best_feasible_final_bankroll": None,
                "best_feasible_max_drawdown": None,
                "bankroll_gap_to_min": None,
                "min_bets_gap": min_bets_gap,
                "best_over_bet_floor_strategy": None,
                "best_over_bet_floor_bets": None,
                "best_over_bet_floor_final_bankroll": None,
                "max_bets_any_candidate": int(best_support_candidate.get("bets") or 0),
            }
        )

    diagnostics = {
        "status_counts": dict(status_counts),
        "feasible_final_bankroll_summary": _summarize_numeric(feasible_bankroll_values),
        "blocked_bankroll_gap_summary": _summarize_numeric(bankroll_gap_values),
        "min_bets_gap_summary": _summarize_numeric(min_bets_gap_values),
        "min_bets_blocked_max_bets_summary": _summarize_numeric(max_bets_values),
    }
    return diagnostics, fold_rows


def _derive_report_label(path: Path, payload: dict[str, Any]) -> str:
    run_context = payload.get("source_run_context") if isinstance(payload.get("source_run_context"), dict) else {}
    config = Path(str(run_context.get("config") or path.stem)).stem
    start_date = str(run_context.get("start_date") or "unknown")
    end_date = str(run_context.get("end_date") or "unknown")
    return f"{config}:{start_date}..{end_date}"


def _get_threshold_analysis(payload: dict[str, Any], threshold: int) -> dict[str, Any] | None:
    analyses = payload.get("threshold_analyses") if isinstance(payload.get("threshold_analyses"), list) else []
    for analysis in analyses:
        if not isinstance(analysis, dict):
            continue
        constraints = analysis.get("policy_constraints") if isinstance(analysis.get("policy_constraints"), dict) else {}
        if int(constraints.get("min_bets_abs") or -1) == int(threshold):
            return analysis
    return None


def _build_comparison(report_paths: list[Path], thresholds: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    report_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    source_cache: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}

    for path in report_paths:
        payload = _load_report(path)
        label = _derive_report_label(path, payload)
        stability_context = payload.get("stability_context") if isinstance(payload.get("stability_context"), dict) else {}
        strictest = payload.get("strictest_threshold_passing_feasible_fold_targets")
        if not isinstance(strictest, dict):
            strictest = {}

        report_rows.append(
            {
                "label": label,
                "report": str(path.relative_to(ROOT)),
                "strictest_for_1_fold": strictest.get("1"),
                "strictest_for_3_folds": strictest.get("3"),
                "strictest_for_5_folds": strictest.get("5"),
                "warning_count": len(stability_context.get("warnings") or []),
                "warnings": list(stability_context.get("warnings") or []),
                "wf_summary_stability_assessment": stability_context.get("wf_summary_stability_assessment"),
                "valid_fold_stability_counts": stability_context.get("valid_fold_stability_counts"),
                "test_fold_stability_counts": stability_context.get("test_fold_stability_counts"),
            }
        )

        for threshold in thresholds:
            analysis = _get_threshold_analysis(payload, threshold)
            if analysis is None:
                continue
            support_summary = analysis.get("best_feasible_bet_support_summary") if isinstance(analysis.get("best_feasible_bet_support_summary"), dict) else {}
            bankroll_diagnostics, threshold_fold_rows = _build_threshold_fold_diagnostics(path, payload, threshold, source_cache)
            status_counts = bankroll_diagnostics.get("status_counts") if isinstance(bankroll_diagnostics.get("status_counts"), dict) else {}
            feasible_bankroll_summary = bankroll_diagnostics.get("feasible_final_bankroll_summary") if isinstance(bankroll_diagnostics.get("feasible_final_bankroll_summary"), dict) else {}
            blocked_bankroll_gap_summary = bankroll_diagnostics.get("blocked_bankroll_gap_summary") if isinstance(bankroll_diagnostics.get("blocked_bankroll_gap_summary"), dict) else {}
            min_bets_gap_summary = bankroll_diagnostics.get("min_bets_gap_summary") if isinstance(bankroll_diagnostics.get("min_bets_gap_summary"), dict) else {}
            threshold_rows.append(
                {
                    "label": label,
                    "report": str(path.relative_to(ROOT)),
                    "min_bets_abs": int(threshold),
                    "feasible_fold_count": int(analysis.get("feasible_fold_count") or 0),
                    "feasible_candidate_count_total": int(analysis.get("feasible_candidate_count_total") or 0),
                    "dominant_failure_reason": analysis.get("dominant_failure_reason"),
                    "support_min_bets": support_summary.get("min"),
                    "support_median_bets": support_summary.get("median"),
                    "support_max_bets": support_summary.get("max"),
                    "status_feasible_count": int(status_counts.get("feasible") or 0),
                    "status_min_bets_count": int(status_counts.get("min_bets") or 0),
                    "status_min_final_bankroll_count": int(status_counts.get("min_final_bankroll") or 0),
                    "status_max_drawdown_count": int(status_counts.get("max_drawdown") or 0),
                    "feasible_bankroll_min": feasible_bankroll_summary.get("min"),
                    "feasible_bankroll_median": feasible_bankroll_summary.get("median"),
                    "feasible_bankroll_max": feasible_bankroll_summary.get("max"),
                    "blocked_bankroll_gap_min": blocked_bankroll_gap_summary.get("min"),
                    "blocked_bankroll_gap_median": blocked_bankroll_gap_summary.get("median"),
                    "blocked_bankroll_gap_max": blocked_bankroll_gap_summary.get("max"),
                    "min_bets_gap_min": min_bets_gap_summary.get("min"),
                    "min_bets_gap_median": min_bets_gap_summary.get("median"),
                    "min_bets_gap_max": min_bets_gap_summary.get("max"),
                }
            )
            for fold_row in threshold_fold_rows:
                fold_rows.append(
                    {
                        "label": label,
                        "report": str(path.relative_to(ROOT)),
                        "min_bets_abs": int(threshold),
                        **fold_row,
                    }
                )

    comparison = {
        "reports": report_rows,
        "threshold_snapshots": threshold_rows,
        "fold_snapshots": fold_rows,
    }
    return comparison, pd.DataFrame(threshold_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="+", required=True)
    parser.add_argument("--thresholds", default="100,60,55,45,34")
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_compare.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_compare.csv")
    args = parser.parse_args()

    report_paths = _parse_path_list(args.reports)
    thresholds = _parse_thresholds(args.thresholds)
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = (ROOT / output_path).resolve()
    summary_csv_path = Path(args.summary_csv)
    if not summary_csv_path.is_absolute():
        summary_csv_path = (ROOT / summary_csv_path).resolve()

    comparison, threshold_df = _build_comparison(report_paths, thresholds)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, comparison)
    threshold_df.to_csv(summary_csv_path, index=False)

    print(f"saved threshold comparison to {output_path.relative_to(ROOT)}")
    print(f"saved threshold comparison table to {summary_csv_path.relative_to(ROOT)}")
    for report in comparison.get("reports") or []:
        print(
            "label={label} strictest1={strictest_for_1_fold} strictest3={strictest_for_3_folds} "
            "strictest5={strictest_for_5_folds} warnings={warning_count}".format(**report)
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
