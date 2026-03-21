from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json


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
                }
            )

    comparison = {
        "reports": report_rows,
        "threshold_snapshots": threshold_rows,
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
