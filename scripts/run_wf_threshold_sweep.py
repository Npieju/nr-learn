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


def _parse_int_list(raw: str) -> list[int]:
    values: list[int] = []
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("at least one min_bets_abs value is required")
    deduped = sorted(set(values), reverse=True)
    return deduped


def _parse_target_fold_counts(raw: str, *, max_fold_count: int) -> list[int]:
    values: list[int] = []
    for part in (raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        value = int(token)
        if value <= 0:
            continue
        values.append(min(value, max_fold_count))
    if not values:
        values = [1]
    return sorted(set(values))


def _load_summary(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"WF summary is not a JSON object: {path}")
    return payload


def _build_stability_context(wf_summary: dict[str, Any], fold_meta: dict[int, dict[str, Any]]) -> dict[str, Any]:
    valid_counts: Counter[str] = Counter()
    test_counts: Counter[str] = Counter()
    for fold in fold_meta.values():
        valid_counts[str(fold.get("valid_stability_assessment") or "unknown")] += 1
        test_counts[str(fold.get("test_stability_assessment") or "unknown")] += 1

    warnings: list[str] = []
    overall_assessment = str(wf_summary.get("stability_assessment") or "unknown")
    if overall_assessment != "representative":
        warnings.append(
            "WF summary is not representative; threshold sweep results should not be used for promotion decisions."
        )
    if valid_counts and valid_counts.get("probe_only", 0) == len(fold_meta):
        warnings.append(
            "All WF validation folds are probe_only; threshold sweep is directional analysis, not long-horizon evidence."
        )
    if test_counts and test_counts.get("probe_only", 0) == len(fold_meta):
        warnings.append(
            "All WF test folds are probe_only; threshold sweep does not replace representative out-of-sample validation."
        )

    return {
        "wf_summary_stability_assessment": overall_assessment,
        "valid_fold_stability_counts": dict(valid_counts),
        "test_fold_stability_counts": dict(test_counts),
        "warnings": warnings,
    }


def _resolve_detail_csv(summary_path: Path, explicit_path: str | None) -> Path:
    if explicit_path:
        return (ROOT / explicit_path).resolve() if not Path(explicit_path).is_absolute() else Path(explicit_path)
    candidate = summary_path.with_suffix(".csv")
    if not candidate.exists():
        raise FileNotFoundError(f"WF detail CSV is missing: {candidate}")
    return candidate


def _default_output_path(summary_path: Path) -> Path:
    stem = summary_path.stem
    if stem.startswith("wf_feasibility_diag_"):
        stem = stem[len("wf_feasibility_diag_") :]
    return ROOT / "artifacts" / "reports" / f"wf_threshold_sweep_{stem}.json"


def _default_csv_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".csv")


def _value_or_none(value: Any) -> Any:
    if isinstance(value, list):
        return [_value_or_none(item) for item in value]
    if isinstance(value, dict):
        return {key: _value_or_none(item) for key, item in value.items()}
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return value
    return value


def _serialize_candidate(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    payload: dict[str, Any] = {}
    for key, value in row.items():
        converted = _value_or_none(value)
        if converted is None:
            continue
        if isinstance(converted, float) and pd.isna(converted):
            continue
        if key == "gate_failures":
            payload[key] = list(converted or [])
            continue
        payload[key] = converted
    return payload


def _closest_key(row: dict[str, Any], *, min_bets_required: int, constraints: PolicyConstraints) -> tuple[float, float, float, float, float]:
    bets_gap = max(0, min_bets_required - int(row.get("bets") or 0))
    drawdown_gap = max(0.0, float(row.get("max_drawdown") or 0.0) - constraints.max_drawdown)
    bankroll_gap = max(0.0, constraints.min_final_bankroll - float(row.get("final_bankroll") or 0.0))
    base_score = float(row.get("base_score")) if row.get("base_score") is not None else float("-inf")
    return (
        float(len(row.get("gate_failures", []))),
        float(bets_gap),
        float(drawdown_gap),
        float(bankroll_gap),
        -base_score,
    )


def _binding_source(min_bets_required: int, *, ratio_required: int, absolute_required: int) -> str:
    if min_bets_required <= 0:
        return "unknown"
    if min_bets_required == absolute_required and absolute_required >= ratio_required:
        return "absolute"
    if min_bets_required == ratio_required and ratio_required > absolute_required:
        return "ratio"
    return "mixed"


def _analyze_threshold(
    detail_df: pd.DataFrame,
    fold_meta: dict[int, dict[str, Any]],
    *,
    constraints: PolicyConstraints,
    min_feasible_folds: int,
) -> dict[str, Any]:
    fold_summaries: list[dict[str, Any]] = []
    failure_reason_counts_total: Counter[str] = Counter()
    binding_source_counts: Counter[str] = Counter()
    feasible_folds: list[int] = []

    for fold_index in sorted(fold_meta):
        fold_rows_df = detail_df.loc[detail_df["fold"] == fold_index].copy()
        if fold_rows_df.empty:
            continue

        valid_races = int(fold_meta[fold_index].get("valid_races") or 0)
        min_bets_required = constraints.min_bets_required(valid_races)
        ratio_required = int(valid_races * constraints.min_bet_ratio)
        binding_source = _binding_source(
            min_bets_required,
            ratio_required=ratio_required,
            absolute_required=int(constraints.min_bets_abs),
        )
        binding_source_counts[binding_source] += 1

        candidate_rows: list[dict[str, Any]] = []
        best_feasible: dict[str, Any] | None = None
        best_feasible_score = float("-inf")
        best_fallback: dict[str, Any] | None = None
        best_fallback_score = float("-inf")

        for raw_row in fold_rows_df.to_dict(orient="records"):
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

            for reason in row["gate_failures"]:
                failure_reason_counts_total[str(reason)] += 1

            if base_score is not None and base_score > best_fallback_score:
                best_fallback_score = base_score
                best_fallback = row
            if selection_score is not None and selection_score > best_feasible_score:
                best_feasible_score = float(selection_score)
                best_feasible = row

        feasible_candidates = [row for row in candidate_rows if bool(row.get("is_feasible"))]
        if feasible_candidates:
            feasible_folds.append(int(fold_index))

        closest_infeasible = [
            _serialize_candidate(row)
            for row in sorted(
                [row for row in candidate_rows if not bool(row.get("is_feasible"))],
                key=lambda row: _closest_key(row, min_bets_required=min_bets_required, constraints=constraints),
            )[:3]
        ]

        fold_failure_counts = Counter(
            reason for row in candidate_rows for reason in row.get("gate_failures", [])
        )
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "valid_races": valid_races,
                "min_bets_required": int(min_bets_required),
                "ratio_bets_required": int(ratio_required),
                "binding_min_bets_source": binding_source,
                "total_candidates": int(len(candidate_rows)),
                "feasible_candidates": int(len(feasible_candidates)),
                "failure_reason_counts": dict(fold_failure_counts),
                "best_feasible": _serialize_candidate(best_feasible),
                "best_fallback": _serialize_candidate(best_fallback),
                "closest_infeasible": closest_infeasible,
            }
        )

    dominant_failure_reason = failure_reason_counts_total.most_common(1)[0][0] if failure_reason_counts_total else None
    dominant_failure_count = failure_reason_counts_total.most_common(1)[0][1] if failure_reason_counts_total else 0

    return {
        "policy_constraints": constraints.to_dict(),
        "feasible_fold_count": int(len(feasible_folds)),
        "blocked_fold_count": int(max(len(fold_summaries) - len(feasible_folds), 0)),
        "feasible_folds": feasible_folds,
        "passes_min_feasible_folds_preview": bool(len(feasible_folds) >= min_feasible_folds),
        "dominant_failure_reason": dominant_failure_reason,
        "dominant_failure_count": int(dominant_failure_count),
        "failure_reason_counts_total": dict(failure_reason_counts_total),
        "binding_min_bets_source_counts": dict(binding_source_counts),
        "folds": fold_summaries,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wf-summary", required=True)
    parser.add_argument("--wf-detail-csv", default=None)
    parser.add_argument("--min-bets-abs-values", default="100,80,60,40")
    parser.add_argument("--min-feasible-folds", type=int, default=1)
    parser.add_argument("--target-feasible-fold-counts", default="1,3,5")
    parser.add_argument("--output", default=None)
    parser.add_argument("--summary-csv", default=None)
    args = parser.parse_args()

    summary_path = (ROOT / args.wf_summary).resolve() if not Path(args.wf_summary).is_absolute() else Path(args.wf_summary)
    detail_csv_path = _resolve_detail_csv(summary_path, args.wf_detail_csv)
    output_path = (
        (ROOT / args.output).resolve() if args.output and not Path(args.output).is_absolute() else Path(args.output)
    ) if args.output else _default_output_path(summary_path)
    summary_csv_path = (
        (ROOT / args.summary_csv).resolve() if args.summary_csv and not Path(args.summary_csv).is_absolute() else Path(args.summary_csv)
    ) if args.summary_csv else _default_csv_output_path(output_path)

    wf_summary = _load_summary(summary_path)
    detail_df = pd.read_csv(detail_csv_path)
    if "fold" not in detail_df.columns:
        raise ValueError(f"WF detail CSV does not contain 'fold': {detail_csv_path}")
    detail_df["fold"] = pd.to_numeric(detail_df["fold"], errors="coerce").fillna(0).astype(int)

    policy_constraints = wf_summary.get("policy_constraints") if isinstance(wf_summary.get("policy_constraints"), dict) else {}
    base_constraints = PolicyConstraints(
        min_bet_ratio=float(policy_constraints.get("min_bet_ratio") or 0.05),
        min_bets_abs=int(policy_constraints.get("min_bets_abs") or 100),
        max_drawdown=float(policy_constraints.get("max_drawdown") or 0.40),
        min_final_bankroll=float(policy_constraints.get("min_final_bankroll") or 0.85),
        selection_mode=str(policy_constraints.get("selection_mode") or "gate_then_roi"),
    )
    min_bets_abs_values = _parse_int_list(args.min_bets_abs_values)

    folds_payload = wf_summary.get("folds") if isinstance(wf_summary.get("folds"), list) else []
    fold_meta = {
        int(fold.get("fold") or 0): fold
        for fold in folds_payload
        if isinstance(fold, dict) and int(fold.get("fold") or 0) > 0
    }
    if not fold_meta:
        raise ValueError(f"WF summary does not contain fold metadata: {summary_path}")
    stability_context = _build_stability_context(wf_summary, fold_meta)
    target_feasible_fold_counts = _parse_target_fold_counts(
        args.target_feasible_fold_counts,
        max_fold_count=len(fold_meta),
    )

    analyses: list[dict[str, Any]] = []
    preview_rows: list[dict[str, Any]] = []
    for min_bets_abs in min_bets_abs_values:
        sweep_constraints = PolicyConstraints(
            min_bet_ratio=base_constraints.min_bet_ratio,
            min_bets_abs=int(min_bets_abs),
            max_drawdown=base_constraints.max_drawdown,
            min_final_bankroll=base_constraints.min_final_bankroll,
            selection_mode=base_constraints.selection_mode,
        )
        analysis = _analyze_threshold(
            detail_df,
            fold_meta,
            constraints=sweep_constraints,
            min_feasible_folds=int(args.min_feasible_folds),
        )
        analyses.append(analysis)
        preview_rows.append(
            {
                "min_bets_abs": int(min_bets_abs),
                "feasible_fold_count": int(analysis["feasible_fold_count"]),
                "blocked_fold_count": int(analysis["blocked_fold_count"]),
                "passes_min_feasible_folds_preview": bool(analysis["passes_min_feasible_folds_preview"]),
                "dominant_failure_reason": analysis.get("dominant_failure_reason"),
                "dominant_failure_count": int(analysis.get("dominant_failure_count") or 0),
                "binding_min_bets_source_counts": str(analysis.get("binding_min_bets_source_counts") or {}),
                "feasible_folds": ",".join(str(fold) for fold in analysis.get("feasible_folds") or []),
            }
        )
        for target_fold_count in target_feasible_fold_counts:
            preview_rows[-1][f"passes_{target_fold_count}_feasible_folds"] = bool(
                int(analysis["feasible_fold_count"]) >= int(target_fold_count)
            )

    threshold_to_feasible = {
        int(analysis["policy_constraints"]["min_bets_abs"]): int(analysis["feasible_fold_count"])
        for analysis in analyses
    }
    fold_first_feasible_threshold: dict[str, int | None] = {}
    for fold_index in sorted(fold_meta):
        first_threshold: int | None = None
        for analysis in sorted(analyses, key=lambda item: int(item["policy_constraints"]["min_bets_abs"])):
            fold_summary = next((fold for fold in analysis.get("folds", []) if int(fold.get("fold") or 0) == fold_index), None)
            if isinstance(fold_summary, dict) and int(fold_summary.get("feasible_candidates") or 0) > 0:
                first_threshold = int(analysis["policy_constraints"]["min_bets_abs"])
                break
        fold_first_feasible_threshold[str(fold_index)] = first_threshold

    first_threshold_with_any_feasible_fold = next(
        (
            int(analysis["policy_constraints"]["min_bets_abs"])
            for analysis in sorted(analyses, key=lambda item: int(item["policy_constraints"]["min_bets_abs"]))
            if int(analysis.get("feasible_fold_count") or 0) > 0
        ),
        None,
    )
    first_threshold_passing_preview = next(
        (
            int(analysis["policy_constraints"]["min_bets_abs"])
            for analysis in sorted(analyses, key=lambda item: int(item["policy_constraints"]["min_bets_abs"]))
            if bool(analysis.get("passes_min_feasible_folds_preview"))
        ),
        None,
    )
    strictest_threshold_passing_feasible_fold_targets: dict[str, int | None] = {}
    for target_fold_count in target_feasible_fold_counts:
        strictest_threshold_passing_feasible_fold_targets[str(target_fold_count)] = next(
            (
                int(analysis["policy_constraints"]["min_bets_abs"])
                for analysis in sorted(
                    analyses,
                    key=lambda item: int(item["policy_constraints"]["min_bets_abs"]),
                    reverse=True,
                )
                if int(analysis.get("feasible_fold_count") or 0) >= int(target_fold_count)
            ),
            None,
        )

    report = {
        "run_context": {
            "wf_summary": str(summary_path.relative_to(ROOT)),
            "wf_detail_csv": str(detail_csv_path.relative_to(ROOT)),
            "min_bets_abs_values": min_bets_abs_values,
            "min_feasible_folds_preview": int(args.min_feasible_folds),
            "target_feasible_fold_counts": target_feasible_fold_counts,
        },
        "source_run_context": wf_summary.get("run_context"),
        "stability_context": stability_context,
        "baseline_policy_constraints": base_constraints.to_dict(),
        "feasible_fold_counts_by_threshold": threshold_to_feasible,
        "first_threshold_with_any_feasible_fold": first_threshold_with_any_feasible_fold,
        "first_threshold_passing_min_feasible_folds_preview": first_threshold_passing_preview,
        "strictest_threshold_passing_feasible_fold_targets": strictest_threshold_passing_feasible_fold_targets,
        "fold_first_feasible_min_bets_abs": fold_first_feasible_threshold,
        "threshold_analyses": analyses,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    pd.DataFrame(preview_rows).to_csv(summary_csv_path, index=False)

    print(f"saved sweep report to {output_path.relative_to(ROOT)}")
    print(f"saved sweep table to {summary_csv_path.relative_to(ROOT)}")
    for warning in stability_context.get("warnings") or []:
        print(f"warning={warning}")
    for row in preview_rows:
        print(
            "min_bets_abs={min_bets_abs} feasible_folds={feasible_fold_count} blocked_folds={blocked_fold_count} "
            "passes_preview={passes_min_feasible_folds_preview} dominant_failure={dominant_failure_reason}".format(**row)
        )
    print(f"strictest_threshold_passing_feasible_fold_targets={strictest_threshold_passing_feasible_fold_targets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
