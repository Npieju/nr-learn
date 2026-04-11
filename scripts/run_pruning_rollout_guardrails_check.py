from __future__ import annotations

import argparse
from math import isclose
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, utc_now_iso, write_json
from racing_ml.common.config import load_yaml


DEFAULT_BASELINE_FEATURE_CONFIG = "configs/features_catboost_rich_high_coverage_diag.yaml"
DEFAULT_CANDIDATE_FEATURE_CONFIG = (
    "configs/features_catboost_rich_high_coverage_diag_"
    "pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_"
    "class_rest_surface_jt_id_combo_dispatch_meta.yaml"
)
DEFAULT_FEATURE_GAP_SUMMARY = (
    "artifacts/reports/feature_gap_summary_"
    "pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_"
    "class_rest_surface_jt_id_combo_dispatch_meta_v1.json"
)
DEFAULT_EVALUATION_SUMMARY = (
    "artifacts/reports/evaluation_summary_"
    "catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_"
    "model_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_"
    "class_rest_surface_jt_id_combo_dispatch_meta_v1_wf_full_nested.json"
)
DEFAULT_PROMOTION_GATE = "artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json"
DEFAULT_WF_SUMMARY = "artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json"
DEFAULT_SEP_COMPARE = "artifacts/reports/serving_smoke_compare_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json"
DEFAULT_SEP_BANKROLL = "artifacts/reports/serving_stateful_bankroll_sweep_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json"
DEFAULT_DEC_COMPARE = "artifacts/reports/serving_smoke_compare_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json"
DEFAULT_DEC_BANKROLL = "artifacts/reports/serving_stateful_bankroll_sweep_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json"
DEFAULT_OUTPUT = "artifacts/reports/pruning_rollout_guardrails_stage7_check.json"

EXPECTED_RETAINED_ANCHORS = [
    "owner_last_50_win_rate",
    "競争条件",
    "リステッド・重賞競走",
    "障害区分",
    "sex",
]
EXPECTED_DECLARED_AND_EXCLUDED = [
    "発走時刻",
    "東西・外国・地方区分",
]


def log(message: str) -> None:
    print(f"[pruning-guardrails] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _normalize_string_list(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        text = str(values).strip()
        return [text] if text else []
    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            output.append(text)
    return output


def _build_check(name: str, ok: bool, details: dict[str, object] | None = None) -> dict[str, object]:
    payload: dict[str, object] = {"name": name, "ok": bool(ok)}
    if details:
        payload.update(details)
    return payload


def _load_required_json(path_text: str | Path, *, label: str) -> dict[str, object]:
    path = _resolve_path(path_text)
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {artifact_display_path(path, workspace_root=ROOT)}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {artifact_display_path(path, workspace_root=ROOT)}")
    return payload


def _sorted_unique(values: list[str]) -> list[str]:
    return sorted({value for value in values if str(value).strip()})


def _extract_feasible_fold_count(promotion_payload: dict[str, object]) -> int | None:
    checks = promotion_payload.get("checks")
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, dict) and check.get("name") == "wf_min_feasible_folds":
                value = check.get("feasible_fold_count")
                if value is not None:
                    return int(value)
    summary = promotion_payload.get("summary")
    if isinstance(summary, dict) and summary.get("wf_feasible_fold_count") is not None:
        return int(summary.get("wf_feasible_fold_count") or 0)
    return None


def _extract_wf_fold_counts(wf_payload: dict[str, object]) -> tuple[int, int]:
    folds = wf_payload.get("folds")
    if not isinstance(folds, list):
        return 0, 0
    fold_count = len(folds)
    feasible_fold_count = sum(
        1
        for fold in folds
        if isinstance(fold, dict) and int(fold.get("feasible_candidates") or 0) > 0
    )
    return fold_count, feasible_fold_count


def _compare_float(actual: float, expected: float, *, tolerance: float = 1e-12) -> bool:
    return isclose(float(actual), float(expected), rel_tol=0.0, abs_tol=tolerance)


def _validate_compare_payload(
    *,
    payload: dict[str, object],
    expected_bets: int,
    expected_net: float,
    label: str,
) -> tuple[list[dict[str, object]], list[str]]:
    checks: list[dict[str, object]] = []
    errors: list[str] = []
    comparison = payload.get("comparison")
    aggregates = comparison.get("shared_ok_aggregates") if isinstance(comparison, dict) else None
    if not isinstance(aggregates, dict):
        message = f"{label} compare payload is missing shared_ok_aggregates"
        checks.append(_build_check(f"{label}_shared_ok_aggregates_present", False))
        errors.append(message)
        return checks, errors

    total_bets_delta = int(aggregates.get("total_policy_bets_delta") or 0)
    total_net_delta = float(aggregates.get("total_policy_net_delta") or 0.0)
    left_total_bets = int(aggregates.get("left_total_policy_bets") or 0)
    right_total_bets = int(aggregates.get("right_total_policy_bets") or 0)
    left_total_net = float(aggregates.get("left_total_policy_net") or 0.0)
    right_total_net = float(aggregates.get("right_total_policy_net") or 0.0)

    checks.append(
        _build_check(
            f"{label}_compare_equivalence",
            total_bets_delta == 0 and _compare_float(total_net_delta, 0.0),
            {
                "left_total_policy_bets": left_total_bets,
                "right_total_policy_bets": right_total_bets,
                "left_total_policy_net": left_total_net,
                "right_total_policy_net": right_total_net,
            },
        )
    )
    if total_bets_delta != 0 or not _compare_float(total_net_delta, 0.0):
        errors.append(f"{label} compare is not equivalent")

    checks.append(
        _build_check(
            f"{label}_compare_expected_totals",
            left_total_bets == expected_bets and right_total_bets == expected_bets
            and _compare_float(left_total_net, expected_net)
            and _compare_float(right_total_net, expected_net),
            {
                "expected_total_bets": expected_bets,
                "expected_total_net": expected_net,
            },
        )
    )
    if left_total_bets != expected_bets or right_total_bets != expected_bets or not _compare_float(left_total_net, expected_net) or not _compare_float(right_total_net, expected_net):
        errors.append(f"{label} compare totals do not match expected values")
    return checks, errors


def _validate_bankroll_payload(
    *,
    payload: dict[str, object],
    expected_final_bankroll: float,
    expected_total_bets: int,
    label: str,
) -> tuple[list[dict[str, object]], list[str]]:
    checks: list[dict[str, object]] = []
    errors: list[str] = []
    baseline_only = payload.get("baseline_only")
    best_result = payload.get("best_result")
    if not isinstance(baseline_only, dict) or not isinstance(best_result, dict):
        message = f"{label} bankroll payload is missing baseline_only or best_result"
        checks.append(_build_check(f"{label}_bankroll_payload_present", False))
        errors.append(message)
        return checks, errors

    baseline_bankroll = float(baseline_only.get("final_bankroll") or 0.0)
    best_bankroll = float(best_result.get("final_bankroll") or 0.0)
    baseline_bets = int(baseline_only.get("total_bets") or 0)
    best_bets = int(best_result.get("total_bets") or 0)

    checks.append(
        _build_check(
            f"{label}_bankroll_equivalence",
            _compare_float(baseline_bankroll, best_bankroll) and baseline_bets == best_bets,
            {
                "baseline_final_bankroll": baseline_bankroll,
                "best_result_final_bankroll": best_bankroll,
                "baseline_total_bets": baseline_bets,
                "best_result_total_bets": best_bets,
            },
        )
    )
    if not _compare_float(baseline_bankroll, best_bankroll) or baseline_bets != best_bets:
        errors.append(f"{label} bankroll sweep is not equivalent")

    checks.append(
        _build_check(
            f"{label}_bankroll_expected_totals",
            _compare_float(baseline_bankroll, expected_final_bankroll)
            and _compare_float(best_bankroll, expected_final_bankroll)
            and baseline_bets == expected_total_bets
            and best_bets == expected_total_bets,
            {
                "expected_final_bankroll": expected_final_bankroll,
                "expected_total_bets": expected_total_bets,
            },
        )
    )
    if not _compare_float(baseline_bankroll, expected_final_bankroll) or not _compare_float(best_bankroll, expected_final_bankroll) or baseline_bets != expected_total_bets or best_bets != expected_total_bets:
        errors.append(f"{label} bankroll sweep totals do not match expected values")
    return checks, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-feature-config", default=DEFAULT_BASELINE_FEATURE_CONFIG)
    parser.add_argument("--candidate-feature-config", default=DEFAULT_CANDIDATE_FEATURE_CONFIG)
    parser.add_argument("--feature-gap-summary", default=DEFAULT_FEATURE_GAP_SUMMARY)
    parser.add_argument("--evaluation-summary", default=DEFAULT_EVALUATION_SUMMARY)
    parser.add_argument("--promotion-gate", default=DEFAULT_PROMOTION_GATE)
    parser.add_argument("--wf-summary", default=DEFAULT_WF_SUMMARY)
    parser.add_argument("--sep-compare", default=DEFAULT_SEP_COMPARE)
    parser.add_argument("--sep-bankroll", default=DEFAULT_SEP_BANKROLL)
    parser.add_argument("--dec-compare", default=DEFAULT_DEC_COMPARE)
    parser.add_argument("--dec-bankroll", default=DEFAULT_DEC_BANKROLL)
    parser.add_argument("--expected-removal-count", type=int, default=45)
    parser.add_argument("--expected-feature-count", type=int, default=64)
    parser.add_argument("--expected-categorical-count", type=int, default=25)
    parser.add_argument("--expected-feasible-fold-count", type=int, default=3)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    output_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)

    checks: list[dict[str, object]] = []
    blocking_reasons: list[str] = []

    baseline_cfg = load_yaml(_resolve_path(args.baseline_feature_config))
    candidate_cfg = load_yaml(_resolve_path(args.candidate_feature_config))
    baseline_selection = baseline_cfg.get("selection", {}) if isinstance(baseline_cfg, dict) else {}
    candidate_selection = candidate_cfg.get("selection", {}) if isinstance(candidate_cfg, dict) else {}

    baseline_force_include = _sorted_unique(_normalize_string_list(baseline_selection.get("force_include_columns")))
    baseline_force_categorical = _sorted_unique(_normalize_string_list(baseline_selection.get("force_categorical_columns")))
    candidate_force_include = _sorted_unique(_normalize_string_list(candidate_selection.get("force_include_columns")))
    candidate_force_categorical = _sorted_unique(_normalize_string_list(candidate_selection.get("force_categorical_columns")))
    candidate_excluded = _sorted_unique(_normalize_string_list(candidate_selection.get("exclude_columns")))

    baseline_forced_surface = set(baseline_force_include) | set(baseline_force_categorical)
    candidate_declared_surface = set(candidate_force_include) | set(candidate_force_categorical)
    effective_retained_anchors = sorted(
        column
        for column in baseline_forced_surface
        if column in candidate_declared_surface and column not in candidate_excluded
    )
    removal_set = sorted(column for column in baseline_forced_surface if column not in effective_retained_anchors)
    declared_and_excluded = sorted(column for column in candidate_declared_surface if column in candidate_excluded)

    checks.append(
        _build_check(
            "config_effective_anchor_set",
            effective_retained_anchors == sorted(EXPECTED_RETAINED_ANCHORS),
            {
                "effective_retained_anchors": effective_retained_anchors,
                "candidate_force_include_count": len(candidate_force_include),
                "candidate_force_categorical_count": len(candidate_force_categorical),
            },
        )
    )
    if effective_retained_anchors != sorted(EXPECTED_RETAINED_ANCHORS):
        blocking_reasons.append("effective retained anchors do not match the expected 5-column set")

    checks.append(
        _build_check(
            "config_removal_count",
            len(removal_set) == int(args.expected_removal_count),
            {"removal_count": len(removal_set)},
        )
    )
    if len(removal_set) != int(args.expected_removal_count):
        blocking_reasons.append("effective removal set does not match the expected 45 columns")

    checks.append(
        _build_check(
            "config_declared_and_excluded_columns",
            declared_and_excluded == sorted(EXPECTED_DECLARED_AND_EXCLUDED),
            {"declared_and_excluded": declared_and_excluded},
        )
    )
    if declared_and_excluded != sorted(EXPECTED_DECLARED_AND_EXCLUDED):
        blocking_reasons.append("declared-and-excluded columns drifted from the known dispatch metadata pair")

    feature_gap_summary = _load_required_json(args.feature_gap_summary, label="feature gap summary")
    feature_gap_run_context = feature_gap_summary.get("run_context") if isinstance(feature_gap_summary.get("run_context"), dict) else {}
    feature_gap_summary_block = feature_gap_summary.get("summary") if isinstance(feature_gap_summary.get("summary"), dict) else {}
    checks.append(
        _build_check(
            "feature_gap_expected_shape",
            int(feature_gap_run_context.get("selected_feature_count") or 0) == int(args.expected_feature_count)
            and int(feature_gap_run_context.get("categorical_feature_count") or 0) == int(args.expected_categorical_count),
            {
                "selected_feature_count": feature_gap_run_context.get("selected_feature_count"),
                "categorical_feature_count": feature_gap_run_context.get("categorical_feature_count"),
            },
        )
    )
    if int(feature_gap_run_context.get("selected_feature_count") or 0) != int(args.expected_feature_count) or int(feature_gap_run_context.get("categorical_feature_count") or 0) != int(args.expected_categorical_count):
        blocking_reasons.append("feature gap summary shape does not match stage-7 expectations")

    checks.append(
        _build_check(
            "feature_gap_force_include_clean",
            feature_gap_summary_block.get("priority_missing_raw_columns") == []
            and feature_gap_summary_block.get("missing_force_include_features") == []
            and feature_gap_summary_block.get("empty_force_include_features") == []
            and feature_gap_summary_block.get("low_coverage_force_include_features") == [],
            {
                "priority_missing_raw_columns": feature_gap_summary_block.get("priority_missing_raw_columns"),
                "missing_force_include_features": feature_gap_summary_block.get("missing_force_include_features"),
                "empty_force_include_features": feature_gap_summary_block.get("empty_force_include_features"),
                "low_coverage_force_include_features": feature_gap_summary_block.get("low_coverage_force_include_features"),
            },
        )
    )
    if feature_gap_summary_block.get("priority_missing_raw_columns") != [] or feature_gap_summary_block.get("missing_force_include_features") != [] or feature_gap_summary_block.get("empty_force_include_features") != [] or feature_gap_summary_block.get("low_coverage_force_include_features") != []:
        blocking_reasons.append("feature gap summary is not clean for the stage-7 candidate")

    evaluation_summary = _load_required_json(args.evaluation_summary, label="evaluation summary")
    eval_run_context = evaluation_summary.get("run_context") if isinstance(evaluation_summary.get("run_context"), dict) else {}
    checks.append(
        _build_check(
            "evaluation_summary_expected_shape",
            int(eval_run_context.get("feature_count") or 0) == int(args.expected_feature_count)
            and int(eval_run_context.get("categorical_feature_count") or 0) == int(args.expected_categorical_count),
            {
                "feature_count": eval_run_context.get("feature_count"),
                "categorical_feature_count": eval_run_context.get("categorical_feature_count"),
                "auc": evaluation_summary.get("auc"),
                "top1_roi": evaluation_summary.get("top1_roi"),
                "ev_top1_roi": evaluation_summary.get("ev_top1_roi"),
            },
        )
    )
    if int(eval_run_context.get("feature_count") or 0) != int(args.expected_feature_count) or int(eval_run_context.get("categorical_feature_count") or 0) != int(args.expected_categorical_count):
        blocking_reasons.append("evaluation summary shape does not match stage-7 expectations")

    promotion_gate = _load_required_json(args.promotion_gate, label="promotion gate")
    feasible_fold_count = _extract_feasible_fold_count(promotion_gate)
    checks.append(
        _build_check(
            "promotion_gate_status",
            str(promotion_gate.get("status") or "") == "pass" and str(promotion_gate.get("decision") or "") == "promote",
            {
                "status": promotion_gate.get("status"),
                "decision": promotion_gate.get("decision"),
            },
        )
    )
    if str(promotion_gate.get("status") or "") != "pass" or str(promotion_gate.get("decision") or "") != "promote":
        blocking_reasons.append("promotion gate is not pass/promote")

    checks.append(
        _build_check(
            "promotion_gate_feasible_folds",
            feasible_fold_count == int(args.expected_feasible_fold_count),
            {"feasible_fold_count": feasible_fold_count},
        )
    )
    if feasible_fold_count != int(args.expected_feasible_fold_count):
        blocking_reasons.append("promotion gate feasible fold count does not match stage-7 expectation")

    wf_summary = _load_required_json(args.wf_summary, label="walk-forward feasibility summary")
    wf_fold_count, wf_feasible_fold_count = _extract_wf_fold_counts(wf_summary)
    checks.append(
        _build_check(
            "wf_summary_expected_shape",
            str(wf_summary.get("stability_assessment") or "") == "representative"
            and wf_feasible_fold_count == int(args.expected_feasible_fold_count)
            and wf_fold_count == 5,
            {
                "stability_assessment": wf_summary.get("stability_assessment"),
                "feasible_fold_count": wf_feasible_fold_count,
                "fold_count": wf_fold_count,
            },
        )
    )
    if str(wf_summary.get("stability_assessment") or "") != "representative" or wf_feasible_fold_count != int(args.expected_feasible_fold_count) or wf_fold_count != 5:
        blocking_reasons.append("walk-forward feasibility summary drifted from the accepted stage-7 read")

    sep_compare = _load_required_json(args.sep_compare, label="September compare")
    sep_checks, sep_errors = _validate_compare_payload(
        payload=sep_compare,
        expected_bets=33,
        expected_net=-20.0,
        label="sep25",
    )
    checks.extend(sep_checks)
    blocking_reasons.extend(sep_errors)

    sep_bankroll = _load_required_json(args.sep_bankroll, label="September bankroll sweep")
    sep_bankroll_checks, sep_bankroll_errors = _validate_bankroll_payload(
        payload=sep_bankroll,
        expected_final_bankroll=0.3931722898269604,
        expected_total_bets=33,
        label="sep25",
    )
    checks.extend(sep_bankroll_checks)
    blocking_reasons.extend(sep_bankroll_errors)

    dec_compare = _load_required_json(args.dec_compare, label="December compare")
    dec_checks, dec_errors = _validate_compare_payload(
        payload=dec_compare,
        expected_bets=17,
        expected_net=-5.199999999999999,
        label="dec25",
    )
    checks.extend(dec_checks)
    blocking_reasons.extend(dec_errors)

    dec_bankroll = _load_required_json(args.dec_bankroll, label="December bankroll sweep")
    dec_bankroll_checks, dec_bankroll_errors = _validate_bankroll_payload(
        payload=dec_bankroll,
        expected_final_bankroll=0.7886889523160848,
        expected_total_bets=17,
        label="dec25",
    )
    checks.extend(dec_bankroll_checks)
    blocking_reasons.extend(dec_bankroll_errors)

    blocking_reasons = _sorted_unique(blocking_reasons)
    payload = {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": "pass" if not blocking_reasons else "block",
        "decision": "review_ready" if not blocking_reasons else "hold",
        "surface": "stage7_pruning_rollout_guardrails",
        "artifacts": {
            "baseline_feature_config": artifact_display_path(_resolve_path(args.baseline_feature_config), workspace_root=ROOT),
            "candidate_feature_config": artifact_display_path(_resolve_path(args.candidate_feature_config), workspace_root=ROOT),
            "feature_gap_summary": artifact_display_path(_resolve_path(args.feature_gap_summary), workspace_root=ROOT),
            "evaluation_summary": artifact_display_path(_resolve_path(args.evaluation_summary), workspace_root=ROOT),
            "promotion_gate": artifact_display_path(_resolve_path(args.promotion_gate), workspace_root=ROOT),
            "wf_summary": artifact_display_path(_resolve_path(args.wf_summary), workspace_root=ROOT),
            "sep_compare": artifact_display_path(_resolve_path(args.sep_compare), workspace_root=ROOT),
            "sep_bankroll": artifact_display_path(_resolve_path(args.sep_bankroll), workspace_root=ROOT),
            "dec_compare": artifact_display_path(_resolve_path(args.dec_compare), workspace_root=ROOT),
            "dec_bankroll": artifact_display_path(_resolve_path(args.dec_bankroll), workspace_root=ROOT),
        },
        "summary": {
            "effective_retained_anchors": effective_retained_anchors,
            "declared_and_excluded_columns": declared_and_excluded,
            "removal_count": len(removal_set),
            "expected_feature_count": int(args.expected_feature_count),
            "expected_categorical_count": int(args.expected_categorical_count),
            "expected_feasible_fold_count": int(args.expected_feasible_fold_count),
        },
        "checks": checks,
        "blocking_reasons": blocking_reasons,
    }
    write_json(output_path, payload)

    log(f"report saved: {artifact_display_path(output_path, workspace_root=ROOT)}")
    log(f"status: {payload['status']}")
    if blocking_reasons:
        log(f"blocking_reasons: {blocking_reasons}")
        return 1
    log("stage-7 rollout guardrails are mechanically review-ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())