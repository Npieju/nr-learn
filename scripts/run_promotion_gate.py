from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.evaluation.manifest_validation import (
    DEFAULT_EVALUATION_MANIFEST,
    display_artifact_path,
    resolve_artifact_path,
    validate_evaluation_manifest_safe,
)


DEFAULT_OUTPUT = "artifacts/reports/promotion_gate_report.json"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[promotion-gate {now}] {message}", flush=True)


def _build_check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = {
        "name": name,
        "ok": bool(ok),
    }
    if details:
        payload.update(details)
    return payload


def _append_result(
    checks: list[dict[str, Any]],
    blocking_reasons: list[str],
    check: dict[str, Any],
    error_message: str | None = None,
) -> None:
    checks.append(check)
    if not check.get("ok") and error_message:
        blocking_reasons.append(error_message)


def _read_json_dict(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = read_json(path)
    return payload if isinstance(payload, dict) else None


def _resolve_matching_wf_summary(
    *,
    report_dir: Path,
    config: str | None,
    data_config: str | None,
    feature_config: str | None,
) -> tuple[Path | None, dict[str, Any] | None]:
    if not config:
        return None, None

    latest_match: tuple[float, Path, dict[str, Any]] | None = None
    for path in sorted(report_dir.glob("wf_feasibility_diag_*.json")):
        payload = _read_json_dict(path)
        if payload is None:
            continue
        run_context = payload.get("run_context") if isinstance(payload.get("run_context"), dict) else {}
        if run_context.get("config") != config:
            continue
        if data_config is not None and run_context.get("data_config") != data_config:
            continue
        if feature_config is not None and run_context.get("feature_config") != feature_config:
            continue
        candidate = (path.stat().st_mtime, path, payload)
        if latest_match is None or candidate[0] > latest_match[0]:
            latest_match = candidate

    if latest_match is None:
        return None, None
    return latest_match[1], latest_match[2]


def _summarize_wf_feasibility_diagnostics(
    folds: list[dict[str, Any]],
    *,
    policy_constraints: dict[str, Any] | None = None,
) -> dict[str, Any]:
    failure_reason_counts: Counter[str] = Counter()
    min_bets_required_by_fold: dict[str, int] = {}
    ratio_bets_required_by_fold: dict[str, int] = {}
    binding_min_bets_source_by_fold: dict[str, str] = {}
    binding_source_counts: Counter[str] = Counter()
    best_fallback_by_fold: list[dict[str, Any]] = []
    closest_infeasible_by_fold: list[dict[str, Any]] = []
    max_infeasible_bets = 0
    absolute_min_bets_required = 0
    min_bet_ratio = 0.0

    if isinstance(policy_constraints, dict):
        absolute_min_bets_required = int(policy_constraints.get("min_bets_abs") or 0)
        min_bet_ratio = float(policy_constraints.get("min_bet_ratio") or 0.0)

    for fold in folds:
        fold_index = int((fold or {}).get("fold") or 0)
        failure_counts = (fold or {}).get("failure_reason_counts")
        if isinstance(failure_counts, dict):
            for reason, count in failure_counts.items():
                try:
                    failure_reason_counts[str(reason)] += int(count)
                except Exception:
                    continue

        min_bets_required = int((fold or {}).get("min_bets_required") or 0)
        min_bets_required_by_fold[str(fold_index)] = min_bets_required
        valid_races = int((fold or {}).get("valid_races") or 0)
        ratio_required = int(valid_races * min_bet_ratio)
        ratio_bets_required_by_fold[str(fold_index)] = ratio_required
        if min_bets_required <= 0:
            binding_source = "unknown"
        elif min_bets_required == absolute_min_bets_required and absolute_min_bets_required >= ratio_required:
            binding_source = "absolute"
        elif min_bets_required == ratio_required and ratio_required > absolute_min_bets_required:
            binding_source = "ratio"
        else:
            binding_source = "mixed"
        binding_min_bets_source_by_fold[str(fold_index)] = binding_source
        binding_source_counts[binding_source] += 1

        best_fallback = (fold or {}).get("best_fallback")
        if isinstance(best_fallback, dict):
            best_fallback_by_fold.append(
                {
                    "fold": fold_index,
                    "strategy_kind": best_fallback.get("strategy_kind"),
                    "bets": int(best_fallback.get("bets") or 0),
                    "roi": best_fallback.get("roi"),
                    "final_bankroll": best_fallback.get("final_bankroll"),
                    "max_drawdown": best_fallback.get("max_drawdown"),
                    "gate_failures": list(best_fallback.get("gate_failures") or []),
                }
            )
            max_infeasible_bets = max(max_infeasible_bets, int(best_fallback.get("bets") or 0))

        closest_infeasible = (fold or {}).get("closest_infeasible")
        if isinstance(closest_infeasible, list) and closest_infeasible:
            first_candidate = closest_infeasible[0]
            if isinstance(first_candidate, dict):
                closest_infeasible_by_fold.append(
                    {
                        "fold": fold_index,
                        "strategy_kind": first_candidate.get("strategy_kind"),
                        "bets": int(first_candidate.get("bets") or 0),
                        "roi": first_candidate.get("roi"),
                        "final_bankroll": first_candidate.get("final_bankroll"),
                        "max_drawdown": first_candidate.get("max_drawdown"),
                        "gate_failures": list(first_candidate.get("gate_failures") or []),
                    }
                )
                max_infeasible_bets = max(max_infeasible_bets, int(first_candidate.get("bets") or 0))

    dominant_failure_reason = failure_reason_counts.most_common(1)[0][0] if failure_reason_counts else None
    dominant_failure_count = failure_reason_counts.most_common(1)[0][1] if failure_reason_counts else 0

    return {
        "dominant_failure_reason": dominant_failure_reason,
        "dominant_failure_count": int(dominant_failure_count),
        "failure_reason_counts_total": dict(failure_reason_counts),
        "min_bet_ratio": float(min_bet_ratio),
        "min_bets_abs": int(absolute_min_bets_required),
        "min_bets_required_by_fold": min_bets_required_by_fold,
        "ratio_bets_required_by_fold": ratio_bets_required_by_fold,
        "binding_min_bets_source_by_fold": binding_min_bets_source_by_fold,
        "binding_min_bets_source_counts": dict(binding_source_counts),
        "max_infeasible_bets_observed": int(max_infeasible_bets),
        "best_fallback_by_fold": best_fallback_by_fold,
        "closest_infeasible_by_fold": closest_infeasible_by_fold,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evaluation-manifest", default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--wf-summary", default=None)
    parser.add_argument("--min-feasible-folds", type=int, default=1)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[promotion-gate]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="loading evaluation manifest")

        manifest_path = resolve_artifact_path(ROOT, args.evaluation_manifest)
        if manifest_path is None:
            raise ValueError("Evaluation manifest path is required")

        with Heartbeat("[promotion-gate]", f"validating {manifest_path.name}", logger=log_progress):
            manifest_report = validate_evaluation_manifest_safe(manifest_path, root=ROOT)
        manifest_payload = _read_json_dict(manifest_path)
        if manifest_payload is None:
            raise RuntimeError(f"Evaluation manifest is unreadable: {manifest_path}")
        progress.update(message="loading evaluation summary")

        files_payload = manifest_payload.get("files") if isinstance(manifest_payload.get("files"), dict) else {}
        summary_path = resolve_artifact_path(ROOT, files_payload.get("latest_summary"))
        evaluation_summary = _read_json_dict(summary_path)

        progress.update(message="resolving walk-forward feasibility summary")
        wf_summary_path = resolve_artifact_path(ROOT, args.wf_summary)
        wf_summary_payload = _read_json_dict(wf_summary_path)
        auto_resolved_wf_path: Path | None = None
        if wf_summary_payload is None:
            auto_resolved_wf_path, wf_summary_payload = _resolve_matching_wf_summary(
                report_dir=ROOT / "artifacts" / "reports",
                config=manifest_payload.get("config"),
                data_config=manifest_payload.get("data_config"),
                feature_config=manifest_payload.get("feature_config"),
            )
            if auto_resolved_wf_path is not None:
                wf_summary_path = auto_resolved_wf_path

        progress.update(message="evaluating gate checks")
        checks: list[dict[str, Any]] = []
        blocking_reasons: list[str] = []
        warnings: list[str] = []

        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "evaluation_manifest_valid",
                manifest_report.get("status") == "ok",
                {"status": manifest_report.get("status"), "manifest": manifest_report.get("manifest")},
            ),
            f"Evaluation manifest validation failed: {manifest_report.get('errors')}",
        )

        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "evaluation_summary_present",
                evaluation_summary is not None,
                {"summary_path": str(summary_path) if summary_path else None},
            ),
            "Evaluation summary is missing",
        )

        eval_stability_assessment = manifest_payload.get("stability_assessment")
        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "evaluation_representative",
                eval_stability_assessment == "representative",
                {"assessment": eval_stability_assessment},
            ),
            f"Evaluation stability is not representative: {eval_stability_assessment}",
        )

        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "wf_summary_present",
                wf_summary_payload is not None,
                {
                    "wf_summary": display_artifact_path(ROOT, wf_summary_path) if wf_summary_path else None,
                    "auto_resolved": bool(auto_resolved_wf_path),
                },
            ),
            "Matching walk-forward feasibility summary is missing",
        )

        wf_run_context = wf_summary_payload.get("run_context") if isinstance((wf_summary_payload or {}).get("run_context"), dict) else {}
        wf_config_match = (
            wf_summary_payload is not None
            and wf_run_context.get("config") == manifest_payload.get("config")
            and wf_run_context.get("data_config") == manifest_payload.get("data_config")
            and wf_run_context.get("feature_config") == manifest_payload.get("feature_config")
        )
        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "wf_summary_matches_evaluation_config",
                wf_config_match,
                {
                    "evaluation_config": manifest_payload.get("config"),
                    "wf_config": wf_run_context.get("config") if wf_summary_payload else None,
                    "evaluation_data_config": manifest_payload.get("data_config"),
                    "wf_data_config": wf_run_context.get("data_config") if wf_summary_payload else None,
                    "evaluation_feature_config": manifest_payload.get("feature_config"),
                    "wf_feature_config": wf_run_context.get("feature_config") if wf_summary_payload else None,
                },
            ),
            "Walk-forward feasibility summary does not match the evaluation config tuple",
        )

        wf_stability_assessment = (wf_summary_payload or {}).get("stability_assessment")
        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "wf_summary_representative",
                wf_stability_assessment == "representative",
                {"assessment": wf_stability_assessment},
            ),
            f"Walk-forward feasibility stability is not representative: {wf_stability_assessment}",
        )

        folds = (wf_summary_payload or {}).get("folds") if isinstance((wf_summary_payload or {}).get("folds"), list) else []
        _append_result(
            checks,
            blocking_reasons,
            _build_check("wf_has_folds", bool(folds), {"fold_count": len(folds)}),
            "Walk-forward feasibility summary has no folds",
        )

        feasible_fold_count = sum(1 for fold in folds if int((fold or {}).get("feasible_candidates") or 0) > 0)
        _append_result(
            checks,
            blocking_reasons,
            _build_check(
                "wf_min_feasible_folds",
                feasible_fold_count >= max(0, int(args.min_feasible_folds)),
                {
                    "min_feasible_folds": int(args.min_feasible_folds),
                    "feasible_fold_count": int(feasible_fold_count),
                },
            ),
            f"Walk-forward feasible fold count is below threshold: {feasible_fold_count} < {int(args.min_feasible_folds)}",
        )

        valid_probe_only_count = sum(1 for fold in folds if (fold or {}).get("valid_stability_assessment") == "probe_only")
        test_probe_only_count = sum(1 for fold in folds if (fold or {}).get("test_stability_assessment") == "probe_only")
        wf_diagnostics = _summarize_wf_feasibility_diagnostics(
            folds,
            policy_constraints=(wf_summary_payload or {}).get("policy_constraints") if isinstance((wf_summary_payload or {}).get("policy_constraints"), dict) else None,
        )
        if folds and valid_probe_only_count == len(folds) and test_probe_only_count == len(folds):
            warnings.append("All walk-forward valid/test slices are probe_only; use fold-level ROI only as directional evidence.")
        if feasible_fold_count == 0 and wf_diagnostics.get("dominant_failure_reason") is not None:
            warnings.append(
                "No feasible walk-forward folds were found; dominant gate failure reason="
                f"{wf_diagnostics['dominant_failure_reason']}"
            )

        output_payload = {
            "status": "pass" if not blocking_reasons else "block",
            "decision": "promote" if not blocking_reasons else "hold",
            "evaluation_manifest": display_artifact_path(ROOT, manifest_path),
            "evaluation_manifest_report": manifest_report,
            "evaluation_summary": display_artifact_path(ROOT, summary_path) if summary_path else None,
            "wf_summary": display_artifact_path(ROOT, wf_summary_path) if wf_summary_path else None,
            "checks": checks,
            "blocking_reasons": blocking_reasons,
            "warnings": warnings,
            "wf_diagnostics": wf_diagnostics,
            "summary": {
                "profile": manifest_payload.get("profile"),
                "config": manifest_payload.get("config"),
                "evaluation_stability_assessment": eval_stability_assessment,
                "evaluation_n_races": ((manifest_payload.get("stability_guardrail") or {}).get("observed") or {}).get("n_races"),
                "evaluation_n_dates": ((manifest_payload.get("stability_guardrail") or {}).get("observed") or {}).get("n_dates"),
                "evaluation_ev_threshold_1_0_bets": ((manifest_payload.get("stability_guardrail") or {}).get("observed") or {}).get("ev_threshold_1_0_bets"),
                "wf_stability_assessment": wf_stability_assessment,
                "wf_fold_count": len(folds),
                "wf_feasible_fold_count": int(feasible_fold_count),
                "wf_valid_probe_only_count": int(valid_probe_only_count),
                "wf_test_probe_only_count": int(test_probe_only_count),
                "wf_dominant_failure_reason": wf_diagnostics.get("dominant_failure_reason"),
                "wf_binding_min_bets_source_counts": wf_diagnostics.get("binding_min_bets_source_counts"),
                "wf_max_infeasible_bets_observed": wf_diagnostics.get("max_infeasible_bets_observed"),
                "auto_resolved_wf_summary": bool(auto_resolved_wf_path),
            },
        }

        output_path = resolve_artifact_path(ROOT, args.output)
        if output_path is None:
            raise ValueError("Output path could not be resolved")
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        with Heartbeat("[promotion-gate]", "writing report", logger=log_progress):
            write_json(output_path, output_payload)
        progress.complete(message=f"gate completed status={output_payload['status']}")

        print(f"[promotion-gate] report saved: {output_path}")
        print(f"[promotion-gate] decision: {output_payload['decision']}")
        print(f"[promotion-gate] status: {output_payload['status']}")
        if blocking_reasons:
            print(f"[promotion-gate] blocking_reasons: {blocking_reasons}")
            return 1
        if warnings:
            print(f"[promotion-gate] warnings: {warnings}")
        return 0
    except KeyboardInterrupt:
        print("[promotion-gate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[promotion-gate] failed: {error}")
        return 1
    except Exception as error:
        print(f"[promotion-gate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())