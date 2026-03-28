from __future__ import annotations

import argparse
from pathlib import Path
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, utc_now_iso, write_json


DEFAULT_REFERENCE = "current_recommended_serving_2025_latest"
DEFAULT_REVISION = "r20260325_current_recommended_serving_2025_latest_benchmark_refresh"
DEFAULT_PROMOTION = "artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json"
DEFAULT_REVISION_MANIFEST = "artifacts/reports/revision_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json"
DEFAULT_EVALUATION_MANIFEST = (
    "artifacts/reports/evaluation_manifest_"
    "catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_"
    "model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json"
)
DEFAULT_EVALUATION_SUMMARY = (
    "artifacts/reports/evaluation_summary_"
    "catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_"
    "model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json"
)
DEFAULT_PUBLIC_DOC = "docs/public_benchmark_snapshot.md"


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _read_required_payload(path: Path, *, label: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {artifact_display_path(path, workspace_root=ROOT)}")
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {artifact_display_path(path, workspace_root=ROOT)}")
    return payload


def _extract_metric(summary_payload: dict[str, object], promotion_payload: dict[str, object]) -> dict[str, object]:
    checks = promotion_payload.get("checks")
    promotion_summary = promotion_payload.get("summary")
    wf_diagnostics = promotion_payload.get("wf_diagnostics")
    feasible_fold_count = None
    if isinstance(checks, list):
        for check in checks:
            if isinstance(check, dict) and check.get("name") == "wf_min_feasible_folds":
                feasible_fold_count = check.get("feasible_fold_count")
                break

    if feasible_fold_count is None and isinstance(promotion_summary, dict):
        feasible_fold_count = promotion_summary.get("wf_feasible_fold_count")

    return {
        "decision": promotion_payload.get("decision"),
        "status": promotion_payload.get("status"),
        "stability_assessment": summary_payload.get("stability_assessment"),
        "auc": summary_payload.get("auc"),
        "top1_roi": summary_payload.get("top1_roi"),
        "ev_top1_roi": summary_payload.get("ev_top1_roi"),
        "nested_wf_weighted_test_roi": summary_payload.get("wf_nested_test_weighted_roi") or summary_payload.get("wf_nested_weighted_test_roi"),
        "nested_wf_bets_total": summary_payload.get("wf_nested_test_bets_total"),
        "formal_benchmark_weighted_roi": wf_diagnostics.get("best_fallback_weighted_roi") if isinstance(wf_diagnostics, dict) else None,
        "formal_benchmark_feasible_folds": feasible_fold_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", default=DEFAULT_REFERENCE)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--promotion-manifest", default=DEFAULT_PROMOTION)
    parser.add_argument("--revision-manifest", default=DEFAULT_REVISION_MANIFEST)
    parser.add_argument("--evaluation-manifest", default=DEFAULT_EVALUATION_MANIFEST)
    parser.add_argument("--evaluation-summary", default=DEFAULT_EVALUATION_SUMMARY)
    parser.add_argument("--public-doc", default=DEFAULT_PUBLIC_DOC)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    output = args.output or f"artifacts/reports/public_benchmark_reference_{args.reference}.json"
    output_path = _resolve_path(output)
    promotion_path = _resolve_path(args.promotion_manifest)
    revision_path = _resolve_path(args.revision_manifest)
    evaluation_manifest_path = _resolve_path(args.evaluation_manifest)
    evaluation_summary_path = _resolve_path(args.evaluation_summary)
    public_doc_path = _resolve_path(args.public_doc)

    try:
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        promotion_payload = _read_required_payload(promotion_path, label="promotion manifest")
        revision_payload = _read_required_payload(revision_path, label="revision manifest")
        evaluation_manifest_payload = _read_required_payload(evaluation_manifest_path, label="evaluation manifest")
        evaluation_summary_payload = _read_required_payload(evaluation_summary_path, label="evaluation summary")

        payload = {
            "started_at": utc_now_iso(),
            "finished_at": utc_now_iso(),
            "status": "completed",
            "reference_kind": "public_benchmark_reference",
            "reference": args.reference,
            "revision": args.revision,
            "universe": "jra",
            "source_scope": "jra_only",
            "public_doc": artifact_display_path(public_doc_path, workspace_root=ROOT),
            "artifacts": {
                "reference_manifest": artifact_display_path(output_path, workspace_root=ROOT),
                "promotion_manifest": artifact_display_path(promotion_path, workspace_root=ROOT),
                "revision_manifest": artifact_display_path(revision_path, workspace_root=ROOT),
                "evaluation_manifest": artifact_display_path(evaluation_manifest_path, workspace_root=ROOT),
                "evaluation_summary": artifact_display_path(evaluation_summary_path, workspace_root=ROOT),
            },
            "metrics": _extract_metric(evaluation_summary_payload, promotion_payload),
            "promotion_summary": {
                "decision": promotion_payload.get("decision"),
                "status": promotion_payload.get("status"),
                "blocking_reasons": promotion_payload.get("blocking_reasons"),
                "warnings": promotion_payload.get("warnings"),
            },
            "evaluation_summary": {
                "profile": evaluation_manifest_payload.get("profile"),
                "config": evaluation_manifest_payload.get("config"),
                "data_config": evaluation_manifest_payload.get("data_config"),
                "feature_config": evaluation_manifest_payload.get("feature_config"),
                "stability_assessment": evaluation_manifest_payload.get("stability_assessment"),
            },
            "revision_summary": {
                "decision": revision_payload.get("decision"),
                "status": revision_payload.get("status"),
                "profile": revision_payload.get("profile"),
            },
        }
        write_json(output_path, payload)
        print(f"[public-benchmark-reference] saved: {artifact_display_path(output_path, workspace_root=ROOT)}", flush=True)
        return 0
    except KeyboardInterrupt:
        print("[public-benchmark-reference] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[public-benchmark-reference] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[public-benchmark-reference] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())