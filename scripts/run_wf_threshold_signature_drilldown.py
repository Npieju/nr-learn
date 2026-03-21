from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.evaluation.policy import PolicyConstraints, apply_selection_mode, evaluate_candidate_gate


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[wf-signature-drilldown {now}] {message}", flush=True)


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


def _candidate_signature(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in SIGNATURE_FIELDS:
        parts.append(f"{field}={_value_or_none(row.get(field))}")
    return "|".join(parts)


def _snapshot(row: dict[str, Any] | None, *, valid_races: int | None = None) -> dict[str, Any]:
    if row is None:
        return {
            "signature": None,
            "strategy_kind": None,
            "bets": None,
            "bet_ratio": None,
            "roi": None,
            "final_bankroll": None,
            "max_drawdown": None,
            "base_score": None,
            "selection_score": None,
            "blend_weight": None,
            "min_edge": None,
            "min_prob": None,
            "fractional_kelly": None,
            "max_fraction": None,
            "odds_min": None,
            "odds_max": None,
            "top_k": None,
            "min_expected_value": None,
        }
    bets = int(row.get("bets") or 0)
    return {
        "signature": _candidate_signature(row),
        "strategy_kind": _value_or_none(row.get("strategy_kind")),
        "bets": bets,
        "bet_ratio": float(bets / valid_races) if valid_races and valid_races > 0 else None,
        "roi": _value_or_none(row.get("roi")),
        "final_bankroll": _value_or_none(row.get("final_bankroll")),
        "max_drawdown": _value_or_none(row.get("max_drawdown")),
        "base_score": _value_or_none(row.get("base_score")),
        "selection_score": _value_or_none(row.get("selection_score")),
        "blend_weight": _value_or_none(row.get("blend_weight")),
        "min_edge": _value_or_none(row.get("min_edge")),
        "min_prob": _value_or_none(row.get("min_prob")),
        "fractional_kelly": _value_or_none(row.get("fractional_kelly")),
        "max_fraction": _value_or_none(row.get("max_fraction")),
        "odds_min": _value_or_none(row.get("odds_min")),
        "odds_max": _value_or_none(row.get("odds_max")),
        "top_k": _value_or_none(row.get("top_k")),
        "min_expected_value": _value_or_none(row.get("min_expected_value")),
    }


def _snapshot_from_compare_row(row: dict[str, Any] | None, *, prefix: str, valid_races: int | None = None) -> dict[str, Any]:
    if row is None:
        return _snapshot(None, valid_races=valid_races)
    bets = _value_or_none(row.get(f"{prefix}_bets"))
    return {
        "signature": _value_or_none(row.get(f"{prefix}_signature")),
        "strategy_kind": _value_or_none(row.get(f"{prefix}_strategy")),
        "bets": bets,
        "bet_ratio": float(bets / valid_races) if bets is not None and valid_races and valid_races > 0 else None,
        "roi": _value_or_none(row.get(f"{prefix}_roi")),
        "final_bankroll": _value_or_none(row.get(f"{prefix}_final_bankroll")),
        "max_drawdown": _value_or_none(row.get(f"{prefix}_max_drawdown")),
        "base_score": _value_or_none(row.get(f"{prefix}_base_score")),
        "selection_score": _value_or_none(row.get(f"{prefix}_selection_score")),
        "blend_weight": _value_or_none(row.get(f"{prefix}_blend_weight")),
        "min_edge": _value_or_none(row.get(f"{prefix}_min_edge")),
        "min_prob": _value_or_none(row.get(f"{prefix}_min_prob")),
        "fractional_kelly": _value_or_none(row.get(f"{prefix}_fractional_kelly")),
        "max_fraction": _value_or_none(row.get(f"{prefix}_max_fraction")),
        "odds_min": _value_or_none(row.get(f"{prefix}_odds_min")),
        "odds_max": _value_or_none(row.get(f"{prefix}_odds_max")),
        "top_k": _value_or_none(row.get(f"{prefix}_top_k")),
        "min_expected_value": _value_or_none(row.get(f"{prefix}_min_expected_value")),
    }


def _resolve_signature(compare_payload: dict[str, Any], requested: str | None) -> str:
    if requested:
        return requested
    fold_rows = compare_payload.get("fold_snapshots") if isinstance(compare_payload.get("fold_snapshots"), list) else []
    counts: Counter[str] = Counter(
        str(row.get("best_over_bet_floor_signature") or "")
        for row in fold_rows
        if isinstance(row, dict) and str(row.get("status") or "") in {"min_bets", "min_final_bankroll", "max_drawdown", "other"}
        and str(row.get("best_over_bet_floor_signature") or "")
    )
    if not counts:
        raise ValueError("no blocked signature found in compare report")
    return counts.most_common(1)[0][0]


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
        row["min_bets_required"] = int(gate_result.get("min_bets_required") or constraints.min_bets_required(valid_races))
        row["signature"] = _candidate_signature(row)
        rows.append(row)
    return rows


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


def _flatten(prefix: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in snapshot.items()}


def _build_drilldown(compare_payload: dict[str, Any], signature: str) -> tuple[dict[str, Any], pd.DataFrame]:
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
    all_fold_rows = [row for row in fold_rows if isinstance(row, dict)]
    occurrence_rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    threshold_counts: Counter[int] = Counter()
    best_feasible_signature_counts: Counter[str] = Counter()
    same_strategy_feasible_signature_counts: Counter[str] = Counter()
    recovery_signature_counts: Counter[str] = Counter()
    target_bet_ratio_values: list[float] = []
    target_bankroll_values: list[float] = []
    target_bankroll_gap_values: list[float] = []
    delta_bankroll_to_feasible_values: list[float] = []
    delta_bets_to_feasible_values: list[float] = []

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
        feasible_candidates = [row for row in scored_rows if bool(row.get("is_feasible"))]
        best_feasible = _pick_best(feasible_candidates)

        strategy_kind = str(target_candidate.get("strategy_kind") or occurrence.get("best_over_bet_floor_strategy") or "") if target_candidate else str(occurrence.get("best_over_bet_floor_strategy") or "")
        same_strategy_alternatives = [
            row for row in scored_rows
            if str(row.get("strategy_kind") or "") == strategy_kind and str(row.get("signature") or "") != signature
        ]
        same_strategy_feasible = [row for row in same_strategy_alternatives if bool(row.get("is_feasible"))]
        best_same_strategy_feasible = _pick_best(same_strategy_feasible)
        best_same_strategy_any = _pick_best(same_strategy_alternatives)
        label = str(occurrence.get("label") or "unknown")

        target_snapshot = _snapshot(target_candidate, valid_races=valid_races)
        feasible_snapshot = _snapshot(best_feasible, valid_races=valid_races)
        same_strategy_feasible_snapshot = _snapshot(best_same_strategy_feasible, valid_races=valid_races)
        same_strategy_any_snapshot = _snapshot(best_same_strategy_any, valid_races=valid_races)

        recovery_rows = [
            row for row in all_fold_rows
            if str(row.get("label") or "") == label
            and int(row.get("fold") or 0) == fold
            and int(row.get("min_bets_abs") or 0) < threshold
            and str(row.get("status") or "") == "feasible"
            and row.get("best_feasible_signature")
        ]
        recovery_row = None
        if recovery_rows:
            recovery_row = max(recovery_rows, key=lambda row: int(row.get("min_bets_abs") or 0))
        recovery_snapshot = _snapshot_from_compare_row(recovery_row, prefix="best_feasible", valid_races=valid_races)

        status = str(occurrence.get("status") or "other")
        status_counts[status] += 1
        label_counts[label] += 1
        threshold_counts[threshold] += 1

        if target_snapshot.get("bet_ratio") is not None:
            target_bet_ratio_values.append(float(target_snapshot["bet_ratio"]))
        if target_snapshot.get("final_bankroll") is not None:
            target_bankroll_values.append(float(target_snapshot["final_bankroll"]))
        if occurrence.get("bankroll_gap_to_min") is not None:
            target_bankroll_gap_values.append(float(occurrence.get("bankroll_gap_to_min")))
        if feasible_snapshot.get("signature"):
            best_feasible_signature_counts[str(feasible_snapshot["signature"])] += 1
            if feasible_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None:
                delta_bankroll_to_feasible_values.append(float(feasible_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"]))
            if feasible_snapshot.get("bets") is not None and target_snapshot.get("bets") is not None:
                delta_bets_to_feasible_values.append(float(feasible_snapshot["bets"]) - float(target_snapshot["bets"]))
        if same_strategy_feasible_snapshot.get("signature"):
            same_strategy_feasible_signature_counts[str(same_strategy_feasible_snapshot["signature"])] += 1
        if recovery_snapshot.get("signature"):
            recovery_signature_counts[str(recovery_snapshot["signature"])] += 1

        occurrence_rows.append(
            {
                "label": label,
                "report": str(occurrence.get("report") or ""),
                "min_bets_abs": threshold,
                "fold": fold,
                "status": status,
                "valid_races": valid_races,
                "min_bets_required": int(occurrence.get("min_bets_required") or constraints.min_bets_required(valid_races)),
                "bankroll_gap_to_min": occurrence.get("bankroll_gap_to_min"),
                "min_bets_gap": occurrence.get("min_bets_gap"),
                **_flatten("target", target_snapshot),
                **_flatten("best_feasible", feasible_snapshot),
                **_flatten("best_same_strategy_feasible", same_strategy_feasible_snapshot),
                **_flatten("best_same_strategy_any", same_strategy_any_snapshot),
                "recovery_threshold": int(recovery_row.get("min_bets_abs") or 0) if recovery_row is not None else None,
                **_flatten("recovery", recovery_snapshot),
                "delta_bankroll_target_to_best_feasible": (
                    float(feasible_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"])
                    if feasible_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None
                    else None
                ),
                "delta_bets_target_to_best_feasible": (
                    float(feasible_snapshot["bets"]) - float(target_snapshot["bets"])
                    if feasible_snapshot.get("bets") is not None and target_snapshot.get("bets") is not None
                    else None
                ),
                "delta_bankroll_target_to_same_strategy_feasible": (
                    float(same_strategy_feasible_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"])
                    if same_strategy_feasible_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None
                    else None
                ),
                "delta_bets_target_to_same_strategy_feasible": (
                    float(same_strategy_feasible_snapshot["bets"]) - float(target_snapshot["bets"])
                    if same_strategy_feasible_snapshot.get("bets") is not None and target_snapshot.get("bets") is not None
                    else None
                ),
                "delta_bankroll_target_to_recovery": (
                    float(recovery_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"])
                    if recovery_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None
                    else None
                ),
                "delta_bets_target_to_recovery": (
                    float(recovery_snapshot["bets"]) - float(target_snapshot["bets"])
                    if recovery_snapshot.get("bets") is not None and target_snapshot.get("bets") is not None
                    else None
                ),
            }
        )

    report = {
        "signature": signature,
        "blocked_occurrence_count": int(len(occurrence_rows)),
        "status_counts": dict(status_counts),
        "label_counts": dict(label_counts),
        "threshold_counts": {str(key): int(value) for key, value in threshold_counts.items()},
        "target_bet_ratio_summary": _summarize_numeric(target_bet_ratio_values),
        "target_final_bankroll_summary": _summarize_numeric(target_bankroll_values),
        "target_bankroll_gap_summary": _summarize_numeric(target_bankroll_gap_values),
        "delta_bankroll_to_best_feasible_summary": _summarize_numeric(delta_bankroll_to_feasible_values),
        "delta_bets_to_best_feasible_summary": _summarize_numeric(delta_bets_to_feasible_values),
        "best_feasible_signature_counts": dict(best_feasible_signature_counts),
        "best_same_strategy_feasible_signature_counts": dict(same_strategy_feasible_signature_counts),
        "recovery_signature_counts": dict(recovery_signature_counts),
        "occurrences": occurrence_rows,
    }
    return report, pd.DataFrame(occurrence_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-report", default="artifacts/reports/wf_threshold_compare_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--signature", default=None)
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_signature_drilldown.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_signature_drilldown.csv")
    args = parser.parse_args()

    try:
        compare_report_path = _normalize_path(args.compare_report)
        output_path = _normalize_path(args.output)
        summary_csv_path = _normalize_path(args.summary_csv)
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        artifact_ensure_output_file_path(summary_csv_path, label="summary csv", workspace_root=ROOT)

        progress = ProgressBar(total=3, prefix="[wf-signature-drilldown]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="loading compare report")
        compare_payload = _load_compare_report(compare_report_path)
        signature = _resolve_signature(compare_payload, args.signature)
        with Heartbeat("[wf-signature-drilldown]", "building drilldown report", logger=log_progress):
            report, summary_df = _build_drilldown(compare_payload, signature)
        progress.update(message=f"drilldown built occurrences={report.get('blocked_occurrence_count')}")
        with Heartbeat("[wf-signature-drilldown]", "writing drilldown outputs", logger=log_progress):
            write_json(output_path, report)
            write_csv_file(summary_csv_path, summary_df, index=False)

        print(f"saved threshold signature drilldown to {output_path.relative_to(ROOT)}")
        print(f"saved threshold signature drilldown table to {summary_csv_path.relative_to(ROOT)}")
        print(f"signature={signature}")
        print(f"blocked_occurrence_count={report['blocked_occurrence_count']}")
        print(f"status_counts={report['status_counts']}")
        print(f"best_feasible_signature_counts={report['best_feasible_signature_counts']}")
        progress.complete(message="signature drilldown completed")
        return 0
    except KeyboardInterrupt:
        print("[wf-signature-drilldown] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[wf-signature-drilldown] failed: {error}")
        return 1
    except Exception as error:
        print(f"[wf-signature-drilldown] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())