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
    print(f"[wf-mitigation-focus {now}] {message}", flush=True)


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
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


def _candidate_signature(row: dict[str, Any]) -> str:
    parts: list[str] = []
    for field in SIGNATURE_FIELDS:
        parts.append(f"{field}={_value_or_none(row.get(field))}")
    return "|".join(parts)


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
        row["signature"] = _candidate_signature(row)
        rows.append(row)
    return rows


def _snapshot(row: dict[str, Any] | None) -> dict[str, Any]:
    if row is None:
        return {
            "signature": None,
            "strategy_kind": None,
            "blend_weight": None,
            "min_prob": None,
            "fractional_kelly": None,
            "max_fraction": None,
            "top_k": None,
            "min_expected_value": None,
            "bets": None,
            "roi": None,
            "final_bankroll": None,
            "base_score": None,
            "selection_score": None,
            "is_feasible": None,
        }
    return {
        "signature": row.get("signature"),
        "strategy_kind": row.get("strategy_kind"),
        "blend_weight": row.get("blend_weight"),
        "min_prob": row.get("min_prob"),
        "fractional_kelly": row.get("fractional_kelly"),
        "max_fraction": row.get("max_fraction"),
        "top_k": row.get("top_k"),
        "min_expected_value": row.get("min_expected_value"),
        "bets": row.get("bets"),
        "roi": row.get("roi"),
        "final_bankroll": row.get("final_bankroll"),
        "base_score": row.get("base_score"),
        "selection_score": row.get("selection_score"),
        "is_feasible": row.get("is_feasible"),
    }


def _flatten(prefix: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in snapshot.items()}


def _resolve_candidate_signatures(shortlist_payload: dict[str, Any], ranks: list[int]) -> list[str]:
    ranked_candidates = shortlist_payload.get("ranked_candidates") if isinstance(shortlist_payload.get("ranked_candidates"), list) else []
    signatures: list[str] = []
    for rank in ranks:
        matched = next((row for row in ranked_candidates if isinstance(row, dict) and int(row.get("rank") or 0) == rank), None)
        if matched is None:
            raise ValueError(f"candidate rank not found in shortlist: {rank}")
        signature = str(matched.get("signature") or "")
        if not signature:
            raise ValueError(f"candidate rank has empty signature: {rank}")
        signatures.append(signature)
    return signatures


def _recommendation(target: dict[str, Any], candidate_a: dict[str, Any], candidate_b: dict[str, Any]) -> str:
    a_bankroll = candidate_a.get("final_bankroll")
    b_bankroll = candidate_b.get("final_bankroll")
    a_bets = candidate_a.get("bets")
    b_bets = candidate_b.get("bets")
    if a_bankroll is None or b_bankroll is None or a_bets is None or b_bets is None:
        return "insufficient_data"
    if float(a_bankroll) >= float(b_bankroll) and float(a_bets) <= float(b_bets):
        return "candidate_a_dominates"
    if float(b_bankroll) >= float(a_bankroll) and float(b_bets) <= float(a_bets):
        return "candidate_b_dominates"
    if float(a_bankroll) > float(b_bankroll):
        return "candidate_a_higher_bankroll"
    if float(b_bankroll) > float(a_bankroll):
        return "candidate_b_higher_bankroll"
    return "tie_or_mixed"


def _build_focus(compare_payload: dict[str, Any], shortlist_payload: dict[str, Any], candidate_ranks: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    target_signature = str(compare_payload.get("fold_snapshots", [{}])[0].get("best_over_bet_floor_signature") if compare_payload.get("fold_snapshots") else shortlist_payload.get("target_signature") or "")
    signatures = _resolve_candidate_signatures(shortlist_payload, candidate_ranks)
    candidate_a_signature, candidate_b_signature = signatures[0], signatures[1]

    fold_rows = compare_payload.get("fold_snapshots") if isinstance(compare_payload.get("fold_snapshots"), list) else []
    target_rows = [
        row for row in fold_rows
        if isinstance(row, dict)
        and str(row.get("best_over_bet_floor_signature") or "") == target_signature
        and str(row.get("status") or "") in {"min_bets", "min_final_bankroll", "max_drawdown", "other"}
    ]

    threshold_report_cache: dict[str, dict[str, Any]] = {}
    source_cache: dict[str, tuple[dict[str, Any], pd.DataFrame]] = {}
    occurrence_rows: list[dict[str, Any]] = []
    recommendation_counts: Counter[str] = Counter()
    candidate_a_bankroll_gain_values: list[float] = []
    candidate_b_bankroll_gain_values: list[float] = []

    for occurrence in target_rows:
        threshold_report_path = _normalize_path(str(occurrence.get("report") or ""))
        threshold_report = _load_json(threshold_report_path)
        summary, detail_df = _load_source_inputs(threshold_report_path, threshold_report, source_cache)
        threshold = int(occurrence.get("min_bets_abs") or 0)
        fold = int(occurrence.get("fold") or 0)
        valid_races = int(occurrence.get("valid_races") or 0)
        constraints = _build_constraints(summary, threshold)
        scored_rows = _score_fold_candidates(detail_df.loc[detail_df["fold"] == fold].copy(), valid_races=valid_races, constraints=constraints)

        target = _pick_best([row for row in scored_rows if str(row.get("signature") or "") == target_signature])
        candidate_a = _pick_best([row for row in scored_rows if str(row.get("signature") or "") == candidate_a_signature])
        candidate_b = _pick_best([row for row in scored_rows if str(row.get("signature") or "") == candidate_b_signature])
        target_snapshot = _snapshot(target)
        candidate_a_snapshot = _snapshot(candidate_a)
        candidate_b_snapshot = _snapshot(candidate_b)
        recommendation = _recommendation(target_snapshot, candidate_a_snapshot, candidate_b_snapshot)
        recommendation_counts[recommendation] += 1

        if candidate_a_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None:
            candidate_a_bankroll_gain_values.append(float(candidate_a_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"]))
        if candidate_b_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None:
            candidate_b_bankroll_gain_values.append(float(candidate_b_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"]))

        occurrence_rows.append(
            {
                "label": occurrence.get("label"),
                "report": occurrence.get("report"),
                "min_bets_abs": threshold,
                "fold": fold,
                "status": occurrence.get("status"),
                "bankroll_gap_to_min": occurrence.get("bankroll_gap_to_min"),
                "min_bets_gap": occurrence.get("min_bets_gap"),
                **_flatten("target", target_snapshot),
                **_flatten("candidate_a", candidate_a_snapshot),
                **_flatten("candidate_b", candidate_b_snapshot),
                "delta_bankroll_target_to_candidate_a": (
                    float(candidate_a_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"])
                    if candidate_a_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None else None
                ),
                "delta_bankroll_target_to_candidate_b": (
                    float(candidate_b_snapshot["final_bankroll"]) - float(target_snapshot["final_bankroll"])
                    if candidate_b_snapshot.get("final_bankroll") is not None and target_snapshot.get("final_bankroll") is not None else None
                ),
                "delta_bets_target_to_candidate_a": (
                    float(candidate_a_snapshot["bets"]) - float(target_snapshot["bets"])
                    if candidate_a_snapshot.get("bets") is not None and target_snapshot.get("bets") is not None else None
                ),
                "delta_bets_target_to_candidate_b": (
                    float(candidate_b_snapshot["bets"]) - float(target_snapshot["bets"])
                    if candidate_b_snapshot.get("bets") is not None and target_snapshot.get("bets") is not None else None
                ),
                "delta_bankroll_candidate_a_to_candidate_b": (
                    float(candidate_b_snapshot["final_bankroll"]) - float(candidate_a_snapshot["final_bankroll"])
                    if candidate_a_snapshot.get("final_bankroll") is not None and candidate_b_snapshot.get("final_bankroll") is not None else None
                ),
                "delta_bets_candidate_a_to_candidate_b": (
                    float(candidate_b_snapshot["bets"]) - float(candidate_a_snapshot["bets"])
                    if candidate_a_snapshot.get("bets") is not None and candidate_b_snapshot.get("bets") is not None else None
                ),
                "recommendation": recommendation,
            }
        )

    report = {
        "target_signature": target_signature,
        "candidate_a_signature": candidate_a_signature,
        "candidate_b_signature": candidate_b_signature,
        "occurrence_count": int(len(occurrence_rows)),
        "recommendation_counts": dict(recommendation_counts),
        "candidate_a_bankroll_gain_summary": _summarize_numeric(candidate_a_bankroll_gain_values),
        "candidate_b_bankroll_gain_summary": _summarize_numeric(candidate_b_bankroll_gain_values),
        "occurrences": occurrence_rows,
    }
    return report, pd.DataFrame(occurrence_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-report", default="artifacts/reports/wf_threshold_compare_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--shortlist-report", default="artifacts/reports/wf_threshold_mitigation_shortlist_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--candidate-ranks", default="1,3")
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_mitigation_focus.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_mitigation_focus.csv")
    args = parser.parse_args()

    try:
        candidate_ranks = [int(value.strip()) for value in str(args.candidate_ranks).split(",") if value.strip()]
        if len(candidate_ranks) != 2:
            raise ValueError("--candidate-ranks must contain exactly two ranks")

        compare_payload = _load_json(_normalize_path(args.compare_report))
        shortlist_payload = _load_json(_normalize_path(args.shortlist_report))
        progress = ProgressBar(total=3, prefix="[wf-mitigation-focus]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="loading compare and shortlist reports")
        with Heartbeat("[wf-mitigation-focus]", "building focus comparison", logger=log_progress):
            report, summary_df = _build_focus(compare_payload, shortlist_payload, candidate_ranks)

        output_path = _normalize_path(args.output)
        summary_csv_path = _normalize_path(args.summary_csv)
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        artifact_ensure_output_file_path(summary_csv_path, label="summary csv", workspace_root=ROOT)
        progress.update(message=f"focus built occurrences={report.get('occurrence_count')}")
        with Heartbeat("[wf-mitigation-focus]", "writing focus outputs", logger=log_progress):
            write_json(output_path, report)
            write_csv_file(summary_csv_path, summary_df, index=False)

        print(f"saved mitigation focus to {output_path.relative_to(ROOT)}")
        print(f"saved mitigation focus table to {summary_csv_path.relative_to(ROOT)}")
        print(f"recommendation_counts={report['recommendation_counts']}")
        progress.complete(message="mitigation focus completed")
        return 0
    except KeyboardInterrupt:
        print("[wf-mitigation-focus] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[wf-mitigation-focus] failed: {error}")
        return 1
    except Exception as error:
        print(f"[wf-mitigation-focus] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())