from __future__ import annotations

import argparse
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


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[wf-mitigation-shortlist {now}] {message}", flush=True)


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


def _value_or_none(value: Any) -> Any:
    if pd.isna(value):
        return None
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


def _build_family_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("variant_overview") if isinstance(payload.get("variant_overview"), list) else []
    candidates: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict) or bool(row.get("is_target_variant")):
            continue
        candidates.append(
            {
                "candidate_type": "portfolio_family_variant",
                "candidate_id": str(row.get("variant_label") or ""),
                "signature": str(row.get("signature_example") or row.get("variant_label") or ""),
                "strategy_kind": row.get("strategy_kind"),
                "blend_weight": row.get("blend_weight"),
                "min_prob": row.get("min_prob"),
                "fractional_kelly": row.get("fractional_kelly"),
                "max_fraction": row.get("max_fraction"),
                "top_k": row.get("top_k"),
                "min_expected_value": row.get("min_expected_value"),
                "occurrence_count": int(row.get("occurrence_count") or 0),
                "evidence_count": int(row.get("higher_bankroll_lower_bets_count") or 0),
                "best_count": int(row.get("best_family_candidate_count") or 0),
                "status_feasible_count": int(row.get("status_feasible_count") or 0),
                "status_min_bets_count": int(row.get("status_min_bets_count") or 0),
                "status_min_final_bankroll_count": int(row.get("status_min_final_bankroll_count") or 0),
                "delta_bets_median": _value_or_none((row.get("delta_bets_vs_target_summary") or {}).get("median")),
                "delta_final_bankroll_median": _value_or_none((row.get("delta_final_bankroll_vs_target_summary") or {}).get("median")),
                "delta_base_score_median": _value_or_none((row.get("delta_base_score_vs_target_summary") or {}).get("median")),
                "notes": "same-threshold family variant compared against the dominant blocked portfolio signature",
            }
        )
    return candidates


def _build_recovery_candidates(payload: dict[str, Any]) -> list[dict[str, Any]]:
    rows = payload.get("occurrences") if isinstance(payload.get("occurrences"), list) else []
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        signature = str(row.get("recovery_signature") or "")
        if not signature:
            continue
        entry = grouped.setdefault(
            signature,
            {
                "candidate_type": "recovery_signature",
                "candidate_id": signature,
                "signature": signature,
                "occurrence_count": 0,
                "recovery_threshold_values": [],
                "delta_bets_values": [],
                "delta_final_bankroll_values": [],
            },
        )
        entry["occurrence_count"] += 1
        if row.get("recovery_threshold") is not None:
            entry["recovery_threshold_values"].append(float(row.get("recovery_threshold")))
        if row.get("delta_bets_target_to_recovery") is not None:
            entry["delta_bets_values"].append(float(row.get("delta_bets_target_to_recovery")))
        if row.get("delta_bankroll_target_to_recovery") is not None:
            entry["delta_final_bankroll_values"].append(float(row.get("delta_bankroll_target_to_recovery")))

    candidates: list[dict[str, Any]] = []
    for signature, entry in grouped.items():
        parsed = _parse_signature(signature)
        candidates.append(
            {
                "candidate_type": entry["candidate_type"],
                "candidate_id": entry["candidate_id"],
                "signature": signature,
                "strategy_kind": parsed.get("strategy_kind"),
                "blend_weight": parsed.get("blend_weight"),
                "min_prob": parsed.get("min_prob"),
                "fractional_kelly": parsed.get("fractional_kelly"),
                "max_fraction": parsed.get("max_fraction"),
                "top_k": parsed.get("top_k"),
                "min_expected_value": parsed.get("min_expected_value"),
                "occurrence_count": int(entry["occurrence_count"]),
                "evidence_count": int(entry["occurrence_count"]),
                "best_count": 0,
                "status_feasible_count": int(entry["occurrence_count"]),
                "status_min_bets_count": 0,
                "status_min_final_bankroll_count": 0,
                "delta_bets_median": _summarize_numeric(entry["delta_bets_values"]).get("median"),
                "delta_final_bankroll_median": _summarize_numeric(entry["delta_final_bankroll_values"]).get("median"),
                "delta_base_score_median": None,
                "median_recovery_threshold": _summarize_numeric(entry["recovery_threshold_values"]).get("median"),
                "notes": "lower-threshold recovery signature observed in drilldown",
            }
        )
    return candidates


def _sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    delta_bankroll = row.get("delta_final_bankroll_median")
    delta_bets = row.get("delta_bets_median")
    return (
        0 if row.get("candidate_type") == "portfolio_family_variant" else 1,
        -int(row.get("evidence_count") or 0),
        -(float(delta_bankroll) if delta_bankroll is not None else float("-inf")),
        float(delta_bets) if delta_bets is not None else float("inf"),
        -int(row.get("best_count") or 0),
    )


def _build_shortlist(family_payload: dict[str, Any], drilldown_payload: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    candidates = _build_family_candidates(family_payload) + _build_recovery_candidates(drilldown_payload)
    candidates.sort(key=_sort_key)
    ranked_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(candidates, start=1):
        ranked = dict(row)
        ranked["rank"] = rank
        ranked_rows.append(ranked)

    report = {
        "target_signature": family_payload.get("target_signature") or drilldown_payload.get("signature"),
        "candidate_count": int(len(ranked_rows)),
        "ranked_candidates": ranked_rows,
    }
    summary_df = pd.DataFrame(ranked_rows)
    return report, summary_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family-report", default="artifacts/reports/wf_threshold_signature_family_compare_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--drilldown-report", default="artifacts/reports/wf_threshold_signature_drilldown_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_mitigation_shortlist.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_mitigation_shortlist.csv")
    args = parser.parse_args()

    try:
        family_report_path = _normalize_path(args.family_report)
        drilldown_report_path = _normalize_path(args.drilldown_report)
        output_path = _normalize_path(args.output)
        summary_csv_path = _normalize_path(args.summary_csv)
        artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
        artifact_ensure_output_file_path(summary_csv_path, label="summary csv", workspace_root=ROOT)

        progress = ProgressBar(total=3, prefix="[wf-mitigation-shortlist]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="loading family and drilldown reports")
        family_payload = _load_json(family_report_path)
        drilldown_payload = _load_json(drilldown_report_path)
        with Heartbeat("[wf-mitigation-shortlist]", "building shortlist", logger=log_progress):
            report, summary_df = _build_shortlist(family_payload, drilldown_payload)
        progress.update(message=f"shortlist built candidates={report.get('candidate_count')}")
        with Heartbeat("[wf-mitigation-shortlist]", "writing shortlist outputs", logger=log_progress):
            write_json(output_path, report)
            write_csv_file(summary_csv_path, summary_df, index=False)

        print(f"saved mitigation shortlist to {output_path.relative_to(ROOT)}")
        print(f"saved mitigation shortlist table to {summary_csv_path.relative_to(ROOT)}")
        print(f"target_signature={report['target_signature']}")
        print(f"candidate_count={report['candidate_count']}")
        progress.complete(message="mitigation shortlist completed")
        return 0
    except KeyboardInterrupt:
        print("[wf-mitigation-shortlist] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[wf-mitigation-shortlist] failed: {error}")
        return 1
    except Exception as error:
        print(f"[wf-mitigation-shortlist] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())