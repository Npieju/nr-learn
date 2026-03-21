from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json


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


def _load_compare(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"threshold compare payload is not a JSON object: {path}")
    return payload


def _normalize_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _build_signature_report(payload: dict[str, Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    fold_rows = payload.get("fold_snapshots") if isinstance(payload.get("fold_snapshots"), list) else []
    blocked_rows = [
        row for row in fold_rows
        if isinstance(row, dict) and str(row.get("status") or "") in {"min_bets", "min_final_bankroll", "max_drawdown", "other"}
    ]

    snapshot_rows: list[dict[str, Any]] = []
    overall_signature_counts: Counter[str] = Counter()
    overall_status_by_signature: dict[str, Counter[str]] = defaultdict(Counter)
    window_counts_by_signature: dict[str, Counter[str]] = defaultdict(Counter)

    grouped: dict[tuple[str, int, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in blocked_rows:
        signature = str(row.get("best_over_bet_floor_signature") or "")
        if not signature:
            continue
        label = str(row.get("label") or "unknown")
        threshold = int(row.get("min_bets_abs") or 0)
        status = str(row.get("status") or "other")
        grouped[(label, threshold, status, signature)].append(row)
        overall_signature_counts[signature] += 1
        overall_status_by_signature[signature][status] += 1
        window_counts_by_signature[signature][label] += 1

    for (label, threshold, status, signature), rows in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1], item[0][2], -len(item[1]))):
        folds = sorted(int(row.get("fold") or 0) for row in rows)
        bets_values = [float(row.get("best_over_bet_floor_bets")) for row in rows if row.get("best_over_bet_floor_bets") is not None]
        bankroll_values = [float(row.get("best_over_bet_floor_final_bankroll")) for row in rows if row.get("best_over_bet_floor_final_bankroll") is not None]
        bankroll_gap_values = [float(row.get("bankroll_gap_to_min")) for row in rows if row.get("bankroll_gap_to_min") is not None]
        min_bets_gap_values = [float(row.get("min_bets_gap")) for row in rows if row.get("min_bets_gap") is not None]

        sample = rows[0]
        snapshot_rows.append(
            {
                "label": label,
                "min_bets_abs": threshold,
                "status": status,
                "signature": signature,
                "count": int(len(rows)),
                "folds": ",".join(str(fold) for fold in folds),
                "strategy": sample.get("best_over_bet_floor_strategy"),
                "blend_weight": sample.get("best_over_bet_floor_blend_weight"),
                "min_edge": sample.get("best_over_bet_floor_min_edge"),
                "min_prob": sample.get("best_over_bet_floor_min_prob"),
                "fractional_kelly": sample.get("best_over_bet_floor_fractional_kelly"),
                "max_fraction": sample.get("best_over_bet_floor_max_fraction"),
                "odds_min": sample.get("best_over_bet_floor_odds_min"),
                "odds_max": sample.get("best_over_bet_floor_odds_max"),
                "top_k": sample.get("best_over_bet_floor_top_k"),
                "min_expected_value": sample.get("best_over_bet_floor_min_expected_value"),
                "bets_min": _summarize_numeric(bets_values).get("min"),
                "bets_median": _summarize_numeric(bets_values).get("median"),
                "bets_max": _summarize_numeric(bets_values).get("max"),
                "final_bankroll_min": _summarize_numeric(bankroll_values).get("min"),
                "final_bankroll_median": _summarize_numeric(bankroll_values).get("median"),
                "final_bankroll_max": _summarize_numeric(bankroll_values).get("max"),
                "bankroll_gap_min": _summarize_numeric(bankroll_gap_values).get("min"),
                "bankroll_gap_median": _summarize_numeric(bankroll_gap_values).get("median"),
                "bankroll_gap_max": _summarize_numeric(bankroll_gap_values).get("max"),
                "min_bets_gap_min": _summarize_numeric(min_bets_gap_values).get("min"),
                "min_bets_gap_median": _summarize_numeric(min_bets_gap_values).get("median"),
                "min_bets_gap_max": _summarize_numeric(min_bets_gap_values).get("max"),
            }
        )

    overall_rows: list[dict[str, Any]] = []
    for signature, count in overall_signature_counts.most_common():
        samples = [row for row in blocked_rows if row.get("best_over_bet_floor_signature") == signature]
        bankroll_gap_values = [float(row.get("bankroll_gap_to_min")) for row in samples if row.get("bankroll_gap_to_min") is not None]
        min_bets_gap_values = [float(row.get("min_bets_gap")) for row in samples if row.get("min_bets_gap") is not None]
        final_bankroll_values = [float(row.get("best_over_bet_floor_final_bankroll")) for row in samples if row.get("best_over_bet_floor_final_bankroll") is not None]
        first = samples[0]
        overall_rows.append(
            {
                "signature": signature,
                "count": int(count),
                "status_counts": dict(overall_status_by_signature[signature]),
                "window_counts": dict(window_counts_by_signature[signature]),
                "strategy": first.get("best_over_bet_floor_strategy"),
                "blend_weight": first.get("best_over_bet_floor_blend_weight"),
                "min_prob": first.get("best_over_bet_floor_min_prob"),
                "top_k": first.get("best_over_bet_floor_top_k"),
                "min_expected_value": first.get("best_over_bet_floor_min_expected_value"),
                "final_bankroll_summary": _summarize_numeric(final_bankroll_values),
                "bankroll_gap_summary": _summarize_numeric(bankroll_gap_values),
                "min_bets_gap_summary": _summarize_numeric(min_bets_gap_values),
            }
        )

    report = {
        "blocked_signature_count": int(len(overall_rows)),
        "blocked_signature_overview": overall_rows,
        "signature_snapshots": snapshot_rows,
    }
    return report, pd.DataFrame(snapshot_rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compare-report", default="artifacts/reports/wf_threshold_compare_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--output", default="artifacts/reports/wf_threshold_signature_report.json")
    parser.add_argument("--summary-csv", default="artifacts/reports/wf_threshold_signature_report.csv")
    args = parser.parse_args()

    compare_report_path = _normalize_path(args.compare_report)
    output_path = _normalize_path(args.output)
    summary_csv_path = _normalize_path(args.summary_csv)

    payload = _load_compare(compare_report_path)
    report, summary_df = _build_signature_report(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_path, report)
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"saved threshold signature report to {output_path.relative_to(ROOT)}")
    print(f"saved threshold signature table to {summary_csv_path.relative_to(ROOT)}")
    for row in report.get("blocked_signature_overview") or []:
        print(
            f"count={row['count']} strategy={row.get('strategy')} min_prob={row.get('min_prob')} "
            f"blend_weight={row.get('blend_weight')} top_k={row.get('top_k')} min_ev={row.get('min_expected_value')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())