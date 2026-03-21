from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _normalize_path(raw: str | Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    return (ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


_BACKTEST_CACHE: dict[Path, dict[str, Any]] = {}


def _parse_floats(raw: str) -> list[float]:
    values = []
    for part in raw.split(","):
        text = part.strip()
        if not text:
            continue
        values.append(float(text))
    if not values:
        raise ValueError("at least one threshold value is required")
    return values


def _case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    for case in cases:
        if not isinstance(case, dict):
            continue
        date_value = str(case.get("date", "")).strip()
        if date_value:
            mapping[date_value] = case
    return mapping


def _resolve_backtest_path(case: dict[str, Any]) -> Path:
    archived = case.get("archived_artifacts") if isinstance(case.get("archived_artifacts"), dict) else {}
    backtest_path = archived.get("backtest_json") or case.get("backtest_file")
    if not backtest_path:
        raise ValueError(f"backtest path missing for case: {case.get('date')}")
    return _normalize_path(str(backtest_path))


def _load_backtest_metrics(case: dict[str, Any]) -> dict[str, Any]:
    backtest_path = _resolve_backtest_path(case)
    cached = _BACKTEST_CACHE.get(backtest_path)
    if cached is not None:
        return cached
    payload = _load_json(backtest_path)
    _BACKTEST_CACHE[backtest_path] = payload
    return payload


def _daily_multiplier(case: dict[str, Any]) -> float:
    final_bankroll = _load_backtest_metrics(case).get("policy_final_bankroll")
    if final_bankroll is None:
        return 1.0
    return float(final_bankroll)


def _daily_bets(case: dict[str, Any]) -> int:
    return int(_load_backtest_metrics(case).get("policy_bets") or 0)


def _select_stage_index(bankroll: float, floors: list[float]) -> int:
    stage_index = 0
    for index, floor in enumerate(floors, start=1):
        if bankroll < float(floor):
            stage_index = index
    return stage_index


def _simulate_path(
    *,
    ordered_dates: list[str],
    labels: list[str],
    case_maps: list[dict[str, dict[str, Any]]],
    floors: list[float],
    initial_bankroll: float,
) -> dict[str, Any]:
    bankroll = float(initial_bankroll)
    rows: list[dict[str, Any]] = []
    stage_use_counts = {label: 0 for label in labels}

    for date_value in ordered_dates:
        stage_index = _select_stage_index(bankroll, floors)
        case = case_maps[stage_index][date_value]
        start_bankroll = bankroll
        multiplier = _daily_multiplier(case)
        bankroll = bankroll * multiplier
        label = labels[stage_index]
        stage_use_counts[label] += 1
        rows.append(
            {
                "date": date_value,
                "selected_label": label,
                "selected_stage_index": stage_index + 1,
                "start_bankroll": start_bankroll,
                "end_bankroll": bankroll,
                "daily_multiplier": multiplier,
                "policy_bets": _daily_bets(case),
                "policy_name": case.get("policy_name"),
                "policy_strategy_kind": _load_backtest_metrics(case).get("policy_strategy_kind"),
            }
        )

    return {
        "floors": floors,
        "final_bankroll": bankroll,
        "stage_use_counts": stage_use_counts,
        "total_bets": int(sum(int(row["policy_bets"]) for row in rows)),
        "path": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-files", nargs="+", required=True)
    parser.add_argument("--labels", default=None)
    parser.add_argument("--bankroll-floor-values", default="1.01,1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.92,0.90")
    parser.add_argument("--initial-bankroll", type=float, default=1.0)
    parser.add_argument(
        "--output-json",
        default="artifacts/reports/serving_stateful_bankroll_sweep.json",
    )
    parser.add_argument(
        "--output-csv",
        default="artifacts/reports/serving_stateful_bankroll_sweep.csv",
    )
    args = parser.parse_args()

    summary_paths = [_normalize_path(path) for path in args.summary_files]
    summaries = [_load_json(path) for path in summary_paths]
    labels = [label.strip() for label in str(args.labels).split(",")] if args.labels else [path.stem for path in summary_paths]
    if len(labels) != len(summary_paths):
        raise ValueError("labels count must match summary-files count")
    if len(summary_paths) < 2:
        raise ValueError("at least two summaries are required")

    case_maps = [_case_map(summary) for summary in summaries]
    shared_dates = sorted(set.intersection(*(set(case_map.keys()) for case_map in case_maps)))
    if not shared_dates:
        raise ValueError("no shared dates across summaries")

    floor_values = sorted(set(_parse_floats(args.bankroll_floor_values)), reverse=True)
    n_floors = len(summary_paths) - 1
    floor_candidates = [combo for combo in product(floor_values, repeat=n_floors) if list(combo) == sorted(combo, reverse=True)]
    if not floor_candidates:
        raise ValueError("no valid floor combinations generated")

    sweep_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None
    for floors_tuple in floor_candidates:
        floors = [float(value) for value in floors_tuple]
        result = _simulate_path(
            ordered_dates=shared_dates,
            labels=labels,
            case_maps=case_maps,
            floors=floors,
            initial_bankroll=float(args.initial_bankroll),
        )
        row = {
            "floors": floors,
            "final_bankroll": float(result["final_bankroll"]),
            "total_bets": int(result["total_bets"]),
        }
        for label, count in result["stage_use_counts"].items():
            row[f"use_count_{label}"] = int(count)
        sweep_rows.append(row)
        if best_result is None or float(result["final_bankroll"]) > float(best_result["final_bankroll"]):
            best_result = result

    baseline_result = _simulate_path(
        ordered_dates=shared_dates,
        labels=[labels[0]],
        case_maps=[case_maps[0]],
        floors=[],
        initial_bankroll=float(args.initial_bankroll),
    )

    payload = {
        "summary_files": [str(path.relative_to(ROOT)) for path in summary_paths],
        "labels": labels,
        "shared_dates": shared_dates,
        "initial_bankroll": float(args.initial_bankroll),
        "baseline_only": {
            "label": labels[0],
            "final_bankroll": float(baseline_result["final_bankroll"]),
            "total_bets": int(baseline_result["total_bets"]),
        },
        "best_result": best_result,
        "sweep": sweep_rows,
    }

    output_json = _normalize_path(args.output_json)
    output_csv = _normalize_path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with output_json.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    pd.DataFrame(sweep_rows).sort_values("final_bankroll", ascending=False).to_csv(output_csv, index=False)

    print(f"saved sweep json to {output_json.relative_to(ROOT)}")
    print(f"saved sweep csv to {output_csv.relative_to(ROOT)}")
    print(f"shared_dates={len(shared_dates)} best_final_bankroll={payload['best_result']['final_bankroll'] if payload['best_result'] else None}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())