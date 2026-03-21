from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-compare-aggregate {now}] {message}", flush=True)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else (Path.cwd() / path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _safe_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _summary_case_map(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    for case in cases:
        if not isinstance(case, dict):
            continue
        date_value = str(case.get("date") or "").strip()
        if date_value:
            mapping[date_value] = case
    return mapping


def _backtest_path(case: dict[str, Any]) -> Path:
    archived = case.get("archived_artifacts") if isinstance(case.get("archived_artifacts"), dict) else {}
    path_value = archived.get("backtest_json") or case.get("backtest_file")
    if not path_value:
        raise ValueError(f"Missing backtest path for date={case.get('date')}")
    return resolve_path(str(path_value))


def _compute_pure_final_bankroll(summary_path: Path) -> float | None:
    summary_payload = load_json(summary_path)
    bankroll = 1.0
    found = False
    for _, case in sorted(_summary_case_map(summary_payload).items()):
        backtest_payload = load_json(_backtest_path(case))
        multiplier = _safe_float(backtest_payload.get("policy_final_bankroll"))
        if multiplier is None:
            multiplier = 1.0
        bankroll = bankroll * multiplier
        found = True
    return float(bankroll) if found else None


def _extract_row_from_dashboard_summary(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    compare = payload.get("compare") if isinstance(payload.get("compare"), dict) else {}
    bankroll = payload.get("bankroll") if isinstance(payload.get("bankroll"), dict) else {}
    best_result = bankroll.get("best_result") if isinstance(bankroll.get("best_result"), dict) else {}
    baseline_only = bankroll.get("baseline_only") if isinstance(bankroll.get("baseline_only"), dict) else {}
    return {
        "window_label": payload.get("window_label"),
        "source_file": display_path(path),
        "source_kind": "dashboard_summary",
        "left_profile": (payload.get("left") or {}).get("profile"),
        "right_profile": (payload.get("right") or {}).get("profile"),
        "date_count": _safe_int(payload.get("date_count")),
        "shared_ok_date_count": len(compare.get("shared_ok_dates") or []),
        "left_total_policy_bets": _safe_int(compare.get("left_total_policy_bets")),
        "right_total_policy_bets": _safe_int(compare.get("right_total_policy_bets")),
        "bets_delta_right_minus_left": (
            _safe_int(compare.get("right_total_policy_bets")) - _safe_int(compare.get("left_total_policy_bets"))
            if _safe_int(compare.get("right_total_policy_bets")) is not None and _safe_int(compare.get("left_total_policy_bets")) is not None
            else None
        ),
        "left_total_policy_net": _safe_float(compare.get("left_total_policy_net")),
        "right_total_policy_net": _safe_float(compare.get("right_total_policy_net")),
        "net_delta_right_minus_left": _safe_float(compare.get("right_minus_left_total_policy_net")),
        "left_pure_final_bankroll": _safe_float((payload.get("left") or {}).get("pure_path_final_bankroll")),
        "right_pure_final_bankroll": _safe_float((payload.get("right") or {}).get("pure_path_final_bankroll")),
        "pure_bankroll_delta_right_minus_left": _safe_float(bankroll.get("right_minus_left_pure_final_bankroll")),
        "best_sweep_final_bankroll": _safe_float(best_result.get("final_bankroll")),
        "best_sweep_total_bets": _safe_int(best_result.get("total_bets")),
        "best_sweep_floors": best_result.get("floors"),
        "baseline_only_final_bankroll": _safe_float(baseline_only.get("final_bankroll")),
        "baseline_only_total_bets": _safe_int(baseline_only.get("total_bets")),
    }


def _extract_row_from_compare_manifest(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    outputs = payload.get("outputs") if isinstance(payload.get("outputs"), dict) else {}
    compare_payload = load_json(resolve_path(outputs.get("compare_json")))
    sweep_payload = load_json(resolve_path(outputs.get("bankroll_sweep_json"))) if outputs.get("bankroll_sweep_json") else {}
    left_summary_path = resolve_path((payload.get("left") or {}).get("summary_file"))
    right_summary_path = resolve_path((payload.get("right") or {}).get("summary_file"))
    aggregates = compare_payload.get("comparison", {}).get("shared_ok_aggregates", {}) if isinstance(compare_payload.get("comparison"), dict) else {}
    best_result = sweep_payload.get("best_result") if isinstance(sweep_payload.get("best_result"), dict) else {}
    baseline_only = sweep_payload.get("baseline_only") if isinstance(sweep_payload.get("baseline_only"), dict) else {}
    left_pure_final_bankroll = _compute_pure_final_bankroll(left_summary_path)
    right_pure_final_bankroll = _compute_pure_final_bankroll(right_summary_path)
    return {
        "window_label": payload.get("window_label"),
        "source_file": display_path(path),
        "source_kind": "compare_manifest",
        "left_profile": (payload.get("left") or {}).get("profile"),
        "right_profile": (payload.get("right") or {}).get("profile"),
        "date_count": len(payload.get("dates") or []),
        "shared_ok_date_count": len(compare_payload.get("comparison", {}).get("shared_ok_dates") or []),
        "left_total_policy_bets": _safe_int(aggregates.get("left_total_policy_bets")),
        "right_total_policy_bets": _safe_int(aggregates.get("right_total_policy_bets")),
        "bets_delta_right_minus_left": (
            _safe_int(aggregates.get("right_total_policy_bets")) - _safe_int(aggregates.get("left_total_policy_bets"))
            if _safe_int(aggregates.get("right_total_policy_bets")) is not None and _safe_int(aggregates.get("left_total_policy_bets")) is not None
            else None
        ),
        "left_total_policy_net": _safe_float(aggregates.get("left_total_policy_net")),
        "right_total_policy_net": _safe_float(aggregates.get("right_total_policy_net")),
        "net_delta_right_minus_left": (
            _safe_float(aggregates.get("right_total_policy_net")) - _safe_float(aggregates.get("left_total_policy_net"))
            if _safe_float(aggregates.get("right_total_policy_net")) is not None and _safe_float(aggregates.get("left_total_policy_net")) is not None
            else None
        ),
        "left_pure_final_bankroll": left_pure_final_bankroll,
        "right_pure_final_bankroll": right_pure_final_bankroll,
        "pure_bankroll_delta_right_minus_left": (
            float(right_pure_final_bankroll) - float(left_pure_final_bankroll)
            if left_pure_final_bankroll is not None and right_pure_final_bankroll is not None
            else None
        ),
        "best_sweep_final_bankroll": _safe_float(best_result.get("final_bankroll")),
        "best_sweep_total_bets": _safe_int(best_result.get("total_bets")),
        "best_sweep_floors": best_result.get("floors"),
        "baseline_only_final_bankroll": _safe_float(baseline_only.get("final_bankroll")),
        "baseline_only_total_bets": _safe_int(baseline_only.get("total_bets")),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--dashboard-summaries", nargs="+")
    input_group.add_argument("--compare-manifests", nargs="+")
    parser.add_argument("--output-summary", default="artifacts/reports/dashboard/serving_compare_aggregate_summary.json")
    parser.add_argument("--output-chart", default="artifacts/reports/dashboard/serving_compare_aggregate_summary.png")
    parser.add_argument("--output-csv", default="artifacts/reports/dashboard/serving_compare_aggregate_summary.csv")
    args = parser.parse_args()
    progress = ProgressBar(total=4, prefix="[serving-compare-aggregate]", logger=log_progress, min_interval_sec=0.0)

    try:
        if args.dashboard_summaries:
            input_paths = [resolve_path(path) for path in args.dashboard_summaries]
            input_kind = "dashboard summaries"
        else:
            input_paths = [resolve_path(path) for path in args.compare_manifests]
            input_kind = "compare manifests"
        output_summary = resolve_path(args.output_summary)
        output_chart = resolve_path(args.output_chart)
        output_csv = resolve_path(args.output_csv)

        progress.start(message=f"loading {input_kind} count={len(input_paths)}")
        with Heartbeat("[serving-compare-aggregate]", "loading summaries", logger=log_progress):
            payloads = [load_json(path) for path in input_paths]
        progress.update(message=f"{input_kind} loaded")

        rows: list[dict[str, Any]] = []
        for path, payload in zip(input_paths, payloads):
            row = _extract_row_from_dashboard_summary(path, payload) if args.dashboard_summaries else _extract_row_from_compare_manifest(path, payload)
            rows.append(row)

        frame = pd.DataFrame(rows).sort_values(["window_label", "date_count"], ascending=[True, True]).reset_index(drop=True)
        aggregate_payload = {
            "input_files": [display_path(path) for path in input_paths],
            "input_kind": "dashboard_summaries" if args.dashboard_summaries else "compare_manifests",
            "window_count": int(len(frame)),
            "rows": frame.to_dict(orient="records"),
            "summary": {
                "windows_with_positive_net_delta": [
                    str(row["window_label"]) for row in frame.to_dict(orient="records") if _safe_float(row.get("net_delta_right_minus_left")) is not None and float(row["net_delta_right_minus_left"]) > 0.0
                ],
                "windows_with_positive_bankroll_delta": [
                    str(row["window_label"]) for row in frame.to_dict(orient="records") if _safe_float(row.get("pure_bankroll_delta_right_minus_left")) is not None and float(row["pure_bankroll_delta_right_minus_left"]) > 0.0
                ],
                "mean_net_delta_right_minus_left": float(frame["net_delta_right_minus_left"].dropna().mean()) if "net_delta_right_minus_left" in frame.columns and not frame["net_delta_right_minus_left"].dropna().empty else None,
                "mean_pure_bankroll_delta_right_minus_left": float(frame["pure_bankroll_delta_right_minus_left"].dropna().mean()) if "pure_bankroll_delta_right_minus_left" in frame.columns and not frame["pure_bankroll_delta_right_minus_left"].dropna().empty else None,
            },
        }
        progress.update(message=f"aggregate assembled windows={len(frame)}")

        with Heartbeat("[serving-compare-aggregate]", "writing aggregate outputs", logger=log_progress):
            output_summary.parent.mkdir(parents=True, exist_ok=True)
            with output_summary.open("w", encoding="utf-8") as file:
                json.dump(aggregate_payload, file, ensure_ascii=False, indent=2)

            frame.to_csv(output_csv, index=False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
            x_values = list(range(len(frame)))
            x_labels = frame["window_label"].astype(str).tolist()

            axes[0].bar(x_values, frame["net_delta_right_minus_left"].fillna(0.0), color="#b45309")
            axes[0].axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
            axes[0].set_title("Net Delta Right Minus Left")
            axes[0].set_xticks(x_values)
            axes[0].set_xticklabels(x_labels, rotation=30, ha="right")
            axes[0].set_ylabel("policy net delta")

            axes[1].scatter(
                frame["net_delta_right_minus_left"].fillna(0.0),
                frame["pure_bankroll_delta_right_minus_left"].fillna(0.0),
                s=90,
                color="#1d4ed8",
            )
            for _, row in frame.iterrows():
                axes[1].annotate(
                    str(row["window_label"]),
                    (float(row["net_delta_right_minus_left"] or 0.0), float(row["pure_bankroll_delta_right_minus_left"] or 0.0)),
                    textcoords="offset points",
                    xytext=(5, 5),
                )
            axes[1].axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
            axes[1].axvline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
            axes[1].set_title("Net vs Bankroll Delta")
            axes[1].set_xlabel("net delta right minus left")
            axes[1].set_ylabel("pure bankroll delta right minus left")

            plt.tight_layout()
            fig.savefig(output_chart, dpi=140)
            plt.close(fig)
        progress.complete(message=f"aggregate outputs saved windows={len(frame)}")

        print(f"[serving-compare-aggregate] summary saved: {display_path(output_summary)}")
        print(f"[serving-compare-aggregate] chart saved: {display_path(output_chart)}")
        print(f"[serving-compare-aggregate] csv saved: {display_path(output_csv)}")
        print(f"[serving-compare-aggregate] metrics: {aggregate_payload['summary']}")
        return 0
    except KeyboardInterrupt:
        print("[serving-compare-aggregate] interrupted by user")
        return 130
    except Exception as error:
        print(f"[serving-compare-aggregate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())