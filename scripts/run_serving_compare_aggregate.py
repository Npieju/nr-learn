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

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import save_figure, write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-compare-aggregate {now}] {message}", flush=True)


def resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


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
        "source_file": artifact_display_path(path if path.is_absolute() else (Path.cwd() / path).resolve(), workspace_root=ROOT),
        "source_kind": "dashboard_summary",
        "manifest_status": payload.get("manifest_status"),
        "manifest_decision": payload.get("manifest_decision"),
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
        "source_file": artifact_display_path(path if path.is_absolute() else (Path.cwd() / path).resolve(), workspace_root=ROOT),
        "source_kind": "compare_manifest",
        "manifest_status": payload.get("status"),
        "manifest_decision": payload.get("decision"),
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


def _delta_direction(value: Any) -> str:
    numeric_value = _safe_float(value)
    if numeric_value is None:
        return "unknown"
    if numeric_value > 0.0:
        return "positive"
    if numeric_value < 0.0:
        return "negative"
    return "zero"


def _classify_window_tradeoff(row: dict[str, Any]) -> str:
    net_direction = _delta_direction(row.get("net_delta_right_minus_left"))
    bankroll_direction = _delta_direction(row.get("pure_bankroll_delta_right_minus_left"))
    if net_direction == "positive" and bankroll_direction == "positive":
        return "positive_net_positive_bankroll"
    if net_direction == "positive" and bankroll_direction == "negative":
        return "positive_net_negative_bankroll"
    if net_direction == "negative" and bankroll_direction == "positive":
        return "negative_net_positive_bankroll"
    if net_direction == "negative" and bankroll_direction == "negative":
        return "negative_net_negative_bankroll"
    if net_direction == "zero" and bankroll_direction == "zero":
        return "zero_net_zero_bankroll"
    if net_direction == "unknown" or bankroll_direction == "unknown":
        return "unknown"
    return f"{net_direction}_net_{bankroll_direction}_bankroll"


def _read_order() -> list[str]:
    return [
        "serving_compare_aggregate_summary",
        "dashboard_summaries_or_compare_manifests",
        "aggregate_rows",
    ]


def _current_phase(status: str) -> str:
    normalized = str(status or "")
    if normalized == "completed":
        return "completed"
    if normalized == "failed":
        return "aggregate_failed"
    return "building_aggregate"


def _recommended_action(status: str) -> str:
    normalized = str(status or "")
    if normalized == "completed":
        return "review_aggregate_summary"
    if normalized == "failed":
        return "inspect_aggregate_inputs"
    return "inspect_aggregate_progress"


def _highlights(
    *,
    status: str,
    recommended_action: str,
    input_kind: str | None,
    window_count: int | None,
    mean_net_delta: float | None,
    mean_bankroll_delta: float | None,
    error_message: str | None,
) -> list[str]:
    kind = str(input_kind or "unknown_inputs")
    if status == "completed":
        return [
            f"serving compare aggregate assembled from {kind}",
            f"window_count={window_count}, mean_net_delta_right_minus_left={mean_net_delta}, mean_pure_bankroll_delta_right_minus_left={mean_bankroll_delta}",
            f"next operator action: {recommended_action}",
        ]
    if status == "failed":
        return [
            f"serving compare aggregate failed while reading {kind}",
            str(error_message or "aggregate generation failed"),
            f"next operator action: {recommended_action}",
        ]
    return [
        f"serving compare aggregate is in progress for {kind}",
        f"next operator action: {recommended_action}",
    ]


def _build_failure_summary(
    *,
    input_paths: list[Path],
    input_kind: str | None,
    output_summary: Path | None,
    error_message: str,
) -> dict[str, Any]:
    status = "failed"
    recommended_action = _recommended_action(status)
    payload: dict[str, Any] = {
        "input_files": [artifact_display_path(path, workspace_root=ROOT) for path in input_paths],
        "input_kind": input_kind,
        "window_count": 0,
        "rows": [],
        "summary": {
            "manifest_status_counts": {},
            "manifest_decision_counts": {},
            "windows_with_positive_net_delta": [],
            "windows_with_negative_net_delta": [],
            "windows_with_positive_bankroll_delta": [],
            "windows_with_negative_bankroll_delta": [],
            "tradeoff_classification_counts": {},
            "windows_by_tradeoff_classification": {},
            "mean_net_delta_right_minus_left": None,
            "mean_pure_bankroll_delta_right_minus_left": None,
        },
        "status": status,
        "error_message": error_message,
        "read_order": _read_order(),
        "current_phase": _current_phase(status),
        "recommended_action": recommended_action,
        "highlights": _highlights(
            status=status,
            recommended_action=recommended_action,
            input_kind=input_kind,
            window_count=0,
            mean_net_delta=None,
            mean_bankroll_delta=None,
            error_message=error_message,
        ),
    }
    if output_summary is not None:
        payload["summary_file"] = artifact_display_path(output_summary, workspace_root=ROOT)
    return payload


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
    input_paths: list[Path] = []
    input_kind: str | None = None
    output_summary: Path | None = None

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
        artifact_ensure_output_file_path(output_summary, label="output summary", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_chart, label="output chart", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        progress.start(message=f"loading {input_kind} count={len(input_paths)}")
        with Heartbeat("[serving-compare-aggregate]", "loading summaries", logger=log_progress):
            payloads = [load_json(path) for path in input_paths]
        progress.update(message=f"{input_kind} loaded")

        rows: list[dict[str, Any]] = []
        for path, payload in zip(input_paths, payloads):
            row = _extract_row_from_dashboard_summary(path, payload) if args.dashboard_summaries else _extract_row_from_compare_manifest(path, payload)
            row["net_delta_direction"] = _delta_direction(row.get("net_delta_right_minus_left"))
            row["pure_bankroll_delta_direction"] = _delta_direction(row.get("pure_bankroll_delta_right_minus_left"))
            row["tradeoff_classification"] = _classify_window_tradeoff(row)
            rows.append(row)

        frame = pd.DataFrame(rows).sort_values(["window_label", "date_count"], ascending=[True, True]).reset_index(drop=True)
        row_records = frame.to_dict(orient="records")
        aggregate_payload = {
            "input_files": [artifact_display_path(path if path.is_absolute() else (Path.cwd() / path).resolve(), workspace_root=ROOT) for path in input_paths],
            "input_kind": "dashboard_summaries" if args.dashboard_summaries else "compare_manifests",
            "window_count": int(len(frame)),
            "rows": row_records,
            "summary": {
                "manifest_status_counts": frame["manifest_status"].fillna("unknown").value_counts().to_dict() if "manifest_status" in frame.columns else {},
                "manifest_decision_counts": frame["manifest_decision"].fillna("unknown").value_counts().to_dict() if "manifest_decision" in frame.columns else {},
                "windows_with_positive_net_delta": [
                    str(row["window_label"]) for row in row_records if row.get("net_delta_direction") == "positive"
                ],
                "windows_with_negative_net_delta": [
                    str(row["window_label"]) for row in row_records if row.get("net_delta_direction") == "negative"
                ],
                "windows_with_positive_bankroll_delta": [
                    str(row["window_label"]) for row in row_records if row.get("pure_bankroll_delta_direction") == "positive"
                ],
                "windows_with_negative_bankroll_delta": [
                    str(row["window_label"]) for row in row_records if row.get("pure_bankroll_delta_direction") == "negative"
                ],
                "tradeoff_classification_counts": frame["tradeoff_classification"].fillna("unknown").value_counts().to_dict(),
                "windows_by_tradeoff_classification": {
                    classification: [
                        str(row["window_label"]) for row in row_records if row.get("tradeoff_classification") == classification
                    ]
                    for classification in sorted({str(value) for value in frame["tradeoff_classification"].fillna("unknown").tolist()})
                },
                "mean_net_delta_right_minus_left": float(frame["net_delta_right_minus_left"].dropna().mean()) if "net_delta_right_minus_left" in frame.columns and not frame["net_delta_right_minus_left"].dropna().empty else None,
                "mean_pure_bankroll_delta_right_minus_left": float(frame["pure_bankroll_delta_right_minus_left"].dropna().mean()) if "pure_bankroll_delta_right_minus_left" in frame.columns and not frame["pure_bankroll_delta_right_minus_left"].dropna().empty else None,
            },
        }
        aggregate_payload["status"] = "completed"
        aggregate_payload["read_order"] = _read_order()
        aggregate_payload["current_phase"] = _current_phase("completed")
        aggregate_payload["recommended_action"] = _recommended_action("completed")
        aggregate_payload["highlights"] = _highlights(
            status="completed",
            recommended_action=str(aggregate_payload.get("recommended_action") or "review_aggregate_summary"),
            input_kind=str(aggregate_payload.get("input_kind") or input_kind or "unknown_inputs"),
            window_count=int(aggregate_payload.get("window_count") or 0),
            mean_net_delta=aggregate_payload.get("summary", {}).get("mean_net_delta_right_minus_left") if isinstance(aggregate_payload.get("summary"), dict) else None,
            mean_bankroll_delta=aggregate_payload.get("summary", {}).get("mean_pure_bankroll_delta_right_minus_left") if isinstance(aggregate_payload.get("summary"), dict) else None,
            error_message=None,
        )
        progress.update(message=f"aggregate assembled windows={len(frame)}")

        with Heartbeat("[serving-compare-aggregate]", "writing aggregate outputs", logger=log_progress):
            write_json(output_summary, aggregate_payload)
            write_csv_file(output_csv, frame, index=False)

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
            save_figure(output_chart, fig, dpi=140)
            plt.close(fig)
        progress.complete(message=f"aggregate outputs saved windows={len(frame)}")

        print(f"[serving-compare-aggregate] summary saved: {artifact_display_path(output_summary if output_summary.is_absolute() else (Path.cwd() / output_summary).resolve(), workspace_root=ROOT)}")
        print(f"[serving-compare-aggregate] chart saved: {artifact_display_path(output_chart if output_chart.is_absolute() else (Path.cwd() / output_chart).resolve(), workspace_root=ROOT)}")
        print(f"[serving-compare-aggregate] csv saved: {artifact_display_path(output_csv if output_csv.is_absolute() else (Path.cwd() / output_csv).resolve(), workspace_root=ROOT)}")
        print(f"[serving-compare-aggregate] metrics: {aggregate_payload['summary']}")
        return 0
    except KeyboardInterrupt:
        print("[serving-compare-aggregate] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        if output_summary is not None:
            write_json(
                output_summary,
                _build_failure_summary(
                    input_paths=input_paths,
                    input_kind="dashboard_summaries" if args.dashboard_summaries else "compare_manifests",
                    output_summary=output_summary,
                    error_message=str(error),
                ),
            )
        print(f"[serving-compare-aggregate] failed: {error}")
        return 1
    except Exception as error:
        if output_summary is not None:
            write_json(
                output_summary,
                _build_failure_summary(
                    input_paths=input_paths,
                    input_kind="dashboard_summaries" if args.dashboard_summaries else "compare_manifests",
                    output_summary=output_summary,
                    error_message=str(error),
                ),
            )
        print(f"[serving-compare-aggregate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())