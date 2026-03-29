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


def latest_file(path: Path, pattern: str) -> Path:
    files = sorted(path.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {path}/{pattern}")
    return files[-1]


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-compare-dashboard {now}] {message}", flush=True)


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


def _build_bankroll_path(summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bankroll = 1.0
    for date_value, case in sorted(_summary_case_map(summary).items()):
        backtest_payload = load_json(_backtest_path(case))
        multiplier = _safe_float(backtest_payload.get("policy_final_bankroll"))
        if multiplier is None:
            multiplier = 1.0
        start_bankroll = bankroll
        bankroll = bankroll * multiplier
        rows.append(
            {
                "date": date_value,
                "start_bankroll": start_bankroll,
                "end_bankroll": bankroll,
                "daily_multiplier": multiplier,
                "policy_bets": _safe_int(case.get("policy_bets")) or 0,
                "policy_name": case.get("policy_name"),
            }
        )
    return rows


def _find_case_key(row: dict[str, Any], suffix: str, field_name: str) -> str | None:
    candidate = f"{suffix}_{field_name}"
    if candidate in row:
        return candidate
    for key in row:
        if key.endswith(f"_{field_name}") and suffix in key:
            return key
    return None


def _build_case_rows(
    *,
    compare_payload: dict[str, Any],
    left_suffix: str,
    right_suffix: str,
    left_bankroll_path: list[dict[str, Any]],
    right_bankroll_path: list[dict[str, Any]],
    best_path: list[dict[str, Any]] | None,
) -> pd.DataFrame:
    left_bankroll_map = {row["date"]: row for row in left_bankroll_path}
    right_bankroll_map = {row["date"]: row for row in right_bankroll_path}
    best_path_map = {row["date"]: row for row in (best_path or [])}
    records: list[dict[str, Any]] = []
    for row in compare_payload.get("cases", []):
        if not isinstance(row, dict):
            continue
        date_value = str(row.get("date") or "").strip()
        if not date_value:
            continue
        left_bets_key = _find_case_key(row, left_suffix, "policy_bets")
        right_bets_key = _find_case_key(row, right_suffix, "policy_bets")
        left_net_key = _find_case_key(row, left_suffix, "policy_net")
        right_net_key = _find_case_key(row, right_suffix, "policy_net")
        left_policy_key = _find_case_key(row, left_suffix, "policy_name")
        right_policy_key = _find_case_key(row, right_suffix, "policy_name")
        left_bankroll_row = left_bankroll_map.get(date_value, {})
        right_bankroll_row = right_bankroll_map.get(date_value, {})
        best_path_row = best_path_map.get(date_value, {})
        records.append(
            {
                "date": date_value,
                "left_policy_name": row.get(left_policy_key) if left_policy_key else None,
                "right_policy_name": row.get(right_policy_key) if right_policy_key else None,
                "left_policy_bets": _safe_int(row.get(left_bets_key)) if left_bets_key else None,
                "right_policy_bets": _safe_int(row.get(right_bets_key)) if right_bets_key else None,
                "left_policy_net": _safe_float(row.get(left_net_key)) if left_net_key else None,
                "right_policy_net": _safe_float(row.get(right_net_key)) if right_net_key else None,
                "policy_net_delta_left_minus_right": _safe_float(row.get("policy_net_delta")),
                "left_end_bankroll": _safe_float(left_bankroll_row.get("end_bankroll")),
                "right_end_bankroll": _safe_float(right_bankroll_row.get("end_bankroll")),
                "best_path_selected_label": best_path_row.get("selected_label"),
                "best_path_end_bankroll": _safe_float(best_path_row.get("end_bankroll")),
                "best_path_policy_bets": _safe_int(best_path_row.get("policy_bets")),
            }
        )
    return pd.DataFrame(records)


def _read_order() -> list[str]:
    return [
        "serving_compare_dashboard_summary",
        "compare_manifest",
        "compare_json",
        "bankroll_sweep_json",
        "left_summary",
        "right_summary",
    ]


def _current_phase(status: str) -> str:
    normalized = str(status or "")
    if normalized == "completed":
        return "completed"
    if normalized == "failed":
        return "dashboard_failed"
    return "building_dashboard"


def _recommended_action(status: str) -> str:
    normalized = str(status or "")
    if normalized == "completed":
        return "review_dashboard_summary"
    if normalized == "failed":
        return "inspect_dashboard_inputs"
    return "inspect_dashboard_progress"


def _highlights(
    *,
    status: str,
    recommended_action: str,
    window_label: str | None,
    manifest_status: str | None,
    net_delta: float | None,
    bankroll_delta: float | None,
    error_message: str | None,
) -> list[str]:
    label = str(window_label or "unknown_window")
    if status == "completed":
        return [
            f"serving compare dashboard assembled for window={label}",
            f"manifest_status={manifest_status}, net_delta_right_minus_left={net_delta}, pure_bankroll_delta_right_minus_left={bankroll_delta}",
            f"next operator action: {recommended_action}",
        ]
    if status == "failed":
        return [
            f"serving compare dashboard failed for window={label}",
            str(error_message or "dashboard generation failed"),
            f"next operator action: {recommended_action}",
        ]
    return [
        f"serving compare dashboard is in progress for window={label}",
        f"next operator action: {recommended_action}",
    ]


def _build_failure_summary(
    *,
    manifest_path: Path | None,
    compare_path: Path | None,
    bankroll_path: Path | None,
    output_summary: Path | None,
    error_message: str,
) -> dict[str, Any]:
    status = "failed"
    recommended_action = _recommended_action(status)
    payload: dict[str, Any] = {
        "manifest_file": artifact_display_path(manifest_path, workspace_root=ROOT) if manifest_path else None,
        "manifest_status": None,
        "manifest_decision": None,
        "compare_json": artifact_display_path(compare_path, workspace_root=ROOT) if compare_path else None,
        "bankroll_sweep_json": artifact_display_path(bankroll_path, workspace_root=ROOT) if bankroll_path else None,
        "window_label": None,
        "prediction_backend": None,
        "date_count": 0,
        "dates": [],
        "left": {
            "profile": None,
            "artifact_suffix": None,
            "summary_file": None,
            "pure_path_final_bankroll": None,
        },
        "right": {
            "profile": None,
            "artifact_suffix": None,
            "summary_file": None,
            "pure_path_final_bankroll": None,
        },
        "compare": {
            "shared_ok_dates": [],
            "left_total_policy_bets": None,
            "right_total_policy_bets": None,
            "left_total_policy_net": None,
            "right_total_policy_net": None,
            "right_minus_left_total_policy_net": None,
            "left_mean_policy_roi": None,
            "right_mean_policy_roi": None,
            "differing_policy_dates": [],
        },
        "bankroll": {
            "baseline_only": None,
            "best_result": None,
            "right_minus_left_pure_final_bankroll": None,
        },
        "status": status,
        "error_message": error_message,
        "read_order": _read_order(),
        "current_phase": _current_phase(status),
        "recommended_action": recommended_action,
        "highlights": _highlights(
            status=status,
            recommended_action=recommended_action,
            window_label=None,
            manifest_status=None,
            net_delta=None,
            bankroll_delta=None,
            error_message=error_message,
        ),
    }
    if output_summary is not None:
        payload["summary_file"] = artifact_display_path(output_summary, workspace_root=ROOT)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-file", default=None)
    parser.add_argument("--compare-json", default=None)
    parser.add_argument("--bankroll-sweep-json", default=None)
    parser.add_argument("--output-summary", default=None)
    parser.add_argument("--output-chart", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=4, prefix="[serving-compare-dashboard]", logger=log_progress, min_interval_sec=0.0)
    manifest_path: Path | None = None
    compare_path: Path | None = None
    bankroll_path: Path | None = None
    output_summary: Path | None = None

    try:
        report_dir = ROOT / "artifacts" / "reports"
        dashboard_dir = report_dir / "dashboard"
        manifest_path = Path(args.manifest_file) if args.manifest_file else latest_file(report_dir, "serving_smoke_profile_compare_*.json")
        manifest_path = resolve_path(manifest_path)
        progress.start(message=f"resolving inputs manifest={artifact_display_path(manifest_path if manifest_path.is_absolute() else (Path.cwd() / manifest_path).resolve(), workspace_root=ROOT)}")

        manifest_payload = load_json(manifest_path)
        compare_path = resolve_path(args.compare_json or manifest_payload.get("outputs", {}).get("compare_json"))
        bankroll_path = None
        bankroll_path_value = args.bankroll_sweep_json or manifest_payload.get("outputs", {}).get("bankroll_sweep_json")
        if bankroll_path_value:
            bankroll_path = resolve_path(bankroll_path_value)

        summary_stem = manifest_path.stem.replace("serving_smoke_profile_compare_", "serving_compare_dashboard_")
        output_summary = resolve_path(args.output_summary) if args.output_summary else dashboard_dir / f"{summary_stem}.json"
        output_chart = resolve_path(args.output_chart) if args.output_chart else dashboard_dir / f"{summary_stem}.png"
        output_csv = resolve_path(args.output_csv) if args.output_csv else dashboard_dir / f"{summary_stem}.csv"
        artifact_ensure_output_file_path(output_summary, label="output summary", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_chart, label="output chart", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)
        progress.update(message=f"inputs resolved compare={artifact_display_path(compare_path if compare_path.is_absolute() else (Path.cwd() / compare_path).resolve(), workspace_root=ROOT)} bankroll={artifact_display_path(bankroll_path if bankroll_path.is_absolute() else (Path.cwd() / bankroll_path).resolve(), workspace_root=ROOT) if bankroll_path else 'none'}")

        with Heartbeat("[serving-compare-dashboard]", "loading compare artifacts", logger=log_progress):
            compare_payload = load_json(compare_path)
            bankroll_payload = load_json(bankroll_path) if bankroll_path and bankroll_path.exists() else None
            left_summary = load_json(resolve_path(manifest_payload["left"]["summary_file"]))
            right_summary = load_json(resolve_path(manifest_payload["right"]["summary_file"]))
        progress.update(message="compare artifacts loaded")

        left_bankroll_path = _build_bankroll_path(left_summary)
        right_bankroll_path = _build_bankroll_path(right_summary)
        best_path = bankroll_payload.get("best_result", {}).get("path") if isinstance(bankroll_payload, dict) else None
        left_suffix = str(manifest_payload.get("left", {}).get("artifact_suffix") or manifest_payload.get("left", {}).get("profile") or "left")
        right_suffix = str(manifest_payload.get("right", {}).get("artifact_suffix") or manifest_payload.get("right", {}).get("profile") or "right")
        case_df = _build_case_rows(
            compare_payload=compare_payload,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            left_bankroll_path=left_bankroll_path,
            right_bankroll_path=right_bankroll_path,
            best_path=best_path if isinstance(best_path, list) else None,
        )

        aggregates = compare_payload.get("comparison", {}).get("shared_ok_aggregates", {})
        pure_left_final_bankroll = left_bankroll_path[-1]["end_bankroll"] if left_bankroll_path else None
        pure_right_final_bankroll = right_bankroll_path[-1]["end_bankroll"] if right_bankroll_path else None
        best_result = bankroll_payload.get("best_result") if isinstance(bankroll_payload, dict) else None
        baseline_only = bankroll_payload.get("baseline_only") if isinstance(bankroll_payload, dict) else None
        summary_payload = {
            "manifest_file": artifact_display_path(manifest_path if manifest_path.is_absolute() else (Path.cwd() / manifest_path).resolve(), workspace_root=ROOT),
            "manifest_status": manifest_payload.get("status"),
            "manifest_decision": manifest_payload.get("decision"),
            "compare_json": artifact_display_path(compare_path if compare_path.is_absolute() else (Path.cwd() / compare_path).resolve(), workspace_root=ROOT),
            "bankroll_sweep_json": artifact_display_path(bankroll_path if bankroll_path.is_absolute() else (Path.cwd() / bankroll_path).resolve(), workspace_root=ROOT) if bankroll_path else None,
            "window_label": manifest_payload.get("window_label"),
            "prediction_backend": manifest_payload.get("prediction_backend"),
            "date_count": len(manifest_payload.get("dates", [])),
            "dates": manifest_payload.get("dates", []),
            "left": {
                "profile": manifest_payload.get("left", {}).get("profile"),
                "artifact_suffix": left_suffix,
                "summary_file": manifest_payload.get("left", {}).get("summary_file"),
                "pure_path_final_bankroll": pure_left_final_bankroll,
            },
            "right": {
                "profile": manifest_payload.get("right", {}).get("profile"),
                "artifact_suffix": right_suffix,
                "summary_file": manifest_payload.get("right", {}).get("summary_file"),
                "pure_path_final_bankroll": pure_right_final_bankroll,
            },
            "compare": {
                "shared_ok_dates": compare_payload.get("comparison", {}).get("shared_ok_dates", []),
                "left_total_policy_bets": aggregates.get("left_total_policy_bets"),
                "right_total_policy_bets": aggregates.get("right_total_policy_bets"),
                "left_total_policy_net": aggregates.get("left_total_policy_net"),
                "right_total_policy_net": aggregates.get("right_total_policy_net"),
                "right_minus_left_total_policy_net": (
                    _safe_float(aggregates.get("right_total_policy_net")) - _safe_float(aggregates.get("left_total_policy_net"))
                    if _safe_float(aggregates.get("right_total_policy_net")) is not None and _safe_float(aggregates.get("left_total_policy_net")) is not None
                    else None
                ),
                "left_mean_policy_roi": aggregates.get("left_mean_policy_roi"),
                "right_mean_policy_roi": aggregates.get("right_mean_policy_roi"),
                "differing_policy_dates": compare_payload.get("comparison", {}).get("differing_policy_dates", []),
            },
            "bankroll": {
                "baseline_only": baseline_only,
                "best_result": best_result,
                "right_minus_left_pure_final_bankroll": (
                    float(pure_right_final_bankroll) - float(pure_left_final_bankroll)
                    if pure_left_final_bankroll is not None and pure_right_final_bankroll is not None
                    else None
                ),
            },
        }
        summary_payload["status"] = "completed"
        summary_payload["read_order"] = _read_order()
        summary_payload["current_phase"] = _current_phase("completed")
        summary_payload["recommended_action"] = _recommended_action("completed")
        summary_payload["highlights"] = _highlights(
            status="completed",
            recommended_action=str(summary_payload.get("recommended_action") or "review_dashboard_summary"),
            window_label=summary_payload.get("window_label"),
            manifest_status=summary_payload.get("manifest_status"),
            net_delta=summary_payload.get("compare", {}).get("right_minus_left_total_policy_net") if isinstance(summary_payload.get("compare"), dict) else None,
            bankroll_delta=summary_payload.get("bankroll", {}).get("right_minus_left_pure_final_bankroll") if isinstance(summary_payload.get("bankroll"), dict) else None,
            error_message=None,
        )
        progress.update(message=f"summary assembled rows={len(case_df)}")

        with Heartbeat("[serving-compare-dashboard]", "writing dashboard outputs", logger=log_progress):
            write_json(output_summary, summary_payload)
            write_csv_file(output_csv, case_df, index=False)

            fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
            x_values = list(range(len(case_df)))
            x_labels = case_df["date"].tolist()

            axes[0].plot(x_values, case_df["left_policy_net"], marker="o", linewidth=2.0, color="#b91c1c", label=str(summary_payload["left"]["profile"]))
            axes[0].plot(x_values, case_df["right_policy_net"], marker="o", linewidth=2.0, color="#0f766e", label=str(summary_payload["right"]["profile"]))
            axes[0].axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--")
            axes[0].set_title("Daily Policy Net")
            axes[0].set_xticks(x_values)
            axes[0].set_xticklabels(x_labels, rotation=45, ha="right")
            axes[0].set_ylabel("policy net")
            axes[0].legend()

            axes[1].plot(x_values, case_df["left_end_bankroll"], marker="o", linewidth=2.0, color="#b91c1c", label=f"{summary_payload['left']['profile']} pure")
            axes[1].plot(x_values, case_df["right_end_bankroll"], marker="o", linewidth=2.0, color="#0f766e", label=f"{summary_payload['right']['profile']} pure")
            if "best_path_end_bankroll" in case_df.columns and case_df["best_path_end_bankroll"].notna().any():
                axes[1].plot(x_values, case_df["best_path_end_bankroll"], marker="o", linewidth=2.0, color="#1d4ed8", label="best sweep path")
            axes[1].axhline(1.0, color="#6b7280", linewidth=1.0, linestyle="--")
            axes[1].set_title("Bankroll Path")
            axes[1].set_xticks(x_values)
            axes[1].set_xticklabels(x_labels, rotation=45, ha="right")
            axes[1].set_ylabel("bankroll")
            axes[1].legend()

            plt.tight_layout()
            save_figure(output_chart, fig, dpi=140)
            plt.close(fig)
        progress.complete(message=f"dashboard outputs saved stem={summary_stem}")

        print(f"[serving-compare-dashboard] summary saved: {artifact_display_path(output_summary if output_summary.is_absolute() else (Path.cwd() / output_summary).resolve(), workspace_root=ROOT)}")
        print(f"[serving-compare-dashboard] chart saved: {artifact_display_path(output_chart if output_chart.is_absolute() else (Path.cwd() / output_chart).resolve(), workspace_root=ROOT)}")
        print(f"[serving-compare-dashboard] csv saved: {artifact_display_path(output_csv if output_csv.is_absolute() else (Path.cwd() / output_csv).resolve(), workspace_root=ROOT)}")
        print(f"[serving-compare-dashboard] metrics: {summary_payload}")
        return 0
    except KeyboardInterrupt:
        print("[serving-compare-dashboard] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        if output_summary is not None:
            write_json(
                output_summary,
                _build_failure_summary(
                    manifest_path=manifest_path,
                    compare_path=compare_path,
                    bankroll_path=bankroll_path,
                    output_summary=output_summary,
                    error_message=str(error),
                ),
            )
        print(f"[serving-compare-dashboard] failed: {error}")
        return 1
    except Exception as error:
        if output_summary is not None:
            write_json(
                output_summary,
                _build_failure_summary(
                    manifest_path=manifest_path,
                    compare_path=compare_path,
                    bankroll_path=bankroll_path,
                    output_summary=output_summary,
                    error_message=str(error),
                ),
            )
        print(f"[serving-compare-dashboard] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
