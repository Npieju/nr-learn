from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import write_csv_file, write_text_file
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-smoke-compare {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_summary(path_value: str | Path) -> dict[str, Any]:
    path = _resolve_path(path_value)
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "summary"


def _summary_slug(summary: dict[str, Any], fallback_path: Path) -> str:
    artifact_suffix = str(summary.get("artifact_suffix") or "").strip()
    if artifact_suffix:
        return _slugify(artifact_suffix)
    profile = str(summary.get("profile") or "").strip()
    if profile:
        return _slugify(profile)
    return _slugify(fallback_path.stem)


def _case_map(summary: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    mapping: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        date_value = str(case.get("date", "")).strip()
        if not date_value:
            continue
        if date_value in mapping:
            raise ValueError(f"Duplicate date '{date_value}' found in {label}")
        mapping[date_value] = case
    return mapping


def _float_or_none(value: Any) -> float | None:
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


def _int_or_none(value: Any) -> int | None:
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


def _sum_int_field(rows: list[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        value = _int_or_none(row.get(key))
        if value is not None:
            total += value
    return total


def _sum_float_field(rows: list[dict[str, Any]], key: str) -> float:
    total = 0.0
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            total += value
    return float(total)


def _mean_float_field(rows: list[dict[str, Any]], key: str) -> float | None:
    values = []
    for row in rows:
        value = _float_or_none(row.get(key))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return float(sum(values) / len(values))


def _policy_return_or_none(bets: int | None, roi: float | None) -> float | None:
    if bets is None:
        return None
    if bets == 0:
        return 0.0
    if roi is None:
        return None
    return float(bets) * float(roi)


def _policy_net_or_none(bets: int | None, roi: float | None) -> float | None:
    policy_return = _policy_return_or_none(bets, roi)
    if policy_return is None or bets is None:
        return None
    return float(policy_return - bets)


def _shared_row(date_value: str, left_case: dict[str, Any] | None, right_case: dict[str, Any] | None, *, left_label: str, right_label: str) -> dict[str, Any]:
    left_bets = _int_or_none(left_case.get("policy_bets")) if left_case else None
    right_bets = _int_or_none(right_case.get("policy_bets")) if right_case else None
    left_roi = _float_or_none(left_case.get("policy_roi")) if left_case else None
    right_roi = _float_or_none(right_case.get("policy_roi")) if right_case else None
    left_selected = _int_or_none(left_case.get("policy_selected_rows")) if left_case else None
    right_selected = _int_or_none(right_case.get("policy_selected_rows")) if right_case else None
    left_return = _policy_return_or_none(left_bets, left_roi)
    right_return = _policy_return_or_none(right_bets, right_roi)
    left_net = _policy_net_or_none(left_bets, left_roi)
    right_net = _policy_net_or_none(right_bets, right_roi)

    row = {
        "date": date_value,
        f"{left_label}_present": left_case is not None,
        f"{right_label}_present": right_case is not None,
        f"{left_label}_status": left_case.get("status") if left_case else None,
        f"{right_label}_status": right_case.get("status") if right_case else None,
        f"{left_label}_score_source": left_case.get("score_source") if left_case else None,
        f"{right_label}_score_source": right_case.get("score_source") if right_case else None,
        f"{left_label}_policy_name": left_case.get("policy_name") if left_case else None,
        f"{right_label}_policy_name": right_case.get("policy_name") if right_case else None,
        f"{left_label}_policy_selected_rows": left_selected,
        f"{right_label}_policy_selected_rows": right_selected,
        f"{left_label}_policy_bets": left_bets,
        f"{right_label}_policy_bets": right_bets,
        f"{left_label}_policy_roi": left_roi,
        f"{right_label}_policy_roi": right_roi,
        f"{left_label}_policy_return": left_return,
        f"{right_label}_policy_return": right_return,
        f"{left_label}_policy_net": left_net,
        f"{right_label}_policy_net": right_net,
        "shared_date": left_case is not None and right_case is not None,
        "both_ok": (left_case is not None and right_case is not None and left_case.get("status") == "ok" and right_case.get("status") == "ok"),
        "score_source_same": (
            left_case is not None and right_case is not None and str(left_case.get("score_source", "")) == str(right_case.get("score_source", ""))
        ),
        "policy_name_same": (
            left_case is not None and right_case is not None and str(left_case.get("policy_name", "")) == str(right_case.get("policy_name", ""))
        ),
        "policy_bets_delta": (left_bets - right_bets) if left_bets is not None and right_bets is not None else None,
        "policy_selected_rows_delta": (left_selected - right_selected) if left_selected is not None and right_selected is not None else None,
        "policy_roi_delta": (left_roi - right_roi) if left_roi is not None and right_roi is not None else None,
        "policy_return_delta": (left_return - right_return) if left_return is not None and right_return is not None else None,
        "policy_net_delta": (left_net - right_net) if left_net is not None and right_net is not None else None,
    }
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-summary", required=True)
    parser.add_argument("--right-summary", required=True)
    parser.add_argument("--left-label", default=None)
    parser.add_argument("--right-label", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=4, prefix="[serving-smoke-compare]", logger=log_progress, min_interval_sec=0.0)

    try:
        left_summary_path = _resolve_path(args.left_summary)
        right_summary_path = _resolve_path(args.right_summary)
        progress.start(
            message=(
                f"loading left={artifact_display_path(left_summary_path, workspace_root=ROOT)} "
                f"right={artifact_display_path(right_summary_path, workspace_root=ROOT)}"
            )
        )
        with Heartbeat("[serving-smoke-compare]", "loading summaries", logger=log_progress):
            left_summary = _load_summary(left_summary_path)
            right_summary = _load_summary(right_summary_path)

        left_label = str(args.left_label or left_summary.get("profile") or left_summary_path.stem).strip()
        right_label = str(args.right_label or right_summary.get("profile") or right_summary_path.stem).strip()
        left_slug = _summary_slug(left_summary, left_summary_path)
        right_slug = _summary_slug(right_summary, right_summary_path)

        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"serving_smoke_compare_{left_slug}_vs_{right_slug}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"serving_smoke_compare_{left_slug}_vs_{right_slug}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)
        progress.update(message=f"summaries loaded left={left_label} right={right_label}")

        with Heartbeat("[serving-smoke-compare]", "building comparison summary", logger=log_progress):
            left_cases = _case_map(left_summary, left_label)
            right_cases = _case_map(right_summary, right_label)
            dates = sorted(set(left_cases) | set(right_cases))
            rows = [
                _shared_row(date_value, left_cases.get(date_value), right_cases.get(date_value), left_label=left_slug, right_label=right_slug)
                for date_value in dates
            ]

            shared_dates = [row["date"] for row in rows if row["shared_date"]]
            shared_ok_dates = [row["date"] for row in rows if row["both_ok"]]
            differing_score_source_dates = [row["date"] for row in rows if row["shared_date"] and not row["score_source_same"]]
            differing_policy_dates = [row["date"] for row in rows if row["shared_date"] and not row["policy_name_same"]]
            left_only_dates = [row["date"] for row in rows if row[f"{left_slug}_present"] and not row[f"{right_slug}_present"]]
            right_only_dates = [row["date"] for row in rows if row[f"{right_slug}_present"] and not row[f"{left_slug}_present"]]
            shared_ok_rows = [row for row in rows if row["both_ok"]]
            matching_score_source_dates = [row["date"] for row in shared_ok_rows if row["score_source_same"]]
            matching_policy_dates = [row["date"] for row in shared_ok_rows if row["policy_name_same"]]
            nonzero_policy_bets_delta_dates = [row["date"] for row in shared_ok_rows if _int_or_none(row.get("policy_bets_delta")) not in {None, 0}]
            nonzero_policy_selected_rows_delta_dates = [
                row["date"] for row in shared_ok_rows if _int_or_none(row.get("policy_selected_rows_delta")) not in {None, 0}
            ]
            nonzero_policy_roi_delta_dates = [row["date"] for row in shared_ok_rows if _float_or_none(row.get("policy_roi_delta")) not in {None, 0.0}]
            nonzero_policy_return_delta_dates = [row["date"] for row in shared_ok_rows if _float_or_none(row.get("policy_return_delta")) not in {None, 0.0}]
            nonzero_policy_net_delta_dates = [row["date"] for row in shared_ok_rows if _float_or_none(row.get("policy_net_delta")) not in {None, 0.0}]
            comparable_policy_roi_dates = [
                row["date"]
                for row in shared_ok_rows
                if _float_or_none(row.get(f"{left_slug}_policy_roi")) is not None and _float_or_none(row.get(f"{right_slug}_policy_roi")) is not None
            ]

            left_total_policy_bets = _sum_int_field(shared_ok_rows, f"{left_slug}_policy_bets")
            right_total_policy_bets = _sum_int_field(shared_ok_rows, f"{right_slug}_policy_bets")
            left_total_policy_selected_rows = _sum_int_field(shared_ok_rows, f"{left_slug}_policy_selected_rows")
            right_total_policy_selected_rows = _sum_int_field(shared_ok_rows, f"{right_slug}_policy_selected_rows")
            left_total_policy_return = _sum_float_field(shared_ok_rows, f"{left_slug}_policy_return")
            right_total_policy_return = _sum_float_field(shared_ok_rows, f"{right_slug}_policy_return")
            left_total_policy_net = _sum_float_field(shared_ok_rows, f"{left_slug}_policy_net")
            right_total_policy_net = _sum_float_field(shared_ok_rows, f"{right_slug}_policy_net")
            left_mean_policy_roi = _mean_float_field(shared_ok_rows, f"{left_slug}_policy_roi")
            right_mean_policy_roi = _mean_float_field(shared_ok_rows, f"{right_slug}_policy_roi")
        progress.update(message=f"comparison built union_dates={len(rows)} shared_ok_dates={len(shared_ok_dates)}")

        summary = {
            "left": {
                "label": left_label,
                "summary_file": artifact_display_path(left_summary_path, workspace_root=ROOT),
                "profile": left_summary.get("profile"),
                "config_file": left_summary.get("config_file"),
                "num_cases": len(left_cases),
            },
            "right": {
                "label": right_label,
                "summary_file": artifact_display_path(right_summary_path, workspace_root=ROOT),
                "profile": right_summary.get("profile"),
                "config_file": right_summary.get("config_file"),
                "num_cases": len(right_cases),
            },
            "comparison": {
                "num_union_dates": len(rows),
                "num_shared_dates": len(shared_dates),
                "num_shared_ok_dates": len(shared_ok_dates),
                "shared_dates": shared_dates,
                "shared_ok_dates": shared_ok_dates,
                "left_only_dates": left_only_dates,
                "right_only_dates": right_only_dates,
                "differing_score_source_dates": differing_score_source_dates,
                "differing_policy_dates": differing_policy_dates,
                "matching_score_source_dates": matching_score_source_dates,
                "matching_policy_dates": matching_policy_dates,
                "shared_ok_aggregates": {
                    "left_total_policy_bets": left_total_policy_bets,
                    "right_total_policy_bets": right_total_policy_bets,
                    "total_policy_bets_delta": left_total_policy_bets - right_total_policy_bets,
                    "left_total_policy_selected_rows": left_total_policy_selected_rows,
                    "right_total_policy_selected_rows": right_total_policy_selected_rows,
                    "total_policy_selected_rows_delta": left_total_policy_selected_rows - right_total_policy_selected_rows,
                    "left_total_policy_return": left_total_policy_return,
                    "right_total_policy_return": right_total_policy_return,
                    "total_policy_return_delta": left_total_policy_return - right_total_policy_return,
                    "left_total_policy_net": left_total_policy_net,
                    "right_total_policy_net": right_total_policy_net,
                    "total_policy_net_delta": left_total_policy_net - right_total_policy_net,
                    "comparable_policy_roi_dates": comparable_policy_roi_dates,
                    "left_mean_policy_roi": left_mean_policy_roi,
                    "right_mean_policy_roi": right_mean_policy_roi,
                    "mean_policy_roi_delta": (
                        left_mean_policy_roi - right_mean_policy_roi
                        if left_mean_policy_roi is not None and right_mean_policy_roi is not None
                        else None
                    ),
                    "nonzero_policy_bets_delta_dates": nonzero_policy_bets_delta_dates,
                    "nonzero_policy_selected_rows_delta_dates": nonzero_policy_selected_rows_delta_dates,
                    "nonzero_policy_roi_delta_dates": nonzero_policy_roi_delta_dates,
                    "nonzero_policy_return_delta_dates": nonzero_policy_return_delta_dates,
                    "nonzero_policy_net_delta_dates": nonzero_policy_net_delta_dates,
                },
            },
            "cases": rows,
        }

        with Heartbeat("[serving-smoke-compare]", "writing comparison outputs", logger=log_progress):
            write_text_file(output_json, json.dumps(summary, ensure_ascii=False, indent=2), label="output json")
            write_csv_file(output_csv, pd.DataFrame(rows), index=False, label="output csv")
        progress.update(message=f"outputs saved json={artifact_display_path(output_json, workspace_root=ROOT)} csv={artifact_display_path(output_csv, workspace_root=ROOT)}")
        progress.complete(message="comparison flow finished")

        print(f"[serving-smoke-compare] json saved: {artifact_display_path(output_json, workspace_root=ROOT)}")
        print(f"[serving-smoke-compare] csv saved: {artifact_display_path(output_csv, workspace_root=ROOT)}")
        print(f"[serving-smoke-compare] shared_dates={shared_dates}")
        print(f"[serving-smoke-compare] differing_score_source_dates={differing_score_source_dates}")
        print(f"[serving-smoke-compare] differing_policy_dates={differing_policy_dates}")
        print(
            "[serving-smoke-compare] shared_ok_policy_bets="
            f"{left_label}:{left_total_policy_bets} {right_label}:{right_total_policy_bets} delta={left_total_policy_bets - right_total_policy_bets}"
        )
        print(
            "[serving-smoke-compare] shared_ok_mean_policy_roi="
            f"{left_label}:{left_mean_policy_roi} {right_label}:{right_mean_policy_roi} "
            f"delta={summary['comparison']['shared_ok_aggregates']['mean_policy_roi_delta']}"
        )
        print(
            "[serving-smoke-compare] shared_ok_policy_net="
            f"{left_label}:{left_total_policy_net} {right_label}:{right_total_policy_net} "
            f"delta={summary['comparison']['shared_ok_aggregates']['total_policy_net_delta']}"
        )
        return 0
    except KeyboardInterrupt:
        print("[serving-smoke-compare] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[serving-smoke-compare] failed: {error}")
        return 1
    except Exception as error:
        print(f"[serving-smoke-compare] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())