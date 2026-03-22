from __future__ import annotations

import argparse
import json
from collections import Counter
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
from racing_ml.common.artifacts import write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-stage-compare {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_summary(path_value: str | Path) -> dict[str, Any]:
    path = _resolve_path(path_value)
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _display_summary_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", value.strip()).strip("_").lower()
    return slug or "summary"


def _parse_summary_arg(raw: str) -> tuple[str, Path]:
    if "=" not in raw:
        raise ValueError(f"summary must be label=path: {raw}")
    label, path_value = raw.split("=", 1)
    normalized_label = str(label).strip()
    if not normalized_label:
        raise ValueError(f"summary label is empty: {raw}")
    return normalized_label, _resolve_path(path_value.strip())


def _case_map(summary: dict[str, Any], label: str) -> dict[str, dict[str, Any]]:
    cases = summary.get("cases") if isinstance(summary.get("cases"), list) else []
    mapping: dict[str, dict[str, Any]] = {}
    for case in cases:
        if not isinstance(case, dict):
            continue
        date_value = str(case.get("date") or "").strip()
        if not date_value:
            continue
        if date_value in mapping:
            raise ValueError(f"duplicate date '{date_value}' in {label}")
        mapping[date_value] = case
    return mapping


def _normalize_stage_names(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _normalize_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return sorted({str(item).strip() for item in value if str(item).strip()})


def _stage_signature(value: Any) -> str:
    stage_names = _normalize_stage_names(value)
    if not stage_names:
        return "(none)"
    return "|".join(stage_names)


def _list_signature(value: Any) -> str:
    normalized = _normalize_string_list(value)
    if not normalized:
        return "(none)"
    return "|".join(normalized)


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


def _build_row(date_value: str, labels: list[str], case_maps: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    row: dict[str, Any] = {"date": date_value}
    ok_policy_names: list[str] = []
    ok_stage_signatures: list[str] = []
    ok_fallback_reason_signatures: list[str] = []
    ok_stage_trace_signatures: list[str] = []
    shared_all = True
    shared_ok_all = True

    for label in labels:
        slug = _slugify(label)
        case = case_maps[label].get(date_value)
        present = case is not None
        status = str(case.get("status") or "") if case else ""
        bets = _int_or_none(case.get("policy_bets")) if case else None
        roi = _float_or_none(case.get("policy_roi")) if case else None
        policy_name = str(case.get("policy_name") or "") if case else ""
        stage_signature = _stage_signature(case.get("policy_stage_names") if case else None)
        fallback_reason_signature = _list_signature(case.get("policy_stage_fallback_reasons") if case else None)
        stage_trace_signature = _list_signature(case.get("policy_stage_traces") if case else None)

        row[f"{slug}_present"] = present
        row[f"{slug}_status"] = status if present else None
        row[f"{slug}_policy_name"] = policy_name if present else None
        row[f"{slug}_policy_stage_signature"] = stage_signature if present else None
        row[f"{slug}_policy_stage_names"] = _normalize_stage_names(case.get("policy_stage_names") if case else None)
        row[f"{slug}_policy_stage_trace_signature"] = stage_trace_signature if present else None
        row[f"{slug}_policy_stage_traces"] = _normalize_string_list(case.get("policy_stage_traces") if case else None)
        row[f"{slug}_policy_stage_fallback_reason_signature"] = fallback_reason_signature if present else None
        row[f"{slug}_policy_stage_fallback_reasons"] = _normalize_string_list(case.get("policy_stage_fallback_reasons") if case else None)
        row[f"{slug}_policy_bets"] = bets
        row[f"{slug}_policy_roi"] = roi
        row[f"{slug}_policy_net"] = _policy_net_or_none(bets, roi)

        if not present:
            shared_all = False
            shared_ok_all = False
            continue
        if status != "ok":
            shared_ok_all = False
            continue

        ok_policy_names.append(policy_name)
        ok_stage_signatures.append(stage_signature)
        ok_fallback_reason_signatures.append(fallback_reason_signature)
        ok_stage_trace_signatures.append(stage_trace_signature)

    row["shared_all"] = shared_all
    row["shared_ok_all"] = shared_ok_all
    row["policy_name_unique_count"] = len(set(ok_policy_names)) if shared_ok_all else None
    row["stage_signature_unique_count"] = len(set(ok_stage_signatures)) if shared_ok_all else None
    row["fallback_reason_signature_unique_count"] = len(set(ok_fallback_reason_signatures)) if shared_ok_all else None
    row["stage_trace_signature_unique_count"] = len(set(ok_stage_trace_signatures)) if shared_ok_all else None
    row["policy_name_same_all"] = shared_ok_all and len(set(ok_policy_names)) <= 1
    row["stage_signature_same_all"] = shared_ok_all and len(set(ok_stage_signatures)) <= 1
    row["fallback_reason_signature_same_all"] = shared_ok_all and len(set(ok_fallback_reason_signatures)) <= 1
    row["stage_trace_signature_same_all"] = shared_ok_all and len(set(ok_stage_trace_signatures)) <= 1
    return row


def _label_summary(label: str, case_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ok_cases = [case for case in case_map.values() if str(case.get("status") or "") == "ok"]
    stage_counts: Counter[str] = Counter()
    stage_trace_counts: Counter[str] = Counter()
    stage_fallback_reason_counts: Counter[str] = Counter()
    total_bets = 0
    total_net = 0.0

    for case in ok_cases:
        total_bets += int(case.get("policy_bets") or 0)
        total_net += float(_policy_net_or_none(_int_or_none(case.get("policy_bets")), _float_or_none(case.get("policy_roi"))) or 0.0)
        for stage_name in _normalize_stage_names(case.get("policy_stage_names")):
            stage_counts[stage_name] += 1
        for stage_trace in _normalize_string_list(case.get("policy_stage_traces")):
            stage_trace_counts[stage_trace] += 1
        for reason in _normalize_string_list(case.get("policy_stage_fallback_reasons")):
            stage_fallback_reason_counts[reason] += 1

    return {
        "label": label,
        "num_cases": int(len(case_map)),
        "num_ok_cases": int(len(ok_cases)),
        "total_policy_bets": int(total_bets),
        "total_policy_net": float(total_net),
        "stage_name_counts": dict(stage_counts),
        "stage_trace_counts": dict(stage_trace_counts),
        "stage_fallback_reason_counts": dict(stage_fallback_reason_counts),
    }


def _build_comparison(summary_specs: list[tuple[str, Path]]) -> tuple[dict[str, Any], pd.DataFrame]:
    loaded: dict[str, dict[str, Any]] = {}
    case_maps: dict[str, dict[str, dict[str, Any]]] = {}
    labels = [label for label, _ in summary_specs]

    for label, path in summary_specs:
        loaded[label] = _load_summary(path)
        case_maps[label] = _case_map(loaded[label], label)

    dates = sorted(set().union(*(set(case_map.keys()) for case_map in case_maps.values())))
    rows = [_build_row(date_value, labels, case_maps) for date_value in dates]
    row_df = pd.DataFrame(rows)

    shared_dates_all = [row["date"] for row in rows if row["shared_all"]]
    shared_ok_dates_all = [row["date"] for row in rows if row["shared_ok_all"]]
    differing_policy_dates = [row["date"] for row in rows if row["shared_ok_all"] and not row["policy_name_same_all"]]
    differing_stage_dates = [row["date"] for row in rows if row["shared_ok_all"] and not row["stage_signature_same_all"]]
    differing_stage_fallback_reason_dates = [
        row["date"] for row in rows if row["shared_ok_all"] and not row["fallback_reason_signature_same_all"]
    ]
    differing_stage_trace_dates = [
        row["date"] for row in rows if row["shared_ok_all"] and not row["stage_trace_signature_same_all"]
    ]

    payload = {
        "summaries": [
            {
                "label": label,
                "summary_file": _display_summary_path(path),
                "artifact_suffix": str(loaded[label].get("artifact_suffix") or ""),
                "profile": str(loaded[label].get("profile") or ""),
                "config_file": str(loaded[label].get("config_file") or ""),
            }
            for label, path in summary_specs
        ],
        "comparison": {
            "num_summaries": int(len(summary_specs)),
            "num_union_dates": int(len(dates)),
            "num_shared_dates_all": int(len(shared_dates_all)),
            "num_shared_ok_dates_all": int(len(shared_ok_dates_all)),
            "shared_dates_all": shared_dates_all,
            "shared_ok_dates_all": shared_ok_dates_all,
            "differing_policy_dates": differing_policy_dates,
            "differing_stage_dates": differing_stage_dates,
            "differing_stage_fallback_reason_dates": differing_stage_fallback_reason_dates,
            "differing_stage_trace_dates": differing_stage_trace_dates,
            "label_summaries": [_label_summary(label, case_maps[label]) for label in labels],
        },
        "rows": rows,
    }
    return payload, row_df


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="append", required=True, help="label=path to serving smoke summary JSON; repeatable")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[serving-stage-compare]", logger=log_progress, min_interval_sec=0.0)
    try:
        summary_specs = [_parse_summary_arg(raw) for raw in args.summary]
        if len(summary_specs) < 2:
            raise ValueError("at least two --summary entries are required")

        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / "serving_stage_path_compare.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / "serving_stage_path_compare.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        progress.start(message=f"loading summaries count={len(summary_specs)}")
        with Heartbeat("[serving-stage-compare]", "building stage comparison", logger=log_progress):
            payload, row_df = _build_comparison(summary_specs)
        progress.update(message=f"comparison built rows={len(row_df)} shared_ok={payload['comparison']['num_shared_ok_dates_all']}")

        with Heartbeat("[serving-stage-compare]", "writing stage comparison outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, row_df, index=False)

        progress.update(
            message=(
                f"outputs saved json={artifact_display_path(output_json, workspace_root=ROOT)} "
                f"csv={artifact_display_path(output_csv, workspace_root=ROOT)}"
            )
        )
        print(f"saved serving stage comparison to {output_json.relative_to(ROOT)}")
        print(f"saved serving stage comparison table to {output_csv.relative_to(ROOT)}")
        print(f"differing_stage_dates={payload['comparison']['differing_stage_dates']}")
        print(f"differing_stage_fallback_reason_dates={payload['comparison']['differing_stage_fallback_reason_dates']}")
        progress.complete(message="serving stage comparison completed")
        return 0
    except KeyboardInterrupt:
        print("[serving-stage-compare] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[serving-stage-compare] failed: {error}")
        return 1
    except Exception as error:
        print(f"[serving-stage-compare] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())