from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import write_csv_file, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.serving.replay_backtest import compute_prediction_backtest_metrics


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[staged-date-rows-sweep {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def _normalize_int_list(raw_values: list[str] | None, *, default: list[int]) -> list[int]:
    if not raw_values:
        return list(default)
    values: list[int] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            text = part.strip()
            if not text:
                continue
            values.append(int(text))
    if not values:
        raise ValueError("threshold list is empty")
    return values


def _prediction_path_for_date(date_value: str) -> Path:
    compact = pd.Timestamp(date_value).strftime("%Y%m%d")
    return ROOT / "artifacts" / "predictions" / f"predictions_{compact}.csv"


def _policy_net_or_none(metrics: dict[str, Any]) -> float | None:
    bets = metrics.get("policy_bets")
    roi = metrics.get("policy_roi")
    if bets is None:
        return None
    try:
        bets_value = int(bets)
    except (TypeError, ValueError):
        return None
    if bets_value == 0:
        return 0.0
    if roi is None:
        return None
    try:
        return float((bets_value * float(roi)) - bets_value)
    except (TypeError, ValueError):
        return None


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def _stage_policy(config: dict[str, Any]) -> list[dict[str, Any]]:
    serving = config.get("serving")
    if not isinstance(serving, dict):
        raise ValueError("serving config is missing")
    overrides = serving.get("policy_regime_overrides")
    if isinstance(overrides, list):
        for override in overrides:
            if not isinstance(override, dict):
                continue
            policy = override.get("policy")
            if not isinstance(policy, dict):
                continue
            if str(policy.get("strategy_kind", "")).strip().lower() == "staged":
                stages = policy.get("stages")
                if isinstance(stages, list) and stages:
                    return stages
    policy = serving.get("policy")
    if not isinstance(policy, dict):
        raise ValueError("serving.policy is missing")
    if str(policy.get("strategy_kind", "")).strip().lower() != "staged":
        raise ValueError("serving policy must be staged")
    stages = policy.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("staged policy must contain stages")
    return stages


def _variant_config(base_config: dict[str, Any], threshold: int) -> dict[str, Any]:
    config = deepcopy(base_config)
    stages = _stage_policy(config)
    stage1 = stages[0]
    fallback_when = stage1.get("fallback_when") if isinstance(stage1.get("fallback_when"), dict) else {}
    fallback_when = dict(fallback_when)
    fallback_when["date_selected_rows_at_most"] = int(threshold)
    stage1["fallback_when"] = fallback_when
    return config


def _variant_label(threshold: int) -> str:
    return f"date_rows_at_most_{threshold}"


def _aggregate_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_bets = 0
    total_net = 0.0
    stage_name_counts: Counter[str] = Counter()
    fallback_reason_counts: Counter[str] = Counter()
    stage_trace_counts: Counter[str] = Counter()
    kelly_dates: list[str] = []
    reason_dates: list[str] = []

    for row in rows:
        total_bets += int(row.get("policy_bets") or 0)
        total_net += float(row.get("policy_net") or 0.0)
        stage_names = _string_list(row.get("policy_stage_names"))
        fallback_reasons = _string_list(row.get("policy_stage_fallback_reasons"))
        stage_traces = _string_list(row.get("policy_stage_traces"))
        for stage_name in stage_names:
            stage_name_counts[stage_name] += 1
        for reason in fallback_reasons:
            fallback_reason_counts[reason] += 1
        for trace in stage_traces:
            stage_trace_counts[trace] += 1
        if any(stage_name.startswith("kelly_fallback") for stage_name in stage_names):
            kelly_dates.append(str(row["date"]))
        if fallback_reasons:
            reason_dates.append(str(row["date"]))

    return {
        "num_dates": int(len(rows)),
        "total_policy_bets": int(total_bets),
        "total_policy_net": float(total_net),
        "kelly_fallback_dates": sorted(kelly_dates),
        "fallback_reason_dates": sorted(reason_dates),
        "stage_name_counts": dict(stage_name_counts),
        "stage_trace_counts": dict(stage_trace_counts),
        "stage_fallback_reason_counts": dict(fallback_reason_counts),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_sep_baseline_date_selected_rows_kelly_candidate.yaml",
    )
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--window-label", required=True)
    parser.add_argument("--date-selected-rows-threshold", action="append", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[staged-date-rows-sweep]", logger=log_progress, min_interval_sec=0.0)
    try:
        base_config_path = _resolve_path(args.base_config)
        base_config = _load_yaml(base_config_path)
        _stage_policy(base_config)

        dates = [str(pd.Timestamp(date_value).date()) for date_value in (args.date or [])]
        thresholds = _normalize_int_list(args.date_selected_rows_threshold, default=[2, 3, 4, 5, 6, 7, 8])
        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"staged_date_selected_rows_sweep_{args.window_label}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"staged_date_selected_rows_sweep_{args.window_label}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        prediction_paths = {date_value: _prediction_path_for_date(date_value) for date_value in dates}
        for date_value, prediction_path in prediction_paths.items():
            if not prediction_path.exists():
                raise FileNotFoundError(f"prediction CSV not found for {date_value}: {prediction_path}")

        progress.start(message=f"loaded config and dates count={len(dates)}")

        variant_rows: list[dict[str, Any]] = []
        variant_summaries: list[dict[str, Any]] = []
        variant_progress = ProgressBar(total=max(len(thresholds), 1), prefix="[staged-date-rows-sweep variants]", logger=log_progress, min_interval_sec=0.0)
        variant_progress.start(message="evaluating date-selected-rows thresholds")

        for threshold in thresholds:
            label = _variant_label(threshold)
            config = _variant_config(base_config, threshold)
            per_date_rows: list[dict[str, Any]] = []
            with Heartbeat("[staged-date-rows-sweep]", f"evaluating {label}", logger=log_progress):
                for date_value in dates:
                    prediction_path = prediction_paths[date_value]
                    frame = pd.read_csv(prediction_path)
                    metrics = compute_prediction_backtest_metrics(
                        frame,
                        model_config=config,
                        config_path=base_config_path,
                        prediction_path=prediction_path,
                        workspace_root=ROOT,
                    )
                    row = {
                        "window_label": args.window_label,
                        "variant": label,
                        "date_selected_rows_at_most": int(threshold),
                        "date": date_value,
                        "policy_name": str(metrics.get("policy_name") or ""),
                        "policy_bets": int(metrics.get("policy_bets") or 0),
                        "policy_roi": metrics.get("policy_roi"),
                        "policy_net": _policy_net_or_none(metrics),
                        "policy_final_bankroll": metrics.get("policy_final_bankroll"),
                        "policy_stage_names": _string_list(metrics.get("policy_stage_names")),
                        "policy_stage_traces": _string_list(metrics.get("policy_stage_traces")),
                        "policy_stage_fallback_reasons": _string_list(metrics.get("policy_stage_fallback_reasons")),
                    }
                    per_date_rows.append(row)
                    variant_rows.append(row)

            summary = {
                "variant": label,
                "date_selected_rows_at_most": int(threshold),
            }
            summary.update(_aggregate_variant(per_date_rows))
            variant_summaries.append(summary)
            variant_progress.update(message=f"{label} net={summary['total_policy_net']:.4f} bets={summary['total_policy_bets']}")

        progress.update(message=f"variants evaluated count={len(variant_summaries)}")

        summary_df = pd.DataFrame(variant_summaries).sort_values(
            ["total_policy_net", "total_policy_bets", "date_selected_rows_at_most"],
            ascending=[False, True, True],
        )
        rows_df = pd.DataFrame(variant_rows)
        payload = {
            "base_config": artifact_display_path(base_config_path, workspace_root=ROOT),
            "window_label": args.window_label,
            "dates": dates,
            "date_selected_rows_thresholds": thresholds,
            "variant_summaries": variant_summaries,
            "rows": variant_rows,
        }

        with Heartbeat("[staged-date-rows-sweep]", "writing sweep outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, rows_df, index=False)
            write_csv_file(output_csv.with_name(output_csv.stem + "_summary.csv"), summary_df, index=False)

        best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        print(f"saved staged date-selected-rows sweep json to {output_json.relative_to(ROOT)}")
        print(f"saved staged date-selected-rows sweep csv to {output_csv.relative_to(ROOT)}")
        if best:
            print(
                "best_variant="
                f"{best.get('variant')} net={float(best.get('total_policy_net') or 0.0):.4f} "
                f"bets={int(best.get('total_policy_bets') or 0)} kelly_dates={best.get('kelly_fallback_dates')}"
            )
        progress.complete(message="staged date-selected-rows threshold sweep completed")
        return 0
    except KeyboardInterrupt:
        print("[staged-date-rows-sweep] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[staged-date-rows-sweep] failed: {error}")
        return 1
    except Exception as error:
        print(f"[staged-date-rows-sweep] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())