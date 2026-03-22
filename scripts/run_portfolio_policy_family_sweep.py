from __future__ import annotations

import argparse
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
from racing_ml.serving.runtime_policy import resolve_runtime_policy


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[portfolio-family-sweep {now}] {message}", flush=True)


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


def _normalize_float_list(raw_values: list[str] | None, *, default: list[float]) -> list[float]:
    if not raw_values:
        return list(default)
    values: list[float] = []
    for raw in raw_values:
        for part in str(raw).split(","):
            text = part.strip()
            if not text:
                continue
            values.append(float(text))
    if not values:
        raise ValueError("parameter list is empty")
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


def _base_policy_for_date(base_config: dict[str, Any], date_value: str) -> tuple[str, dict[str, Any]]:
    frame = pd.DataFrame({"date": [str(pd.Timestamp(date_value).date())]})
    policy_resolution = resolve_runtime_policy(base_config, frame=frame)
    if policy_resolution is None:
        raise ValueError("runtime policy is missing")
    policy_name, policy_config = policy_resolution
    if str(policy_config.get("strategy_kind", "")).strip().lower() != "portfolio":
        raise ValueError("resolved runtime policy must be portfolio")
    return policy_name, deepcopy(policy_config)


def _variant_config(
    base_config: dict[str, Any],
    *,
    policy_name: str,
    base_policy: dict[str, Any],
    blend_weight: float,
    min_prob: float,
    min_expected_value: float,
) -> dict[str, Any]:
    config = deepcopy(base_config)
    serving = config.get("serving") if isinstance(config.get("serving"), dict) else {}
    serving = deepcopy(serving)
    policy = deepcopy(base_policy)
    policy["name"] = policy_name
    policy["blend_weight"] = float(blend_weight)
    policy["min_prob"] = float(min_prob)
    policy["min_expected_value"] = float(min_expected_value)
    policy["strategy_kind"] = "portfolio"
    serving["policy"] = policy
    serving.pop("policy_regime_overrides", None)
    config["serving"] = serving
    return config


def _variant_label(blend_weight: float, min_prob: float, min_expected_value: float) -> str:
    return f"blend_{blend_weight:.2f}_prob_{min_prob:.2f}_ev_{min_expected_value:.2f}"


def _aggregate_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_bets = 0
    total_net = 0.0
    profitable_dates: list[str] = []
    losing_dates: list[str] = []
    zero_bet_dates: list[str] = []

    for row in rows:
        bets = int(row.get("policy_bets") or 0)
        net = float(row.get("policy_net") or 0.0)
        total_bets += bets
        total_net += net
        if bets <= 0:
            zero_bet_dates.append(str(row["date"]))
        elif net > 0.0:
            profitable_dates.append(str(row["date"]))
        else:
            losing_dates.append(str(row["date"]))

    return {
        "num_dates": int(len(rows)),
        "total_policy_bets": int(total_bets),
        "total_policy_net": float(total_net),
        "profitable_dates": sorted(profitable_dates),
        "losing_dates": sorted(losing_dates),
        "zero_bet_dates": sorted(zero_bet_dates),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
    )
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--window-label", required=True)
    parser.add_argument("--reference-date", default=None)
    parser.add_argument("--blend-weight", action="append", default=None)
    parser.add_argument("--min-prob", action="append", default=None)
    parser.add_argument("--min-expected-value", action="append", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[portfolio-family-sweep]", logger=log_progress, min_interval_sec=0.0)
    try:
        base_config_path = _resolve_path(args.base_config)
        base_config = _load_yaml(base_config_path)

        dates = [str(pd.Timestamp(date_value).date()) for date_value in (args.date or [])]
        if not dates:
            raise ValueError("at least one date is required")
        reference_date = str(pd.Timestamp(args.reference_date).date()) if args.reference_date else dates[0]
        resolved_policy_name, resolved_policy = _base_policy_for_date(base_config, reference_date)

        blend_weights = _normalize_float_list(args.blend_weight, default=[0.8, 0.6])
        min_probs = _normalize_float_list(args.min_prob, default=[0.03, 0.05])
        min_expected_values = _normalize_float_list(args.min_expected_value, default=[0.95, 1.0])

        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"portfolio_policy_family_sweep_{args.window_label}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"portfolio_policy_family_sweep_{args.window_label}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        prediction_paths = {date_value: _prediction_path_for_date(date_value) for date_value in dates}
        for date_value, prediction_path in prediction_paths.items():
            if not prediction_path.exists():
                raise FileNotFoundError(f"prediction CSV not found for {date_value}: {prediction_path}")

        progress.start(message=f"loaded config and dates count={len(dates)}")

        variant_rows: list[dict[str, Any]] = []
        variant_summaries: list[dict[str, Any]] = []
        total_variants = max(len(blend_weights) * len(min_probs) * len(min_expected_values), 1)
        variant_progress = ProgressBar(total=total_variants, prefix="[portfolio-family-sweep variants]", logger=log_progress, min_interval_sec=0.0)
        variant_progress.start(message="evaluating portfolio policy family variants")

        for blend_weight in blend_weights:
            for min_prob in min_probs:
                for min_expected_value in min_expected_values:
                    label = _variant_label(blend_weight, min_prob, min_expected_value)
                    config = _variant_config(
                        base_config,
                        policy_name=f"{resolved_policy_name}_{label}",
                        base_policy=resolved_policy,
                        blend_weight=blend_weight,
                        min_prob=min_prob,
                        min_expected_value=min_expected_value,
                    )
                    per_date_rows: list[dict[str, Any]] = []
                    with Heartbeat("[portfolio-family-sweep]", f"evaluating {label}", logger=log_progress):
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
                                "blend_weight": float(blend_weight),
                                "min_prob": float(min_prob),
                                "min_expected_value": float(min_expected_value),
                                "date": date_value,
                                "policy_name": str(metrics.get("policy_name") or ""),
                                "policy_bets": int(metrics.get("policy_bets") or 0),
                                "policy_roi": metrics.get("policy_roi"),
                                "policy_net": _policy_net_or_none(metrics),
                                "policy_final_bankroll": metrics.get("policy_final_bankroll"),
                            }
                            per_date_rows.append(row)
                            variant_rows.append(row)

                    summary = {
                        "variant": label,
                        "blend_weight": float(blend_weight),
                        "min_prob": float(min_prob),
                        "min_expected_value": float(min_expected_value),
                    }
                    summary.update(_aggregate_variant(per_date_rows))
                    variant_summaries.append(summary)
                    variant_progress.update(message=f"{label} net={summary['total_policy_net']:.4f} bets={summary['total_policy_bets']}")

        progress.update(message=f"variants evaluated count={len(variant_summaries)}")

        summary_df = pd.DataFrame(variant_summaries).sort_values(
            ["total_policy_net", "total_policy_bets", "blend_weight", "min_prob", "min_expected_value"],
            ascending=[False, True, False, True, True],
        )
        rows_df = pd.DataFrame(variant_rows)
        payload = {
            "base_config": artifact_display_path(base_config_path, workspace_root=ROOT),
            "reference_date": reference_date,
            "resolved_policy_name": resolved_policy_name,
            "resolved_policy": resolved_policy,
            "window_label": args.window_label,
            "dates": dates,
            "blend_weights": blend_weights,
            "min_probs": min_probs,
            "min_expected_values": min_expected_values,
            "variant_summaries": variant_summaries,
            "rows": variant_rows,
        }

        with Heartbeat("[portfolio-family-sweep]", "writing sweep outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, rows_df, index=False)
            write_csv_file(output_csv.with_name(output_csv.stem + "_summary.csv"), summary_df, index=False)

        best = summary_df.iloc[0].to_dict() if not summary_df.empty else {}
        print(f"saved portfolio family sweep json to {output_json.relative_to(ROOT)}")
        print(f"saved portfolio family sweep csv to {output_csv.relative_to(ROOT)}")
        if best:
            print(
                "best_variant="
                f"{best.get('variant')} net={float(best.get('total_policy_net') or 0.0):.4f} "
                f"bets={int(best.get('total_policy_bets') or 0)} profitable_dates={best.get('profitable_dates')}"
            )
        progress.complete(message="portfolio policy family sweep completed")
        return 0
    except KeyboardInterrupt:
        print("[portfolio-family-sweep] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[portfolio-family-sweep] failed: {error}")
        return 1
    except Exception as error:
        print(f"[portfolio-family-sweep] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())