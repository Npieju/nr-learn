from __future__ import annotations

import argparse
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
from racing_ml.evaluation.scoring import resolve_odds_column
from racing_ml.serving.runtime_policy import annotate_runtime_policy, resolve_runtime_policy


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[policy-date-signal {now}] {message}", flush=True)


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


def _prediction_path_for_date(date_value: str) -> Path:
    compact = pd.Timestamp(date_value).strftime("%Y%m%d")
    return ROOT / "artifacts" / "predictions" / f"predictions_{compact}.csv"


def _float_or_none(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _selected_signal_summary(selected: pd.DataFrame) -> dict[str, Any]:
    ev = pd.to_numeric(selected.get("policy_expected_value"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    edge = pd.to_numeric(selected.get("policy_edge"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
    prob = pd.to_numeric(selected.get("policy_prob"), errors="coerce") if not selected.empty else pd.Series(dtype=float)

    return {
        "selected_count": int(len(selected)),
        "ev_min": _float_or_none(ev.min() if not ev.empty else None),
        "ev_mean": _float_or_none(ev.mean() if not ev.empty else None),
        "ev_median": _float_or_none(ev.median() if not ev.empty else None),
        "ev_max": _float_or_none(ev.max() if not ev.empty else None),
        "edge_min": _float_or_none(edge.min() if not edge.empty else None),
        "edge_mean": _float_or_none(edge.mean() if not edge.empty else None),
        "edge_median": _float_or_none(edge.median() if not edge.empty else None),
        "edge_max": _float_or_none(edge.max() if not edge.empty else None),
        "prob_min": _float_or_none(prob.min() if not prob.empty else None),
        "prob_mean": _float_or_none(prob.mean() if not prob.empty else None),
        "prob_median": _float_or_none(prob.median() if not prob.empty else None),
        "prob_max": _float_or_none(prob.max() if not prob.empty else None),
        "count_ev_ge_1_0": int((ev >= 1.0).sum()) if not ev.empty else 0,
        "count_ev_ge_1_01": int((ev >= 1.01).sum()) if not ev.empty else 0,
        "count_edge_pos": int((edge > 0).sum()) if not edge.empty else 0,
        "count_edge_ge_0_01": int((edge >= 0.01).sum()) if not edge.empty else 0,
        "share_ev_ge_1_0": _float_or_none((ev >= 1.0).mean() if not ev.empty else None),
        "share_edge_pos": _float_or_none((edge > 0).mean() if not edge.empty else None),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
    )
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--window-label", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[policy-date-signal]", logger=log_progress, min_interval_sec=0.0)
    try:
        config_path = _resolve_path(args.config)
        config = _load_yaml(config_path)
        dates = [str(pd.Timestamp(date_value).date()) for date_value in (args.date or [])]
        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"policy_date_signal_report_{args.window_label}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"policy_date_signal_report_{args.window_label}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        prediction_paths = {date_value: _prediction_path_for_date(date_value) for date_value in dates}
        for date_value, prediction_path in prediction_paths.items():
            if not prediction_path.exists():
                raise FileNotFoundError(f"prediction CSV not found for {date_value}: {prediction_path}")

        progress.start(message=f"loaded config and dates count={len(dates)}")

        rows: list[dict[str, Any]] = []
        date_progress = ProgressBar(total=max(len(dates), 1), prefix="[policy-date-signal dates]", logger=log_progress, min_interval_sec=0.0)
        date_progress.start(message="building per-date signal report")

        for date_value in dates:
            prediction_path = prediction_paths[date_value]
            with Heartbeat("[policy-date-signal]", f"processing {date_value}", logger=log_progress):
                frame = pd.read_csv(prediction_path)
                odds_col = resolve_odds_column(frame)
                if odds_col is None:
                    raise ValueError(f"odds column not found: {prediction_path}")
                policy_resolution = resolve_runtime_policy(config, frame=frame)
                if policy_resolution is None:
                    raise ValueError(f"runtime policy not found: {config_path}")
                policy_name, policy_config = policy_resolution
                annotated = annotate_runtime_policy(frame, odds_col=odds_col, policy_name=policy_name, policy_config=policy_config, score_col="score")
                selected = annotated[annotated["policy_selected"].fillna(False).astype(bool)].copy()
                rank = pd.to_numeric(selected.get("rank"), errors="coerce") if not selected.empty else pd.Series(dtype=float)
                odds = pd.to_numeric(selected.get(odds_col), errors="coerce") if not selected.empty else pd.Series(dtype=float)
                weights = pd.to_numeric(selected.get("policy_weight"), errors="coerce").fillna(0.0) if not selected.empty else pd.Series(dtype=float)
                realized_return_units = float((weights[rank == 1] * odds[rank == 1]).sum()) if not selected.empty else 0.0
                realized_stake_units = float(weights.sum()) if not selected.empty else 0.0
                row = {
                    "date": date_value,
                    "config": artifact_display_path(config_path, workspace_root=ROOT),
                    "prediction_file": artifact_display_path(prediction_path, workspace_root=ROOT),
                    "policy_name": str(policy_name),
                    "policy_strategy_kind": str(policy_config.get("strategy_kind") or ""),
                    "policy_bets": int(len(selected)),
                    "policy_return_units": realized_return_units,
                    "policy_net_units": float(realized_return_units - realized_stake_units),
                    "policy_stage_names": selected["policy_stage_name"].dropna().astype(str).unique().tolist() if "policy_stage_name" in selected.columns else [],
                    "policy_stage_fallback_reasons": annotated["policy_stage_fallback_reasons"].dropna().astype(str).unique().tolist() if "policy_stage_fallback_reasons" in annotated.columns else [],
                }
                row.update(_selected_signal_summary(selected))
                rows.append(row)
            date_progress.update(message=f"processed {date_value}")

        progress.update(message=f"report rows complete count={len(rows)}")

        rows_df = pd.DataFrame(rows).sort_values(["date"], ascending=[True])
        payload = {
            "config": artifact_display_path(config_path, workspace_root=ROOT),
            "window_label": args.window_label,
            "dates": dates,
            "rows": rows,
        }

        with Heartbeat("[policy-date-signal]", "writing outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, rows_df, index=False)

        print(f"saved policy date signal report json to {output_json.relative_to(ROOT)}")
        print(f"saved policy date signal report csv to {output_csv.relative_to(ROOT)}")
        progress.complete(message="policy date signal report completed")
        return 0
    except KeyboardInterrupt:
        print("[policy-date-signal] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[policy-date-signal] failed: {error}")
        return 1
    except Exception as error:
        print(f"[policy-date-signal] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())