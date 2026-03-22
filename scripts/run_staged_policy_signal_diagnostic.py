from __future__ import annotations

import argparse
from collections import Counter
import json
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
from racing_ml.serving.runtime_policy import _annotate_single_runtime_policy
from racing_ml.serving.runtime_policy import _evaluate_stage_fallback
from racing_ml.serving.runtime_policy import annotate_runtime_policy, resolve_runtime_policy


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[staged-signal-diagnostic {now}] {message}", flush=True)


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


def _string_or_none(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _resolve_staged_policy(config: dict[str, Any], *, date_value: str) -> tuple[str, dict[str, Any], list[dict[str, Any]]]:
    policy_resolution = resolve_runtime_policy(config, frame=pd.DataFrame({"date": [date_value]}))
    if policy_resolution is None:
        raise ValueError("serving runtime policy is missing")
    policy_name, policy_config = policy_resolution
    if str(policy_config.get("strategy_kind", "")).strip().lower() != "staged":
        raise ValueError("serving runtime policy must be staged")
    stages = policy_config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("staged runtime policy must define stages")
    return str(policy_name), policy_config, stages


def _selected_rows(stage_race: pd.DataFrame) -> pd.DataFrame:
    return stage_race[stage_race["policy_selected"].fillna(False).astype(bool)].copy()


def _weighted_selected_return(selected: pd.DataFrame, *, odds_col: str) -> float:
    if selected.empty or odds_col not in selected.columns:
        return 0.0
    winners = selected[pd.to_numeric(selected.get("rank"), errors="coerce") == 1]
    if winners.empty:
        return 0.0
    weights = pd.to_numeric(winners["policy_weight"], errors="coerce").fillna(0.0)
    odds = pd.to_numeric(winners[odds_col], errors="coerce").fillna(0.0)
    return float((weights * odds).sum())


def _race_stage_row(
    stage_race: pd.DataFrame,
    *,
    date_value: str,
    race_id: str,
    stage_index: int,
    stage_name: str,
    stage_cfg: dict[str, Any],
    odds_col: str,
    final_stage_name: str | None,
    final_stage_trace: str | None,
    final_stage_fallback_reasons: str | None,
    stage_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = _selected_rows(stage_race)
    fallback_state = _evaluate_stage_fallback(stage_race, stage_cfg, stage_context=stage_context)
    selected_count = int(fallback_state["selected_count"])
    selected_stake_units = float(pd.to_numeric(selected.get("policy_weight"), errors="coerce").fillna(0.0).sum()) if not selected.empty else 0.0
    selected_return_units = _weighted_selected_return(selected, odds_col=odds_col)
    selected_net_units = float(selected_return_units - selected_stake_units)
    return {
        "date": date_value,
        "race_id": str(race_id),
        "stage_index": int(stage_index),
        "stage_name": stage_name,
        "selected_count": selected_count,
        "selected_stake_units": selected_stake_units,
        "selected_return_units": selected_return_units,
        "selected_net_units": selected_net_units,
        "selected_hit": bool(selected_return_units > 0.0),
        "max_expected_value": _float_or_none(pd.to_numeric(selected.get("policy_expected_value"), errors="coerce").max() if not selected.empty else None),
        "max_prob": _float_or_none(pd.to_numeric(selected.get("policy_prob"), errors="coerce").max() if not selected.empty else None),
        "max_edge": _float_or_none(pd.to_numeric(selected.get("policy_edge"), errors="coerce").max() if not selected.empty else None),
        "fallback": bool(fallback_state["fallback"]),
        "fallback_reasons": list(fallback_state["reasons"]),
        "final_stage_name": final_stage_name,
        "final_stage_trace": final_stage_trace,
        "final_stage_fallback_reasons": final_stage_fallback_reasons,
        "selected_horse_ids": [str(value) for value in selected.get("horse_id", pd.Series(dtype=object)).tolist()],
        "selected_horse_names": [str(value) for value in selected.get("horse_name", pd.Series(dtype=object)).tolist()],
        "stage_is_final": bool(final_stage_name == stage_name and selected_count > 0 and not fallback_state["fallback"]),
        "ev_guard": _float_or_none((stage_cfg.get("fallback_when") or {}).get("max_expected_value_below") if isinstance(stage_cfg.get("fallback_when"), dict) else None),
        "prob_guard": _float_or_none((stage_cfg.get("fallback_when") or {}).get("max_prob_below") if isinstance(stage_cfg.get("fallback_when"), dict) else None),
        "edge_guard": _float_or_none((stage_cfg.get("fallback_when") or {}).get("max_edge_below") if isinstance(stage_cfg.get("fallback_when"), dict) else None),
        "selected_rows_guard": _float_or_none((stage_cfg.get("fallback_when") or {}).get("selected_rows_at_most") if isinstance(stage_cfg.get("fallback_when"), dict) else None),
        "date_selected_rows_guard": _float_or_none((stage_cfg.get("fallback_when") or {}).get("date_selected_rows_at_most") if isinstance(stage_cfg.get("fallback_when"), dict) else None),
        "date_selected_count": _float_or_none((stage_context or {}).get("date_selected_count")),
    }


def _aggregate_stage_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fallback_reason_counts: Counter[str] = Counter()
    final_stage_counts: Counter[str] = Counter()
    fallback_race_count = 0
    no_selection_race_count = 0
    selected_race_count = 0
    hit_race_count = 0
    stage_is_final_count = 0
    total_selected_stake_units = 0.0
    total_selected_return_units = 0.0
    total_selected_net_units = 0.0
    max_expected_values: list[float] = []
    max_probs: list[float] = []
    max_edges: list[float] = []
    ev_guard_pass_race_count = 0

    for row in rows:
        selected_count = int(row.get("selected_count") or 0)
        if selected_count > 0:
            selected_race_count += 1
        else:
            no_selection_race_count += 1
        if bool(row.get("fallback")):
            fallback_race_count += 1
        if bool(row.get("selected_hit")):
            hit_race_count += 1
        if bool(row.get("stage_is_final")):
            stage_is_final_count += 1
        total_selected_stake_units += float(row.get("selected_stake_units") or 0.0)
        total_selected_return_units += float(row.get("selected_return_units") or 0.0)
        total_selected_net_units += float(row.get("selected_net_units") or 0.0)

        max_expected_value = _float_or_none(row.get("max_expected_value"))
        if max_expected_value is not None:
            max_expected_values.append(max_expected_value)
            ev_guard = _float_or_none(row.get("ev_guard"))
            if ev_guard is None or max_expected_value >= ev_guard:
                ev_guard_pass_race_count += 1
        max_prob = _float_or_none(row.get("max_prob"))
        if max_prob is not None:
            max_probs.append(max_prob)
        max_edge = _float_or_none(row.get("max_edge"))
        if max_edge is not None:
            max_edges.append(max_edge)

        for reason in row.get("fallback_reasons") or []:
            fallback_reason_counts[str(reason)] += 1
        final_stage_name = _string_or_none(row.get("final_stage_name"))
        if final_stage_name is not None:
            final_stage_counts[final_stage_name] += 1

    num_races = len(rows)
    return {
        "num_races": int(num_races),
        "selected_race_count": int(selected_race_count),
        "fallback_race_count": int(fallback_race_count),
        "no_selection_race_count": int(no_selection_race_count),
        "hit_race_count": int(hit_race_count),
        "stage_is_final_count": int(stage_is_final_count),
        "total_selected_stake_units": float(total_selected_stake_units),
        "total_selected_return_units": float(total_selected_return_units),
        "total_selected_net_units": float(total_selected_net_units),
        "mean_selected_count": float(pd.Series([int(row.get("selected_count") or 0) for row in rows], dtype=float).mean()) if rows else 0.0,
        "ev_guard_pass_race_count": int(ev_guard_pass_race_count),
        "fallback_reason_counts": dict(fallback_reason_counts),
        "final_stage_counts": dict(final_stage_counts),
        "max_expected_value_min": float(min(max_expected_values)) if max_expected_values else None,
        "max_expected_value_median": float(pd.Series(max_expected_values, dtype=float).median()) if max_expected_values else None,
        "max_expected_value_max": float(max(max_expected_values)) if max_expected_values else None,
        "max_prob_min": float(min(max_probs)) if max_probs else None,
        "max_prob_median": float(pd.Series(max_probs, dtype=float).median()) if max_probs else None,
        "max_prob_max": float(max(max_probs)) if max_probs else None,
        "max_edge_min": float(min(max_edges)) if max_edges else None,
        "max_edge_median": float(pd.Series(max_edges, dtype=float).median()) if max_edges else None,
        "max_edge_max": float(max(max_edges)) if max_edges else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_staged_mitigation_ev_guard_probe.yaml",
    )
    parser.add_argument("--date", action="append", required=True)
    parser.add_argument("--window-label", required=True)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    progress = ProgressBar(total=4, prefix="[staged-signal-diagnostic]", logger=log_progress, min_interval_sec=0.0)
    try:
        config_path = _resolve_path(args.config)
        config = _load_yaml(config_path)
        dates = [str(pd.Timestamp(date_value).date()) for date_value in (args.date or [])]

        output_json = _resolve_path(args.output_json) if args.output_json else ROOT / "artifacts" / "reports" / f"staged_policy_signal_diagnostic_{args.window_label}.json"
        output_csv = _resolve_path(args.output_csv) if args.output_csv else ROOT / "artifacts" / "reports" / f"staged_policy_signal_diagnostic_{args.window_label}.csv"
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        prediction_paths = {date_value: _prediction_path_for_date(date_value) for date_value in dates}
        for date_value, prediction_path in prediction_paths.items():
            if not prediction_path.exists():
                raise FileNotFoundError(f"prediction CSV not found for {date_value}: {prediction_path}")

        progress.start(message=f"loaded config and dates count={len(dates)}")

        raw_rows: list[dict[str, Any]] = []
        summary_rows: list[dict[str, Any]] = []
        date_progress = ProgressBar(total=max(len(dates), 1), prefix="[staged-signal-diagnostic dates]", logger=log_progress, min_interval_sec=0.0)
        date_progress.start(message="replaying staged runtime diagnostics")

        for date_value in dates:
            prediction_path = prediction_paths[date_value]
            with Heartbeat("[staged-signal-diagnostic]", f"processing {date_value}", logger=log_progress):
                frame = pd.read_csv(prediction_path)
                odds_col = resolve_odds_column(frame)
                if odds_col is None:
                    raise ValueError(f"odds column not found: {prediction_path}")
                policy_name, policy_config, stages = _resolve_staged_policy(config, date_value=date_value)

                annotated = annotate_runtime_policy(frame, odds_col=odds_col, policy_name=policy_name, policy_config=policy_config, score_col="score")
                stage_results: list[tuple[int, str, dict[str, Any], pd.DataFrame]] = []
                for stage_index, stage_cfg in enumerate(stages, start=1):
                    stage_name = str(stage_cfg.get("name", f"stage_{stage_index}")).strip() or f"stage_{stage_index}"
                    stage_policy = stage_cfg.get("policy") if isinstance(stage_cfg.get("policy"), dict) else stage_cfg
                    if not isinstance(stage_policy, dict):
                        continue
                    stage_result = _annotate_single_runtime_policy(frame, odds_col=odds_col, policy_name=stage_name, policy_config=stage_policy, score_col="score")
                    stage_selected_mask = stage_result["policy_selected"].fillna(False).astype(bool)
                    stage_context = {
                        "date_selected_count": int(stage_selected_mask.sum()),
                    }
                    stage_results.append((stage_index, stage_name, stage_cfg, stage_result, stage_context))

                date_stage_rows: dict[str, list[dict[str, Any]]] = {str(stage_cfg.get("name", f"stage_{index}")): [] for index, stage_cfg in enumerate(stages, start=1)}
                for race_id, race_group in frame.groupby("race_id", sort=False):
                    race_index = race_group.index
                    race_annotated = annotated.loc[race_index]
                    selected_final = race_annotated[race_annotated["policy_selected"].fillna(False).astype(bool)]
                    final_stage_name = _string_or_none(selected_final["policy_stage_name"].dropna().astype(str).iloc[0] if not selected_final.empty and "policy_stage_name" in selected_final.columns else None)
                    final_stage_trace = _string_or_none(race_annotated["policy_stage_trace"].dropna().astype(str).iloc[0] if "policy_stage_trace" in race_annotated.columns and race_annotated["policy_stage_trace"].dropna().any() else None)
                    final_stage_fallback_reasons = _string_or_none(race_annotated["policy_stage_fallback_reasons"].dropna().astype(str).iloc[0] if "policy_stage_fallback_reasons" in race_annotated.columns and race_annotated["policy_stage_fallback_reasons"].dropna().any() else None)

                    for stage_index, stage_name, stage_cfg, stage_result, stage_context in stage_results:
                        row = _race_stage_row(
                            stage_result.loc[race_index],
                            date_value=date_value,
                            race_id=str(race_id),
                            stage_index=stage_index,
                            stage_name=stage_name,
                            stage_cfg=stage_cfg,
                            odds_col=odds_col,
                            final_stage_name=final_stage_name,
                            final_stage_trace=final_stage_trace,
                            final_stage_fallback_reasons=final_stage_fallback_reasons,
                            stage_context=stage_context,
                        )
                        raw_rows.append(row)
                        date_stage_rows[stage_name].append(row)

                for stage_index, stage_cfg in enumerate(stages, start=1):
                    stage_name = str(stage_cfg.get("name", f"stage_{stage_index}")).strip() or f"stage_{stage_index}"
                    summary = {
                        "date": date_value,
                        "stage_index": int(stage_index),
                        "stage_name": stage_name,
                        "config": artifact_display_path(config_path, workspace_root=ROOT),
                        "prediction_file": artifact_display_path(prediction_path, workspace_root=ROOT),
                    }
                    summary.update(_aggregate_stage_rows(date_stage_rows.get(stage_name, [])))
                    summary_rows.append(summary)

            date_progress.update(message=f"processed {date_value}")

        progress.update(message=f"date diagnostics complete count={len(summary_rows)}")

        summary_df = pd.DataFrame(summary_rows).sort_values(["date", "stage_index"], ascending=[True, True])
        raw_df = pd.DataFrame(raw_rows).sort_values(["date", "race_id", "stage_index"], ascending=[True, True, True])
        payload = {
            "config": artifact_display_path(config_path, workspace_root=ROOT),
            "window_label": args.window_label,
            "dates": dates,
            "summary_rows": summary_rows,
            "raw_row_count": int(len(raw_rows)),
            "summary_row_count": int(len(summary_rows)),
            "stage_names": sorted({str(row.get("stage_name") or "") for row in summary_rows if str(row.get("stage_name") or "").strip()}),
        }

        with Heartbeat("[staged-signal-diagnostic]", "writing outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, raw_df, index=False)
            write_csv_file(output_csv.with_name(output_csv.stem + "_summary.csv"), summary_df, index=False)

        print(f"saved staged signal diagnostic json to {output_json.relative_to(ROOT)}")
        print(f"saved staged signal diagnostic csv to {output_csv.relative_to(ROOT)}")
        progress.complete(message="staged policy signal diagnostic completed")
        return 0
    except KeyboardInterrupt:
        print("[staged-signal-diagnostic] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError, RuntimeError) as error:
        print(f"[staged-signal-diagnostic] failed: {error}")
        return 1
    except Exception as error:
        print(f"[staged-signal-diagnostic] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())