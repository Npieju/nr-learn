from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import read_json, resolve_output_artifacts, write_csv_file, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table_for_feature_build
from racing_ml.evaluation.policy import add_market_signals
from racing_ml.evaluation.scoring import generate_prediction_outputs, prepare_scored_frame, resolve_odds_column
from racing_ml.evaluation.walk_forward import fit_isotonic
from racing_ml.features.builder import build_features
from racing_ml.features.selection import prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection
from racing_ml.serving.runtime_policy import annotate_runtime_policy


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[wf-date-report {now}] {message}", flush=True)


def _resolve_path(path_text: str | Path) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else (ROOT / path)


def _sanitize_output_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "report"


def _build_policy_config(fold_row: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    strategy_kind = str(fold_row.get("strategy_kind") or "").strip().lower()
    if strategy_kind not in {"kelly", "portfolio"}:
        raise ValueError(f"unsupported fold strategy_kind: {strategy_kind}")

    policy: dict[str, Any] = {
        "strategy_kind": strategy_kind,
        "blend_weight": float(fold_row.get("blend_weight") or 0.0),
        "min_prob": float(fold_row.get("min_prob") or 0.0),
        "odds_min": float(fold_row.get("odds_min") or 1.0),
        "odds_max": float(fold_row.get("odds_max") or 999.0),
    }
    if strategy_kind == "kelly":
        policy.update(
            {
                "min_edge": float(fold_row.get("min_edge") or 0.0),
                "fractional_kelly": float(fold_row.get("fractional_kelly") or 0.25),
                "max_fraction": float(fold_row.get("max_fraction") or 0.02),
            }
        )
    else:
        policy.update(
            {
                "top_k": int(fold_row.get("top_k") or 1),
                "min_expected_value": float(fold_row.get("min_expected_value") or 1.0),
            }
        )
    policy_name = f"fold_{int(fold_row.get('fold') or 0)}_{strategy_kind}"
    return policy_name, policy


def _slice_by_window(frame: pd.DataFrame, *, start_date: str, end_date: str) -> pd.DataFrame:
    date_series = pd.to_datetime(frame["date"], errors="coerce")
    mask = date_series.notna() & (date_series >= pd.Timestamp(start_date)) & (date_series <= pd.Timestamp(end_date))
    sliced = frame.loc[mask].copy()
    if sliced.empty:
        raise RuntimeError(f"no rows in window {start_date}..{end_date}")
    return sliced


def _median_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return float(pd.Series(values, dtype=float).median())


def _selected_rows_bucket(value: int) -> str:
    if value <= 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 4:
        return "3-4"
    if value <= 8:
        return "5-8"
    return "9+"


def _field_size_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value <= 10:
        return "<=10"
    if value <= 14:
        return "11-14"
    return "15+"


def _mean_odds_bucket(value: float | None) -> str:
    if value is None:
        return "unknown"
    if value < 3.0:
        return "<3"
    if value < 5.0:
        return "3-5"
    if value < 10.0:
        return "5-10"
    return "10+"


def _aggregate_negative_date_breakdown(rows: list[dict[str, Any]], key: str) -> dict[str, dict[str, float | int]]:
    negative_rows = [row for row in rows if (row.get("net_units") is not None and float(row["net_units"]) < 0)]
    count_by_key: Counter[str] = Counter()
    net_by_key: dict[str, float] = {}
    for row in negative_rows:
        label = str(row.get(key) or "unknown")
        count_by_key[label] += 1
        net_by_key[label] = float(net_by_key.get(label, 0.0) + float(row["net_units"]))
    return {
        label: {
            "negative_date_count": int(count_by_key[label]),
            "net_units": float(net_by_key[label]),
        }
        for label in sorted(count_by_key)
    }


def _build_fold_date_distribution(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "negative_date_count": int(sum(1 for row in rows if row.get("net_units") is not None and float(row["net_units"]) < 0)),
        "zero_date_count": int(sum(1 for row in rows if row.get("net_units") is not None and float(row["net_units"]) == 0)),
        "positive_date_count": int(sum(1 for row in rows if row.get("net_units") is not None and float(row["net_units"]) > 0)),
        "negative_by_dominant_track": _aggregate_negative_date_breakdown(rows, "dominant_track"),
        "negative_by_selected_rows_bucket": _aggregate_negative_date_breakdown(rows, "selected_rows_bucket"),
        "negative_by_field_size_bucket": _aggregate_negative_date_breakdown(rows, "mean_field_size_bucket"),
        "negative_by_mean_odds_bucket": _aggregate_negative_date_breakdown(rows, "mean_selected_odds_bucket"),
    }


def _simulate_policy_by_date(annotated: pd.DataFrame, *, odds_col: str, initial_bankroll: float = 1.0) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if annotated.empty:
        return [], {
            "policy_bets": 0,
            "selected_rows": 0,
            "stake_units": 0.0,
            "return_units": 0.0,
            "net_units": 0.0,
            "hit_races": 0,
            "final_bankroll": float(initial_bankroll),
            "max_drawdown": 0.0,
        }

    work = annotated.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    total_stake = 0.0
    total_return = 0.0
    total_selected_rows = 0
    total_policy_bets = 0
    total_hit_races = 0
    rows: list[dict[str, Any]] = []

    for date_value, date_group in work.sort_values(["date", "race_id"]).groupby("date", sort=True):
        date_start_bankroll = float(bankroll)
        date_stake = 0.0
        date_return = 0.0
        date_selected_rows = 0
        date_policy_bets = 0
        date_hit_races = 0
        track_counts: Counter[str] = Counter()
        field_sizes: list[float] = []
        selected_odds_values: list[float] = []

        for _, race_group in date_group.groupby("race_id", sort=False):
            selected = race_group[race_group["policy_selected"].fillna(False).astype(bool)].copy()
            if selected.empty:
                continue

            strategy_values = selected["policy_selected_strategy_kind"].dropna().astype(str).str.strip().str.lower().unique().tolist()
            strategy_kind = strategy_values[0] if strategy_values else str(selected["policy_strategy_kind"].iloc[0]).strip().lower()
            date_selected_rows += int(len(selected))
            date_policy_bets += 1
            track_values = race_group["track"].dropna().astype(str).str.strip().tolist() if "track" in race_group.columns else []
            if track_values:
                track_counts[track_values[0]] += 1
            field_sizes.append(float(len(race_group)))
            selected_odds = pd.to_numeric(selected.get(odds_col), errors="coerce").dropna().astype(float).tolist()
            selected_odds_values.extend(selected_odds)

            if strategy_kind == "kelly":
                pick = selected.sort_values("policy_weight", ascending=False).iloc[0]
                weight = float(pd.to_numeric(pd.Series([pick.get("policy_weight")]), errors="coerce").iloc[0] or 0.0)
                stake = bankroll * max(weight, 0.0)
                payout = 0.0
                rank = pd.to_numeric(pd.Series([pick.get("rank")]), errors="coerce").iloc[0]
                odds = pd.to_numeric(pd.Series([pick.get(odds_col)]), errors="coerce").iloc[0]
                if pd.notna(rank) and int(rank) == 1 and pd.notna(odds) and float(odds) > 0:
                    payout = stake * float(odds)
                    date_hit_races += 1
            else:
                stake = 1.0
                payout = 0.0
                for _, pick in selected.iterrows():
                    rank = pd.to_numeric(pd.Series([pick.get("rank")]), errors="coerce").iloc[0]
                    odds = pd.to_numeric(pd.Series([pick.get(odds_col)]), errors="coerce").iloc[0]
                    weight = float(pd.to_numeric(pd.Series([pick.get("policy_weight")]), errors="coerce").iloc[0] or 0.0)
                    if pd.notna(rank) and int(rank) == 1 and pd.notna(odds) and float(odds) > 0:
                        payout += stake * weight * float(odds)
                if payout > 0:
                    date_hit_races += 1

            date_stake += float(stake)
            date_return += float(payout)
            bankroll = bankroll - float(stake) + float(payout)
            peak_bankroll = max(peak_bankroll, bankroll)
            if peak_bankroll > 0:
                max_drawdown = max(max_drawdown, (peak_bankroll - bankroll) / peak_bankroll)

        total_stake += date_stake
        total_return += date_return
        total_selected_rows += date_selected_rows
        total_policy_bets += date_policy_bets
        total_hit_races += date_hit_races
        mean_field_size = (float(sum(field_sizes) / len(field_sizes)) if field_sizes else None)
        dominant_track = None
        if track_counts:
            dominant_track = sorted(track_counts.items(), key=lambda item: (-item[1], item[0]))[0][0]
        rows.append(
            {
                "date": str(date_value.date()),
                "policy_bets": int(date_policy_bets),
                "selected_rows": int(date_selected_rows),
                "hit_races": int(date_hit_races),
                "stake_units": float(date_stake),
                "return_units": float(date_return),
                "net_units": float(date_return - date_stake),
                "roi": (float(date_return / date_stake) if date_stake > 0 else None),
                "bankroll_start": float(date_start_bankroll),
                "bankroll_end": float(bankroll),
                "dominant_track": dominant_track,
                "unique_track_count": int(len(track_counts)),
                "track_counts": json.dumps(dict(track_counts), ensure_ascii=False, sort_keys=True),
                "mean_field_size": mean_field_size,
                "median_field_size": _median_or_none(field_sizes),
                "mean_field_size_bucket": _field_size_bucket(mean_field_size),
                "mean_selected_odds": (float(sum(selected_odds_values) / len(selected_odds_values)) if selected_odds_values else None),
                "median_selected_odds": _median_or_none(selected_odds_values),
                "max_selected_odds": (float(max(selected_odds_values)) if selected_odds_values else None),
                "selected_rows_bucket": _selected_rows_bucket(int(date_selected_rows)),
                "mean_selected_odds_bucket": _mean_odds_bucket(float(sum(selected_odds_values) / len(selected_odds_values)) if selected_odds_values else None),
            }
        )

    summary = {
        "policy_bets": int(total_policy_bets),
        "selected_rows": int(total_selected_rows),
        "stake_units": float(total_stake),
        "return_units": float(total_return),
        "net_units": float(total_return - total_stake),
        "roi": (float(total_return / total_stake) if total_stake > 0 else None),
        "hit_races": int(total_hit_races),
        "hit_rate": (float(total_hit_races / total_policy_bets) if total_policy_bets > 0 else None),
        "final_bankroll": float(bankroll),
        "max_drawdown": float(max_drawdown),
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="artifacts/reports/evaluation_summary.json")
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    args = parser.parse_args()

    try:
        summary_path = _resolve_path(args.summary)
        summary_payload = read_json(summary_path)
        if not isinstance(summary_payload, dict):
            raise ValueError(f"summary must be a JSON object: {summary_path}")
        folds = summary_payload.get("wf_nested_folds")
        if not isinstance(folds, list) or not folds:
            raise ValueError("summary does not contain wf_nested_folds")

        run_context = summary_payload.get("run_context") if isinstance(summary_payload.get("run_context"), dict) else {}
        config_path = _resolve_path(str(run_context.get("config") or ""))
        data_config_path = _resolve_path(str(run_context.get("data_config") or ""))
        feature_config_path = _resolve_path(str(run_context.get("feature_config") or ""))
        pre_feature_max_rows = int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else run_context.get("pre_feature_max_rows")
        model_cfg = load_yaml(config_path)
        data_cfg = load_yaml(data_config_path)
        feature_cfg = load_yaml(feature_config_path)
        dataset_cfg = data_cfg.get("dataset", {})
        label_col = str(model_cfg.get("label", "is_win"))

        output_slug = _sanitize_output_slug(Path(summary_path).stem)
        output_json = _resolve_path(args.output_json) if args.output_json else (ROOT / "artifacts" / "reports" / f"wf_selected_candidate_date_report_{output_slug}.json")
        output_csv = _resolve_path(args.output_csv) if args.output_csv else (ROOT / "artifacts" / "reports" / f"wf_selected_candidate_date_report_{output_slug}.csv")
        artifact_ensure_output_file_path(output_json, label="output json", workspace_root=ROOT)
        artifact_ensure_output_file_path(output_csv, label="output csv", workspace_root=ROOT)

        progress = ProgressBar(total=6, prefix="[wf-date-report]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="configs loaded")

        with Heartbeat("[wf-date-report]", "loading training table", logger=log_progress):
            load_result = load_training_table_for_feature_build(
                dataset_cfg.get("raw_dir", "data/raw"),
                pre_feature_max_rows=int(pre_feature_max_rows) if pre_feature_max_rows is not None else None,
                dataset_config=dataset_cfg,
                base_dir=ROOT,
            )
        frame = load_result.frame
        progress.update(message=f"training table loaded rows={load_result.loaded_rows:,}")

        with Heartbeat("[wf-date-report]", "building features", logger=log_progress):
            frame = build_features(frame)
        progress.update(message=f"features ready rows={len(frame):,}")

        output_artifacts = resolve_output_artifacts(model_cfg.get("output", {}))
        model_path = _resolve_path(output_artifacts.model_path)
        with Heartbeat("[wf-date-report]", f"loading model {model_path.name}", logger=log_progress):
            model = joblib.load(model_path)
        fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
        feature_selection = resolve_model_feature_selection(model, fallback_selection)
        x_eval = prepare_model_input_frame(frame, feature_selection.feature_columns, feature_selection.categorical_columns)
        odds_col = resolve_odds_column(frame)
        if odds_col is None:
            raise RuntimeError("odds column is required for date report")
        with Heartbeat("[wf-date-report]", "running model inference", logger=log_progress):
            outputs = generate_prediction_outputs(model, x_eval, race_ids=frame["race_id"])
        pred = prepare_scored_frame(frame, outputs.score, odds_col=odds_col, score_col="score")
        pred = add_market_signals(pred, score_col="score", odds_col=odds_col)
        progress.update(message="scored frame ready")

        report_rows: list[dict[str, Any]] = []
        fold_reports: list[dict[str, Any]] = []
        fold_progress = ProgressBar(total=max(len(folds), 1), prefix="[wf-date-report folds]", logger=log_progress, min_interval_sec=0.0)
        fold_progress.start(message="building fold date reports")

        for fold_row in folds:
            if not isinstance(fold_row, dict):
                continue
            fold_index = int(fold_row.get("fold") or 0)
            train_df = _slice_by_window(pred, start_date=str(fold_row.get("train_start_date")), end_date=str(fold_row.get("train_end_date")))
            test_df = _slice_by_window(pred, start_date=str(fold_row.get("test_start_date")), end_date=str(fold_row.get("test_end_date")))

            train_scores = train_df["score"].to_numpy()
            train_labels = train_df[label_col].astype(int).to_numpy()
            test_annotated = test_df.copy()
            test_annotated["iso_prob"] = fit_isotonic(train_scores, train_labels, test_annotated["score"].to_numpy())

            policy_name, policy_config = _build_policy_config(fold_row)
            test_annotated = annotate_runtime_policy(
                test_annotated,
                odds_col=odds_col,
                policy_name=policy_name,
                policy_config=policy_config,
                score_col="iso_prob",
            )
            date_rows, totals = _simulate_policy_by_date(test_annotated, odds_col=odds_col, initial_bankroll=1.0)

            top_loss_dates = sorted([row for row in date_rows if row.get("net_units") is not None], key=lambda row: float(row["net_units"]))[:10]
            top_gain_dates = sorted([row for row in date_rows if row.get("net_units") is not None], key=lambda row: float(row["net_units"]), reverse=True)[:10]
            fold_reports.append(
                {
                    "fold": fold_index,
                    "strategy_kind": fold_row.get("strategy_kind"),
                    "test_window": {
                        "start_date": fold_row.get("test_start_date"),
                        "end_date": fold_row.get("test_end_date"),
                    },
                    "policy": policy_config,
                    "replayed_totals": totals,
                    "date_distribution": _build_fold_date_distribution(date_rows),
                    "evaluation_summary_test": {
                        "test_roi": fold_row.get("test_roi"),
                        "test_bets": fold_row.get("test_bets"),
                        "test_hit_rate": fold_row.get("test_hit_rate"),
                        "test_final_bankroll": fold_row.get("test_final_bankroll"),
                        "test_max_drawdown": fold_row.get("test_max_drawdown"),
                    },
                    "top_loss_dates": top_loss_dates,
                    "top_gain_dates": top_gain_dates,
                }
            )

            for row in date_rows:
                report_rows.append(
                    {
                        "fold": fold_index,
                        "strategy_kind": fold_row.get("strategy_kind"),
                        "test_start_date": fold_row.get("test_start_date"),
                        "test_end_date": fold_row.get("test_end_date"),
                        **row,
                    }
                )
            fold_progress.update(message=f"fold={fold_index} dates={len(date_rows)}")

        progress.update(message="fold reports built")
        rows_df = pd.DataFrame(report_rows).sort_values(["fold", "date"], ascending=[True, True])
        payload = {
            "summary": str(summary_path.relative_to(ROOT)) if summary_path.is_relative_to(ROOT) else str(summary_path),
            "run_context": {
                "config": str(config_path.relative_to(ROOT)) if config_path.is_relative_to(ROOT) else str(config_path),
                "data_config": str(data_config_path.relative_to(ROOT)) if data_config_path.is_relative_to(ROOT) else str(data_config_path),
                "feature_config": str(feature_config_path.relative_to(ROOT)) if feature_config_path.is_relative_to(ROOT) else str(feature_config_path),
                "pre_feature_max_rows": int(pre_feature_max_rows) if pre_feature_max_rows is not None else None,
            },
            "fold_reports": fold_reports,
        }
        with Heartbeat("[wf-date-report]", "writing outputs", logger=log_progress):
            write_json(output_json, payload)
            write_csv_file(output_csv, rows_df, index=False)
        progress.complete(message="date report completed")
        print(f"saved json to {output_json}")
        print(f"saved csv to {output_csv}")
        return 0
    except KeyboardInterrupt:
        print("[wf-date-report] interrupted by user")
        return 130
    except Exception as error:
        print(f"[wf-date-report] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())