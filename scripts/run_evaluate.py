import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.artifacts import resolve_output_artifacts
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.leakage import run_leakage_audit
from racing_ml.evaluation.policy import (
    PolicyConstraints,
    add_market_signals,
    blend_prob,
    compute_market_prob,
    ev_top1_roi_from_prob,
    evaluate_fixed_stake_summary,
    run_policy_strategy,
    simulate_fractional_kelly,
)
from racing_ml.evaluation.scoring import (
    generate_prediction_outputs,
    prepare_scored_frame,
    resolve_odds_column,
    topk_hit_rate,
)
from racing_ml.evaluation.walk_forward import (
    build_nested_wf_slices,
    fit_isotonic,
    fit_platt,
    optimize_blend_weight,
    optimize_roi_strategy,
    split_for_calibration,
    split_three_way_time,
)
from racing_ml.features.builder import build_features


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[evaluate {now}] {message}", flush=True)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    return float(roc_auc_score(y_true, y_score))


def _base_summary(pred: pd.DataFrame, odds_col: str | None, score_col: str = "score") -> dict[str, float | int | None]:
    policy_summary = evaluate_fixed_stake_summary(pred, odds_col=odds_col, score_col=score_col, stake=100.0)
    return {
        "n_rows": int(len(pred)),
        "n_races": int(pred["race_id"].nunique()),
        "top1_hit_rate": topk_hit_rate(pred, 1),
        "top3_hit_rate": topk_hit_rate(pred, 3),
        "top5_hit_rate": topk_hit_rate(pred, 5),
        **policy_summary,
    }


def _append_top3_auc(summary: dict[str, object], pred: pd.DataFrame, top3_probs: dict[str, np.ndarray] | None) -> None:
    if top3_probs is None or "rank" not in pred.columns:
        return

    rank_series = pd.to_numeric(pred["rank"], errors="coerce")
    for pos, key in [(1, "p_rank1"), (2, "p_rank2"), (3, "p_rank3")]:
        y_pos = (rank_series == pos).astype(int).to_numpy()
        auc_value = _safe_auc(y_pos, top3_probs[key])
        if auc_value is not None:
            summary[f"auc_rank{pos}"] = auc_value


def _record_single_wf_summary(
    summary: dict[str, object],
    best_params: dict[str, float],
    best_valid_metrics: dict[str, float | int | None],
    wf_test_metrics: dict[str, float | int | None],
) -> None:
    summary["wf_strategy_kind"] = str(best_params.get("strategy_kind", "kelly"))
    summary["wf_best_blend_weight"] = float(best_params.get("blend_weight", 0.0))
    summary["wf_best_min_prob"] = float(best_params.get("min_prob", 0.05))
    summary["wf_best_odds_min"] = float(best_params.get("odds_min", 1.0))
    summary["wf_best_odds_max"] = float(best_params.get("odds_max", 999.0))

    if summary["wf_strategy_kind"] == "kelly":
        summary["wf_best_min_edge"] = float(best_params.get("min_edge", 0.03))
        summary["wf_best_fractional_kelly"] = float(best_params.get("fractional_kelly", 0.5))
        summary["wf_best_max_fraction"] = float(best_params.get("max_fraction", 0.05))
        summary["wf_valid_roi"] = best_valid_metrics.get("kelly_roi")
        summary["wf_valid_bets"] = best_valid_metrics.get("kelly_bets")
        summary["wf_valid_hit_rate"] = best_valid_metrics.get("kelly_hit_rate")
        summary["wf_test_roi"] = wf_test_metrics.get("kelly_roi")
        summary["wf_test_bets"] = wf_test_metrics.get("kelly_bets")
        summary["wf_test_hit_rate"] = wf_test_metrics.get("kelly_hit_rate")
        summary["wf_test_final_bankroll"] = wf_test_metrics.get("kelly_final_bankroll")
        summary["wf_valid_max_drawdown"] = best_valid_metrics.get("kelly_max_drawdown")
        summary["wf_test_max_drawdown"] = wf_test_metrics.get("kelly_max_drawdown")
    else:
        summary["wf_best_top_k"] = int(best_params.get("top_k", 2))
        summary["wf_best_min_expected_value"] = float(best_params.get("min_expected_value", 1.0))
        summary["wf_valid_roi"] = best_valid_metrics.get("portfolio_roi")
        summary["wf_valid_bets"] = best_valid_metrics.get("portfolio_bets")
        summary["wf_valid_hit_rate"] = best_valid_metrics.get("portfolio_hit_rate")
        summary["wf_test_roi"] = wf_test_metrics.get("portfolio_roi")
        summary["wf_test_bets"] = wf_test_metrics.get("portfolio_bets")
        summary["wf_test_hit_rate"] = wf_test_metrics.get("portfolio_hit_rate")
        summary["wf_test_avg_synthetic_odds"] = wf_test_metrics.get("portfolio_avg_synthetic_odds")
        summary["wf_valid_final_bankroll"] = best_valid_metrics.get("portfolio_final_bankroll")
        summary["wf_test_final_bankroll"] = wf_test_metrics.get("portfolio_final_bankroll")
        summary["wf_valid_max_drawdown"] = best_valid_metrics.get("portfolio_max_drawdown")
        summary["wf_test_max_drawdown"] = wf_test_metrics.get("portfolio_max_drawdown")


def _fold_row(
    strategy_kind: str,
    best_params: dict[str, float],
    best_valid_metrics: dict[str, float | int | None],
    fold_test_metrics: dict[str, float | int | None],
    fold_index: int,
) -> dict[str, float | int | str | None]:
    if strategy_kind == "kelly":
        return {
            "fold": int(fold_index),
            "strategy_kind": strategy_kind,
            "valid_roi": best_valid_metrics.get("kelly_roi"),
            "valid_bets": best_valid_metrics.get("kelly_bets"),
            "valid_hit_rate": best_valid_metrics.get("kelly_hit_rate"),
            "valid_final_bankroll": best_valid_metrics.get("kelly_final_bankroll"),
            "valid_max_drawdown": best_valid_metrics.get("kelly_max_drawdown"),
            "test_roi": fold_test_metrics.get("kelly_roi"),
            "test_bets": fold_test_metrics.get("kelly_bets"),
            "test_hit_rate": fold_test_metrics.get("kelly_hit_rate"),
            "test_final_bankroll": fold_test_metrics.get("kelly_final_bankroll"),
            "test_max_drawdown": fold_test_metrics.get("kelly_max_drawdown"),
            "blend_weight": float(best_params.get("blend_weight", 0.0)),
        }
    return {
        "fold": int(fold_index),
        "strategy_kind": strategy_kind,
        "valid_roi": best_valid_metrics.get("portfolio_roi"),
        "valid_bets": best_valid_metrics.get("portfolio_bets"),
        "valid_hit_rate": best_valid_metrics.get("portfolio_hit_rate"),
        "valid_final_bankroll": best_valid_metrics.get("portfolio_final_bankroll"),
        "valid_max_drawdown": best_valid_metrics.get("portfolio_max_drawdown"),
        "test_roi": fold_test_metrics.get("portfolio_roi"),
        "test_bets": fold_test_metrics.get("portfolio_bets"),
        "test_hit_rate": fold_test_metrics.get("portfolio_hit_rate"),
        "test_final_bankroll": fold_test_metrics.get("portfolio_final_bankroll"),
        "test_max_drawdown": fold_test_metrics.get("portfolio_max_drawdown"),
        "blend_weight": float(best_params.get("blend_weight", 0.0)),
    }


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--max-rows", type=int, default=120000)
    parser.add_argument("--wf-mode", choices=["off", "fast", "full"], default="fast")
    parser.add_argument("--wf-scheme", choices=["single", "nested"], default="nested")
    parser.add_argument("--progress-interval-sec", type=float, default=5.0)
    args = parser.parse_args()

    try:
        model_cfg = load_yaml(ROOT / args.config)
        data_cfg = load_yaml(ROOT / args.data_config)
        feature_cfg = load_yaml(ROOT / args.feature_config)

        task = str(model_cfg.get("task", "classification")).strip().lower()
        evaluation_cfg = model_cfg.get("evaluation", {})
        output_cfg = model_cfg.get("output", {})
        output_artifacts = resolve_output_artifacts(output_cfg)
        policy_constraints = PolicyConstraints.from_config(evaluation_cfg)
        leakage_cfg = evaluation_cfg.get("leakage_audit", {})
        leakage_enabled = bool(leakage_cfg.get("enabled", True))

        raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
        label_col = model_cfg.get("label", "is_win")
        feature_columns = feature_cfg.get("features", {}).get("base", []) + feature_cfg.get("features", {}).get("history", [])

        log_progress("Loading training table...")
        frame = load_training_table(str(ROOT / raw_dir))
        log_progress("Building features...")
        frame = build_features(frame)
        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()
            print(f"[evaluate] using tail rows: {len(frame)}")

        available_features = [column for column in feature_columns if column in frame.columns]
        if not available_features:
            raise RuntimeError("No features available for evaluation")
        if label_col not in frame.columns:
            raise RuntimeError(f"Missing label column: {label_col}")

        model_path = output_artifacts.model_path if output_artifacts.model_path.is_absolute() else (ROOT / output_artifacts.model_path)
        model = joblib.load(model_path)

        x_eval = frame[available_features]
        y_eval = frame[label_col].astype(int).to_numpy()
        odds_col = resolve_odds_column(frame)

        log_progress("Running model inference...")
        outputs = generate_prediction_outputs(model, x_eval, race_ids=frame["race_id"])
        pred = prepare_scored_frame(frame, outputs.score, odds_col=odds_col, score_col="score")
        if odds_col is not None:
            pred = add_market_signals(pred, score_col="score", odds_col=odds_col)

        score_is_prob = bool(np.nanmin(outputs.score) >= 0.0 and np.nanmax(outputs.score) <= 1.0)
        compute_prob_metrics = task in {"classification", "ranking", "multi_position"}
        probabilistic_flow = bool(compute_prob_metrics and score_is_prob)

        summary = {
            **_base_summary(pred, odds_col=odds_col, score_col="score"),
            "n_dates": int(pred["date"].nunique()) if "date" in pred.columns else None,
            "auc": _safe_auc(y_eval, outputs.score) if compute_prob_metrics else None,
            "logloss": float(log_loss(y_eval, np.clip(outputs.score, 1e-12, 1 - 1e-12), labels=[0, 1])) if (compute_prob_metrics and score_is_prob) else None,
            "score_is_probability": score_is_prob,
            "task": task,
            "evaluation_flow": "probability_market" if probabilistic_flow else "roi_direct",
        }

        summary["run_context"] = {
            "config": str(args.config),
            "data_config": str(args.data_config),
            "feature_config": str(args.feature_config),
            "task": task,
            "label_column": label_col,
            "max_rows": int(args.max_rows),
            "wf_mode": args.wf_mode,
            "wf_scheme": args.wf_scheme,
            "progress_interval_sec": float(args.progress_interval_sec),
            "rows_total_after_tail": int(len(frame)),
            "artifact_manifest": output_artifacts.manifest_path.as_posix() if (ROOT / output_artifacts.manifest_path).exists() else None,
        }
        summary["policy_constraints"] = policy_constraints.to_dict()

        summary["leakage_audit"] = (
            run_leakage_audit(frame=frame, feature_columns=available_features, label_column=label_col)
            if leakage_enabled
            else {"enabled": False}
        )

        if odds_col is not None and probabilistic_flow:
            market_prob = compute_market_prob(pred, odds_col=odds_col).to_numpy(dtype=float)
            model_prob = np.clip(np.asarray(outputs.score, dtype=float), 1e-6, 1 - 1e-6)
            market_prob_clip = np.clip(market_prob, 1e-6, 1 - 1e-6)
            odds_values = pd.to_numeric(pred[odds_col], errors="coerce").to_numpy(dtype=float)
            edge = model_prob * odds_values - 1.0
            delta_logit = np.log(model_prob / (1.0 - model_prob)) - np.log(market_prob_clip / (1.0 - market_prob_clip))
            summary["market_prob_corr"] = float(np.corrcoef(model_prob, market_prob_clip)[0, 1]) if len(model_prob) > 1 else None
            summary["edge_mean"] = float(np.nanmean(edge)) if np.isfinite(np.nanmean(edge)) else None
            summary["delta_logit_mean"] = float(np.nanmean(delta_logit)) if np.isfinite(np.nanmean(delta_logit)) else None

        _append_top3_auc(summary, pred, outputs.top3_probs)

        log_progress("Starting calibration/evaluation block...")
        calib_train, calib_test = split_for_calibration(pred, date_col="date", train_ratio=0.7)
        if probabilistic_flow and len(calib_train) >= 1000 and len(calib_test) >= 1000:
            train_scores = calib_train["score"].to_numpy()
            train_labels = calib_train[label_col].astype(int).to_numpy()
            test_scores = calib_test["score"].to_numpy()

            platt_scores = fit_platt(train_scores, train_labels, test_scores)
            isotonic_scores = fit_isotonic(train_scores, train_labels, test_scores)

            platt_df = prepare_scored_frame(calib_test, platt_scores, odds_col=odds_col, score_col="calibrated_score")
            isotonic_df = prepare_scored_frame(calib_test, isotonic_scores, odds_col=odds_col, score_col="calibrated_score")
            if odds_col is not None:
                platt_df = add_market_signals(platt_df, score_col="calibrated_score", odds_col=odds_col)
                isotonic_df = add_market_signals(isotonic_df, score_col="calibrated_score", odds_col=odds_col)

            platt_metrics = evaluate_fixed_stake_summary(platt_df, odds_col=odds_col, score_col="calibrated_score", stake=100.0)
            isotonic_metrics = evaluate_fixed_stake_summary(isotonic_df, odds_col=odds_col, score_col="calibrated_score", stake=100.0)

            summary["calibration_eval_rows"] = int(len(calib_test))
            summary["calibration_eval_races"] = int(calib_test["race_id"].nunique())
            summary["platt_top1_roi"] = platt_metrics.get("top1_roi")
            summary["platt_ev_top1_roi"] = platt_metrics.get("ev_top1_roi")
            summary["isotonic_top1_roi"] = isotonic_metrics.get("top1_roi")
            summary["isotonic_ev_top1_roi"] = isotonic_metrics.get("ev_top1_roi")
            if summary.get("top1_roi") is not None and summary.get("platt_top1_roi") is not None:
                summary["platt_top1_roi_lift"] = float(summary["platt_top1_roi"] - summary["top1_roi"])
            if summary.get("top1_roi") is not None and summary.get("isotonic_top1_roi") is not None:
                summary["isotonic_top1_roi_lift"] = float(summary["isotonic_top1_roi"] - summary["top1_roi"])

            if odds_col is not None:
                calib_train_b = calib_train.copy()
                calib_test_b = calib_test.copy()
                calib_train_b["isotonic_prob"] = fit_isotonic(train_scores, train_labels, train_scores)
                calib_test_b["isotonic_prob"] = isotonic_scores
                calib_train_b["market_prob"] = compute_market_prob(calib_train_b, odds_col=odds_col)
                calib_test_b["market_prob"] = compute_market_prob(calib_test_b, odds_col=odds_col)

                blend_weight = optimize_blend_weight(
                    frame=calib_train_b,
                    label_col=label_col,
                    model_prob_col="isotonic_prob",
                    market_prob_col="market_prob",
                )
                calib_test_b["benter_prob"] = blend_prob(
                    calib_test_b["isotonic_prob"],
                    calib_test_b["market_prob"],
                    weight=blend_weight,
                )

                summary["benter_blend_weight"] = float(blend_weight)
                summary["benter_ev_top1_roi"] = ev_top1_roi_from_prob(
                    calib_test_b,
                    prob_col="benter_prob",
                    odds_col=odds_col,
                )
                kelly = simulate_fractional_kelly(
                    calib_test_b,
                    prob_col="benter_prob",
                    odds_col=odds_col,
                    min_edge=0.03,
                    min_prob=0.05,
                    fractional_kelly=0.5,
                    max_fraction=0.05,
                    initial_bankroll=1.0,
                )
                summary["benter_kelly_roi"] = kelly["kelly_roi"]
                summary["benter_kelly_bets"] = kelly["kelly_bets"]
                summary["benter_kelly_hit_rate"] = kelly["kelly_hit_rate"]
                summary["benter_kelly_final_bankroll"] = kelly["kelly_final_bankroll"]
                summary["benter_kelly_max_drawdown"] = kelly.get("kelly_max_drawdown")

                wf_train, wf_valid, wf_test = split_three_way_time(pred, date_col="date", train_ratio=0.5, valid_ratio=0.25)
                summary["wf_train_rows"] = int(len(wf_train))
                summary["wf_valid_rows"] = int(len(wf_valid))
                summary["wf_test_rows"] = int(len(wf_test))
                summary["wf_mode"] = args.wf_mode
                summary["wf_scheme"] = args.wf_scheme
                summary["wf_constraints_min_bet_ratio"] = float(policy_constraints.min_bet_ratio)
                summary["wf_constraints_min_bets_abs"] = int(policy_constraints.min_bets_abs)
                summary["wf_constraints_max_drawdown"] = float(policy_constraints.max_drawdown)
                summary["wf_constraints_min_final_bankroll"] = float(policy_constraints.min_final_bankroll)
                summary["wf_enabled"] = bool(args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500)

                if args.wf_mode != "off" and len(wf_train) >= 1000 and len(wf_valid) >= 500 and len(wf_test) >= 500:
                    if args.wf_scheme == "single":
                        log_progress("Walk-forward optimization started (single split)...")
                        best_params, best_valid_metrics = optimize_roi_strategy(
                            train_df=wf_train,
                            valid_df=wf_valid,
                            label_col=label_col,
                            odds_col=odds_col,
                            constraints=policy_constraints,
                            mode=args.wf_mode,
                            progress_interval_sec=float(args.progress_interval_sec),
                            logger=log_progress,
                        )

                        wf_train_scores = wf_train["score"].to_numpy()
                        wf_train_labels = wf_train[label_col].astype(int).to_numpy()
                        wf_test = wf_test.copy()
                        wf_test["iso_prob"] = fit_isotonic(wf_train_scores, wf_train_labels, wf_test["score"].to_numpy())
                        wf_test["market_prob"] = compute_market_prob(wf_test, odds_col=odds_col)
                        wf_test["blend_prob"] = blend_prob(
                            wf_test["iso_prob"],
                            wf_test["market_prob"],
                            weight=float(best_params.get("blend_weight", 0.0)),
                        )
                        wf_test_metrics = run_policy_strategy(wf_test, prob_col="blend_prob", odds_col=odds_col, params=best_params)
                        log_progress("Walk-forward optimization finished (single split).")
                        _record_single_wf_summary(summary, best_params, best_valid_metrics, wf_test_metrics)
                    else:
                        n_folds = 5 if args.wf_mode == "full" else 3
                        nested_slices = build_nested_wf_slices(
                            pred,
                            date_col="date",
                            n_folds=n_folds,
                            valid_ratio=0.15,
                            test_ratio=0.15,
                            min_train_rows=1000,
                            min_valid_rows=500,
                            min_test_rows=500,
                        )
                        summary["wf_nested_target_folds"] = int(n_folds)
                        summary["wf_nested_actual_folds"] = int(len(nested_slices))

                        if nested_slices:
                            fold_rows: list[dict[str, float | int | str | None]] = []
                            log_progress(f"Nested WF started: folds={len(nested_slices)}")
                            for fold_index, (fold_train, fold_valid, fold_test) in enumerate(nested_slices, start=1):
                                log_progress(f"Nested WF fold {fold_index}/{len(nested_slices)}: optimizing on inner valid...")
                                best_params, best_valid_metrics = optimize_roi_strategy(
                                    train_df=fold_train,
                                    valid_df=fold_valid,
                                    label_col=label_col,
                                    odds_col=odds_col,
                                    constraints=policy_constraints,
                                    mode=args.wf_mode,
                                    progress_interval_sec=float(args.progress_interval_sec),
                                    logger=log_progress,
                                )

                                fold_train_scores = fold_train["score"].to_numpy()
                                fold_train_labels = fold_train[label_col].astype(int).to_numpy()
                                fold_test = fold_test.copy()
                                fold_test["iso_prob"] = fit_isotonic(fold_train_scores, fold_train_labels, fold_test["score"].to_numpy())
                                fold_test["market_prob"] = compute_market_prob(fold_test, odds_col=odds_col)
                                fold_test["blend_prob"] = blend_prob(
                                    fold_test["iso_prob"],
                                    fold_test["market_prob"],
                                    weight=float(best_params.get("blend_weight", 0.0)),
                                )
                                fold_test_metrics = run_policy_strategy(fold_test, prob_col="blend_prob", odds_col=odds_col, params=best_params)
                                strategy_kind = str(best_params.get("strategy_kind", "kelly"))
                                fold_rows.append(_fold_row(strategy_kind, best_params, best_valid_metrics, fold_test_metrics, fold_index))

                            summary["wf_nested_folds"] = fold_rows
                            test_roi_values = [float(row["test_roi"]) for row in fold_rows if row.get("test_roi") is not None]
                            test_bets_values = [int(row["test_bets"] or 0) for row in fold_rows]
                            weighted_numerator = 0.0
                            weighted_denominator = 0.0
                            for row in fold_rows:
                                row_roi = row.get("test_roi")
                                row_bets = int(row.get("test_bets") or 0)
                                if row_roi is None or row_bets <= 0:
                                    continue
                                weighted_numerator += float(row_roi) * row_bets
                                weighted_denominator += row_bets

                            summary["wf_nested_test_roi_mean"] = float(np.mean(test_roi_values)) if test_roi_values else None
                            summary["wf_nested_test_roi_weighted"] = float(weighted_numerator / weighted_denominator) if weighted_denominator > 0 else None
                            summary["wf_nested_test_bets_total"] = int(np.sum(test_bets_values))
                            summary["wf_nested_test_bets_mean"] = float(np.mean(test_bets_values)) if test_bets_values else None
                            summary["wf_nested_completed"] = True
                            log_progress("Nested WF finished.")
                        else:
                            summary["wf_nested_completed"] = False
                            summary["wf_nested_reason"] = "insufficient_data_for_nested_folds"
        else:
            if not probabilistic_flow:
                summary["calibration_skipped_reason"] = "non_probability_task_or_score"
                summary["wf_enabled"] = False
                summary["wf_mode"] = args.wf_mode
                summary["wf_scheme"] = args.wf_scheme
                summary["wf_skipped_reason"] = "non_probability_task_or_score"
            else:
                summary["calibration_skipped_reason"] = "insufficient_calibration_rows"
                summary["wf_enabled"] = False
                summary["wf_mode"] = args.wf_mode
                summary["wf_scheme"] = args.wf_scheme
                summary["wf_skipped_reason"] = "insufficient_calibration_rows"

        by_date_rows: list[dict[str, object]] = []
        if "date" in pred.columns:
            date_series = pd.to_datetime(pred["date"], errors="coerce")
            for date_value, date_df in pred.groupby(date_series.dt.date):
                date_summary = _base_summary(date_df, odds_col=odds_col, score_col="score")
                by_date_rows.append({"date": str(date_value), **date_summary})

        by_date = pd.DataFrame(by_date_rows).sort_values("date") if by_date_rows else pd.DataFrame()

        report_dir = ROOT / "artifacts/reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "evaluation_summary.json"
        by_date_path = report_dir / "evaluation_by_date.csv"

        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        if not by_date.empty:
            by_date.to_csv(by_date_path, index=False)

        print(f"[evaluate] summary saved: {summary_path}")
        if not by_date.empty:
            print(f"[evaluate] by-date saved: {by_date_path}")
            print(by_date.tail(5).to_string(index=False))
        print(f"[evaluate] summary: {json.dumps(summary, ensure_ascii=False)}")
        return 0
    except KeyboardInterrupt:
        print("[evaluate] interrupted by user")
        return 130
    except Exception as error:
        print(f"[evaluate] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())