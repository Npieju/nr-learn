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


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.probability import normalize_position_probabilities
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.leakage import run_leakage_audit
from racing_ml.features.builder import build_features
from racing_ml.models.trainer import train_and_evaluate
from racing_ml.serving.predict_batch import _predict_multi_position


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[tune {now}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    seconds_int = max(int(seconds), 0)
    minutes, sec = divmod(seconds_int, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{sec:02d}s"
    return f"{minutes:d}m{sec:02d}s"


def _time_valid_slice(
    frame: pd.DataFrame,
    valid_start: str,
    valid_end: str,
    max_valid_rows: int,
) -> pd.DataFrame:
    valid_start_ts = pd.to_datetime(valid_start)
    valid_end_ts = pd.to_datetime(valid_end)
    work = frame[(frame["date"] >= valid_start_ts) & (frame["date"] <= valid_end_ts)].copy()
    if max_valid_rows and len(work) > max_valid_rows:
        work = work.tail(max_valid_rows).copy()
    return work


def _predict_top1_score(model: object, features: pd.DataFrame, race_ids: pd.Series) -> np.ndarray:
    if isinstance(model, dict) and model.get("kind") == "multi_position_top3":
        outputs = _predict_multi_position(model, features)
        work = pd.DataFrame({"race_id": race_ids.to_numpy(copy=False)})
        work["p_rank1_raw"] = outputs["p_rank1"]
        work = normalize_position_probabilities(
            work,
            raw_columns=["p_rank1_raw"],
            race_id_col="race_id",
            output_prefix="",
        )
        return work["p_rank1_raw"].to_numpy(dtype=float)

    if hasattr(model, "predict_proba"):
        return model.predict_proba(features)[:, 1]
    if hasattr(model, "predict"):
        return np.asarray(model.predict(features), dtype=float)
    raise RuntimeError("Model does not support prediction for ROI tuning")


def _strategy_roi(
    scored: pd.DataFrame,
    mode: str,
    threshold: float = 1.0,
    odds_min: float = 1.0,
    odds_max: float = 80.0,
    min_score: float = 0.0,
) -> tuple[float | None, int, float | None, float, float]:
    total_bet = 0.0
    total_return = 0.0
    bets = 0
    hits = 0
    bankroll = 1.0
    peak_bankroll = 1.0
    max_drawdown = 0.0

    for _, group in scored.groupby("race_id"):
        if bankroll <= 0:
            break

        if mode == "top1":
            pick = group.sort_values("score", ascending=False).iloc[0]
        elif mode == "top1_filtered":
            pick = group.sort_values("score", ascending=False).iloc[0]
            score_value = pd.to_numeric(pd.Series([pick.get("score")]), errors="coerce").iloc[0]
            odds_value = pd.to_numeric(pd.Series([pick.get("odds")]), errors="coerce").iloc[0]
            if pd.isna(score_value) or pd.isna(odds_value):
                continue
            if float(score_value) < float(min_score):
                continue
            if float(odds_value) < float(odds_min) or float(odds_value) > float(odds_max):
                continue
        elif mode == "ev":
            candidates = group[(group["expected_value"] >= threshold) & (group["odds"] >= odds_min) & (group["odds"] <= odds_max)]
            if candidates.empty:
                continue
            pick = candidates.sort_values("expected_value", ascending=False).iloc[0]
        elif mode == "edge":
            candidates = group[(group["edge"] >= threshold) & (group["odds"] >= odds_min) & (group["odds"] <= odds_max)]
            if candidates.empty:
                continue
            pick = candidates.sort_values("edge", ascending=False).iloc[0]
        else:
            raise ValueError(f"Unsupported strategy mode: {mode}")

        stake = min(1.0, float(bankroll))
        if stake <= 0:
            break

        bets += 1
        total_bet += stake
        rank = pd.to_numeric(pd.Series([pick.get("rank")]), errors="coerce").iloc[0]
        odds = pd.to_numeric(pd.Series([pick.get("odds")]), errors="coerce").iloc[0]
        payout = 0.0
        if pd.notna(rank) and int(rank) == 1 and pd.notna(odds) and float(odds) > 0:
            hits += 1
            payout = float(odds) * stake
            total_return += payout

        bankroll = bankroll - stake + payout
        bankroll = max(float(bankroll), 0.0)
        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, min(drawdown, 1.0))

    if total_bet == 0:
        return None, 0, None, float(bankroll), float(max_drawdown)
    roi = float(total_return / total_bet)
    hit_rate = float(hits / bets) if bets > 0 else None
    return roi, int(bets), hit_rate, float(bankroll), float(max_drawdown)


def objective_score(
    model_path: Path,
    valid_frame: pd.DataFrame,
    feature_columns: list[str],
    min_bet_ratio: float = 0.02,
    min_bets_abs: int = 30,
    max_drawdown: float = 0.45,
    min_final_bankroll: float = 0.85,
    selection_mode: str = "roi_first",
) -> tuple[float, dict[str, float | int | str | None]]:
    model = joblib.load(model_path)
    x_valid = valid_frame[feature_columns]
    score = _predict_top1_score(model, x_valid, valid_frame["race_id"])

    scored = valid_frame.copy()
    scored["score"] = score
    scored["odds"] = pd.to_numeric(scored.get("odds"), errors="coerce")
    scored["expected_value"] = scored["score"] * scored["odds"]
    implied = 1.0 / scored["odds"].replace(0, np.nan)
    market_prob = implied / implied.groupby(scored["race_id"]).transform("sum")
    scored["market_prob"] = market_prob.fillna(0.0)
    scored["edge"] = scored["score"] - scored["market_prob"]

    n_races = int(scored["race_id"].nunique())
    min_bets = max(int(n_races * min_bet_ratio), int(min_bets_abs))

    candidates: list[dict[str, float | int | str | None]] = []

    top1_roi, top1_bets, top1_hit, top1_final_bankroll, top1_max_drawdown = _strategy_roi(scored, mode="top1")
    candidates.append(
        {
            "strategy": "top1",
            "threshold": None,
            "roi": top1_roi,
            "bets": top1_bets,
            "hit_rate": top1_hit,
            "final_bankroll": top1_final_bankroll,
            "max_drawdown": top1_max_drawdown,
        }
    )

    for min_score in [0.16, 0.18, 0.20, 0.22, 0.24, 0.27]:
        for odds_min in [1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                roi, bets, hit, final_bankroll, candidate_drawdown = _strategy_roi(
                    scored,
                    mode="top1_filtered",
                    min_score=min_score,
                    odds_min=odds_min,
                    odds_max=odds_max,
                )
                candidates.append(
                    {
                        "strategy": "top1_filtered",
                        "threshold": float(min_score),
                        "odds_min": float(odds_min),
                        "odds_max": float(odds_max),
                        "roi": roi,
                        "bets": bets,
                        "hit_rate": hit,
                        "final_bankroll": final_bankroll,
                        "max_drawdown": candidate_drawdown,
                    }
                )

    for threshold in [1.00, 1.05, 1.10, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00]:
        for odds_min in [1.2, 1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                roi, bets, hit, final_bankroll, candidate_drawdown = _strategy_roi(
                    scored,
                    mode="ev",
                    threshold=threshold,
                    odds_min=odds_min,
                    odds_max=odds_max,
                )
                candidates.append(
                    {
                        "strategy": "ev",
                        "threshold": float(threshold),
                        "odds_min": float(odds_min),
                        "odds_max": float(odds_max),
                        "roi": roi,
                        "bets": bets,
                        "hit_rate": hit,
                        "final_bankroll": final_bankroll,
                        "max_drawdown": candidate_drawdown,
                    }
                )

    for edge_threshold in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for odds_min in [1.2, 1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                roi, bets, hit, final_bankroll, candidate_drawdown = _strategy_roi(
                    scored,
                    mode="edge",
                    threshold=edge_threshold,
                    odds_min=odds_min,
                    odds_max=odds_max,
                )
                candidates.append(
                    {
                        "strategy": "edge",
                        "threshold": float(edge_threshold),
                        "odds_min": float(odds_min),
                        "odds_max": float(odds_max),
                        "roi": roi,
                        "bets": bets,
                        "hit_rate": hit,
                        "final_bankroll": final_bankroll,
                        "max_drawdown": candidate_drawdown,
                    }
                )

    best = {
        "strategy": "top1",
        "threshold": None,
        "roi": 0.0,
        "bets": 0,
        "hit_rate": None,
        "final_bankroll": 1.0,
        "max_drawdown": 0.0,
    }
    fallback_best = dict(best)
    fallback_score = -1e9
    best_score = -1e9
    feasible_count = 0
    for row in candidates:
        roi = row["roi"]
        bets = int(row["bets"] or 0)
        candidate_drawdown = float(row.get("max_drawdown") or 1.0)
        candidate_final_bankroll = float(row.get("final_bankroll") or 0.0)
        if roi is None:
            continue
        unconstrained_score = float(roi)
        if bets < min_bets:
            unconstrained_score -= 1.0
        if unconstrained_score > fallback_score:
            fallback_score = unconstrained_score
            fallback_best = row
        score_value = float(roi)
        if bets < min_bets:
            score_value -= 1.0

        risk_ok = (candidate_drawdown <= max_drawdown) and (candidate_final_bankroll >= min_final_bankroll)
        if risk_ok:
            feasible_count += 1

        if selection_mode == "risk_first":
            if not risk_ok:
                continue
        else:
            if candidate_drawdown > max_drawdown:
                score_value -= 0.10 * float(candidate_drawdown - max_drawdown)
            if candidate_final_bankroll < min_final_bankroll:
                score_value -= 0.20 * float(min_final_bankroll - candidate_final_bankroll)

        if score_value > best_score:
            best_score = score_value
            best = row

    if selection_mode == "risk_first" and feasible_count == 0:
        best = fallback_best
        best_score = fallback_score

    best["min_bets_required"] = int(min_bets)
    best["n_races"] = int(n_races)
    best["constraints_max_drawdown"] = float(max_drawdown)
    best["constraints_min_final_bankroll"] = float(min_final_bankroll)
    best["selection_mode"] = selection_mode
    best["feasible_candidate_count"] = int(feasible_count)
    best["is_feasible"] = bool(feasible_count > 0)
    return float(best_score), best


def build_model_candidates(base_params: dict) -> list[dict]:
    candidates = [
        {},
        {"learning_rate": 0.02, "num_leaves": 96, "min_data_in_leaf": 40, "feature_fraction": 0.85},
        {"learning_rate": 0.02, "num_leaves": 128, "min_data_in_leaf": 30, "feature_fraction": 0.9, "bagging_fraction": 0.9},
        {"learning_rate": 0.04, "num_leaves": 64, "min_data_in_leaf": 60, "feature_fraction": 0.8},
        {"learning_rate": 0.03, "num_leaves": 80, "min_data_in_leaf": 35, "feature_fraction": 0.9, "bagging_fraction": 0.85},
        {"learning_rate": 0.015, "num_leaves": 160, "min_data_in_leaf": 25, "feature_fraction": 0.95, "bagging_fraction": 0.9},
        {"learning_rate": 0.02, "num_leaves": 96, "min_data_in_leaf": 30, "feature_fraction": 0.9, "bagging_fraction": 0.9, "lambda_l2": 1.0},
        {"learning_rate": 0.02, "num_leaves": 96, "min_data_in_leaf": 25, "feature_fraction": 0.9, "bagging_fraction": 0.9, "lambda_l2": 3.0},
        {"learning_rate": 0.025, "num_leaves": 96, "min_data_in_leaf": 30, "feature_fraction": 0.88, "bagging_fraction": 0.9, "lambda_l2": 1.0},
        {"learning_rate": 0.02, "num_leaves": 80, "min_data_in_leaf": 20, "feature_fraction": 0.92, "bagging_fraction": 0.9, "lambda_l2": 1.0},
        {"learning_rate": 0.02, "num_leaves": 64, "min_data_in_leaf": 25, "feature_fraction": 0.92, "bagging_fraction": 0.9, "lambda_l2": 2.0},
        {"learning_rate": 0.03, "num_leaves": 96, "min_data_in_leaf": 30, "feature_fraction": 0.88, "bagging_fraction": 0.85, "lambda_l2": 1.0},
    ]
    merged: list[dict] = []
    for patch in candidates:
        params = dict(base_params)
        params.update(patch)

        if "min_data_in_leaf" in params and "min_child_samples" not in params:
            params["min_child_samples"] = params.pop("min_data_in_leaf")
        if "feature_fraction" in params and "colsample_bytree" not in params:
            params["colsample_bytree"] = params.pop("feature_fraction")
        if "bagging_fraction" in params and "subsample" not in params:
            params["subsample"] = params.pop("bagging_fraction")
        if "bagging_freq" in params and "subsample_freq" not in params:
            params["subsample_freq"] = params.pop("bagging_freq")
        if "lambda_l2" in params and "reg_lambda" not in params:
            params["reg_lambda"] = params.pop("lambda_l2")
        params.setdefault("verbosity", -1)

        merged.append(params)
    return merged


def build_row_profiles(training_cfg: dict) -> list[dict[str, int]]:
    base_train = int(training_cfg.get("max_train_rows", 300000) or 300000)
    base_valid = int(training_cfg.get("max_valid_rows", 100000) or 100000)

    profiles = [
        {"max_train_rows": base_train, "max_valid_rows": base_valid},
        {"max_train_rows": max(200000, int(base_train * 0.75)), "max_valid_rows": max(80000, int(base_valid * 0.8))},
        {"max_train_rows": min(700000, int(base_train * 1.5)), "max_valid_rows": min(220000, int(base_valid * 1.6))},
    ]

    unique_profiles: list[dict[str, int]] = []
    seen: set[tuple[int, int]] = set()
    for profile in profiles:
        key = (int(profile["max_train_rows"]), int(profile["max_valid_rows"]))
        if key in seen:
            continue
        seen.add(key)
        unique_profiles.append(profile)
    return unique_profiles


def _write_summary(
    out_path: Path,
    args: argparse.Namespace,
    model_candidates: list[dict],
    row_profiles: list[dict[str, int]],
    trial_plan: list[tuple[dict, dict[str, int]]],
    run_rows: list[dict],
    best_row: dict | None,
    failures: list[dict],
    run_context: dict,
    leakage_audit: dict,
    strategy_constraints: dict,
) -> None:
    summary = {
        "base_config": args.config,
        "objective": "validation_roi_with_min_bets_constraint",
        "n_model_candidates": len(model_candidates),
        "n_row_profiles": len(row_profiles),
        "n_trials": len(trial_plan),
        "min_bet_ratio": float(args.min_bet_ratio),
        "min_bets_abs": int(args.min_bets_abs),
        "n_candidates": len(run_rows),
        "best": best_row,
        "runs": run_rows,
        "failures": failures,
        "run_context": run_context,
        "leakage_audit": leakage_audit,
        "strategy_constraints": strategy_constraints,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> int:
    warnings.filterwarnings("ignore", message="X does not have valid feature names, but LGBMClassifier was fitted with feature names")

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_top3.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--max-row-profiles", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=16)
    parser.add_argument("--min-bet-ratio", type=float, default=0.02)
    parser.add_argument("--min-bets-abs", type=int, default=30)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--summary-path", default="artifacts/reports/tune_top3_summary.json")
    args = parser.parse_args()

    log_progress("Loading configs...")
    model_cfg = load_yaml(ROOT / args.config)
    data_cfg = load_yaml(ROOT / args.data_config)
    feature_cfg = load_yaml(ROOT / args.feature_config)

    raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
    split_cfg = data_cfg.get("split", {})
    log_progress("Loading training table...")
    frame = load_training_table(str(ROOT / raw_dir))
    log_progress("Building features...")
    frame = build_features(frame)

    feature_columns = feature_cfg.get("features", {}).get("base", []) + feature_cfg.get("features", {}).get("history", [])
    label_column = model_cfg.get("label", "is_win")
    task = str(model_cfg.get("task", "multi_position"))

    base_params = model_cfg.get("model", {}).get("params", {})
    training_cfg = model_cfg.get("training", {})
    evaluation_cfg = model_cfg.get("evaluation", {})
    output_cfg = model_cfg.get("output", {})
    model_name = model_cfg.get("model", {}).get("name", "lightgbm")
    device_type = str(base_params.get("device_type", "cpu")).strip().lower() or "cpu"
    strategy_constraints_cfg = evaluation_cfg.get("strategy_constraints", {})
    leakage_cfg = evaluation_cfg.get("leakage_audit", {})
    leakage_enabled = bool(leakage_cfg.get("enabled", True))

    constraints = {
        "max_drawdown": float(strategy_constraints_cfg.get("max_drawdown", 0.45)),
        "min_final_bankroll": float(strategy_constraints_cfg.get("min_final_bankroll", 0.85)),
        "selection_mode": str(strategy_constraints_cfg.get("selection_mode", "roi_first")),
    }

    leakage_audit = (
        run_leakage_audit(frame=frame, feature_columns=feature_columns, label_column=label_column)
        if leakage_enabled
        else {"enabled": False}
    )

    run_context = {
        "config": str(args.config),
        "data_config": str(args.data_config),
        "feature_config": str(args.feature_config),
        "task": task,
        "label_column": label_column,
        "max_candidates": int(args.max_candidates),
        "max_row_profiles": int(args.max_row_profiles),
        "max_trials": int(args.max_trials),
        "min_bet_ratio": float(args.min_bet_ratio),
        "min_bets_abs": int(args.min_bets_abs),
        "rows_total": int(len(frame)),
        "device_type": device_type,
    }

    log_progress(f"Runtime device_type from config: {device_type}")

    model_candidates = build_model_candidates(base_params)[: max(1, args.max_candidates)]
    row_profiles = build_row_profiles(training_cfg)[: max(1, args.max_row_profiles)]

    trial_plan: list[tuple[dict, dict[str, int]]] = []
    for model_candidate in model_candidates:
        for row_profile in row_profiles:
            trial_plan.append((model_candidate, row_profile))
    trial_plan = trial_plan[: max(1, args.max_trials)]

    log_progress(
        "Prepared trial plan: "
        f"model_candidates={len(model_candidates)}, row_profiles={len(row_profiles)}, trials={len(trial_plan)}"
    )

    out_path = ROOT / args.summary_path
    run_rows: list[dict] = []
    failures: list[dict] = []
    best_row: dict | None = None
    best_score = float("-inf")

    completed_candidates: set[int] = set()
    if args.resume and out_path.exists():
        try:
            with out_path.open("r", encoding="utf-8") as file:
                prev = json.load(file)
            prev_runs = prev.get("runs", []) if isinstance(prev, dict) else []
            prev_failures = prev.get("failures", []) if isinstance(prev, dict) else []
            for row in prev_runs:
                if isinstance(row, dict) and isinstance(row.get("candidate"), int):
                    run_rows.append(row)
                    completed_candidates.add(int(row["candidate"]))
                    row_score = float(row.get("score", -1e9))
                    if row_score > best_score:
                        best_score = row_score
                        best_row = row
            for failure in prev_failures:
                if isinstance(failure, dict):
                    failures.append(failure)
            log_progress(f"Resume loaded: completed={len(run_rows)}, failures={len(failures)}")
        except Exception as error:
            log_progress(f"Resume failed; starting fresh. reason={error}")

    started_at = time.perf_counter()
    total_trials = len(trial_plan)
    attempted_trials = 0
    log_progress("Tuning started.")

    try:
        for idx, (params, row_profile) in enumerate(trial_plan, start=1):
            if idx in completed_candidates:
                attempted_trials += 1
                elapsed = time.perf_counter() - started_at
                rate = (attempted_trials / elapsed) if elapsed > 0 else 0.0
                remaining = max(total_trials - attempted_trials, 0)
                eta = (remaining / rate) if rate > 0 else float("inf")
                eta_text = format_duration(eta) if eta != float("inf") else "--"
                log_progress(
                    f"Skip candidate {idx}/{total_trials} (already completed), "
                    f"progress={attempted_trials}/{total_trials}, elapsed={format_duration(elapsed)}, eta={eta_text}"
                )
                continue

            model_file = f"tune_top3_{idx}.joblib"
            report_file = f"tune_top3_{idx}.json"
            candidate_started_at = time.perf_counter()
            log_progress(
                f"Candidate {idx}/{total_trials} started: "
                f"rows={row_profile}, lr={params.get('learning_rate')}, leaves={params.get('num_leaves')}, n_estimators={params.get('n_estimators')}"
            )

            try:
                log_progress(f"Candidate {idx}/{total_trials}: training model...")
                result = train_and_evaluate(
                    frame=frame,
                    feature_columns=feature_columns,
                    label_column=label_column,
                    task=task,
                    model_name=model_name,
                    model_params=params,
                    train_end=split_cfg.get("train_end", "2022-12-31"),
                    valid_start=split_cfg.get("valid_start", "2023-01-01"),
                    valid_end=split_cfg.get("valid_end", "2023-12-31"),
                    max_train_rows=int(row_profile["max_train_rows"]),
                    max_valid_rows=int(row_profile["max_valid_rows"]),
                    early_stopping_rounds=training_cfg.get("early_stopping_rounds", 120),
                    allow_fallback=bool(training_cfg.get("allow_fallback_model", False)),
                    model_dir=output_cfg.get("model_dir", "artifacts/models"),
                    report_dir=output_cfg.get("report_dir", "artifacts/reports"),
                    model_file_name=model_file,
                    report_file_name=report_file,
                )

                model_path = ROOT / output_cfg.get("model_dir", "artifacts/models") / model_file
                valid_frame = _time_valid_slice(
                    frame,
                    valid_start=split_cfg.get("valid_start", "2020-01-01"),
                    valid_end=split_cfg.get("valid_end", "2021-07-31"),
                    max_valid_rows=int(row_profile["max_valid_rows"]),
                )
                score, roi_detail = objective_score(
                    model_path,
                    valid_frame,
                    feature_columns=[c for c in feature_columns if c in valid_frame.columns],
                    min_bet_ratio=float(args.min_bet_ratio),
                    min_bets_abs=int(args.min_bets_abs),
                    max_drawdown=constraints["max_drawdown"],
                    min_final_bankroll=constraints["min_final_bankroll"],
                    selection_mode=constraints["selection_mode"],
                )

                row = {
                    "candidate": idx,
                    "score": score,
                    "params": params,
                    "training": row_profile,
                    "roi_detail": roi_detail,
                    "metrics": result.metrics,
                    "model_file": model_file,
                    "report_file": report_file,
                    "status": "ok",
                }
                run_rows.append(row)
                log_progress(
                    f"Candidate {idx}/{total_trials} finished: score={score:.6f}, "
                    f"roi={roi_detail.get('roi')}, bets={roi_detail.get('bets')}, "
                    f"dd={roi_detail.get('max_drawdown')}, feasible={roi_detail.get('is_feasible')}"
                )
                gpu_flag = result.metrics.get("gpu_enabled")
                log_progress(f"Candidate {idx}/{total_trials}: gpu_enabled={gpu_flag}")

                row_feasible = bool((row.get("roi_detail") or {}).get("is_feasible"))
                current_best_feasible = bool((best_row or {}).get("roi_detail", {}).get("is_feasible")) if isinstance(best_row, dict) else False

                should_update = False
                if best_row is None:
                    should_update = True
                elif row_feasible and (not current_best_feasible):
                    should_update = True
                elif row_feasible == current_best_feasible and score > best_score:
                    should_update = True

                if should_update:
                    best_score = score
                    best_row = row
            except Exception as error:
                failure = {
                    "candidate": idx,
                    "params": params,
                    "training": row_profile,
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                    "status": "error",
                }
                failures.append(failure)
                log_progress(f"Candidate {idx}/{total_trials} failed: {error}")
            finally:
                attempted_trials += 1
                elapsed = time.perf_counter() - started_at
                rate = (attempted_trials / elapsed) if elapsed > 0 else 0.0
                remaining = max(total_trials - attempted_trials, 0)
                eta = (remaining / rate) if rate > 0 else float("inf")
                eta_text = format_duration(eta) if eta != float("inf") else "--"
                candidate_elapsed = time.perf_counter() - candidate_started_at
                log_progress(
                    f"Progress {attempted_trials}/{total_trials} ({(attempted_trials / total_trials):.1%}), "
                    f"candidate_elapsed={format_duration(candidate_elapsed)}, total_elapsed={format_duration(elapsed)}, eta={eta_text}"
                )
                log_progress("Writing partial summary...")
                _write_summary(
                    out_path=out_path,
                    args=args,
                    model_candidates=model_candidates,
                    row_profiles=row_profiles,
                    trial_plan=trial_plan,
                    run_rows=run_rows,
                    best_row=best_row,
                    failures=failures,
                    run_context=run_context,
                    leakage_audit=leakage_audit,
                    strategy_constraints=constraints,
                )
    except KeyboardInterrupt:
        log_progress("Interrupted by user; partial summary saved")
        return 130

    if best_row is None:
        raise RuntimeError("No tuning candidate was evaluated")

    _write_summary(
        out_path=out_path,
        args=args,
        model_candidates=model_candidates,
        row_profiles=row_profiles,
        trial_plan=trial_plan,
        run_rows=run_rows,
        best_row=best_row,
        failures=failures,
        run_context=run_context,
        leakage_audit=leakage_audit,
        strategy_constraints=constraints,
    )

    total_elapsed = time.perf_counter() - started_at
    log_progress(f"Summary saved: {out_path}")
    log_progress(f"Best candidate: {best_row['candidate']}")
    log_progress(f"Best score: {best_score:.6f}")
    log_progress(f"Best params: {best_row['params']}")
    log_progress(f"Best metrics: {best_row['metrics']}")
    log_progress(f"Tuning completed in {format_duration(total_elapsed)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
