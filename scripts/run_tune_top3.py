import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import warnings

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.artifacts import (
    build_model_manifest,
    build_training_report_payload,
    derive_manifest_file_name,
    resolve_output_artifacts,
    write_json,
)
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.leakage import run_leakage_audit
from racing_ml.evaluation.policy import PolicyConstraints, add_market_signals, evaluate_flat_strategy_catalog
from racing_ml.evaluation.scoring import generate_prediction_outputs, prepare_scored_frame, resolve_odds_column
from racing_ml.features.builder import build_features
from racing_ml.models.trainer import train_and_evaluate


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


def _override_constraints(base: PolicyConstraints, args: argparse.Namespace) -> PolicyConstraints:
    return PolicyConstraints(
        min_bet_ratio=float(args.min_bet_ratio) if args.min_bet_ratio is not None else float(base.min_bet_ratio),
        min_bets_abs=int(args.min_bets_abs) if args.min_bets_abs is not None else int(base.min_bets_abs),
        max_drawdown=float(base.max_drawdown),
        min_final_bankroll=float(base.min_final_bankroll),
        selection_mode=str(base.selection_mode),
    )


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


def objective_score(
    model_path: Path,
    valid_frame: pd.DataFrame,
    feature_columns: list[str],
    constraints: PolicyConstraints,
) -> tuple[float, dict[str, float | int | str | None]]:
    if valid_frame.empty:
        return float("-inf"), {
            "strategy": "top1",
            "threshold": None,
            "roi": None,
            "bets": 0,
            "hit_rate": None,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "selection_score": None,
            "is_feasible": False,
            "gate_failures": ["empty_validation_slice"],
        }

    odds_col = resolve_odds_column(valid_frame)
    if odds_col is None:
        return float("-inf"), {
            "strategy": "top1",
            "threshold": None,
            "roi": None,
            "bets": 0,
            "hit_rate": None,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "selection_score": None,
            "is_feasible": False,
            "gate_failures": ["missing_odds_column"],
        }

    model = joblib.load(model_path)
    x_valid = valid_frame[feature_columns]
    outputs = generate_prediction_outputs(model, x_valid, race_ids=valid_frame["race_id"])
    scored = prepare_scored_frame(valid_frame, outputs.score, odds_col=odds_col, score_col="score")
    scored = add_market_signals(scored, score_col="score", odds_col=odds_col)

    best_detail, candidate_rows = evaluate_flat_strategy_catalog(
        scored,
        constraints=constraints,
        score_col="score",
        odds_col=odds_col,
        stake_per_bet=1.0,
    )
    best_detail = dict(best_detail)
    best_detail["candidate_count"] = int(len(candidate_rows))

    selection_score = best_detail.get("selection_score")
    if selection_score is None:
        roi_value = best_detail.get("roi")
        if roi_value is None:
            return float("-inf"), best_detail
        return float(roi_value), best_detail
    return float(selection_score), best_detail


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
    policy_constraints: dict,
) -> None:
    summary = {
        "base_config": args.config,
        "objective": "validation_policy_gate_then_roi",
        "n_model_candidates": len(model_candidates),
        "n_row_profiles": len(row_profiles),
        "n_trials": len(trial_plan),
        "n_candidates": len(run_rows),
        "best": best_row,
        "runs": run_rows,
        "failures": failures,
        "run_context": run_context,
        "leakage_audit": leakage_audit,
        "policy_constraints": policy_constraints,
        "strategy_constraints": policy_constraints,
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
    parser.add_argument("--min-bet-ratio", type=float)
    parser.add_argument("--min-bets-abs", type=int)
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
    output_artifacts = resolve_output_artifacts(output_cfg)
    model_name = model_cfg.get("model", {}).get("name", "lightgbm")
    device_type = str(base_params.get("device_type", "cpu")).strip().lower() or "cpu"
    leakage_cfg = evaluation_cfg.get("leakage_audit", {})
    leakage_enabled = bool(leakage_cfg.get("enabled", True))

    base_constraints = PolicyConstraints.from_config(evaluation_cfg)
    constraints = _override_constraints(base_constraints, args)

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
        "rows_total": int(len(frame)),
        "device_type": device_type,
        "artifact_model_dir": output_artifacts.model_dir.as_posix(),
        "artifact_report_dir": output_artifacts.report_dir.as_posix(),
        **constraints.to_dict(),
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
                    row_score = float(row.get("score", float("-inf")))
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
            candidate_output_cfg = dict(output_cfg)
            candidate_output_cfg["model_file"] = model_file
            candidate_output_cfg["report_file"] = report_file
            candidate_output_cfg["manifest_file"] = derive_manifest_file_name(model_file)
            candidate_artifacts = resolve_output_artifacts(candidate_output_cfg)
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

                model_path = result.model_path if result.model_path.is_absolute() else (ROOT / result.model_path)
                valid_frame = _time_valid_slice(
                    frame,
                    valid_start=split_cfg.get("valid_start", "2020-01-01"),
                    valid_end=split_cfg.get("valid_end", "2021-07-31"),
                    max_valid_rows=int(row_profile["max_valid_rows"]),
                )
                available_features = [column for column in feature_columns if column in valid_frame.columns]
                score, roi_detail = objective_score(
                    model_path,
                    valid_frame,
                    feature_columns=available_features,
                    constraints=constraints,
                )

                candidate_run_context = {
                    **run_context,
                    "candidate": int(idx),
                    "params": params,
                    "rows_train_max": int(row_profile["max_train_rows"]),
                    "rows_valid_max": int(row_profile["max_valid_rows"]),
                    "split_train_end": split_cfg.get("train_end", "2022-12-31"),
                    "split_valid_start": split_cfg.get("valid_start", "2023-01-01"),
                    "split_valid_end": split_cfg.get("valid_end", "2023-12-31"),
                    "artifact_model": candidate_artifacts.model_path.as_posix(),
                    "artifact_report": candidate_artifacts.report_path.as_posix(),
                    "artifact_manifest": candidate_artifacts.manifest_path.as_posix(),
                }
                report_payload = build_training_report_payload(
                    metrics=result.metrics,
                    run_context=candidate_run_context,
                    leakage_audit=leakage_audit,
                    policy_constraints=constraints.to_dict(),
                    extra_metadata={
                        "objective": "validation_policy_gate_then_roi",
                        "training_profile": row_profile,
                        "roi_detail": roi_detail,
                    },
                )
                write_json(result.report_path, report_payload)

                manifest_abs = candidate_artifacts.manifest_path if candidate_artifacts.manifest_path.is_absolute() else (ROOT / candidate_artifacts.manifest_path)
                manifest_payload = build_model_manifest(
                    workspace_root=ROOT,
                    model_config_path=ROOT / args.config,
                    data_config_path=ROOT / args.data_config,
                    feature_config_path=ROOT / args.feature_config,
                    model_path=result.model_path,
                    report_path=result.report_path,
                    task=task,
                    label_column=label_column,
                    model_name=model_name,
                    used_features=result.used_features,
                    metrics=result.metrics,
                    run_context=candidate_run_context,
                    leakage_audit=leakage_audit,
                    policy_constraints=constraints.to_dict(),
                    extra_metadata={
                        "candidate": int(idx),
                        "params": params,
                        "training_profile": row_profile,
                        "objective": "validation_policy_gate_then_roi",
                        "roi_detail": roi_detail,
                    },
                )
                write_json(manifest_abs, manifest_payload)

                row = {
                    "candidate": idx,
                    "score": score,
                    "params": params,
                    "training": row_profile,
                    "roi_detail": roi_detail,
                    "metrics": result.metrics,
                    "model_file": model_file,
                    "report_file": report_file,
                    "manifest_file": candidate_artifacts.manifest_path.as_posix(),
                    "status": "ok",
                }
                run_rows.append(row)
                log_progress(
                    f"Candidate {idx}/{total_trials} finished: score={score:.6f}, "
                    f"roi={roi_detail.get('roi')}, bets={roi_detail.get('bets')}, "
                    f"dd={roi_detail.get('max_drawdown')}, feasible={roi_detail.get('is_feasible')}"
                )
                log_progress(f"Candidate {idx}/{total_trials}: gpu_enabled={result.metrics.get('gpu_enabled')}")

                row_feasible = bool((row.get("roi_detail") or {}).get("is_feasible"))
                current_best_feasible = bool((best_row or {}).get("roi_detail", {}).get("is_feasible")) if isinstance(best_row, dict) else False

                should_update = False
                if best_row is None:
                    should_update = True
                elif row_feasible and not current_best_feasible:
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
                    policy_constraints=constraints.to_dict(),
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
        policy_constraints=constraints.to_dict(),
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