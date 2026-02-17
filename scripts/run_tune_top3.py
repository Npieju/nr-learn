import argparse
import json
from pathlib import Path
import sys
import traceback

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
from racing_ml.features.builder import build_features
from racing_ml.models.trainer import train_and_evaluate
from racing_ml.serving.predict_batch import _predict_multi_position


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
) -> tuple[float | None, int, float | None]:
    total_bet = 0.0
    total_return = 0.0
    bets = 0
    hits = 0

    for _, group in scored.groupby("race_id"):
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
        else:
            raise ValueError(f"Unsupported strategy mode: {mode}")

        bets += 1
        total_bet += 1.0
        rank = pd.to_numeric(pd.Series([pick.get("rank")]), errors="coerce").iloc[0]
        odds = pd.to_numeric(pd.Series([pick.get("odds")]), errors="coerce").iloc[0]
        if pd.notna(rank) and int(rank) == 1 and pd.notna(odds) and float(odds) > 0:
            hits += 1
            total_return += float(odds)

    if total_bet == 0:
        return None, 0, None
    roi = float(total_return / total_bet)
    hit_rate = float(hits / bets) if bets > 0 else None
    return roi, int(bets), hit_rate


def objective_score(
    model_path: Path,
    valid_frame: pd.DataFrame,
    feature_columns: list[str],
    min_bet_ratio: float = 0.02,
    min_bets_abs: int = 30,
) -> tuple[float, dict[str, float | int | str | None]]:
    model = joblib.load(model_path)
    x_valid = valid_frame[feature_columns]
    score = _predict_top1_score(model, x_valid, valid_frame["race_id"])

    scored = valid_frame.copy()
    scored["score"] = score
    scored["odds"] = pd.to_numeric(scored.get("odds"), errors="coerce")
    scored["expected_value"] = scored["score"] * scored["odds"]

    n_races = int(scored["race_id"].nunique())
    min_bets = max(int(n_races * min_bet_ratio), int(min_bets_abs))

    candidates: list[dict[str, float | int | str | None]] = []

    top1_roi, top1_bets, top1_hit = _strategy_roi(scored, mode="top1")
    candidates.append(
        {
            "strategy": "top1",
            "threshold": None,
            "roi": top1_roi,
            "bets": top1_bets,
            "hit_rate": top1_hit,
        }
    )

    for min_score in [0.16, 0.18, 0.20, 0.22, 0.24, 0.27]:
        for odds_min in [1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                roi, bets, hit = _strategy_roi(
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
                    }
                )

    for threshold in [1.00, 1.05, 1.10, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00]:
        for odds_min in [1.2, 1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                roi, bets, hit = _strategy_roi(
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
                    }
                )

    best = {
        "strategy": "top1",
        "threshold": None,
        "roi": 0.0,
        "bets": 0,
        "hit_rate": None,
    }
    best_score = -1e9
    for row in candidates:
        roi = row["roi"]
        bets = int(row["bets"] or 0)
        if roi is None:
            continue
        score_value = float(roi)
        if bets < min_bets:
            score_value -= 1.0
        if score_value > best_score:
            best_score = score_value
            best = row

    best["min_bets_required"] = int(min_bets)
    best["n_races"] = int(n_races)
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
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)


def main() -> int:
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

    model_cfg = load_yaml(ROOT / args.config)
    data_cfg = load_yaml(ROOT / args.data_config)
    feature_cfg = load_yaml(ROOT / args.feature_config)

    raw_dir = data_cfg.get("dataset", {}).get("raw_dir", "data/raw")
    split_cfg = data_cfg.get("split", {})
    frame = load_training_table(str(ROOT / raw_dir))
    frame = build_features(frame)

    feature_columns = feature_cfg.get("features", {}).get("base", []) + feature_cfg.get("features", {}).get("history", [])
    label_column = model_cfg.get("label", "is_win")
    task = str(model_cfg.get("task", "multi_position"))

    base_params = model_cfg.get("model", {}).get("params", {})
    training_cfg = model_cfg.get("training", {})
    output_cfg = model_cfg.get("output", {})
    model_name = model_cfg.get("model", {}).get("name", "lightgbm")

    model_candidates = build_model_candidates(base_params)[: max(1, args.max_candidates)]
    row_profiles = build_row_profiles(training_cfg)[: max(1, args.max_row_profiles)]

    trial_plan: list[tuple[dict, dict[str, int]]] = []
    for model_candidate in model_candidates:
        for row_profile in row_profiles:
            trial_plan.append((model_candidate, row_profile))
    trial_plan = trial_plan[: max(1, args.max_trials)]

    out_path = ROOT / args.summary_path
    run_rows: list[dict] = []
    failures: list[dict] = []
    best_row: dict | None = None
    best_score = -1e9

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
            print(f"[tune] resume: loaded {len(run_rows)} completed candidates and {len(failures)} failures")
        except Exception as error:
            print(f"[tune] resume failed; start fresh. reason={error}")

    try:
        for idx, (params, row_profile) in enumerate(trial_plan, start=1):
            if idx in completed_candidates:
                print(f"[tune] skip candidate {idx}/{len(trial_plan)} (already completed)")
                continue

            model_file = f"tune_top3_{idx}.joblib"
            report_file = f"tune_top3_{idx}.json"
            print(
                f"[tune] candidate {idx}/{len(trial_plan)}: "
                f"params={params}, rows={row_profile}"
            )

            try:
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
                print(
                    f"[tune] candidate={idx} score={score:.6f} "
                    f"roi={roi_detail.get('roi')} bets={roi_detail.get('bets')} "
                    f"strategy={roi_detail.get('strategy')} threshold={roi_detail.get('threshold')}"
                )

                if score > best_score:
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
                print(f"[tune] candidate={idx} failed: {error}")
            finally:
                _write_summary(
                    out_path=out_path,
                    args=args,
                    model_candidates=model_candidates,
                    row_profiles=row_profiles,
                    trial_plan=trial_plan,
                    run_rows=run_rows,
                    best_row=best_row,
                    failures=failures,
                )
    except KeyboardInterrupt:
        print("[tune] interrupted; partial summary saved")
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
    )

    print(f"[tune] summary saved: {out_path}")
    print(f"[tune] best candidate: {best_row['candidate']}")
    print(f"[tune] best score: {best_score:.6f}")
    print(f"[tune] best params: {best_row['params']}")
    print(f"[tune] best metrics: {best_row['metrics']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
