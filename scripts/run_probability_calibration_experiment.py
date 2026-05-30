from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from racing_ml.common.artifacts import display_path, ensure_output_file_path, utc_now_iso, write_json
from racing_ml.serving.runtime_policy import annotate_runtime_policy


def _resolve_glob(pattern: str) -> list[Path]:
    paths = sorted(ROOT.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return paths


def _read_prediction_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = display_path(path, workspace_root=ROOT)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _require_columns(frame: pd.DataFrame, columns: list[str], *, label: str) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")


def _fit_calibrator(method: str, train_score: np.ndarray, train_label: np.ndarray) -> Any:
    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip")
        model.fit(train_score, train_label)
        return model
    if method == "platt":
        model = LogisticRegression(max_iter=1000)
        model.fit(train_score.reshape(-1, 1), train_label)
        return model
    raise ValueError(f"Unsupported calibration method: {method}")


def _predict_calibrated(method: str, model: Any, score: np.ndarray) -> np.ndarray:
    if method == "isotonic":
        return np.asarray(model.transform(score), dtype=float)
    if method == "platt":
        return np.asarray(model.predict_proba(score.reshape(-1, 1))[:, 1], dtype=float)
    raise ValueError(f"Unsupported calibration method: {method}")


def _extract_policy_config(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    strategy = str(frame.get("policy_strategy_kind", pd.Series(["portfolio"])).dropna().iloc[0]).strip().lower()
    policy_name = str(frame.get("policy_name", pd.Series(["calibrated_policy"])).dropna().iloc[0]).strip() or "calibrated_policy"
    config: dict[str, Any] = {
        "strategy_kind": strategy,
        "blend_weight": float(_numeric(frame, "policy_blend_weight").dropna().iloc[0]),
        "min_prob": float(_numeric(frame, "policy_min_prob").dropna().iloc[0]),
        "odds_min": float(_numeric(frame, "policy_odds_min").dropna().iloc[0]),
        "odds_max": float(_numeric(frame, "policy_odds_max").dropna().iloc[0]),
    }
    if strategy == "kelly":
        config.update(
            {
                "min_edge": float(_numeric(frame, "policy_min_edge").dropna().iloc[0]),
                "fractional_kelly": float(_numeric(frame, "policy_fractional_kelly").dropna().iloc[0]),
                "max_fraction": float(_numeric(frame, "policy_max_fraction").dropna().iloc[0]),
            }
        )
    elif strategy == "portfolio":
        config.update(
            {
                "top_k": int(_numeric(frame, "policy_top_k").dropna().iloc[0]),
                "min_expected_value": float(_numeric(frame, "policy_min_expected_value").dropna().iloc[0]),
            }
        )
    else:
        raise ValueError(f"Unsupported policy_strategy_kind for calibration replay: {strategy}")
    return policy_name, config


def _apply_exposure_guards(
    frame: pd.DataFrame,
    calibrated: np.ndarray,
    *,
    base_col: str,
    top_popularity_max: int | None,
    non_top_max_lift: float,
    shrinkage: float,
) -> np.ndarray:
    base = _numeric(frame, base_col).to_numpy(dtype=float)
    raw = np.clip(np.asarray(calibrated, dtype=float), 1e-6, 1.0 - 1e-6)
    blended = ((1.0 - shrinkage) * base) + (shrinkage * raw)

    popularity = _numeric(frame, "popularity")
    if top_popularity_max is not None:
        top_mask = (popularity <= top_popularity_max).fillna(False).to_numpy(dtype=bool)
        guarded = base.copy()
        guarded[top_mask] = blended[top_mask]
        lift_cap = base * (1.0 + max(non_top_max_lift, 0.0))
        guarded[~top_mask] = np.minimum(blended[~top_mask], lift_cap[~top_mask])
        return np.clip(guarded, 1e-6, 1.0 - 1e-6)

    return np.clip(blended, 1e-6, 1.0 - 1e-6)


def _bucket_summary(frame: pd.DataFrame, score_col: str) -> list[dict[str, Any]]:
    popularity = _numeric(frame, "popularity")
    buckets = {
        "favorite": popularity == 1,
        "second_favorite": popularity == 2,
        "third_favorite": popularity == 3,
        "pop_4_6": (popularity >= 4) & (popularity <= 6),
        "pop_7_10": (popularity >= 7) & (popularity <= 10),
        "pop_11_plus": popularity >= 11,
    }
    rows: list[dict[str, Any]] = []
    rank = _numeric(frame, "rank")
    odds = _numeric(frame, "odds")
    score = _numeric(frame, score_col)
    policy_ev = _numeric(frame, "policy_expected_value")
    selected = frame["policy_selected"].fillna(False).astype(bool) if "policy_selected" in frame.columns else pd.Series(False, index=frame.index)
    for bucket, mask in buckets.items():
        group = frame.loc[mask.fillna(False)].copy()
        if group.empty:
            continue
        idx = group.index
        rows.append(
            {
                "bucket": bucket,
                "rows": int(len(group)),
                "win_rate": float((rank.loc[idx] == 1).mean()),
                "mean_odds": float(odds.loc[idx].mean()),
                "mean_score": float(score.loc[idx].mean()),
                "mean_policy_expected_value": float(policy_ev.loc[idx].mean()),
                "mean_return": float(odds.loc[idx].where(rank.loc[idx] == 1, 0.0).mean()),
                "policy_selected_rows": int(selected.loc[idx].sum()),
            }
        )
    return rows


def _race_sum_summary(frame: pd.DataFrame, score_col: str) -> dict[str, float | int]:
    race_sum = _numeric(frame, score_col).groupby(frame["race_id"]).sum(min_count=1)
    return {
        "race_count": int(race_sum.shape[0]),
        "sum_mean": float(race_sum.mean()),
        "sum_abs_error_mean": float((race_sum - 1.0).abs().mean()),
        "sum_min": float(race_sum.min()),
        "sum_max": float(race_sum.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and replay a strict train/test probability calibration experiment.")
    parser.add_argument("--train-glob", required=True)
    parser.add_argument("--test-glob", required=True)
    parser.add_argument("--method", choices=["isotonic", "platt"], default="isotonic")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--label-col", default="rank")
    parser.add_argument("--top-popularity-max", type=int, default=3)
    parser.add_argument("--non-top-max-lift", type=float, default=0.0)
    parser.add_argument("--shrinkage", type=float, default=1.0)
    parser.add_argument("--output-dir", default="artifacts/predictions/calibration_experiment")
    parser.add_argument("--artifact-suffix", default="prob_calibration_experiment")
    parser.add_argument("--output-report", default="artifacts/reports/probability_calibration_experiment.json")
    args = parser.parse_args()

    train_paths = _resolve_glob(args.train_glob)
    test_paths = _resolve_glob(args.test_glob)
    train = _read_prediction_files(train_paths)
    test_all = _read_prediction_files(test_paths)
    _require_columns(train, [args.score_col, args.label_col], label="train")
    _require_columns(test_all, [args.score_col, "race_id", "odds"], label="test")

    train_score = _numeric(train, args.score_col)
    if args.label_col == "rank":
        train_label = (_numeric(train, args.label_col) == 1).astype(int)
    else:
        train_label = _numeric(train, args.label_col).astype(int)
    valid = train_score.notna() & train_label.notna()
    if int(valid.sum()) < 100:
        raise ValueError(f"Not enough calibration rows: {int(valid.sum())}")

    calibrator = _fit_calibrator(args.method, train_score.loc[valid].to_numpy(dtype=float), train_label.loc[valid].to_numpy(dtype=int))

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[str] = []
    replay_frames: list[pd.DataFrame] = []
    for source_file, test in test_all.groupby("__source_file", sort=False):
        test = test.copy()
        raw_calibrated = _predict_calibrated(args.method, calibrator, _numeric(test, args.score_col).to_numpy(dtype=float))
        test["calibrated_score_raw"] = raw_calibrated
        test["calibrated_score"] = _apply_exposure_guards(
            test,
            raw_calibrated,
            base_col=args.score_col,
            top_popularity_max=args.top_popularity_max,
            non_top_max_lift=args.non_top_max_lift,
            shrinkage=args.shrinkage,
        )
        policy_name, policy_config = _extract_policy_config(test)
        replay = annotate_runtime_policy(
            test,
            odds_col="odds",
            policy_name=f"{policy_name}_calibrated",
            policy_config=policy_config,
            score_col="calibrated_score",
        )
        replay["expected_value"] = _numeric(replay, "calibrated_score") * _numeric(replay, "odds")
        replay["score"] = replay["calibrated_score"]
        replay["pred_rank"] = replay.groupby("race_id")["score"].rank(method="first", ascending=False).astype("Int64")
        replay["ev_rank"] = replay.groupby("race_id")["expected_value"].rank(method="first", ascending=False).astype("Int64")
        source_name = Path(str(source_file)).name
        output_path = output_dir / source_name.replace(".csv", f"_{args.artifact_suffix}.csv")
        replay.drop(columns=["__source_file"], errors="ignore").to_csv(output_path, index=False)
        output_paths.append(display_path(output_path, workspace_root=ROOT))
        replay_frames.append(replay)

    replay_all = pd.concat(replay_frames, ignore_index=True)
    selected = replay_all["policy_selected"].fillna(False).astype(bool)
    report = {
        "created_at": utc_now_iso(),
        "method": args.method,
        "train_files": [display_path(path, workspace_root=ROOT) for path in train_paths],
        "test_files": [display_path(path, workspace_root=ROOT) for path in test_paths],
        "output_files": output_paths,
        "train_rows": int(len(train)),
        "calibration_rows": int(valid.sum()),
        "test_rows": int(len(replay_all)),
        "test_races": int(replay_all["race_id"].nunique()),
        "top_popularity_max": args.top_popularity_max,
        "non_top_max_lift": args.non_top_max_lift,
        "shrinkage": args.shrinkage,
        "policy_selected_rows": int(selected.sum()),
        "policy_selected_races": int(replay_all.loc[selected, "race_id"].nunique()) if selected.any() else 0,
        "selected_roi": None
        if not selected.any()
        else float(_numeric(replay_all.loc[selected], "odds").where(_numeric(replay_all.loc[selected], "rank") == 1, 0.0).mean()),
        "race_sum": _race_sum_summary(replay_all, "score"),
        "bucket_summary": _bucket_summary(replay_all, "score"),
    }
    output_report = ensure_output_file_path(args.output_report, label="output-report", workspace_root=ROOT)
    write_json(output_report, report)
    print(f"wrote {display_path(output_report, workspace_root=ROOT)}")
    for path in output_paths:
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
