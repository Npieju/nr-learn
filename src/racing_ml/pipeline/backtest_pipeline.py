from __future__ import annotations

import json
from pathlib import Path
import time

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.evaluation.policy import run_policy_strategy
from racing_ml.evaluation.scoring import resolve_odds_column
from racing_ml.serving.runtime_policy import annotate_runtime_policy, resolve_runtime_policy


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[backtest {now}] {message}", flush=True)


def _latest_prediction_file(predictions_dir: Path) -> Path:
    files = sorted(predictions_dir.glob("predictions_*.csv"))
    if not files:
        raise FileNotFoundError(f"No prediction files found under {predictions_dir}")
    return files[-1]


def _topk_hit_rate(frame: pd.DataFrame, k: int) -> float:
    if "rank" not in frame.columns:
        return float("nan")
    race_hits = []
    for _, group in frame.groupby("race_id"):
        picked = group[group["pred_rank"] <= k]
        hit = int((picked["rank"] == 1).any())
        race_hits.append(hit)
    return float(sum(race_hits) / len(race_hits)) if race_hits else float("nan")


def _simple_win_roi(frame: pd.DataFrame, stake_per_race: float = 100.0) -> float:
    if "rank" not in frame.columns:
        return float("nan")

    total_bet = 0.0
    total_return = 0.0

    for _, group in frame.groupby("race_id"):
        pick = group.sort_values("pred_rank").iloc[0]
        total_bet += stake_per_race
        rank = pd.to_numeric(pick.get("rank"), errors="coerce")
        if pd.notna(rank) and int(rank) == 1:
            odds = pd.to_numeric(pick.get("odds", 0), errors="coerce")
            if pd.isna(odds) or odds <= 0:
                odds = 1.0
            total_return += float(stake_per_race * odds)

    if total_bet == 0:
        return float("nan")
    return float(total_return / total_bet)


def _ev_top1_roi(frame: pd.DataFrame, stake_per_race: float = 100.0) -> float:
    if "rank" not in frame.columns or "expected_value" not in frame.columns:
        return float("nan")

    total_bet = 0.0
    total_return = 0.0

    for _, group in frame.groupby("race_id"):
        valid = group.dropna(subset=["expected_value"]) 
        if valid.empty:
            continue
        pick = valid.sort_values("expected_value", ascending=False).iloc[0]
        total_bet += stake_per_race
        rank = pd.to_numeric(pick.get("rank"), errors="coerce")
        if pd.notna(rank) and int(rank) == 1:
            odds = pd.to_numeric(pick.get("odds", 0), errors="coerce")
            if pd.isna(odds) or odds <= 0:
                odds = 1.0
            total_return += float(stake_per_race * odds)

    if total_bet == 0:
        return float("nan")
    return float(total_return / total_bet)


def _plot_backtest(frame: pd.DataFrame, out_path: Path) -> None:
    mean_scores = (
        frame.sort_values("pred_rank")
        .groupby("pred_rank", as_index=False)["score"]
        .mean()
        .head(10)
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(mean_scores["pred_rank"], mean_scores["score"], marker="o", color="#2563eb")
    axes[0].set_title("Mean score by predicted rank")
    axes[0].set_xlabel("Predicted rank")
    axes[0].set_ylabel("Mean score")

    top1 = frame[frame["pred_rank"] == 1].copy()
    if "rank" in top1.columns:
        top1["is_hit"] = (top1["rank"] == 1).astype(int)
        rolling = top1["is_hit"].rolling(window=50, min_periods=10).mean()
        axes[1].plot(rolling.index, rolling.values, color="#16a34a")
        axes[1].set_title("Rolling Top1 hit rate (window=50)")
        axes[1].set_xlabel("Index")
        axes[1].set_ylabel("Hit rate")
    else:
        axes[1].text(0.5, 0.5, "rank column not found", ha="center", va="center")
        axes[1].set_title("Hit rate chart unavailable")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_backtest(config_path: str, predictions_file: str | None = None, profile_name: str | None = None) -> None:
    resolved_config_path = Path(config_path)
    if not resolved_config_path.is_absolute():
        resolved_config_path = Path.cwd() / resolved_config_path
    model_config = load_yaml(resolved_config_path)
    predictions_dir = Path("artifacts/predictions")
    target_file = Path(predictions_file) if predictions_file else _latest_prediction_file(predictions_dir)
    if not target_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {target_file}")
    progress = ProgressBar(total=4, prefix="[backtest]", logger=log_progress, min_interval_sec=0.0)
    progress.start("starting backtest")
    progress.update(message=f"prediction target resolved file={target_file.name}")

    with Heartbeat("[backtest]", "loading prediction frame", logger=log_progress):
        frame = pd.read_csv(target_file)
    progress.update(message=f"prediction frame loaded rows={len(frame):,}")

    required_cols = {"race_id", "score", "pred_rank"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")

    with Heartbeat("[backtest]", "computing backtest metrics", logger=log_progress):
        frame["pred_rank"] = pd.to_numeric(frame["pred_rank"], errors="coerce")
        frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
        if "odds" in frame.columns:
            frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")
        if "expected_value" not in frame.columns and "odds" in frame.columns:
            frame["expected_value"] = frame["score"] * frame["odds"]
        frame = frame.dropna(subset=["pred_rank", "score"])

        metrics = {
            "prediction_file": str(target_file),
            "config_file": str(resolved_config_path),
            "num_rows": int(len(frame)),
            "num_races": int(frame["race_id"].nunique()),
            "top1_hit_rate": _topk_hit_rate(frame, 1),
            "top3_hit_rate": _topk_hit_rate(frame, 3),
            "top5_hit_rate": _topk_hit_rate(frame, 5),
            "simple_top1_win_roi": _simple_win_roi(frame),
            "ev_top1_win_roi": _ev_top1_roi(frame),
        }
        if profile_name is not None:
            metrics["profile"] = profile_name
        if "score_source" in frame.columns:
            score_source_counts = frame["score_source"].fillna("default").astype(str).value_counts().to_dict()
            metrics["score_source_count"] = int(len(score_source_counts))
            metrics["score_sources"] = {str(key): int(value) for key, value in score_source_counts.items()}

        odds_col = resolve_odds_column(frame)
        policy_resolution = resolve_runtime_policy(model_config, frame=frame)
        if odds_col is not None and policy_resolution is not None:
            policy_name, policy_config = policy_resolution
            policy_frame = annotate_runtime_policy(
                frame,
                odds_col=odds_col,
                policy_name=policy_name,
                policy_config=policy_config,
                score_col="score",
            )
            policy_metrics = run_policy_strategy(
                policy_frame,
                prob_col="policy_prob",
                odds_col=odds_col,
                params=policy_config,
            )
            policy_strategy_kind = str(policy_config.get("strategy_kind", "")).strip().lower()
            selected_mask = policy_frame["policy_selected"].fillna(False).astype(bool)
            metrics["policy_name"] = policy_name
            metrics["policy_strategy_kind"] = policy_strategy_kind
            metrics["policy_blend_weight"] = float(policy_config.get("blend_weight", 1.0))
            metrics["policy_selected_rows"] = int(selected_mask.sum())
            metrics["policy_selected_races"] = int(policy_frame.loc[selected_mask, "race_id"].nunique()) if selected_mask.any() else 0
            if policy_strategy_kind == "portfolio":
                metrics["policy_roi"] = policy_metrics.get("portfolio_roi")
                metrics["policy_bets"] = int(policy_metrics.get("portfolio_bets") or 0)
                metrics["policy_hit_rate"] = policy_metrics.get("portfolio_hit_rate")
                metrics["policy_final_bankroll"] = policy_metrics.get("portfolio_final_bankroll")
                metrics["policy_max_drawdown"] = policy_metrics.get("portfolio_max_drawdown")
                metrics["policy_avg_synthetic_odds"] = policy_metrics.get("portfolio_avg_synthetic_odds")
            else:
                metrics["policy_roi"] = policy_metrics.get("kelly_roi")
                metrics["policy_bets"] = int(policy_metrics.get("kelly_bets") or 0)
                metrics["policy_hit_rate"] = policy_metrics.get("kelly_hit_rate")
                metrics["policy_final_bankroll"] = policy_metrics.get("kelly_final_bankroll")
                metrics["policy_max_drawdown"] = policy_metrics.get("kelly_max_drawdown")
    progress.update(message=f"metrics computed races={metrics['num_races']:,}")

    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = target_file.stem.replace("predictions_", "")
    json_path = report_dir / f"backtest_{stem}.json"
    png_path = report_dir / f"backtest_{stem}.png"

    with Heartbeat("[backtest]", "writing backtest outputs", logger=log_progress):
        with json_path.open("w", encoding="utf-8") as file:
            json.dump(metrics, file, ensure_ascii=False, indent=2)

        _plot_backtest(frame, png_path)
    progress.complete(message="backtest outputs written")

    print(f"[backtest] target predictions: {target_file}")
    print(f"[backtest] report saved: {json_path}")
    print(f"[backtest] chart saved: {png_path}")
    print(f"[backtest] metrics: {metrics}")
