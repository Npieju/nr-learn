from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
        if int(pick.get("rank", 0)) == 1:
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


def run_backtest(config_path: str, predictions_file: str | None = None) -> None:
    _ = Path(config_path)
    predictions_dir = Path("artifacts/predictions")
    target_file = Path(predictions_file) if predictions_file else _latest_prediction_file(predictions_dir)
    if not target_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {target_file}")

    frame = pd.read_csv(target_file)
    required_cols = {"race_id", "score", "pred_rank"}
    missing = required_cols - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in prediction file: {sorted(missing)}")

    frame["pred_rank"] = pd.to_numeric(frame["pred_rank"], errors="coerce")
    frame["score"] = pd.to_numeric(frame["score"], errors="coerce")
    if "odds" in frame.columns:
        frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")
    if "expected_value" not in frame.columns and "odds" in frame.columns:
        frame["expected_value"] = frame["score"] * frame["odds"]
    frame = frame.dropna(subset=["pred_rank", "score"])

    metrics = {
        "prediction_file": str(target_file),
        "num_rows": int(len(frame)),
        "num_races": int(frame["race_id"].nunique()),
        "top1_hit_rate": _topk_hit_rate(frame, 1),
        "top3_hit_rate": _topk_hit_rate(frame, 3),
        "top5_hit_rate": _topk_hit_rate(frame, 5),
        "simple_top1_win_roi": _simple_win_roi(frame),
        "ev_top1_win_roi": _ev_top1_roi(frame),
    }

    report_dir = Path("artifacts/reports")
    report_dir.mkdir(parents=True, exist_ok=True)
    stem = target_file.stem.replace("predictions_", "")
    json_path = report_dir / f"backtest_{stem}.json"
    png_path = report_dir / f"backtest_{stem}.png"

    with json_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, ensure_ascii=False, indent=2)

    _plot_backtest(frame, png_path)

    print(f"[backtest] target predictions: {target_file}")
    print(f"[backtest] report saved: {json_path}")
    print(f"[backtest] chart saved: {png_path}")
    print(f"[backtest] metrics: {metrics}")
