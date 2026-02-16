from __future__ import annotations

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from racing_ml.common.config import load_yaml
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.features.builder import build_features


def _resolve_target_date(frame: pd.DataFrame, race_date: str | None) -> pd.Timestamp:
    available_dates = frame["date"].dropna().sort_values().unique()
    if len(available_dates) == 0:
        raise ValueError("No dated rows found in dataset")

    if race_date is None:
        return pd.to_datetime(available_dates[-1])

    target = pd.to_datetime(race_date)
    candidates = [date for date in available_dates if pd.to_datetime(date) <= target]
    if candidates:
        return pd.to_datetime(candidates[-1])
    return pd.to_datetime(available_dates[0])


def _plot_predictions(predictions: pd.DataFrame, out_path: Path) -> None:
    if predictions.empty:
        return

    race_counts = predictions["race_id"].value_counts()
    target_race_id = race_counts.index[0]
    race_frame = predictions[predictions["race_id"] == target_race_id].copy()
    race_frame = race_frame.sort_values("score", ascending=False).head(12)

    labels = race_frame["horse_id"].astype(str)
    scores = race_frame["score"].astype(float)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].barh(labels.iloc[::-1], scores.iloc[::-1], color="#3b82f6")
    axes[0].set_title(f"Top Scores: race_id={target_race_id}")
    axes[0].set_xlabel("Predicted win score")

    axes[1].hist(predictions["score"].astype(float), bins=20, color="#10b981", alpha=0.85)
    axes[1].set_title("Score Distribution")
    axes[1].set_xlabel("Predicted win score")
    axes[1].set_ylabel("Count")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def run_predict(
    model_config_path: str,
    data_config_path: str,
    feature_config_path: str,
    race_date: str | None = None,
) -> None:
    model_config = load_yaml(Path(model_config_path))
    data_config = load_yaml(Path(data_config_path))
    feature_config = load_yaml(Path(feature_config_path))

    raw_dir = data_config.get("dataset", {}).get("raw_dir", "data/raw")
    frame = load_training_table(raw_dir)
    frame = build_features(frame)

    target_date = _resolve_target_date(frame, race_date)
    pred_frame = frame[frame["date"] == target_date].copy()
    if pred_frame.empty:
        raise ValueError(f"No races found for target date: {target_date}")

    features_cfg = feature_config.get("features", {})
    feature_columns = features_cfg.get("base", []) + features_cfg.get("history", [])
    available_features = [column for column in feature_columns if column in pred_frame.columns]
    if not available_features:
        raise ValueError("No feature columns available for prediction")

    model_dir = Path(model_config.get("output", {}).get("model_dir", "artifacts/models"))
    model_path = model_dir / "baseline_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    pred_frame["score"] = model.predict_proba(pred_frame[available_features])[:, 1]
    pred_frame["pred_rank"] = (
        pred_frame.groupby("race_id")["score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    if "odds" in pred_frame.columns:
        pred_frame["odds"] = pd.to_numeric(pred_frame["odds"], errors="coerce")
        pred_frame["expected_value"] = pred_frame["score"] * pred_frame["odds"]
        ev_rank = pred_frame.groupby("race_id")["expected_value"].rank(method="first", ascending=False)
        pred_frame["ev_rank"] = ev_rank.astype("Int64")

    columns = ["date", "race_id", "horse_id"]
    if "horse_name" in pred_frame.columns:
        columns.append("horse_name")
    if "rank" in pred_frame.columns:
        columns.append("rank")
    if "odds" in pred_frame.columns:
        columns.append("odds")
    columns += ["score", "pred_rank"]
    if "expected_value" in pred_frame.columns:
        columns += ["expected_value", "ev_rank"]

    out_dir = Path("artifacts/predictions")
    out_dir.mkdir(parents=True, exist_ok=True)

    date_tag = pd.Timestamp(target_date).strftime("%Y%m%d")
    csv_path = out_dir / f"predictions_{date_tag}.csv"
    png_path = out_dir / f"predictions_{date_tag}.png"

    output = pred_frame[columns].sort_values(["race_id", "pred_rank"]).reset_index(drop=True)
    output.to_csv(csv_path, index=False)
    _plot_predictions(output, png_path)

    print(f"[predict] target date: {pd.Timestamp(target_date).date()}")
    print(f"[predict] predictions saved: {csv_path}")
    print(f"[predict] chart saved: {png_path}")
    print(f"[predict] records: {len(output)}")
