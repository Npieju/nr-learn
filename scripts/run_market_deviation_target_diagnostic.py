import argparse
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table_for_feature_build


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[target-diag {now}] {message}", flush=True)


def _compute_market_prob(frame: pd.DataFrame, *, odds_clip: float, market_prob_floor: float) -> pd.Series:
    odds = pd.to_numeric(frame["odds"], errors="coerce").clip(lower=1.01, upper=float(odds_clip))
    implied = 1.0 / odds.replace(0, np.nan)
    denom = implied.groupby(frame["race_id"]).transform("sum")
    return (implied / denom.replace(0, np.nan)).clip(
        lower=float(market_prob_floor),
        upper=1.0 - float(market_prob_floor),
    )


def _compute_observed_prob(frame: pd.DataFrame, *, label_column: str, market_prob_floor: float) -> pd.Series:
    if "rank" in frame.columns:
        win_label = (pd.to_numeric(frame["rank"], errors="coerce") == 1).astype(float)
    elif label_column in frame.columns:
        win_label = pd.to_numeric(frame[label_column], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    else:
        raise ValueError("market_deviation diagnostic requires 'rank' or label column")
    eps = float(market_prob_floor)
    return win_label.clip(lower=eps, upper=1.0 - eps)


def _quantiles(values: pd.Series, points: list[float]) -> dict[str, float | None]:
    numeric = pd.to_numeric(values, errors="coerce")
    if not bool(numeric.notna().any()):
        return {str(point): None for point in points}
    return {str(point): float(numeric.quantile(point)) for point in points}


def _summarize_series(values: pd.Series, *, lower_clip: float | None = None, upper_clip: float | None = None) -> dict[str, object]:
    numeric = pd.to_numeric(values, errors="coerce")
    finite = numeric[np.isfinite(numeric.to_numpy(dtype=float, na_value=np.nan))]
    if finite.empty:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "positive_rate": None,
            "zero_or_above_rate": None,
            "quantiles": {},
            "lower_clip_rate": None,
            "upper_clip_rate": None,
        }

    values_np = finite.to_numpy(dtype=float)
    summary = {
        "count": int(values_np.size),
        "mean": float(np.mean(values_np)),
        "std": float(np.std(values_np)),
        "min": float(np.min(values_np)),
        "max": float(np.max(values_np)),
        "positive_rate": float(np.mean(values_np > 0.0)),
        "zero_or_above_rate": float(np.mean(values_np >= 0.0)),
        "quantiles": _quantiles(finite, [0.001, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 0.999]),
        "lower_clip_rate": None,
        "upper_clip_rate": None,
    }
    if lower_clip is not None:
        summary["lower_clip_rate"] = float(np.mean(values_np <= float(lower_clip)))
    if upper_clip is not None:
        summary["upper_clip_rate"] = float(np.mean(values_np >= float(upper_clip)))
    return summary


def _build_report(frame: pd.DataFrame, *, label_column: str, odds_clip: float, market_prob_floor: float, target_clip: float) -> dict[str, object]:
    market_prob = _compute_market_prob(frame, odds_clip=odds_clip, market_prob_floor=market_prob_floor)
    observed_prob = _compute_observed_prob(frame, label_column=label_column, market_prob_floor=market_prob_floor)
    observed_logit = np.log(observed_prob / (1.0 - observed_prob))
    market_logit = np.log(market_prob / (1.0 - market_prob))
    raw_target = pd.Series((observed_logit - market_logit).astype(float), index=frame.index, name="raw_target")
    clipped_target = raw_target.clip(lower=-float(target_clip), upper=float(target_clip)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    race_mean = clipped_target.groupby(frame["race_id"]).transform("mean")
    race_std = clipped_target.groupby(frame["race_id"]).transform("std").replace(0.0, np.nan)
    race_normalized_preview = ((clipped_target - race_mean) / race_std).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    race_counts = frame.groupby("race_id").size()
    race_clip_stats = pd.DataFrame(
        {
            "race_id": frame["race_id"].astype(str),
            "lower_clipped": (raw_target <= -float(target_clip)).astype(int),
            "upper_clipped": (raw_target >= float(target_clip)).astype(int),
        }
    ).groupby("race_id").sum()

    return {
        "run_context": {
            "rows": int(len(frame)),
            "n_races": int(frame["race_id"].nunique()),
            "label_column": label_column,
            "odds_clip": float(odds_clip),
            "market_prob_floor": float(market_prob_floor),
            "target_clip": float(target_clip),
        },
        "market_prob": _summarize_series(market_prob),
        "raw_target": _summarize_series(raw_target, lower_clip=-float(target_clip), upper_clip=float(target_clip)),
        "clipped_target": _summarize_series(clipped_target, lower_clip=-float(target_clip), upper_clip=float(target_clip)),
        "race_normalized_preview": _summarize_series(race_normalized_preview),
        "clip_surface": {
            "races_with_lower_clip": int((race_clip_stats["lower_clipped"] > 0).sum()),
            "races_with_upper_clip": int((race_clip_stats["upper_clipped"] > 0).sum()),
            "mean_lower_clipped_rows_per_race": float(race_clip_stats["lower_clipped"].mean()),
            "mean_upper_clipped_rows_per_race": float(race_clip_stats["upper_clipped"].mean()),
            "race_size_quantiles": _quantiles(race_counts.astype(float), [0.1, 0.25, 0.5, 0.75, 0.9]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--model-config", default="configs/model_lightgbm_alpha_cpu_diag.yaml")
    parser.add_argument("--pre-feature-max-rows", type=int, default=150000)
    parser.add_argument("--summary-output", default="")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=4, prefix="[target-diag]", logger=log_progress, min_interval_sec=0.0)
        progress.start("starting market deviation target diagnostic")

        data_cfg = load_yaml(ROOT / args.data_config)
        model_cfg = load_yaml(ROOT / args.model_config)
        progress.update(message="configs loaded")

        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        with Heartbeat("[target-diag]", "loading training table", logger=log_progress):
            load_result = load_training_table_for_feature_build(
                raw_dir,
                pre_feature_max_rows=int(args.pre_feature_max_rows) if args.pre_feature_max_rows else None,
                dataset_config=dataset_cfg,
                base_dir=ROOT,
            )
        frame = load_result.frame
        progress.update(message=f"training table loaded rows={len(frame):,} races={frame['race_id'].nunique():,}")

        model_params = dict(model_cfg.get("model", {}).get("params", {}))
        label_column = str(model_cfg.get("label", "is_win"))
        report = _build_report(
            frame,
            label_column=label_column,
            odds_clip=float(model_params.get("odds_clip", 30.0)),
            market_prob_floor=float(model_params.get("market_prob_floor", 1e-4)),
            target_clip=float(model_params.get("target_clip", 8.0)),
        )
        report["run_context"].update(
            {
                "data_config": args.data_config,
                "model_config": args.model_config,
                "loaded_rows": int(load_result.loaded_rows),
                "pre_feature_rows": int(load_result.pre_feature_rows),
                "data_load_strategy": load_result.data_load_strategy,
                "primary_source_rows_total": load_result.primary_source_rows_total,
            }
        )
        progress.update(message="target diagnostic computed")

        summary_output = ROOT / (
            args.summary_output
            or f"artifacts/reports/market_deviation_target_diagnostic_{Path(args.model_config).stem}.json"
        )
        artifact_ensure_output_file_path(summary_output, label="summary output", workspace_root=ROOT)
        with Heartbeat("[target-diag]", "writing report file", logger=log_progress):
            write_json(summary_output, report)
        progress.complete(message="report file written")

        print(f"[target-diag] summary saved: {summary_output}")
        print(f"[target-diag] raw_target positive_rate={report['raw_target']['positive_rate']}")
        print(f"[target-diag] clipped_target positive_rate={report['clipped_target']['positive_rate']}")
        print(f"[target-diag] race_normalized_preview positive_rate={report['race_normalized_preview']['positive_rate']}")
        return 0
    except KeyboardInterrupt:
        print("[target-diag] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[target-diag] failed: {error}")
        return 1
    except Exception as error:
        print(f"[target-diag] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())