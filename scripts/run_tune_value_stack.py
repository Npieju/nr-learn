import argparse
import itertools
import json
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

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.policy import compute_market_prob, evaluate_fixed_stake_summary
from racing_ml.evaluation.scoring import compose_value_blend_probabilities, predict_score, predict_target_values, prepare_scored_frame, topk_hit_rate
from racing_ml.features.builder import build_features
from racing_ml.features.selection import prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection
from racing_ml.models.value_blend import load_component_from_config


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[stack-tune {now}] {message}", flush=True)


def _parse_grid(raw: str | None, default_values: list[float]) -> list[float]:
    if raw is None or not str(raw).strip():
        return [float(value) for value in default_values]

    parsed: list[float] = []
    for token in str(raw).split(","):
        value = token.strip()
        if not value:
            continue
        parsed.append(float(value))
    if not parsed:
        raise ValueError("Parameter grid cannot be empty")
    return parsed


def _dedupe_grid(values: list[float]) -> list[float]:
    output: list[float] = []
    seen: set[float] = set()
    for value in values:
        rounded = round(float(value), 10)
        if rounded in seen:
            continue
        seen.add(rounded)
        output.append(float(value))
    return output


def _centered_grid(default: float, *, lower: float, upper: float, extra: list[float] | None = None) -> list[float]:
    values = [lower, default, upper]
    if extra:
        values.extend(extra)
    return _dedupe_grid([value for value in values if value > 0.0])


def _resolve_default_grid(params: dict, key: str, *, scales: tuple[float, ...], floor: float = 0.05) -> list[float]:
    default = float(params.get(key, 0.0))
    if default <= 0.0:
        return [0.0]
    values = [max(default * scale, floor) for scale in scales]
    values.append(default)
    return _dedupe_grid(sorted(values))


def _resolve_component_predictions(
    *,
    frame: pd.DataFrame,
    component_name: str,
    component_bundle: dict,
    fallback_selection,
) -> np.ndarray:
    model = component_bundle["model"]
    feature_selection = resolve_model_feature_selection(model, fallback_selection)
    x_eval = prepare_model_input_frame(frame, feature_selection.feature_columns, feature_selection.categorical_columns)
    if component_name == "time":
        return np.asarray(predict_target_values(model, x_eval), dtype=float).reshape(-1)
    return np.asarray(predict_score(model, x_eval, race_ids=frame["race_id"]), dtype=float).reshape(-1)


def _candidate_grid(component_names: set[str], params: dict, args: argparse.Namespace) -> list[dict[str, float]]:
    alpha_weight_grid = [0.0]
    alpha_scale_grid = [float(params.get("alpha_scale", 2.0))]
    if "alpha" in component_names:
        alpha_weight_grid = _parse_grid(
            args.alpha_weight_grid,
            _resolve_default_grid(params, "alpha_weight", scales=(0.5, 1.0, 1.5), floor=0.05),
        )
        alpha_scale_grid = _parse_grid(
            args.alpha_scale_grid,
            _centered_grid(float(params.get("alpha_scale", 2.0)), lower=max(float(params.get("alpha_scale", 2.0)) * 0.75, 0.5), upper=float(params.get("alpha_scale", 2.0)) * 1.5),
        )

    roi_weight_grid = [0.0]
    roi_scale_grid = [float(params.get("roi_scale", 2.0))]
    if "roi" in component_names:
        roi_weight_grid = _parse_grid(
            args.roi_weight_grid,
            _resolve_default_grid(params, "roi_weight", scales=(0.5, 1.0, 1.5), floor=0.05),
        )
        roi_scale_grid = _parse_grid(
            args.roi_scale_grid,
            _centered_grid(float(params.get("roi_scale", 2.0)), lower=max(float(params.get("roi_scale", 2.0)) * 0.75, 0.5), upper=float(params.get("roi_scale", 2.0)) * 1.5),
        )

    time_weight_grid = [0.0]
    time_scale_grid = [float(params.get("time_scale", 3.0))]
    if "time" in component_names:
        time_weight_grid = _parse_grid(
            args.time_weight_grid,
            _resolve_default_grid(params, "time_weight", scales=(0.5, 1.0, 1.5, 2.0), floor=0.05),
        )
        time_scale_grid = _parse_grid(
            args.time_scale_grid,
            _centered_grid(float(params.get("time_scale", 3.0)), lower=max(float(params.get("time_scale", 3.0)) * 0.5, 0.5), upper=float(params.get("time_scale", 3.0)) * 1.5),
        )

    market_grid = _parse_grid(
        args.market_blend_weight_grid,
        _centered_grid(float(params.get("market_blend_weight", 1.0)), lower=max(float(params.get("market_blend_weight", 1.0)) - 0.05, 0.70), upper=min(float(params.get("market_blend_weight", 1.0)) + 0.05, 0.98)),
    )

    grid: list[dict[str, float]] = []
    for alpha_weight, alpha_scale, roi_weight, roi_scale, time_weight, time_scale, market_weight in itertools.product(
        alpha_weight_grid,
        alpha_scale_grid,
        roi_weight_grid,
        roi_scale_grid,
        time_weight_grid,
        time_scale_grid,
        market_grid,
    ):
        grid.append(
            {
                "alpha_weight": float(alpha_weight),
                "alpha_scale": float(alpha_scale),
                "roi_weight": float(roi_weight),
                "roi_scale": float(roi_scale),
                "time_weight": float(time_weight),
                "time_scale": float(time_scale),
                "market_blend_weight": float(market_weight),
            }
        )
    return grid


def _merge_candidate_params(base_params: dict, candidate: dict[str, float]) -> dict:
    merged = dict(base_params)
    merged.update(candidate)
    return merged


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_catboost_value_stack_time.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features_catboost_rich.yaml")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--summary-path", default="artifacts/reports/tune_value_stack_summary.json")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--sort-by", default="top1_roi")
    parser.add_argument("--alpha-weight-grid", default=None)
    parser.add_argument("--alpha-scale-grid", default=None)
    parser.add_argument("--roi-weight-grid", default=None)
    parser.add_argument("--roi-scale-grid", default=None)
    parser.add_argument("--time-weight-grid", default=None)
    parser.add_argument("--time-scale-grid", default=None)
    parser.add_argument("--market-blend-weight-grid", default=None)
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=7, prefix="[stack-tune]", logger=log_progress, min_interval_sec=0.0)

        config = load_yaml(ROOT / args.config)
        data_cfg = load_yaml(ROOT / args.data_config)
        feature_cfg = load_yaml(ROOT / args.feature_config)
        progress.start("configs loaded")

        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        with Heartbeat("[stack-tune]", "loading training table", logger=log_progress):
            frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
        progress.update(message=f"training table loaded rows={len(frame):,}")

        with Heartbeat("[stack-tune]", "building features", logger=log_progress):
            frame = build_features(frame)
        progress.update(message=f"features built columns={len(frame.columns):,}")

        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()
        else:
            frame = frame.copy()
        progress.update(message=f"evaluation slice ready rows={len(frame):,} races={frame['race_id'].nunique():,}")

        label_col = str(config.get("label", "is_win"))
        fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
        if not fallback_selection.feature_columns:
            raise RuntimeError("No configured feature columns found for value stack tuning")

        component_cfg = config.get("components", {})
        if not isinstance(component_cfg, dict) or "win" not in component_cfg:
            raise RuntimeError("Value stack config must define at least a win component")

        component_bundles = {
            name: load_component_from_config(workspace_root=ROOT, config_path=path)
            for name, path in component_cfg.items()
        }
        component_predictions: dict[str, np.ndarray] = {}
        with Heartbeat("[stack-tune]", "precomputing component predictions", logger=log_progress):
            for name, bundle in component_bundles.items():
                component_predictions[name] = _resolve_component_predictions(
                    frame=frame,
                    component_name=name,
                    component_bundle=bundle,
                    fallback_selection=fallback_selection,
                )
        progress.update(message=f"component predictions ready components={','.join(component_bundles.keys())}")

        params = dict(config.get("model", {}).get("params", {}))
        candidate_grid = _candidate_grid(set(component_bundles.keys()), params, args)
        market_prob = None
        odds_column = str(params.get("odds_column", "odds"))
        if odds_column in frame.columns:
            market_prob = compute_market_prob(frame, odds_col=odds_column).to_numpy(dtype=float)
        progress.update(message=f"candidate grid ready total={len(candidate_grid):,}")

        candidate_progress = ProgressBar(total=len(candidate_grid), prefix="[stack-tune candidates]", logger=log_progress, min_interval_sec=0.0)
        candidate_progress.start(message=f"candidate search started total={len(candidate_grid):,}")

        base_frame = frame[[column for column in ["race_id", "rank", odds_column] if column in frame.columns]].copy()
        results: list[dict] = []
        with Heartbeat("[stack-tune]", "evaluating candidate grid", logger=log_progress):
            for candidate in candidate_grid:
                candidate_params = _merge_candidate_params(params, candidate)
                blended_prob = compose_value_blend_probabilities(
                    win_prob=component_predictions["win"],
                    params=candidate_params,
                    alpha_raw=component_predictions.get("alpha"),
                    roi_raw=component_predictions.get("roi"),
                    time_raw=component_predictions.get("time"),
                    market_prob=market_prob if candidate_params.get("market_blend_weight", 1.0) < 0.999 else None,
                )
                scored = prepare_scored_frame(base_frame, blended_prob, odds_col=odds_column if odds_column in base_frame.columns else None, score_col="score")
                summary = evaluate_fixed_stake_summary(
                    scored,
                    odds_col=odds_column if odds_column in scored.columns else None,
                    score_col="score",
                    stake=100.0,
                )
                results.append(
                    {
                        **candidate,
                        **summary,
                        "top1_hit_rate": topk_hit_rate(scored, 1),
                        "top3_hit_rate": topk_hit_rate(scored, 3),
                    }
                )
                candidate_progress.update(message=f"candidate top1_roi={results[-1].get('top1_roi')}")
        candidate_progress.complete(message="candidate search finished")
        progress.update(message="candidate grid evaluated")

        result_df = pd.DataFrame(results)
        if result_df.empty:
            raise RuntimeError("Value stack tuning produced no candidate results")

        if args.sort_by not in result_df.columns:
            raise ValueError(f"Unknown sort column: {args.sort_by}")

        sort_columns = [args.sort_by]
        if args.sort_by != "top1_roi" and "top1_roi" in result_df.columns:
            sort_columns.append("top1_roi")
        ascending = [False] * len(sort_columns)
        result_df = result_df.sort_values(sort_columns, ascending=ascending, na_position="last").reset_index(drop=True)
        progress.update(message=f"results ranked sort_by={args.sort_by}")

        summary_path = ROOT / args.summary_path
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        csv_path = summary_path.with_suffix(".csv")
        best_params = _merge_candidate_params(params, result_df.iloc[0][[
            "alpha_weight",
            "alpha_scale",
            "roi_weight",
            "roi_scale",
            "time_weight",
            "time_scale",
            "market_blend_weight",
        ]].to_dict())
        payload = {
            "config": args.config,
            "data_config": args.data_config,
            "feature_config": args.feature_config,
            "max_rows": int(len(frame)),
            "sort_by": args.sort_by,
            "candidate_count": int(len(result_df)),
            "best_params": best_params,
            "top_results": result_df.head(args.top_n).to_dict(orient="records"),
        }
        with Heartbeat("[stack-tune]", "writing tuning outputs", logger=log_progress):
            result_df.to_csv(csv_path, index=False)
            with summary_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, ensure_ascii=False, indent=2)
        progress.complete(message=f"tuning outputs written candidates={len(result_df):,}")

        print(f"[stack-tune] summary saved: {summary_path}")
        print(f"[stack-tune] csv saved: {csv_path}")
        print(result_df.head(args.top_n).to_string(index=False))
        print(f"[stack-tune] best_params={best_params}")
        return 0
    except KeyboardInterrupt:
        print("[stack-tune] interrupted by user")
        return 130
    except Exception as error:
        print(f"[stack-tune] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())