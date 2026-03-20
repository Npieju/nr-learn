import argparse
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import resolve_output_artifacts, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles, resolve_model_run_profile
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.evaluation.policy import add_market_signals, evaluate_fixed_stake_summary
from racing_ml.evaluation.scoring import generate_prediction_outputs, prepare_scored_frame, resolve_odds_column, topk_hit_rate
from racing_ml.features.builder import build_features
from racing_ml.features.selection import FeatureSelection, prepare_model_input_frame, resolve_feature_selection, resolve_model_feature_selection


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[ab {now}] {message}", flush=True)


def _display_path(path: Path) -> str:
    resolved = path if path.is_absolute() else (ROOT / path).resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _date_window_payload(frame: pd.DataFrame) -> dict[str, str | int | None]:
    if "date" not in frame.columns:
        return {
            "start_date": None,
            "end_date": None,
            "start_month": None,
            "end_month": None,
        }

    date_series = pd.to_datetime(frame["date"], errors="coerce").dropna().sort_values().reset_index(drop=True)
    if date_series.empty:
        return {
            "start_date": None,
            "end_date": None,
            "start_month": None,
            "end_month": None,
        }

    start_ts = pd.Timestamp(date_series.iloc[0])
    end_ts = pd.Timestamp(date_series.iloc[-1])
    return {
        "start_date": str(start_ts.date()),
        "end_date": str(end_ts.date()),
        "start_month": int(start_ts.month),
        "end_month": int(end_ts.month),
    }


def _resolve_shared_config_path(
    explicit_path: str | None,
    candidate_paths: list[str],
    *,
    kind: str,
) -> str:
    if explicit_path:
        return explicit_path

    unique_candidates = sorted({path for path in candidate_paths if path})
    if not unique_candidates:
        raise ValueError(f"Could not resolve shared {kind} config")
    if len(unique_candidates) > 1:
        raise ValueError(
            f"Multiple {kind} configs resolved from selected profiles: {unique_candidates}. "
            f"Specify --{kind}-config explicitly."
        )
    return unique_candidates[0]


def _compare_target_slug(profile_name: str | None, config_path: str) -> str:
    return profile_name or Path(config_path).stem


def _build_output_manifest(
    *,
    base_profile: str | None,
    challenger_profile: str | None,
    base_config: str,
    challenger_config: str,
    row_count: int,
    date_window: dict[str, Any],
) -> tuple[Path, dict[str, str]]:
    report_dir = ROOT / "artifacts" / "reports"
    latest_path = report_dir / "ab_compare_summary.json"
    base_slug = _compare_target_slug(base_profile, base_config)
    challenger_slug = _compare_target_slug(challenger_profile, challenger_config)
    date_bits: list[str] = []
    if date_window.get("start_date"):
        date_bits.append(str(date_window["start_date"]).replace("-", ""))
    if date_window.get("end_date"):
        date_bits.append(str(date_window["end_date"]).replace("-", ""))
    date_slug = f"_{'_'.join(date_bits)}" if date_bits else ""
    versioned_path = report_dir / f"ab_compare_summary_{base_slug}_vs_{challenger_slug}{date_slug}_rows_{row_count}.json"
    return versioned_path, {
        "latest_summary": _display_path(latest_path),
        "versioned_summary": _display_path(versioned_path),
    }


def evaluate_model(name: str, model_cfg: dict, frame: pd.DataFrame, feature_cfg: dict, label_col: str) -> dict[str, Any]:
    output_cfg = model_cfg.get("output", {})
    output_artifacts = resolve_output_artifacts(output_cfg)
    model_path = output_artifacts.model_path if output_artifacts.model_path.is_absolute() else (ROOT / output_artifacts.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
    feature_selection = resolve_model_feature_selection(model, fallback_selection)
    eval_frame = frame.copy()
    x_eval = prepare_model_input_frame(eval_frame, feature_selection.feature_columns, feature_selection.categorical_columns)
    odds_col = resolve_odds_column(eval_frame)
    outputs = generate_prediction_outputs(model, x_eval, race_ids=eval_frame["race_id"])
    eval_frame = prepare_scored_frame(eval_frame, outputs.score, odds_col=odds_col, score_col="score")
    if odds_col is not None:
        eval_frame = add_market_signals(eval_frame, score_col="score", odds_col=odds_col)

    policy_summary = evaluate_fixed_stake_summary(eval_frame, odds_col=odds_col, score_col="score", stake=100.0)
    metrics = {
        "model": name,
        "label_column": label_col,
        "model_file": str(model_path.relative_to(ROOT)),
        "manifest_file": output_artifacts.manifest_path.as_posix() if (ROOT / output_artifacts.manifest_path).exists() else None,
        "feature_count": int(len(feature_selection.feature_columns)),
        "categorical_feature_count": int(len(feature_selection.categorical_columns)),
        "n_rows": int(len(eval_frame)),
        "n_races": int(eval_frame["race_id"].nunique()),
        "top1_hit_rate": topk_hit_rate(eval_frame, 1),
        "top3_hit_rate": topk_hit_rate(eval_frame, 3),
        "top1_roi": policy_summary.get("top1_roi"),
    }

    if label_col in eval_frame.columns:
        y_true = eval_frame[label_col].astype(int).to_numpy()
        if len(np.unique(y_true)) > 1:
            metrics["auc"] = float(roc_auc_score(y_true, eval_frame["score"].to_numpy()))
        else:
            metrics["auc"] = None

    return metrics


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--base-profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--challenger-profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--base-config", default=None)
    parser.add_argument("--challenger-config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--max-rows", type=int, default=30000)
    args = parser.parse_args()

    try:
        if args.list_profiles:
            print(format_model_run_profiles())
            return 0

        if args.base_profile and args.base_config is not None:
            raise ValueError("--base-profile cannot be combined with --base-config")
        if args.challenger_profile and args.challenger_config is not None:
            raise ValueError("--challenger-profile cannot be combined with --challenger-config")

        base_profile, base_config_path, base_data_config, base_feature_config = resolve_model_run_profile(
            args.base_profile,
            default_model_config=args.base_config or "configs/model.yaml",
            default_data_config=args.data_config or "configs/data.yaml",
            default_feature_config=args.feature_config or "configs/features.yaml",
        )
        challenger_profile, challenger_config_path, challenger_data_config, challenger_feature_config = resolve_model_run_profile(
            args.challenger_profile,
            default_model_config=args.challenger_config or "configs/model_ranker.yaml",
            default_data_config=args.data_config or "configs/data.yaml",
            default_feature_config=args.feature_config or "configs/features.yaml",
        )
        data_config_path = _resolve_shared_config_path(
            args.data_config,
            [base_data_config, challenger_data_config],
            kind="data",
        )
        feature_config_path = _resolve_shared_config_path(
            args.feature_config,
            [base_feature_config, challenger_feature_config],
            kind="feature",
        )

        base_cfg = load_yaml(ROOT / base_config_path)
        challenger_cfg = load_yaml(ROOT / challenger_config_path)
        data_cfg = load_yaml(ROOT / data_config_path)
        feature_cfg = load_yaml(ROOT / feature_config_path)
        progress = ProgressBar(total=7, prefix="[ab]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"configs loaded base={base_profile or base_config_path} "
                f"challenger={challenger_profile or challenger_config_path}"
            )
        )

        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        with Heartbeat("[ab]", "loading training table", logger=log_progress):
            frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
        progress.update(message=f"training table loaded rows={len(frame):,}")

        with Heartbeat("[ab]", "building features", logger=log_progress):
            frame = build_features(frame)
        progress.update(message=f"features built columns={len(frame.columns):,}")

        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(args.max_rows).copy()
        progress.update(message=f"evaluation slice ready rows={len(frame):,}")

        base_label_col = str(base_cfg.get("label", "is_win"))
        challenger_label_col = str(challenger_cfg.get("label", "is_win"))
        base_selection = resolve_feature_selection(frame, feature_cfg, label_column=base_label_col)
        challenger_selection = resolve_feature_selection(frame, feature_cfg, label_column=challenger_label_col)
        if not base_selection.feature_columns:
            raise RuntimeError("No configured feature columns found for baseline comparison")
        if not challenger_selection.feature_columns:
            raise RuntimeError("No configured feature columns found for challenger comparison")
        progress.update(
            message=(
                f"feature selection ready base={len(base_selection.feature_columns):,} "
                f"challenger={len(challenger_selection.feature_columns):,}"
            )
        )

        with Heartbeat("[ab]", "evaluating baseline model", logger=log_progress):
            base_metrics = evaluate_model("baseline", base_cfg, frame, feature_cfg, base_label_col)
        progress.update(message="baseline evaluated")

        with Heartbeat("[ab]", "evaluating challenger model", logger=log_progress):
            challenger_metrics = evaluate_model("challenger", challenger_cfg, frame, feature_cfg, challenger_label_col)
        progress.update(message="challenger evaluated")

        labels_match = base_label_col == challenger_label_col
        same_model_artifact = base_metrics.get("model_file") == challenger_metrics.get("model_file")
        same_manifest_artifact = (
            base_metrics.get("manifest_file") == challenger_metrics.get("manifest_file")
            if base_metrics.get("manifest_file") and challenger_metrics.get("manifest_file")
            else None
        )
        warnings: list[str] = []
        if not labels_match:
            warnings.append(
                f"label mismatch: baseline={base_label_col} challenger={challenger_label_col}; auc delta is omitted"
            )
        if same_model_artifact:
            warnings.append(
                "baseline and challenger resolved to the same model_file; "
                "metric deltas reflect identical underlying artifacts unless downstream config behavior differs"
            )
        if same_manifest_artifact:
            warnings.append(
                "baseline and challenger resolved to the same manifest_file; compare is artifact-identical"
            )

        date_window = _date_window_payload(frame)
        versioned_out_path, artifact_manifest = _build_output_manifest(
            base_profile=base_profile,
            challenger_profile=challenger_profile,
            base_config=base_config_path,
            challenger_config=challenger_config_path,
            row_count=int(len(frame)),
            date_window=date_window,
        )

        summary = {
            "base_profile": base_profile,
            "challenger_profile": challenger_profile,
            "base_config": base_config_path,
            "challenger_config": challenger_config_path,
            "data_config": data_config_path,
            "feature_config": feature_config_path,
            "max_rows": int(len(frame)),
            "label_columns_match": labels_match,
            "distinct_model_artifacts": not same_model_artifact,
            "distinct_manifest_artifacts": (not same_manifest_artifact) if same_manifest_artifact is not None else None,
            "comparison_warnings": warnings,
            "date_window": date_window,
            "run_context": {
                "base_profile": base_profile,
                "challenger_profile": challenger_profile,
                "base_config": base_config_path,
                "challenger_config": challenger_config_path,
                "data_config": data_config_path,
                "feature_config": feature_config_path,
                "requested_max_rows": args.max_rows,
                "actual_rows": int(len(frame)),
            },
            "artifact_manifest": artifact_manifest,
            "baseline": base_metrics,
            "challenger": challenger_metrics,
            "delta": {
                "auc": (
                    (challenger_metrics.get("auc") or 0.0) - (base_metrics.get("auc") or 0.0)
                    if labels_match and challenger_metrics.get("auc") is not None and base_metrics.get("auc") is not None
                    else None
                ),
                "top1_hit_rate": (
                    (challenger_metrics.get("top1_hit_rate") or 0.0) - (base_metrics.get("top1_hit_rate") or 0.0)
                    if challenger_metrics.get("top1_hit_rate") is not None and base_metrics.get("top1_hit_rate") is not None
                    else None
                ),
                "top1_roi": (
                    (challenger_metrics.get("top1_roi") or 0.0) - (base_metrics.get("top1_roi") or 0.0)
                    if challenger_metrics.get("top1_roi") is not None and base_metrics.get("top1_roi") is not None
                    else None
                ),
            },
        }

        out_path = ROOT / "artifacts/reports/ab_compare_summary.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with Heartbeat("[ab]", "writing comparison summary", logger=log_progress):
            write_json(out_path, summary)
            write_json(versioned_out_path, summary)
        progress.complete(message="comparison summary written")

        print(f"[ab] summary saved: {out_path}")
        print(f"[ab] versioned summary saved: {versioned_out_path}")
        print(f"[ab] baseline: {base_metrics}")
        print(f"[ab] challenger: {challenger_metrics}")
        print(f"[ab] delta: {summary['delta']}")
        if warnings:
            print(f"[ab] warnings: {warnings}")
        return 0
    except KeyboardInterrupt:
        print("[ab] interrupted by user")
        return 130
    except Exception as error:
        print(f"[ab] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())