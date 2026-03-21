from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import pandas as pd

from racing_ml.common.config import load_yaml
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.common.regime import resolve_regime_name
from racing_ml.pipeline.backtest_pipeline import run_backtest
from racing_ml.serving.predict_batch import prepare_prediction_frame, run_predict_from_frame
from racing_ml.serving.runtime_policy import resolve_runtime_policy


PROFILE_PRESETS: dict[str, dict[str, Any]] = {
    "best_policy_may": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "policy_may",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-07-20",
                "score_source": "default",
                "policy_name": "july_runtime_kelly",
            },
            {
                "date": "2024-08-10",
                "score_source": "default",
                "policy_name": "aug_runtime_portfolio",
            },
            {
                "date": "2024-09-14",
                "score_source": "default",
                "policy_name": "sep_runtime_portfolio",
            },
        ],
    },
    "best_policy_may_window": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "policy_may_window",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-05-26",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-01",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-02",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-08",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-09",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-16",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-22",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-23",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
        ],
    },
    "best_policy_may_may_weekends": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "best_policy_may_may_weekends",
        "cases": [
            {"date": "2024-05-04", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-05", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-11", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-12", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-18", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-19", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-25", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-26", "score_source": "may_runtime_liquidity", "policy_name": "may_runtime_kelly"},
        ],
    },
    "best_policy_may_test_partition_window": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may_test_partition.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "policy_may_test_partition_window",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-05-26",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-01",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-02",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-08",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-09",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "may_runtime_liquidity",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-16",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-22",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-23",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
        ],
    },
    "fallback_hybrid": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "fallback_hybrid",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-07-20",
                "score_source": "default",
                "policy_name": "july_runtime_kelly",
            },
            {
                "date": "2024-08-10",
                "score_source": "default",
                "policy_name": "aug_runtime_portfolio",
            },
            {
                "date": "2024-09-14",
                "score_source": "default",
                "policy_name": "sep_runtime_portfolio",
            },
        ],
    },
    "fallback_hybrid_june_strict": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "fallback_hybrid_june_strict",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "default",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-07-20",
                "score_source": "default",
                "policy_name": "july_runtime_kelly",
            },
            {
                "date": "2024-08-10",
                "score_source": "default",
                "policy_name": "aug_runtime_portfolio",
            },
            {
                "date": "2024-09-14",
                "score_source": "default",
                "policy_name": "sep_runtime_portfolio",
            },
        ],
    },
    "fallback_hybrid_window": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "fallback_hybrid_window",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-05-26",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-01",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-02",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-08",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-09",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-16",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-22",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
            {
                "date": "2024-06-23",
                "score_source": "default",
                "policy_name": "may_june_runtime_kelly",
            },
        ],
    },
    "fallback_hybrid_june_strict_window": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "fallback_hybrid_june_strict_window",
        "cases": [
            {
                "date": "2024-05-25",
                "score_source": "default",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-05-26",
                "score_source": "default",
                "policy_name": "may_runtime_kelly",
            },
            {
                "date": "2024-06-01",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-02",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-08",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-09",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-15",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-16",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-22",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
            {
                "date": "2024-06-23",
                "score_source": "default",
                "policy_name": "june_runtime_kelly",
            },
        ],
    },
    "fallback_hybrid_june_strict_may_weekends": {
        "config": "configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
        "data_config": "configs/data.yaml",
        "feature_config": "configs/features_catboost_rich_high_coverage_diag.yaml",
        "artifact_suffix": "fallback_hybrid_june_strict_may_weekends",
        "cases": [
            {"date": "2024-05-04", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-05", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-11", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-12", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-18", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-19", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-25", "score_source": "default", "policy_name": "may_runtime_kelly"},
            {"date": "2024-05-26", "score_source": "default", "policy_name": "may_runtime_kelly"},
        ],
    },
}

PROFILE_ALIASES: dict[str, str] = {
    "current_best_eval": "best_policy_may",
    "current_best_eval_window": "best_policy_may_window",
    "current_best_eval_may_weekends": "best_policy_may_may_weekends",
    "current_recommended_serving": "fallback_hybrid_june_strict",
    "current_recommended_serving_window": "fallback_hybrid_june_strict_window",
    "current_recommended_serving_may_weekends": "fallback_hybrid_june_strict_may_weekends",
}

LOCK_FILE = ROOT / "artifacts" / "reports" / "run_serving_smoke.lock"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-smoke {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _resolve_profile_name(profile_name: str) -> str:
    return PROFILE_ALIASES.get(profile_name, profile_name)


def _resolve_profile_defaults(resolved_profile: str) -> dict[str, str]:
    if resolved_profile in PROFILE_PRESETS:
        preset = PROFILE_PRESETS[resolved_profile]
        return {
            "config": str(preset["config"]),
            "data_config": str(preset["data_config"]),
            "feature_config": str(preset["feature_config"]),
            "artifact_suffix": str(preset["artifact_suffix"]),
        }

    model_profile = MODEL_RUN_PROFILES.get(resolved_profile)
    if model_profile is None:
        raise ValueError(f"Unknown serving smoke profile: {resolved_profile}")

    return {
        "config": model_profile.model_config,
        "data_config": model_profile.data_config,
        "feature_config": model_profile.feature_config,
        "artifact_suffix": resolved_profile,
    }


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _acquire_run_lock(lock_path: Path) -> None:
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            existing_pid = 0
            try:
                payload = json.loads(lock_path.read_text(encoding="utf-8"))
                existing_pid = int(payload.get("pid", 0) or 0)
            except Exception:
                existing_pid = 0

            if _pid_is_running(existing_pid):
                raise RuntimeError(
                    "Another run_serving_smoke.py process is already running "
                    f"(pid={existing_pid}). Wait for it to finish before starting another run."
                )

            lock_path.unlink(missing_ok=True)
            continue

        try:
            payload = {
                "pid": os.getpid(),
                "command": "run_serving_smoke.py",
                "workspace": ROOT.as_posix(),
            }
            os.write(fd, json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"))
        finally:
            os.close(fd)
        return


def _release_run_lock(lock_path: Path) -> None:
    try:
        payload = json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        payload = {}

    if int(payload.get("pid", 0) or 0) == os.getpid():
        lock_path.unlink(missing_ok=True)


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _date_tag(date_value: str) -> str:
    return pd.Timestamp(date_value).strftime("%Y%m%d")


def _prediction_paths(date_value: str) -> tuple[Path, Path]:
    date_tag = _date_tag(date_value)
    base_dir = ROOT / "artifacts" / "predictions"
    return base_dir / f"predictions_{date_tag}.csv", base_dir / f"predictions_{date_tag}.png"


def _backtest_paths(date_value: str) -> tuple[Path, Path]:
    date_tag = _date_tag(date_value)
    base_dir = ROOT / "artifacts" / "reports"
    return base_dir / f"backtest_{date_tag}.json", base_dir / f"backtest_{date_tag}.png"


def _copy_with_suffix(path: Path, suffix: str) -> str | None:
    if not path.exists():
        return None
    destination = path.with_name(f"{path.stem}_{suffix}{path.suffix}")
    shutil.copy2(path, destination)
    return _display_path(destination)


def _single_value(values: list[str], label: str) -> str:
    unique_values = sorted({value for value in values if value})
    if not unique_values:
        raise ValueError(f"Missing {label} in artifact")
    if len(unique_values) != 1:
        raise ValueError(f"Expected a single {label}, got {unique_values}")
    return unique_values[0]


def _prediction_summary(prediction_file: Path) -> dict[str, Any]:
    frame = pd.read_csv(prediction_file)
    if frame.empty:
        raise ValueError(f"Prediction output is empty: {prediction_file}")

    resolved_date = _single_value(frame["date"].astype(str).str[:10].tolist(), "prediction date")
    score_source = "default"
    if "score_source" in frame.columns:
        score_source = _single_value(frame["score_source"].fillna("default").astype(str).tolist(), "score_source")

    policy_name = ""
    if "policy_name" in frame.columns:
        policy_values = frame["policy_name"].dropna().astype(str).tolist()
        if policy_values:
            policy_name = _single_value(policy_values, "policy_name")

    score_source_model_config = ""
    if "score_source_model_config" in frame.columns:
        model_config_values = frame["score_source_model_config"].dropna().astype(str).tolist()
        if model_config_values:
            score_source_model_config = _single_value(model_config_values, "score_source_model_config")

    selected_rows = 0
    if "policy_selected" in frame.columns:
        selected_rows = int(frame["policy_selected"].fillna(False).astype(bool).sum())

    return {
        "resolved_date": resolved_date,
        "score_source": score_source,
        "score_source_model_config": score_source_model_config,
        "policy_name": policy_name,
        "policy_selected_rows": selected_rows,
        "num_rows": int(len(frame)),
    }


def _backtest_summary(report_file: Path) -> dict[str, Any]:
    with report_file.open("r", encoding="utf-8") as file:
        return json.load(file)


def _single_date_frame(date_value: str) -> pd.DataFrame:
    return pd.DataFrame({"date": [pd.Timestamp(date_value)]})


def _resolve_expected_score_source(model_config: dict[str, Any], date_value: str) -> str:
    serving_cfg = model_config.get("serving", {}) if isinstance(model_config.get("serving", {}), dict) else {}
    evaluation_cfg = model_config.get("evaluation", {}) if isinstance(model_config.get("evaluation", {}), dict) else {}

    score_regime_overrides = serving_cfg.get("score_regime_overrides", [])
    if not isinstance(score_regime_overrides, list) or not score_regime_overrides:
        score_regime_overrides = evaluation_cfg.get("score_regime_overrides", [])

    return resolve_regime_name(
        score_regime_overrides,
        frame=_single_date_frame(date_value),
        default_name="default",
    )


def _resolve_expected_policy_name(model_config: dict[str, Any], date_value: str) -> str:
    policy_resolution = resolve_runtime_policy(model_config, frame=_single_date_frame(date_value))
    if policy_resolution is None:
        raise ValueError(f"Runtime policy could not be resolved for date: {date_value}")
    policy_name, _ = policy_resolution
    return str(policy_name)


def _auto_case_for_date(model_config: dict[str, Any], date_value: str) -> dict[str, str]:
    normalized_date = str(pd.Timestamp(date_value).date())
    return {
        "date": normalized_date,
        "score_source": _resolve_expected_score_source(model_config, normalized_date),
        "policy_name": _resolve_expected_policy_name(model_config, normalized_date),
    }


def _validate_case(
    case: dict[str, Any],
    *,
    prediction_summary: dict[str, Any],
    backtest_summary: dict[str, Any],
    expected_config_path: Path,
) -> None:
    if prediction_summary["resolved_date"] != case["date"]:
        raise ValueError(
            f"Resolved date mismatch for {case['date']}: got {prediction_summary['resolved_date']}"
        )

    if prediction_summary["score_source"] != case["score_source"]:
        raise ValueError(
            f"Score source mismatch for {case['date']}: expected {case['score_source']} got {prediction_summary['score_source']}"
        )

    if prediction_summary["policy_name"] != case["policy_name"]:
        raise ValueError(
            f"Policy mismatch in predictions for {case['date']}: expected {case['policy_name']} got {prediction_summary['policy_name']}"
        )

    if str(backtest_summary.get("policy_name", "")) != case["policy_name"]:
        raise ValueError(
            f"Policy mismatch in backtest for {case['date']}: expected {case['policy_name']} got {backtest_summary.get('policy_name')}"
        )

    score_sources = backtest_summary.get("score_sources") if isinstance(backtest_summary.get("score_sources"), dict) else {}
    if sorted(str(key) for key in score_sources.keys()) != [case["score_source"]]:
        raise ValueError(
            f"Backtest score sources mismatch for {case['date']}: expected {[case['score_source']]} got {sorted(score_sources.keys())}"
        )

    if int(backtest_summary.get("policy_selected_rows", 0) or 0) != int(prediction_summary["policy_selected_rows"]):
        raise ValueError(
            f"Selected-row mismatch for {case['date']}: predictions={prediction_summary['policy_selected_rows']} backtest={backtest_summary.get('policy_selected_rows')}"
        )

    actual_config_path = Path(str(backtest_summary.get("config_file", ""))).resolve()
    if actual_config_path != expected_config_path.resolve():
        raise ValueError(
            f"Backtest config mismatch for {case['date']}: expected {expected_config_path} got {actual_config_path}"
        )


def _select_cases(profile_name: str, requested_dates: list[str] | None, model_config: dict[str, Any]) -> list[dict[str, Any]]:
    preset = PROFILE_PRESETS.get(profile_name)
    cases = list(preset["cases"]) if preset is not None else []
    if not requested_dates:
        if preset is None:
            raise ValueError(f"Profile {profile_name} requires at least one --date because it has no built-in case preset")
        return cases

    normalized_dates: list[str] = []
    seen_dates: set[str] = set()
    for date_value in requested_dates:
        normalized_date = str(pd.Timestamp(date_value).date())
        if normalized_date in seen_dates:
            continue
        seen_dates.add(normalized_date)
        normalized_dates.append(normalized_date)

    case_map = {str(case["date"]): case for case in cases}
    selected: list[dict[str, Any]] = []
    for normalized_date in normalized_dates:
        if normalized_date in case_map:
            selected.append(case_map[normalized_date])
            continue
        selected.append(_auto_case_for_date(model_config, normalized_date))
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(set(PROFILE_PRESETS) | set(PROFILE_ALIASES) | set(MODEL_RUN_PROFILES)), required=True)
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--no-archive-artifacts", action="store_true")
    parser.add_argument("--print-traceback", action="store_true")
    args = parser.parse_args()

    resolved_profile = _resolve_profile_name(args.profile)
    defaults = _resolve_profile_defaults(resolved_profile)
    config_path = _resolve_path(args.config or defaults["config"])
    data_config_path = _resolve_path(args.data_config or defaults["data_config"])
    feature_config_path = _resolve_path(args.feature_config or defaults["feature_config"])
    default_artifact_suffix = defaults["artifact_suffix"] if args.profile == resolved_profile else args.profile
    artifact_suffix = str(args.artifact_suffix or default_artifact_suffix).strip() or args.profile
    output_file = _resolve_path(args.output_file) if args.output_file else ROOT / "artifacts" / "reports" / f"serving_smoke_{args.profile}.json"
    lock_acquired = False

    try:
        progress = ProgressBar(total=4, prefix="[serving-smoke]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            message=(
                f"acquiring run lock profile={args.profile} resolved_profile={resolved_profile} "
                f"output={_display_path(output_file)}"
            )
        )
        _acquire_run_lock(LOCK_FILE)
        lock_acquired = True
        log_progress(f"run lock acquired path={_display_path(LOCK_FILE)}")
        model_config = load_yaml(config_path)
        cases = _select_cases(resolved_profile, args.date, model_config)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        os.chdir(ROOT)
        progress.update(message=f"profile resolved cases={len(cases)} artifact_suffix={artifact_suffix}")

        with Heartbeat("[serving-smoke]", "preparing shared prediction frame", logger=log_progress):
            shared_frame = prepare_prediction_frame(_display_path(data_config_path))
        print(
            f"[serving-smoke] shared frame ready rows={len(shared_frame):,} columns={len(shared_frame.columns):,}",
            flush=True,
        )
        progress.update(message=f"shared frame ready rows={len(shared_frame):,} columns={len(shared_frame.columns):,}")

        results: list[dict[str, Any]] = []
        overall_success = True
        case_progress = ProgressBar(total=max(len(cases), 1), prefix="[serving-smoke cases]", logger=log_progress, min_interval_sec=0.0)
        case_progress.start(message="starting case execution")

        for index, case in enumerate(cases, start=1):
            case_result: dict[str, Any] = {
                "date": case["date"],
                "expected_score_source": case["score_source"],
                "expected_policy_name": case["policy_name"],
            }
            prediction_csv, prediction_png = _prediction_paths(case["date"])
            backtest_json, backtest_png = _backtest_paths(case["date"])

            print(
                f"[serving-smoke] running date={case['date']} expected_score_source={case['score_source']} "
                f"expected_policy={case['policy_name']}",
                flush=True,
            )
            log_progress(f"case {index}/{len(cases)} started date={case['date']}")

            try:
                run_predict_from_frame(
                    model_config_path=_display_path(config_path),
                    feature_config_path=_display_path(feature_config_path),
                    frame=shared_frame,
                    race_date=case["date"],
                )
                if not prediction_csv.exists():
                    raise FileNotFoundError(f"Prediction artifact not found: {prediction_csv}")

                run_backtest(_display_path(config_path), _display_path(prediction_csv))
                if not backtest_json.exists():
                    raise FileNotFoundError(f"Backtest artifact not found: {backtest_json}")

                prediction_summary = _prediction_summary(prediction_csv)
                backtest_summary = _backtest_summary(backtest_json)
                _validate_case(
                    case,
                    prediction_summary=prediction_summary,
                    backtest_summary=backtest_summary,
                    expected_config_path=config_path,
                )

                archived_artifacts = {}
                if not args.no_archive_artifacts:
                    archived_artifacts = {
                        "prediction_csv": _copy_with_suffix(prediction_csv, artifact_suffix),
                        "prediction_png": _copy_with_suffix(prediction_png, artifact_suffix),
                        "backtest_json": _copy_with_suffix(backtest_json, artifact_suffix),
                        "backtest_png": _copy_with_suffix(backtest_png, artifact_suffix),
                    }

                case_result.update(
                    {
                        "status": "ok",
                        "prediction_file": _display_path(prediction_csv),
                        "backtest_file": _display_path(backtest_json),
                        "score_source": prediction_summary["score_source"],
                        "policy_name": prediction_summary["policy_name"],
                        "policy_selected_rows": prediction_summary["policy_selected_rows"],
                        "policy_bets": int(backtest_summary.get("policy_bets", 0) or 0),
                        "policy_roi": backtest_summary.get("policy_roi"),
                        "archived_artifacts": archived_artifacts,
                    }
                )
                print(
                    f"[serving-smoke] ok date={case['date']} score_source={prediction_summary['score_source']} "
                    f"policy={prediction_summary['policy_name']} bets={int(backtest_summary.get('policy_bets', 0) or 0)}",
                    flush=True,
                )
                case_progress.update(current=index, message=f"date={case['date']} status=ok")
            except Exception as error:
                overall_success = False
                case_result.update({"status": "failed", "error": str(error)})
                print(f"[serving-smoke] failed date={case['date']} error={error}", flush=True)
                if args.print_traceback:
                    traceback.print_exc()
                case_progress.update(current=index, message=f"date={case['date']} status=failed")

            results.append(case_result)
        progress.update(message=f"case loop finished success={overall_success}")

        summary = {
            "profile": args.profile,
            "resolved_profile": resolved_profile,
            "config_file": _display_path(config_path),
            "data_config_file": _display_path(data_config_path),
            "feature_config_file": _display_path(feature_config_path),
            "artifact_suffix": artifact_suffix,
            "cases": results,
        }
        with Heartbeat("[serving-smoke]", "writing summary output", logger=log_progress):
            output_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        progress.complete(message=f"summary saved path={_display_path(output_file)}")
        print(f"[serving-smoke] summary saved: {_display_path(output_file)}", flush=True)
        return 0 if overall_success else 1
    except KeyboardInterrupt:
        print("[serving-smoke] interrupted by user")
        return 130
    except Exception as error:
        print(f"[serving-smoke] failed: {error}")
        if args.print_traceback:
            traceback.print_exc()
        return 1
    finally:
        if lock_acquired:
            _release_run_lock(LOCK_FILE)
            log_progress(f"run lock released path={_display_path(LOCK_FILE)}")


if __name__ == "__main__":
    raise SystemExit(main())