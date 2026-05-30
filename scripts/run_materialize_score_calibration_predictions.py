from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
import traceback
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import display_path, ensure_output_file_path, utc_now_iso, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles, resolve_model_run_profile
from racing_ml.common.progress import ProgressBar
from racing_ml.serving.predict_batch import prepare_prediction_frame, run_predict_from_frame

NO_MODEL_ARTIFACT_SUFFIX = "__NO_MODEL_ARTIFACT_SUFFIX__"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[calibration-materialize {now}] {message}", flush=True)


def _date_values(args: argparse.Namespace) -> list[str]:
    if args.dates:
        values = [value.strip() for value in args.dates.split(",") if value.strip()]
        if not values:
            raise ValueError("--dates did not contain any usable date")
        return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in values]

    if not args.start_date or not args.end_date:
        raise ValueError("Either --dates or both --start-date/--end-date are required")
    start = pd.Timestamp(args.start_date)
    end = pd.Timestamp(args.end_date)
    if end < start:
        raise ValueError("--end-date must be on or after --start-date")
    return [date.strftime("%Y-%m-%d") for date in pd.date_range(start, end, freq="D")]


def _resolve_output_suffix(template: str, *, artifact_suffix: str) -> str:
    return template.format(artifact_suffix=artifact_suffix)


def _ensure_non_calibrating_source(model_config_path: str, *, allow_calibrating_profile: bool) -> None:
    config = load_yaml(ROOT / model_config_path)
    serving_cfg = config.get("serving", {})
    calibration_cfg = serving_cfg.get("score_calibration") if isinstance(serving_cfg, dict) else None
    if isinstance(calibration_cfg, dict) and calibration_cfg.get("enabled", False) and not allow_calibrating_profile:
        raise ValueError(
            "Refusing to materialize calibration train predictions from a profile with "
            "serving.score_calibration.enabled=true. Use a non-calibrating source profile "
            "or pass --allow-calibrating-profile intentionally."
        )


def _manifest_payload(
    *,
    args: argparse.Namespace,
    resolved_profile: str | None,
    model_config_path: str,
    data_config_path: str,
    feature_config_path: str,
    dates: list[str],
    output_suffix: str,
    outputs: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    return {
        "created_at": utc_now_iso(),
        "status": status,
        "profile": resolved_profile,
        "model_config": model_config_path,
        "data_config": data_config_path,
        "feature_config": feature_config_path,
        "model_artifact_suffix": None if args.model_artifact_suffix == NO_MODEL_ARTIFACT_SUFFIX else args.model_artifact_suffix,
        "output_file_suffix_template": args.output_file_suffix,
        "output_file_suffix": output_suffix,
        "output_artifact_suffix": args.output_artifact_suffix or args.model_artifact_suffix,
        "dates": dates,
        "date_count": len(dates),
        "outputs": outputs,
        "calibration_train_glob": f"artifacts/predictions/predictions_*_{output_suffix}.csv",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Materialize revision-scoped prediction files for score calibration training.")
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--model-artifact-suffix", required=True)
    parser.add_argument("--output-artifact-suffix", default=None)
    parser.add_argument("--output-file-suffix", required=True)
    parser.add_argument("--dates", default=None, help="Comma-separated YYYY-MM-DD dates.")
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument("--manifest-output", default="artifacts/reports/score_calibration_prediction_materialize_manifest.json")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-calibrating-profile", action="store_true")
    args = parser.parse_args()

    if args.list_profiles:
        print(format_model_run_profiles())
        return 0

    try:
        if args.profile and any(value is not None for value in (args.config, args.data_config, args.feature_config)):
            raise ValueError("--profile cannot be combined with --config, --data-config, or --feature-config")
        resolved_profile, model_config_path, data_config_path, feature_config_path = resolve_model_run_profile(
            args.profile,
            default_model_config=args.config or "configs/model.yaml",
            default_data_config=args.data_config or "configs/data.yaml",
            default_feature_config=args.feature_config or "configs/features.yaml",
        )
        _ensure_non_calibrating_source(model_config_path, allow_calibrating_profile=bool(args.allow_calibrating_profile))
        dates = _date_values(args)
        output_suffix = _resolve_output_suffix(
            args.output_file_suffix,
            artifact_suffix=str(args.output_artifact_suffix or args.model_artifact_suffix),
        )
        manifest_path = ensure_output_file_path(args.manifest_output, label="manifest output", workspace_root=ROOT)

        planned_outputs = [
            {
                "date": date,
                "prediction_file": f"artifacts/predictions/predictions_{pd.Timestamp(date).strftime('%Y%m%d')}_{output_suffix}.csv",
                "summary_file": f"artifacts/predictions/predictions_{pd.Timestamp(date).strftime('%Y%m%d')}_{output_suffix}.summary.json",
            }
            for date in dates
        ]
        if args.dry_run:
            payload = _manifest_payload(
                args=args,
                resolved_profile=resolved_profile,
                model_config_path=model_config_path,
                data_config_path=data_config_path,
                feature_config_path=feature_config_path,
                dates=dates,
                output_suffix=output_suffix,
                outputs=planned_outputs,
                status="planned",
            )
            write_json(manifest_path, payload)
            print(f"[calibration-materialize] dry-run manifest saved: {display_path(manifest_path, workspace_root=ROOT)}")
            print(f"[calibration-materialize] calibration train glob: {payload['calibration_train_glob']}")
            return 0

        progress = ProgressBar(total=len(dates) + 1, prefix="[calibration-materialize]", logger=log_progress, min_interval_sec=0.0)
        progress.start(
            f"loading frame profile={resolved_profile or 'custom'} dates={len(dates)} "
            f"model_artifact_suffix={args.model_artifact_suffix} output_suffix={output_suffix}"
        )
        frame = prepare_prediction_frame(data_config_path)
        progress.update(message=f"frame ready rows={len(frame):,}")

        outputs: list[dict[str, Any]] = []
        for date in dates:
            summary = run_predict_from_frame(
                model_config_path=model_config_path,
                feature_config_path=feature_config_path,
                frame=frame,
                race_date=date,
                profile_name=resolved_profile,
                model_artifact_suffix=None if args.model_artifact_suffix == NO_MODEL_ARTIFACT_SUFFIX else args.model_artifact_suffix,
                output_file_suffix=output_suffix,
            )
            outputs.append({"date": date, **summary})
            progress.update(message=f"materialized date={date}")

        payload = _manifest_payload(
            args=args,
            resolved_profile=resolved_profile,
            model_config_path=model_config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
            dates=dates,
            output_suffix=output_suffix,
            outputs=outputs,
            status="completed",
        )
        write_json(manifest_path, payload)
        progress.complete(message=f"manifest written {display_path(manifest_path, workspace_root=ROOT)}")
        print(f"[calibration-materialize] calibration train glob: {payload['calibration_train_glob']}")
        return 0
    except KeyboardInterrupt:
        print("[calibration-materialize] interrupted by user")
        return 130
    except Exception as error:
        print(f"[calibration-materialize] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
