import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.serving.predict_batch import run_predict
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles, resolve_model_run_profile
from racing_ml.common.progress import ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[predict {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--race-date", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=2, prefix="[predict cli]", logger=log_progress, min_interval_sec=0.0)

    try:
        if args.list_profiles:
            print(format_model_run_profiles())
            return 0

        if args.profile and any(value is not None for value in (args.config, args.data_config, args.feature_config)):
            raise ValueError("--profile cannot be combined with --config, --data-config, or --feature-config")

        resolved_profile, config_path, data_config_path, feature_config_path = resolve_model_run_profile(
            args.profile,
            default_model_config=args.config or "configs/model.yaml",
            default_data_config=args.data_config or "configs/data.yaml",
            default_feature_config=args.feature_config or "configs/features.yaml",
        )
        progress.start(
            message=(
                f"starting profile={resolved_profile or 'custom'} config={config_path} data_config={data_config_path} "
                f"feature_config={feature_config_path} race_date={args.race_date or 'latest'}"
            )
        )
        run_predict(
            model_config_path=config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
            race_date=args.race_date,
            profile_name=resolved_profile,
        )
        progress.complete(message="prediction flow finished")
        return 0
    except KeyboardInterrupt:
        print("[predict] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[predict] failed: {error}")
        return 1
    except Exception as error:
        print(f"[predict] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
