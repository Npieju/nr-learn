import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.pipeline.backtest_pipeline import run_backtest
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, resolve_model_run_profile
from racing_ml.common.progress import ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[backtest {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--predictions-file", default=None)
    args = parser.parse_args()
    progress = ProgressBar(total=2, prefix="[backtest cli]", logger=log_progress, min_interval_sec=0.0)

    try:
        if args.profile and args.config is not None:
            raise ValueError("--profile cannot be combined with --config")

        resolved_profile, config_path, _, _ = resolve_model_run_profile(
            args.profile,
            default_model_config=args.config or "configs/model.yaml",
            default_data_config="configs/data.yaml",
            default_feature_config="configs/features.yaml",
        )
        progress.start(
            message=(
                f"starting profile={resolved_profile or 'custom'} config={config_path} "
                f"predictions_file={args.predictions_file or 'latest'}"
            )
        )
        run_backtest(config_path, args.predictions_file)
        progress.complete(message="backtest flow finished")
        return 0
    except KeyboardInterrupt:
        print("[backtest] interrupted by user")
        return 130
    except Exception as error:
        print(f"[backtest] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
