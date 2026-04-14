import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.pipeline.train_pipeline import run_train
from racing_ml.common.config import load_yaml
from racing_ml.common.execution_capacity import assert_no_conflicting_heavy_processes
from racing_ml.common.local_nankan_trust import require_local_nankan_trust_ready
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles, resolve_model_run_profile
from racing_ml.common.progress import ProgressBar


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[train {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-config", default=None)
    parser.add_argument("--feature-config", default=None)
    parser.add_argument("--artifact-suffix", default=None)
    parser.add_argument("--max-train-rows", type=int, default=None)
    parser.add_argument("--max-valid-rows", type=int, default=None)
    parser.add_argument("--allow-concurrent-heavy-jobs", action="store_true")
    parser.add_argument("--allow-diagnostic-local-nankan", action="store_true")
    args = parser.parse_args()
    progress = ProgressBar(total=2, prefix="[train cli]", logger=log_progress, min_interval_sec=0.0)

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
        data_cfg = load_yaml(ROOT / data_config_path)
        require_local_nankan_trust_ready(
            workspace_root=ROOT,
            data_config=data_cfg,
            data_config_path=data_config_path,
            allow_diagnostic_override=bool(args.allow_diagnostic_local_nankan),
            command_name="train",
            profile_name=resolved_profile,
        )
        if not args.allow_concurrent_heavy_jobs:
            assert_no_conflicting_heavy_processes(current_script_pattern="scripts/run_train.py")
        progress.start(
            message=(
                f"starting profile={resolved_profile or 'custom'} config={config_path} "
                f"data_config={data_config_path} feature_config={feature_config_path} "
                f"artifact_suffix={args.artifact_suffix or 'none'} "
                f"max_train_rows={args.max_train_rows or 'config'} max_valid_rows={args.max_valid_rows or 'config'} "
                f"allow_concurrent_heavy_jobs={args.allow_concurrent_heavy_jobs} "
                f"allow_diagnostic_local_nankan={args.allow_diagnostic_local_nankan}"
            )
        )
        run_train(
            model_config_path=config_path,
            data_config_path=data_config_path,
            feature_config_path=feature_config_path,
            artifact_suffix=args.artifact_suffix,
            max_train_rows_override=args.max_train_rows,
            max_valid_rows_override=args.max_valid_rows,
            profile_name=resolved_profile,
        )
        progress.complete(message="training flow finished")
        return 0
    except KeyboardInterrupt:
        print("[train] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[train] failed: {error}")
        return 1
    except Exception as error:
        print(f"[train] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
