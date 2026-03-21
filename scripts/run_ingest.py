import argparse
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.ingest import download_dataset_if_needed, create_sample_dataset


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[ingest {now}] {message}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    try:
        config = load_yaml(args.config)
        dataset = config.get("dataset", {})
        target = dataset.get("raw_dir", "data/raw")
        source_dataset = dataset.get(
            "source_dataset",
            dataset.get("source_repo", "takamotoki/jra-horse-racing-dataset"),
        )
        external_raw_dirs = dataset.get("external_raw_dirs", [])
        progress = ProgressBar(total=max(3, 2 + len(external_raw_dirs)), prefix="[ingest]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message=f"config loaded source={source_dataset}")

        try:
            with Heartbeat("[ingest]", "preparing raw dataset", logger=log_progress):
                path = download_dataset_if_needed(target, source_dataset)
            print(f"[ingest] raw dataset ready at: {path}")
        except RuntimeError as error:
            print(f"[ingest] warning: {error}")
            with Heartbeat("[ingest]", "creating fallback sample dataset", logger=log_progress):
                sample = create_sample_dataset(target)
            print(f"[ingest] fallback sample dataset created: {sample}")
        progress.update(message="primary dataset ready")

        for raw_dir in external_raw_dirs:
            external_path = Path(raw_dir)
            if not external_path.is_absolute():
                external_path = ROOT / external_path
            external_path.mkdir(parents=True, exist_ok=True)
            print(f"[ingest] external raw dir ready: {external_path}")
            progress.update(message=f"external dir ready {external_path}")
        progress.complete(message="ingest setup completed")
        return 0
    except KeyboardInterrupt:
        print("[ingest] interrupted by user")
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[ingest] failed: {error}")
        return 1
    except Exception as error:
        print(f"[ingest] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
