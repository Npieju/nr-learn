import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.config import load_yaml
from racing_ml.data.ingest import download_dataset_if_needed, create_sample_dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    args = parser.parse_args()

    config = load_yaml(args.config)
    dataset = config.get("dataset", {})
    target = dataset.get("raw_dir", "data/raw")
    source_dataset = dataset.get(
        "source_dataset",
        dataset.get("source_repo", "takamotoki/jra-horse-racing-dataset"),
    )

    try:
        path = download_dataset_if_needed(target, source_dataset)
        print(f"[ingest] raw dataset ready at: {path}")
    except RuntimeError as error:
        print(f"[ingest] warning: {error}")
        sample = create_sample_dataset(target)
        print(f"[ingest] fallback sample dataset created: {sample}")
