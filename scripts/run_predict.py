import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.serving.predict_batch import run_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    parser.add_argument("--race-date", default=None)
    args = parser.parse_args()
    run_predict(
        model_config_path=args.config,
        data_config_path=args.data_config,
        feature_config_path=args.feature_config,
        race_date=args.race_date,
    )
