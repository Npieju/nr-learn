import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.pipeline.train_pipeline import run_train


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features.yaml")
    args = parser.parse_args()

    try:
        run_train(
            model_config_path=args.config,
            data_config_path=args.data_config,
            feature_config_path=args.feature_config,
        )
        return 0
    except KeyboardInterrupt:
        print("[train] interrupted by user")
        return 130
    except Exception as error:
        print(f"[train] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
