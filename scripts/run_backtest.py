import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.pipeline.backtest_pipeline import run_backtest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model.yaml")
    parser.add_argument("--predictions-file", default=None)
    args = parser.parse_args()

    try:
        run_backtest(args.config, args.predictions_file)
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
