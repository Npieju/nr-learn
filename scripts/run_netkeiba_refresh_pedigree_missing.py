from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.progress import Heartbeat, ProgressBar


def log_progress(message: str) -> None:
    print(message, flush=True)


def _normalize_keys(series: pd.Series) -> pd.Series:
    values = series.astype("string").fillna("").str.strip()
    values = values[values != ""]
    return values.drop_duplicates().reset_index(drop=True)


def _read_key_column(path: Path, column: str) -> pd.Series:
    frame = pd.read_csv(path, dtype={column: "string"}, low_memory=False)
    if column not in frame.columns:
        raise ValueError(f"column '{column}' not found in {path}")
    return _normalize_keys(frame[column])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-file", default="data/external/netkeiba/ids/horse_keys.csv")
    parser.add_argument("--target-column", default="horse_key")
    parser.add_argument("--pedigree-file", default="data/external/netkeiba/pedigree/netkeiba_pedigree_crawled.csv")
    parser.add_argument("--pedigree-column", default="horse_key")
    parser.add_argument("--output-file", default="data/external/netkeiba/ids/horse_keys_missing_2025.csv")
    args = parser.parse_args()

    target_file = ROOT / args.target_file
    pedigree_file = ROOT / args.pedigree_file
    output_file = ROOT / args.output_file
    progress = ProgressBar(total=3, prefix="[netkeiba-refresh-pedigree]", logger=log_progress, min_interval_sec=0.0)
    progress.start(
        message=(
            f"starting target_file={target_file} pedigree_file={pedigree_file} "
            f"output_file={output_file}"
        )
    )

    with Heartbeat("[netkeiba-refresh-pedigree]", "loading target and pedigree keys", logger=log_progress):
        target_keys = _read_key_column(target_file, args.target_column)
        existing_keys = _read_key_column(pedigree_file, args.pedigree_column) if pedigree_file.exists() else pd.Series(dtype="string")
    progress.update(message=f"keys loaded target={len(target_keys)} existing={len(existing_keys)}")
    missing_keys = target_keys[~target_keys.isin(set(existing_keys.tolist()))].reset_index(drop=True)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with Heartbeat("[netkeiba-refresh-pedigree]", "writing missing pedigree keys", logger=log_progress):
        pd.DataFrame({args.target_column: missing_keys}).to_csv(output_file, index=False)
    progress.complete(message=f"missing keys written count={len(missing_keys)} path={output_file}")

    print(
        f"target_keys={len(target_keys)} existing_keys={len(existing_keys)} missing_keys={len(missing_keys)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
