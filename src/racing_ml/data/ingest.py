from __future__ import annotations

from pathlib import Path
import shutil

import pandas as pd


def _copy_csv_tree(source_root: Path, target_root: Path) -> int:
    copied = 0
    for csv_file in source_root.rglob("*.csv"):
        rel = csv_file.relative_to(source_root)
        out = target_root / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            shutil.copy2(csv_file, out)
            copied += 1
    return copied


def download_dataset_if_needed(target_dir: str, source_dataset: str) -> Path:
    raw_dir = Path(target_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    if any(raw_dir.glob("**/*.csv")):
        return raw_dir

    try:
        import kagglehub

        cached_dir = Path(kagglehub.dataset_download(source_dataset))
    except Exception as error:
        raise RuntimeError(
            "Failed to download dataset via kagglehub. "
            f"dataset='{source_dataset}'. Check dataset id and Kaggle credentials."
        ) from error

    copied = _copy_csv_tree(cached_dir, raw_dir)
    if copied == 0 and not any(raw_dir.glob("**/*.csv")):
        raise RuntimeError(
            f"Dataset downloaded but no CSV files found under {cached_dir}."
        )

    return raw_dir


def create_sample_dataset(target_dir: str) -> Path:
    raw_dir = Path(target_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_path = raw_dir / "sample_races.csv"

    if sample_path.exists():
        return sample_path

    rows = []
    race_no = 0
    for year in [2021, 2022, 2023]:
        for month in [3, 6, 9, 12]:
            for race_index in range(1, 6):
                race_no += 1
                race_id = f"{year}{month:02d}{race_index:02d}_{race_no:03d}"
                for horse_index in range(1, 11):
                    rank = horse_index
                    rows.append(
                        {
                            "date": f"{year}-{month:02d}-{race_index + 1:02d}",
                            "race_id": race_id,
                            "horse_id": f"H{horse_index:03d}",
                            "horse_name": f"Horse_{horse_index:03d}",
                            "jockey_id": f"J{(horse_index % 8) + 1:03d}",
                            "trainer_id": f"T{(horse_index % 6) + 1:03d}",
                            "rank": rank,
                            "age": 3 + (horse_index % 5),
                            "sex": "M" if horse_index % 2 == 0 else "F",
                            "weight": 430 + horse_index * 5,
                            "track": "Tokyo" if race_index % 2 == 0 else "Kyoto",
                            "distance": 1200 + (race_index % 4) * 400,
                            "weather": "Sunny" if race_index % 3 == 0 else "Cloudy",
                            "ground_condition": "Firm" if race_index % 2 == 0 else "Good",
                        }
                    )

    frame = pd.DataFrame(rows)
    frame["is_win"] = (frame["rank"] == 1).astype(int)
    frame.to_csv(sample_path, index=False)
    return sample_path
