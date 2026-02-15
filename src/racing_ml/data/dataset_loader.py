from __future__ import annotations

from pathlib import Path

import pandas as pd

COLUMN_ALIASES = {
    "date": ["date", "race_date", "held_at", "レース日付"],
    "race_id": ["race_id", "racecode", "race_code", "レースid", "レースID"],
    "horse_id": ["horse_id", "horse_code", "horse_no", "レース馬番id", "レース馬番ID", "馬番"],
    "horse_name": ["horse_name", "name", "馬名"],
    "rank": ["rank", "result", "finish_position", "order", "着順"],
    "jockey_id": ["jockey_id", "jockey_code", "騎手id", "騎手ID", "騎手"],
    "trainer_id": ["trainer_id", "trainer_code", "調教師id", "調教師ID", "調教師"],
    "track": ["track", "course", "venue", "競馬場名"],
    "distance": ["distance", "race_distance", "距離(m)", "距離"],
    "weather": ["weather", "天候"],
    "ground_condition": ["ground_condition", "condition", "track_condition", "馬場状態1", "馬場状態2"],
    "age": ["age", "馬齢"],
    "sex": ["sex", "gender", "性別"],
    "weight": ["weight", "horse_weight", "body_weight", "馬体重", "斤量"],
}


def _pick_dataset(raw_dir: Path) -> Path:
    csv_files = list(raw_dir.glob("**/*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found under {raw_dir}")

    alias_pool = set()
    for aliases in COLUMN_ALIASES.values():
        alias_pool.update(alias.lower() for alias in aliases)

    def score(path: Path) -> tuple[int, int]:
        try:
            frame = pd.read_csv(path, nrows=5, low_memory=False)
            headers = [str(column).strip().lower() for column in frame.columns]
            semantic_hits = sum(1 for header in headers if header in alias_pool)
            has_date = int(any(header in ["date", "race_date", "レース日付"] for header in headers))
            has_rank = int(any(header in ["rank", "着順"] for header in headers))
            return (semantic_hits + has_date * 3 + has_rank * 2, len(frame.columns))
        except Exception:
            return (0, 0)

    return sorted(csv_files, key=score, reverse=True)[0]


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            alias = alias.lower()
            if alias in frame.columns:
                frame = frame.rename(columns={alias: canonical})
                break

    return frame


def _ensure_minimum_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if "date" not in frame.columns:
        raise ValueError("Dataset must include a date-like column")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])

    if "race_id" not in frame.columns:
        frame["race_id"] = frame["date"].dt.strftime("%Y%m%d") + "_" + frame.index.astype(str)

    if "horse_id" not in frame.columns:
        if "horse_name" in frame.columns:
            frame["horse_id"] = frame["horse_name"].astype(str)
        else:
            frame["horse_id"] = frame.index.astype(str)

    if "rank" in frame.columns:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")

    if "weight" in frame.columns:
        frame["weight"] = (
            frame["weight"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")

    if "is_win" not in frame.columns and "rank" in frame.columns:
        frame["is_win"] = (frame["rank"] == 1).astype(int)

    return frame


def load_training_table(raw_dir: str) -> pd.DataFrame:
    raw_path = Path(raw_dir)
    dataset_path = _pick_dataset(raw_path)
    frame = pd.read_csv(dataset_path, low_memory=False)
    frame = _normalize_columns(frame)
    frame = _ensure_minimum_columns(frame)
    frame = frame.sort_values(["date", "race_id"]).reset_index(drop=True)
    return frame
