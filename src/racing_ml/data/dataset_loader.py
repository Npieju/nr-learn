from __future__ import annotations

from pathlib import Path
import re
from typing import Any

import numpy as np
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
    "odds": ["odds", "単勝", "単勝オッズ"],
    "popularity": ["popularity", "人気", "人気順"],
    "finish_time": ["finish_time", "タイム"],
    "closing_time_3f": ["closing_time_3f", "上り"],
    "corner_1_position": ["corner_1_position", "1コーナー"],
    "corner_2_position": ["corner_2_position", "2コーナー"],
    "corner_3_position": ["corner_3_position", "3コーナー"],
    "corner_4_position": ["corner_4_position", "4コーナー"],
    "race_pace_front3f": ["race_pace_front3f", "前半3ハロン"],
    "race_pace_back3f": ["race_pace_back3f", "上がり3ハロン"],
}

TIME_PATTERN = re.compile(r"^(?:(?P<minutes>\d+):)?(?P<seconds>\d+(?:\.\d+)?)$")

DEFAULT_SUPPLEMENTAL_TABLES: list[dict[str, Any]] = [
    {
        "name": "laptime",
        "pattern": "**/*laptime*.csv",
        "join_on": ["race_id"],
        "required_columns": ["race_id"],
        "keep_columns": ["race_id", "race_pace_front3f", "race_pace_back3f"],
        "dedupe_on": ["race_id"],
        "merge_mode": "fill_missing",
    }
]


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


def _resolve_column_aliases(extra_aliases: dict[str, Any] | None = None) -> dict[str, list[str]]:
    resolved: dict[str, list[str]] = {
        canonical: [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
        for canonical, aliases in COLUMN_ALIASES.items()
    }

    if not isinstance(extra_aliases, dict):
        return resolved

    for canonical, aliases in extra_aliases.items():
        canonical_name = str(canonical).strip().lower()
        if not canonical_name:
            continue

        resolved.setdefault(canonical_name, [])
        for alias in _normalize_string_list(aliases):
            alias_name = alias.strip().lower()
            if alias_name and alias_name not in resolved[canonical_name]:
                resolved[canonical_name].append(alias_name)
    return resolved


def _normalize_columns(frame: pd.DataFrame, extra_aliases: dict[str, Any] | None = None) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).strip().lower() for column in frame.columns]

    for canonical, aliases in _resolve_column_aliases(extra_aliases).items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            alias = alias.lower()
            if alias in frame.columns:
                frame = frame.rename(columns={alias: canonical})
                break

    return frame


def _parse_finish_time_seconds(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")

    text = str(value).strip().replace(" ", "")
    if not text or text.lower() in {"nan", "none"}:
        return float("nan")

    match = TIME_PATTERN.match(text)
    if match is None:
        return float("nan")

    minutes = float(match.group("minutes") or 0.0)
    seconds = float(match.group("seconds"))
    return float(minutes * 60.0 + seconds)


def _parse_first_numeric_token(value: object) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("nan")

    text = str(value).strip().replace(" ", "")
    if not text or text.lower() in {"nan", "none"}:
        return float("nan")

    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
        return float("nan")
    return float(match.group(1))


def _load_laptime_summary(raw_dir: Path) -> pd.DataFrame | None:
    candidates = [path for path in raw_dir.glob("**/*.csv") if "laptime" in path.name.lower()]
    if not candidates:
        return None

    laptime_path = sorted(candidates)[0]
    frame = pd.read_csv(
        laptime_path,
        usecols=lambda column: str(column).strip() in {"レースID", "前半3ハロン", "上がり3ハロン"},
        low_memory=False,
    )
    frame = _normalize_columns(frame)
    required = [column for column in ["race_id", "race_pace_front3f", "race_pace_back3f"] if column in frame.columns]
    if "race_id" not in required:
        return None

    frame = frame[required].copy()
    frame = frame.drop_duplicates(subset=["race_id"], keep="first")
    return frame


def _normalize_string_list(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        text = str(values).strip()
        return [text] if text else []

    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            output.append(text)
    return output


def _resolve_path(raw_path: str | Path, base_dir: Path | None) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or base_dir is None:
        return path
    return base_dir / path


def _resolve_dataset_config(dataset_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(dataset_config, dict):
        return {}

    nested = dataset_config.get("dataset")
    if isinstance(nested, dict):
        return nested
    return dataset_config


def _resolve_external_raw_dirs(dataset_config: dict[str, Any] | None, base_dir: Path | None) -> list[Path]:
    dataset_cfg = _resolve_dataset_config(dataset_config)
    raw_dirs = _normalize_string_list(dataset_cfg.get("external_raw_dirs"))
    return [_resolve_path(raw_dir, base_dir) for raw_dir in raw_dirs]


def _resolve_search_roots(
    *,
    raw_dir: Path,
    dataset_config: dict[str, Any] | None,
    base_dir: Path | None,
    include_raw_dir: bool,
) -> list[Path]:
    roots: list[Path] = []
    if include_raw_dir:
        roots.append(raw_dir)

    for external_dir in _resolve_external_raw_dirs(dataset_config, base_dir):
        if external_dir not in roots:
            roots.append(external_dir)
    return roots


def _iter_csv_candidates(search_roots: list[Path], pattern: str) -> list[Path]:
    output: list[Path] = []
    seen: set[str] = set()
    for root in search_roots:
        if not root.exists():
            continue
        for candidate in sorted(root.glob(pattern)):
            if not candidate.is_file() or candidate.suffix.lower() != ".csv":
                continue
            resolved = str(candidate.resolve())
            if resolved in seen:
                continue
            seen.add(resolved)
            output.append(candidate)
    return output


def _select_table_columns(frame: pd.DataFrame, keep_columns: list[str], join_on: list[str]) -> pd.DataFrame:
    if not keep_columns:
        return frame.copy()

    selected = [column for column in keep_columns if column in frame.columns]
    selected = list(dict.fromkeys([*join_on, *selected]))
    if not selected:
        return frame.copy()
    return frame[selected].copy()


def _load_matching_table(
    *,
    table_cfg: dict[str, Any],
    search_roots: list[Path],
    join_on: list[str],
    keep_columns: list[str],
    required_columns: list[str],
) -> pd.DataFrame | None:
    pattern = str(table_cfg.get("pattern", table_cfg.get("path_glob", ""))).strip()
    if not pattern:
        return None

    for candidate in _iter_csv_candidates(search_roots, pattern):
        table = pd.read_csv(candidate, low_memory=False)
        table = _normalize_columns(table, extra_aliases=table_cfg.get("column_aliases"))
        if required_columns and not all(column in table.columns for column in required_columns):
            continue
        if join_on and not all(column in table.columns for column in join_on):
            continue
        return _select_table_columns(table, keep_columns=keep_columns, join_on=join_on)

    return None


def _merge_table(
    *,
    base_frame: pd.DataFrame,
    table: pd.DataFrame,
    join_on: list[str],
    value_columns: list[str],
    merge_mode: str,
) -> pd.DataFrame:
    merged = base_frame.merge(table, on=join_on, how="left", suffixes=("", "_supp"))
    normalized_mode = merge_mode.strip().lower() or "fill_missing"

    for column in value_columns:
        supplemental_column = f"{column}_supp"
        if supplemental_column not in merged.columns:
            continue

        if column not in merged.columns:
            merged = merged.rename(columns={supplemental_column: column})
            continue

        if normalized_mode == "prefer_supplemental":
            merged[column] = merged[supplemental_column].where(merged[supplemental_column].notna(), merged[column])
        else:
            merged[column] = merged[column].where(merged[column].notna(), merged[supplemental_column])
        merged = merged.drop(columns=[supplemental_column])

    return merged


def _append_external_tables(frame: pd.DataFrame, raw_dir: Path, dataset_config: dict[str, Any] | None, base_dir: Path | None) -> pd.DataFrame:
    dataset_cfg = _resolve_dataset_config(dataset_config)
    append_tables = dataset_cfg.get("append_tables", [])
    if not isinstance(append_tables, list) or not append_tables:
        return frame

    for table_cfg in append_tables:
        if not isinstance(table_cfg, dict):
            continue

        search_dirs = _normalize_string_list(table_cfg.get("search_dirs"))
        if search_dirs:
            search_roots = [_resolve_path(search_dir, base_dir) for search_dir in search_dirs]
        else:
            search_roots = _resolve_search_roots(
                raw_dir=raw_dir,
                dataset_config=dataset_cfg,
                base_dir=base_dir,
                include_raw_dir=False,
            )

        required_columns = _normalize_string_list(table_cfg.get("required_columns"))
        keep_columns = _normalize_string_list(table_cfg.get("keep_columns"))
        append_frame = _load_matching_table(
            table_cfg=table_cfg,
            search_roots=search_roots,
            join_on=[],
            keep_columns=keep_columns,
            required_columns=required_columns,
        )
        if append_frame is None or append_frame.empty:
            continue

        combined = pd.concat([frame, append_frame], ignore_index=True, sort=False)
        dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or ["race_id", "horse_id"]
        dedupe_on = [column for column in dedupe_on if column in combined.columns]
        if dedupe_on:
            combined = combined.drop_duplicates(subset=dedupe_on, keep="first")
        frame = combined.reset_index(drop=True)

    return frame


def _merge_supplemental_tables(
    frame: pd.DataFrame,
    raw_dir: Path,
    dataset_config: dict[str, Any] | None = None,
    base_dir: Path | None = None,
) -> pd.DataFrame:
    if "race_id" not in frame.columns:
        return frame

    dataset_cfg = _resolve_dataset_config(dataset_config)
    supplemental_tables = dataset_cfg.get("supplemental_tables")
    if not isinstance(supplemental_tables, list) or not supplemental_tables:
        supplemental_tables = DEFAULT_SUPPLEMENTAL_TABLES

    default_search_roots = _resolve_search_roots(
        raw_dir=raw_dir,
        dataset_config=dataset_cfg,
        base_dir=base_dir,
        include_raw_dir=True,
    )

    for table_cfg in supplemental_tables:
        if not isinstance(table_cfg, dict):
            continue

        join_on = _normalize_string_list(table_cfg.get("join_on"))
        if not join_on or not all(column in frame.columns for column in join_on):
            continue

        search_dirs = _normalize_string_list(table_cfg.get("search_dirs"))
        if search_dirs:
            search_roots = [_resolve_path(search_dir, base_dir) for search_dir in search_dirs]
        else:
            search_roots = default_search_roots

        keep_columns = _normalize_string_list(table_cfg.get("keep_columns"))
        required_columns = _normalize_string_list(table_cfg.get("required_columns")) or join_on
        supplemental_frame = _load_matching_table(
            table_cfg=table_cfg,
            search_roots=search_roots,
            join_on=join_on,
            keep_columns=keep_columns,
            required_columns=required_columns,
        )
        if supplemental_frame is None or supplemental_frame.empty:
            continue

        dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or join_on
        dedupe_on = [column for column in dedupe_on if column in supplemental_frame.columns]
        if dedupe_on:
            supplemental_frame = supplemental_frame.drop_duplicates(subset=dedupe_on, keep="first")

        value_columns = [column for column in supplemental_frame.columns if column not in join_on]
        frame = _merge_table(
            base_frame=frame,
            table=supplemental_frame,
            join_on=join_on,
            value_columns=value_columns,
            merge_mode=str(table_cfg.get("merge_mode", "fill_missing")),
        )

    return frame


def _ensure_minimum_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()

    if "date" not in frame.columns:
        raise ValueError("Dataset must include a date-like column")

    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"])
    date_series = pd.to_datetime(frame["date"], errors="coerce")

    if "race_id" not in frame.columns:
        frame["race_id"] = date_series.dt.strftime("%Y%m%d") + "_" + frame.index.astype(str)

    if "horse_id" not in frame.columns:
        if "horse_name" in frame.columns:
            frame["horse_id"] = frame["horse_name"].astype(str)
        else:
            frame["horse_id"] = frame.index.astype(str)

    if "rank" in frame.columns:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")

    if "distance" in frame.columns:
        frame["distance"] = frame["distance"].astype(str).str.extract(r"(\d+)", expand=False)
        frame["distance"] = pd.to_numeric(frame["distance"], errors="coerce")

    if "weight" in frame.columns:
        frame["weight"] = (
            frame["weight"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        frame["weight"] = pd.to_numeric(frame["weight"], errors="coerce")

    if "odds" in frame.columns:
        frame["odds"] = (
            frame["odds"]
            .astype(str)
            .str.extract(r"(\d+(?:\.\d+)?)", expand=False)
        )
        frame["odds"] = pd.to_numeric(frame["odds"], errors="coerce")

    if "popularity" in frame.columns:
        frame["popularity"] = (
            frame["popularity"]
            .astype(str)
            .str.extract(r"(\d+)", expand=False)
        )
        frame["popularity"] = pd.to_numeric(frame["popularity"], errors="coerce")

    if "finish_time" in frame.columns and "finish_time_sec" not in frame.columns:
        frame["finish_time_sec"] = frame["finish_time"].map(_parse_finish_time_seconds)

    for column in ["closing_time_3f", "race_pace_front3f", "race_pace_back3f"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    for column in ["corner_1_position", "corner_2_position", "corner_3_position", "corner_4_position"]:
        if column in frame.columns:
            frame[column] = frame[column].map(_parse_first_numeric_token)

    if "is_win" not in frame.columns and "rank" in frame.columns:
        frame["is_win"] = (frame["rank"] == 1).astype(int)

    return frame


def load_training_table(
    raw_dir: str | Path,
    dataset_config: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> pd.DataFrame:
    base_path = Path(base_dir) if base_dir is not None else None
    raw_path = _resolve_path(raw_dir, base_path)
    dataset_path = _pick_dataset(raw_path)
    frame = pd.read_csv(dataset_path, low_memory=False)
    frame = _normalize_columns(frame)
    frame = _append_external_tables(frame, raw_path, dataset_config=dataset_config, base_dir=base_path)
    frame = _merge_supplemental_tables(frame, raw_path, dataset_config=dataset_config, base_dir=base_path)
    frame = _ensure_minimum_columns(frame)
    frame = frame.sort_values(["date", "race_id"]).reset_index(drop=True)
    return frame
