from __future__ import annotations

import csv
from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

COLUMN_ALIASES = {
    "date": ["date", "race_date", "held_at", "レース日付"],
    "race_id": ["race_id", "racecode", "race_code", "レースid", "レースID"],
    "horse_id": ["horse_id", "horse_code", "horse_no", "レース馬番id", "レース馬番ID", "馬番"],
    "horse_key": ["horse_key", "horse_uuid", "競走馬id", "競走馬ID", "競走馬コード"],
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
    "weight": ["weight", "horse_weight", "body_weight", "馬体重"],
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
    "frame_no": ["frame_no", "frame_number", "枠番"],
    "gate_no": ["gate_no", "gate_number", "馬番"],
    "owner_name": ["owner_name", "owner", "馬主"],
    "breeder_name": ["breeder_name", "breeder", "生産者"],
    "sire_name": ["sire_name", "sire", "父"],
    "dam_name": ["dam_name", "dam", "母"],
    "damsire_name": ["damsire_name", "dam_sire", "mother_father", "母父"],
}
BASE_RESOLVED_COLUMN_ALIASES: dict[str, list[str]] = {
    canonical: [str(alias).strip().lower() for alias in aliases if str(alias).strip()]
    for canonical, aliases in COLUMN_ALIASES.items()
}

TIME_PATTERN = re.compile(r"^(?:(?P<minutes>\d+):)?(?P<seconds>\d+(?:\.\d+)?)$")
PASSING_ORDER_NUMBER_PATTERN = re.compile(r"\d+")
DATASET_DATE_RANGE_PREFIX_PATTERN = re.compile(r"^(?P<start>\d{8})-(?P<end>\d{8})")

DEFAULT_SUPPLEMENTAL_TABLES: list[dict[str, Any]] = [
    {
        "name": "laptime",
        "pattern": "**/*laptime*.csv",
        "join_on": ["race_id"],
        "required_columns": ["race_id"],
        "keep_columns": ["race_id", "race_pace_front3f", "race_pace_back3f"],
        "dedupe_on": ["race_id"],
        "merge_mode": "fill_missing",
    },
    {
        "name": "corner_passing_order",
        "pattern": "**/*corner_passing_order*.csv",
        "join_on": ["race_id", "gate_no"],
        "required_columns": ["race_id", "gate_no"],
        "keep_columns": [
            "race_id",
            "gate_no",
            "corner_1_position",
            "corner_2_position",
            "corner_3_position",
            "corner_4_position",
        ],
        "dedupe_on": ["race_id", "gate_no"],
        "merge_mode": "fill_missing",
        "table_loader": "corner_passing_order",
    }
]


@dataclass(frozen=True)
class TrainingTableLoadResult:
    frame: pd.DataFrame
    loaded_rows: int
    pre_feature_rows: int
    data_load_strategy: str
    primary_source_rows_total: int | None


def _get_supplemental_table_configs(dataset_config: dict[str, Any] | None) -> list[dict[str, Any]]:
    dataset_cfg = _resolve_dataset_config(dataset_config)
    supplemental_tables = dataset_cfg.get("supplemental_tables")
    if not isinstance(supplemental_tables, list) or not supplemental_tables:
        return list(DEFAULT_SUPPLEMENTAL_TABLES)
    return [table for table in supplemental_tables if isinstance(table, dict)]


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
    if not isinstance(extra_aliases, dict):
        return BASE_RESOLVED_COLUMN_ALIASES

    resolved: dict[str, list[str]] = {
        canonical: list(aliases)
        for canonical, aliases in BASE_RESOLVED_COLUMN_ALIASES.items()
    }

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
    normalized_columns = [str(column).strip().lower() for column in frame.columns]
    columns_changed = list(frame.columns) != normalized_columns
    if columns_changed:
        frame = frame.copy()
        frame.columns = normalized_columns

    for canonical, aliases in _resolve_column_aliases(extra_aliases).items():
        if canonical in frame.columns:
            continue
        for alias in aliases:
            alias = alias.lower()
            if alias in frame.columns:
                if not columns_changed:
                    frame = frame.copy()
                    columns_changed = True
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


def _normalize_digit_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    missing_mask = series.notna() & numeric.isna()
    if not bool(missing_mask.any()):
        return numeric
    extracted = series.loc[missing_mask].astype(str).str.extract(r"(\d+)", expand=False)
    numeric.loc[missing_mask] = pd.to_numeric(extracted, errors="coerce")
    return numeric


def _normalize_decimal_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce")
    numeric = pd.to_numeric(series, errors="coerce")
    missing_mask = series.notna() & numeric.isna()
    if not bool(missing_mask.any()):
        return numeric
    extracted = series.loc[missing_mask].astype(str).str.extract(r"(\d+(?:\.\d+)?)", expand=False)
    numeric.loc[missing_mask] = pd.to_numeric(extracted, errors="coerce")
    return numeric


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


def _parse_passing_order_positions(value: object) -> dict[int, int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return {}

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return {}

    output: dict[int, int] = {}
    for token in PASSING_ORDER_NUMBER_PATTERN.findall(text):
        gate_no = int(token)
        if gate_no not in output:
            output[gate_no] = len(output) + 1
    return output


def _expand_corner_passing_order(frame: pd.DataFrame) -> pd.DataFrame:
    work = _normalize_columns(frame)
    if "race_id" not in work.columns:
        return pd.DataFrame()

    corner_columns = [
        column
        for column in ["corner_1_position", "corner_2_position", "corner_3_position", "corner_4_position"]
        if column in work.columns
    ]
    if not corner_columns:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for values in work[["race_id", *corner_columns]].itertuples(index=False, name=None):
        race_id = values[0]
        maps = {
            column: _parse_passing_order_positions(value)
            for column, value in zip(corner_columns, values[1:])
        }
        gate_nos = sorted({gate_no for mapping in maps.values() for gate_no in mapping.keys()})
        for gate_no in gate_nos:
            row: dict[str, Any] = {"race_id": race_id, "gate_no": gate_no}
            for column in corner_columns:
                row[column] = maps[column].get(gate_no)
            rows.append(row)

    if not rows:
        return pd.DataFrame(columns=["race_id", "gate_no", *corner_columns])

    expanded = pd.DataFrame(rows)
    expanded["race_id"] = pd.to_numeric(expanded["race_id"], errors="coerce")
    expanded["gate_no"] = pd.to_numeric(expanded["gate_no"], errors="coerce")
    expanded = expanded.dropna(subset=["race_id", "gate_no"])
    return expanded.reset_index(drop=True)


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


def _resolve_materialized_candidate(table_cfg: dict[str, Any], base_dir: Path | None) -> Path | None:
    materialized_file = str(table_cfg.get("materialized_file", "")).strip()
    if not materialized_file:
        return None
    return _resolve_path(materialized_file, base_dir)


def _display_path(path: Path, base_dir: Path | None) -> str:
    if base_dir is None:
        return str(path)
    try:
        return str(path.relative_to(base_dir))
    except Exception:
        return str(path)


def _parse_race_id_date_token(value: Any) -> str | None:
    text = str(value).strip()
    if len(text) < 8:
        return None
    token = text[:8]
    return token if token.isdigit() else None


def _resolve_frame_race_id_date_bounds(frame: pd.DataFrame, join_on: list[str]) -> tuple[str | None, str | None]:
    if "race_id" not in join_on or "race_id" not in frame.columns or frame.empty:
        return None, None

    minimum: str | None = None
    maximum: str | None = None
    for value in frame["race_id"].dropna().tolist():
        token = _parse_race_id_date_token(value)
        if token is None:
            continue
        if minimum is None or token < minimum:
            minimum = token
        if maximum is None or token > maximum:
            maximum = token
    return minimum, maximum


def _resolve_candidate_filename_date_bounds(candidate: Path) -> tuple[str | None, str | None]:
    match = DATASET_DATE_RANGE_PREFIX_PATTERN.match(candidate.name)
    if not match:
        return None, None
    return match.group("start"), match.group("end")


def _resolve_materialized_manifest_date_bounds(table_cfg: dict[str, Any], base_dir: Path | None) -> tuple[str | None, str | None]:
    manifest_file = str(table_cfg.get("materialized_manifest_file", "")).strip()
    if not manifest_file:
        return None, None

    manifest_path = _resolve_path(manifest_file, base_dir)
    if not manifest_path.exists() or not manifest_path.is_file():
        return None, None

    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return None, None

    start = str(payload.get("race_id_date_start", "")).strip() or _parse_race_id_date_token(payload.get("race_id_min"))
    end = str(payload.get("race_id_date_end", "")).strip() or _parse_race_id_date_token(payload.get("race_id_max"))
    return (start or None), (end or None)


def _is_candidate_outside_base_race_range(
    *,
    candidate: Path,
    table_cfg: dict[str, Any],
    base_frame: pd.DataFrame | None,
    join_on: list[str],
    base_dir: Path | None,
    is_materialized: bool,
) -> bool:
    if base_frame is None:
        return False

    base_start, base_end = _resolve_frame_race_id_date_bounds(base_frame, join_on)
    if base_start is None or base_end is None:
        return False

    if is_materialized:
        candidate_start, candidate_end = _resolve_materialized_manifest_date_bounds(table_cfg, base_dir)
    else:
        candidate_start, candidate_end = _resolve_candidate_filename_date_bounds(candidate)
    if candidate_start is None or candidate_end is None:
        return False

    return candidate_end < base_start or candidate_start > base_end


def _select_table_columns(frame: pd.DataFrame, keep_columns: list[str], join_on: list[str]) -> pd.DataFrame:
    if not keep_columns:
        return frame

    selected = [column for column in keep_columns if column in frame.columns]
    selected = list(dict.fromkeys([*join_on, *selected]))
    if not selected:
        return frame
    if selected == list(frame.columns):
        return frame
    return frame[selected].copy()


def _build_candidate_usecols(
    table_cfg: dict[str, Any],
    *,
    join_on: list[str],
    keep_columns: list[str],
    required_columns: list[str],
):
    table_loader = str(table_cfg.get("table_loader", "")).strip().lower()
    if table_loader or not join_on:
        return None

    requested = list(dict.fromkeys([*join_on, *keep_columns, *required_columns]))
    if not requested:
        return None

    resolved_aliases = _resolve_column_aliases(table_cfg.get("column_aliases"))
    accepted = set()
    for column in requested:
        canonical = str(column).strip().lower()
        if not canonical:
            continue
        accepted.add(canonical)
        for alias in resolved_aliases.get(canonical, []):
            accepted.add(alias)

    if not accepted:
        return None

    return lambda column: str(column).strip().lower() in accepted


@lru_cache(maxsize=64)
def _read_csv_header_columns(csv_path: str) -> tuple[str, ...]:
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        return tuple(next(reader, ()))


def _resolve_exact_candidate_usecols(
    candidate: Path,
    table_cfg: dict[str, Any],
    *,
    join_on: list[str],
    keep_columns: list[str],
    required_columns: list[str],
) -> list[str] | None:
    table_loader = str(table_cfg.get("table_loader", "")).strip().lower()
    if table_loader or not join_on:
        return None

    requested = list(dict.fromkeys([*join_on, *keep_columns, *required_columns]))
    if not requested:
        return None

    resolved_aliases = _resolve_column_aliases(table_cfg.get("column_aliases"))
    accepted = set()
    for column in requested:
        canonical = str(column).strip().lower()
        if not canonical:
            continue
        accepted.add(canonical)
        for alias in resolved_aliases.get(canonical, []):
            accepted.add(alias)
    if not accepted:
        return None

    try:
        header_columns = _read_csv_header_columns(str(candidate))
    except Exception:
        return None

    exact_usecols = [column for column in header_columns if str(column).strip().lower() in accepted]
    return exact_usecols or None


def _load_candidate_table(
    candidate: Path,
    table_cfg: dict[str, Any],
    *,
    join_on: list[str],
    keep_columns: list[str],
    required_columns: list[str],
) -> pd.DataFrame:
    usecols = _resolve_exact_candidate_usecols(
        candidate,
        table_cfg,
        join_on=join_on,
        keep_columns=keep_columns,
        required_columns=required_columns,
    )
    if usecols is None:
        usecols = _build_candidate_usecols(
            table_cfg,
            join_on=join_on,
            keep_columns=keep_columns,
            required_columns=required_columns,
        )
    try:
        table = pd.read_csv(
            candidate,
            low_memory=False,
            usecols=usecols,
        )
    except pd.errors.EmptyDataError:
        return pd.DataFrame()
    table = _normalize_columns(table, extra_aliases=table_cfg.get("column_aliases"))

    table_loader = str(table_cfg.get("table_loader", "")).strip().lower()
    if table_loader == "corner_passing_order":
        table = _expand_corner_passing_order(table)

    return table


def _load_materialized_candidate_table(candidate: Path, table_cfg: dict[str, Any]) -> pd.DataFrame:
    materialized_cfg = dict(table_cfg)
    materialized_cfg.pop("table_loader", None)
    join_on = _normalize_string_list(materialized_cfg.get("join_on"))
    keep_columns = _normalize_string_list(materialized_cfg.get("keep_columns"))
    required_columns = _normalize_string_list(materialized_cfg.get("required_columns")) or join_on
    return _load_candidate_table(
        candidate,
        materialized_cfg,
        join_on=join_on,
        keep_columns=keep_columns,
        required_columns=required_columns,
    )


def _read_csv_tail(csv_path: Path, tail_rows: int) -> tuple[pd.DataFrame, int]:
    if tail_rows <= 0:
        raise ValueError("tail_rows must be greater than 0")

    chunk_size = max(min(int(tail_rows) * 4, 200000), 50000)
    total_rows = 0
    kept_rows = 0
    max_kept_rows = int(tail_rows)
    chunks: deque[pd.DataFrame] = deque()

    for chunk in pd.read_csv(csv_path, low_memory=False, chunksize=chunk_size):
        total_rows += int(len(chunk))
        if len(chunk) > max_kept_rows:
            chunk = chunk.tail(max_kept_rows)
        chunks.append(chunk)
        kept_rows += int(len(chunk))
        if kept_rows > max_kept_rows:
            tail_frame = pd.concat(chunks, ignore_index=True).tail(max_kept_rows)
            chunks = deque([tail_frame])
            kept_rows = int(len(tail_frame))

    if not chunks:
        return pd.DataFrame(), 0

    tail_frame = chunks[0] if len(chunks) == 1 else pd.concat(chunks, ignore_index=True)
    if len(tail_frame) <= max_kept_rows:
        return tail_frame.reset_index(drop=True), total_rows
    return tail_frame.tail(max_kept_rows).reset_index(drop=True), total_rows


def _resolve_recent_date_floor(frame: pd.DataFrame) -> pd.Timestamp | None:
    if "date" not in frame.columns:
        return None

    date_series = pd.to_datetime(frame["date"], errors="coerce").dropna()
    if date_series.empty:
        return None

    return pd.Timestamp(date_series.min())


def _restrict_table_to_join_keys(table: pd.DataFrame, base_frame: pd.DataFrame, join_on: list[str]) -> pd.DataFrame:
    available_join_on = [column for column in join_on if column in table.columns and column in base_frame.columns]
    if not available_join_on:
        return table

    join_keys = base_frame[available_join_on].drop_duplicates()
    if join_keys.empty:
        return table.iloc[0:0].copy()
    if len(available_join_on) == 1:
        table_keys = pd.Index(table[available_join_on[0]])
        base_keys = pd.Index(join_keys[available_join_on[0]])
    else:
        table_keys = pd.MultiIndex.from_frame(table[available_join_on])
        base_keys = pd.MultiIndex.from_frame(join_keys)

    mask = table_keys.isin(base_keys)
    if bool(np.all(mask)):
        return table
    return table.loc[mask].copy()


def _sort_and_tail(frame: pd.DataFrame, max_rows: int | None) -> pd.DataFrame:
    if max_rows is None or max_rows <= 0 or len(frame) <= max_rows:
        if isinstance(frame.index, pd.RangeIndex) and frame.index.start == 0 and frame.index.step == 1:
            return frame
        return frame.reset_index(drop=True)

    if "date" in frame.columns:
        ordered = frame.copy()
        ordered["_date_order"] = pd.to_datetime(ordered["date"], errors="coerce")
        sort_columns = ["_date_order"]
        if "race_id" in ordered.columns:
            sort_columns.append("race_id")
        ordered = ordered.sort_values(sort_columns, na_position="last")
        ordered = ordered.tail(int(max_rows)).drop(columns=["_date_order"])
        return ordered.reset_index(drop=True)

    return frame.tail(int(max_rows)).reset_index(drop=True)


def _load_matching_table(
    *,
    table_cfg: dict[str, Any],
    search_roots: list[Path],
    join_on: list[str],
    keep_columns: list[str],
    required_columns: list[str],
    base_frame: pd.DataFrame | None = None,
    base_dir: Path | None = None,
) -> pd.DataFrame | None:
    materialized_candidate = _resolve_materialized_candidate(table_cfg, base_dir)
    if materialized_candidate is not None and materialized_candidate.exists() and materialized_candidate.is_file():
        if _is_candidate_outside_base_race_range(
            candidate=materialized_candidate,
            table_cfg=table_cfg,
            base_frame=base_frame,
            join_on=join_on,
            base_dir=base_dir,
            is_materialized=True,
        ):
            return None
        table = _load_materialized_candidate_table(materialized_candidate, table_cfg)
        if required_columns and all(column in table.columns for column in required_columns):
            if not join_on or all(column in table.columns for column in join_on):
                return _select_table_columns(table, keep_columns=keep_columns, join_on=join_on)

    pattern = str(table_cfg.get("pattern", table_cfg.get("path_glob", ""))).strip()
    if not pattern:
        return None

    for candidate in _iter_csv_candidates(search_roots, pattern):
        if _is_candidate_outside_base_race_range(
            candidate=candidate,
            table_cfg=table_cfg,
            base_frame=base_frame,
            join_on=join_on,
            base_dir=base_dir,
            is_materialized=False,
        ):
            continue
        table = _load_candidate_table(
            candidate,
            table_cfg,
            join_on=join_on,
            keep_columns=keep_columns,
            required_columns=required_columns,
        )
        if required_columns and not all(column in table.columns for column in required_columns):
            continue
        if join_on and not all(column in table.columns for column in join_on):
            continue
        return _select_table_columns(table, keep_columns=keep_columns, join_on=join_on)

    return None


def _inspect_table_sources(
    *,
    table_cfg: dict[str, Any],
    default_search_roots: list[Path],
    base_dir: Path | None,
) -> dict[str, Any]:
    search_dirs = _normalize_string_list(table_cfg.get("search_dirs"))
    if search_dirs:
        search_roots = [_resolve_path(search_dir, base_dir) for search_dir in search_dirs]
    else:
        search_roots = default_search_roots

    join_on = _normalize_string_list(table_cfg.get("join_on"))
    required_columns = _normalize_string_list(table_cfg.get("required_columns")) or join_on
    materialized_candidate = _resolve_materialized_candidate(table_cfg, base_dir)
    pattern = str(table_cfg.get("pattern", table_cfg.get("path_glob", ""))).strip()
    candidates = _iter_csv_candidates(search_roots, pattern) if pattern else []

    result: dict[str, Any] = {
        "name": str(table_cfg.get("name", "unnamed")),
        "materialized_file": _display_path(materialized_candidate, base_dir) if materialized_candidate is not None else None,
        "materialized_exists": bool(materialized_candidate is not None and materialized_candidate.exists()),
        "pattern": pattern,
        "search_roots": [_display_path(path, base_dir) for path in search_roots],
        "matched_file_count": int(len(candidates)),
        "matched_files": [_display_path(path, base_dir) for path in candidates[:10]],
        "join_on": join_on,
        "required_columns": required_columns,
        "optional": bool(table_cfg.get("optional", False)),
        "status": "missing",
    }

    if materialized_candidate is not None and materialized_candidate.exists() and materialized_candidate.is_file():
        try:
            table = _load_materialized_candidate_table(materialized_candidate, table_cfg)
        except Exception as error:
            result["status"] = "materialized_read_error"
            result["active_file"] = _display_path(materialized_candidate, base_dir)
            result["error"] = str(error)
            return result

        missing_required = [column for column in required_columns if column not in table.columns]
        missing_join = [column for column in join_on if column not in table.columns]
        if not missing_required and not missing_join:
            dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or join_on
            dedupe_on = [column for column in dedupe_on if column in table.columns]
            duplicate_rows = int(table.duplicated(subset=dedupe_on).sum()) if dedupe_on else 0
            result.update(
                {
                    "status": "ok_materialized",
                    "active_file": _display_path(materialized_candidate, base_dir),
                    "row_count": int(len(table)),
                    "column_count": int(len(table.columns)),
                    "dedupe_on": dedupe_on,
                    "duplicate_rows_on_key": duplicate_rows,
                    "canonical_columns": [str(column) for column in table.columns[:50]],
                }
            )
            return result

    if not candidates:
        if result["optional"]:
            result["status"] = "optional_missing"
        return result

    for candidate in candidates:
        try:
            table = _load_candidate_table(
                candidate,
                table_cfg,
                join_on=join_on,
                keep_columns=_normalize_string_list(table_cfg.get("keep_columns")),
                required_columns=required_columns,
            )
        except Exception as error:
            result["status"] = "read_error"
            result["active_file"] = _display_path(candidate, base_dir)
            result["error"] = str(error)
            continue

        missing_required = [column for column in required_columns if column not in table.columns]
        missing_join = [column for column in join_on if column not in table.columns]
        if missing_required or missing_join:
            result["status"] = "invalid_schema"
            result["active_file"] = _display_path(candidate, base_dir)
            result["missing_required_columns"] = missing_required
            result["missing_join_columns"] = missing_join
            result["canonical_columns"] = [str(column) for column in table.columns[:50]]
            continue

        dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or join_on
        dedupe_on = [column for column in dedupe_on if column in table.columns]
        duplicate_rows = int(table.duplicated(subset=dedupe_on).sum()) if dedupe_on else 0
        result.update(
            {
                "status": "ok",
                "active_file": _display_path(candidate, base_dir),
                "row_count": int(len(table)),
                "column_count": int(len(table.columns)),
                "dedupe_on": dedupe_on,
                "duplicate_rows_on_key": duplicate_rows,
                "canonical_columns": [str(column) for column in table.columns[:50]],
            }
        )
        return result

    return result


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
def _append_external_tables(
    frame: pd.DataFrame,
    raw_dir: Path,
    dataset_config: dict[str, Any] | None,
    base_dir: Path | None,
    *,
    recent_date_floor: pd.Timestamp | None = None,
    max_rows: int | None = None,
) -> pd.DataFrame:
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
            base_frame=frame,
            base_dir=base_dir,
        )
        if append_frame is None or append_frame.empty:
            continue

        if recent_date_floor is not None and "date" in append_frame.columns:
            append_dates = pd.to_datetime(append_frame["date"], errors="coerce")
            recent_append_frame = append_frame.loc[append_dates.notna() & (append_dates >= recent_date_floor)].copy()
            if not recent_append_frame.empty:
                append_frame = recent_append_frame

        combined = pd.concat([frame, append_frame], ignore_index=True, sort=False)
        dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or ["race_id", "horse_id"]
        dedupe_on = [column for column in dedupe_on if column in combined.columns]
        if dedupe_on:
            combined = combined.drop_duplicates(subset=dedupe_on, keep="first", ignore_index=True)
        frame = _sort_and_tail(combined, max_rows)

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
    supplemental_tables = _get_supplemental_table_configs(dataset_cfg)

    default_search_roots = _resolve_search_roots(
        raw_dir=raw_dir,
        dataset_config=dataset_cfg,
        base_dir=base_dir,
        include_raw_dir=True,
    )

    for table_cfg in supplemental_tables:
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
            base_frame=frame,
            base_dir=base_dir,
        )
        if supplemental_frame is None or supplemental_frame.empty:
            continue

        dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or join_on
        dedupe_on = [column for column in dedupe_on if column in supplemental_frame.columns]
        if dedupe_on:
            supplemental_frame = supplemental_frame.drop_duplicates(subset=dedupe_on, keep="first")

        supplemental_frame = _restrict_table_to_join_keys(supplemental_frame, frame, join_on)
        if supplemental_frame.empty:
            continue

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
        frame["distance"] = _normalize_digit_series(frame["distance"])

    for column in ["frame_no", "gate_no"]:
        if column in frame.columns:
            frame[column] = _normalize_digit_series(frame[column])

    if "weight" in frame.columns:
        frame["weight"] = _normalize_digit_series(frame["weight"])

    if "odds" in frame.columns:
        frame["odds"] = _normalize_decimal_series(frame["odds"])

    if "popularity" in frame.columns:
        frame["popularity"] = _normalize_digit_series(frame["popularity"])

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


def load_training_table_tail(
    raw_dir: str | Path,
    *,
    tail_rows: int,
    dataset_config: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> tuple[pd.DataFrame, int]:
    base_path = Path(base_dir) if base_dir is not None else None
    raw_path = _resolve_path(raw_dir, base_path)
    dataset_path = _pick_dataset(raw_path)
    frame, primary_source_rows_total = _read_csv_tail(dataset_path, int(tail_rows))
    frame = _normalize_columns(frame)
    recent_date_floor = _resolve_recent_date_floor(frame)
    frame = _append_external_tables(
        frame,
        raw_path,
        dataset_config=dataset_config,
        base_dir=base_path,
        recent_date_floor=recent_date_floor,
        max_rows=int(tail_rows),
    )
    frame = _merge_supplemental_tables(frame, raw_path, dataset_config=dataset_config, base_dir=base_path)
    frame = _ensure_minimum_columns(frame)
    frame = frame.sort_values(["date", "race_id"]).tail(int(tail_rows)).reset_index(drop=True)
    return frame, int(primary_source_rows_total)


def load_training_table_for_feature_build(
    raw_dir: str | Path,
    *,
    pre_feature_max_rows: int | None = None,
    dataset_config: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> TrainingTableLoadResult:
    if pre_feature_max_rows is not None:
        if int(pre_feature_max_rows) <= 0:
            raise ValueError("pre_feature_max_rows must be greater than 0")
        frame, primary_source_rows_total = load_training_table_tail(
            raw_dir,
            tail_rows=int(pre_feature_max_rows),
            dataset_config=dataset_config,
            base_dir=base_dir,
        )
        data_load_strategy = "tail_training_table"
    else:
        frame = load_training_table(
            raw_dir,
            dataset_config=dataset_config,
            base_dir=base_dir,
        )
        primary_source_rows_total = None
        data_load_strategy = "full_training_table"

    frame = frame.copy()
    loaded_rows = int(len(frame))
    return TrainingTableLoadResult(
        frame=frame,
        loaded_rows=loaded_rows,
        pre_feature_rows=loaded_rows,
        data_load_strategy=data_load_strategy,
        primary_source_rows_total=int(primary_source_rows_total) if primary_source_rows_total is not None else None,
    )


def inspect_dataset_sources(
    raw_dir: str | Path,
    dataset_config: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
) -> dict[str, Any]:
    base_path = Path(base_dir) if base_dir is not None else None
    raw_path = _resolve_path(raw_dir, base_path)
    dataset_cfg = _resolve_dataset_config(dataset_config)

    summary: dict[str, Any] = {
        "raw_dir": _display_path(raw_path, base_path),
        "raw_dir_exists": bool(raw_path.exists()),
        "external_raw_dirs": [],
        "primary_dataset": None,
        "append_tables": [],
        "supplemental_tables": [],
    }

    for external_dir in _resolve_external_raw_dirs(dataset_cfg, base_path):
        csv_count = len(list(external_dir.glob("**/*.csv"))) if external_dir.exists() else 0
        summary["external_raw_dirs"].append(
            {
                "path": _display_path(external_dir, base_path),
                "exists": bool(external_dir.exists()),
                "csv_file_count": int(csv_count),
            }
        )

    try:
        primary_path = _pick_dataset(raw_path)
        summary["primary_dataset"] = {
            "status": "ok",
            "path": _display_path(primary_path, base_path),
        }
    except Exception as error:
        summary["primary_dataset"] = {
            "status": "missing",
            "error": str(error),
        }

    append_defaults = _resolve_search_roots(
        raw_dir=raw_path,
        dataset_config=dataset_cfg,
        base_dir=base_path,
        include_raw_dir=False,
    )
    supplemental_defaults = _resolve_search_roots(
        raw_dir=raw_path,
        dataset_config=dataset_cfg,
        base_dir=base_path,
        include_raw_dir=True,
    )

    append_tables = dataset_cfg.get("append_tables", [])
    if isinstance(append_tables, list):
        summary["append_tables"] = [
            _inspect_table_sources(table_cfg=table_cfg, default_search_roots=append_defaults, base_dir=base_path)
            for table_cfg in append_tables
            if isinstance(table_cfg, dict)
        ]

    summary["supplemental_tables"] = [
        _inspect_table_sources(table_cfg=table_cfg, default_search_roots=supplemental_defaults, base_dir=base_path)
        for table_cfg in _get_supplemental_table_configs(dataset_cfg)
    ]

    return summary


def materialize_supplemental_table(
    raw_dir: str | Path,
    *,
    table_name: str,
    dataset_config: dict[str, Any] | None = None,
    base_dir: str | Path | None = None,
    output_file: str | Path | None = None,
) -> dict[str, Any]:
    base_path = Path(base_dir) if base_dir is not None else None
    raw_path = _resolve_path(raw_dir, base_path)
    dataset_cfg = _resolve_dataset_config(dataset_config)
    supplemental_tables = _get_supplemental_table_configs(dataset_cfg)
    normalized_name = str(table_name).strip()
    table_cfg = next((table for table in supplemental_tables if str(table.get("name", "")).strip() == normalized_name), None)
    if table_cfg is None:
        raise ValueError(f"Unknown supplemental table: {table_name}")

    resolved_output = Path(output_file) if output_file is not None else _resolve_materialized_candidate(table_cfg, base_path)
    if resolved_output is None:
        raise ValueError(f"No output file configured for supplemental table: {table_name}")
    if not resolved_output.is_absolute() and base_path is not None:
        resolved_output = base_path / resolved_output

    search_dirs = _normalize_string_list(table_cfg.get("search_dirs"))
    if search_dirs:
        search_roots = [_resolve_path(search_dir, base_path) for search_dir in search_dirs]
    else:
        search_roots = _resolve_search_roots(
            raw_dir=raw_path,
            dataset_config=dataset_cfg,
            base_dir=base_path,
            include_raw_dir=True,
        )

    source_table_cfg = dict(table_cfg)
    source_table_cfg.pop("materialized_file", None)
    join_on = _normalize_string_list(table_cfg.get("join_on"))
    required_columns = _normalize_string_list(table_cfg.get("required_columns")) or join_on
    keep_columns = _normalize_string_list(table_cfg.get("keep_columns"))
    table = _load_matching_table(
        table_cfg=source_table_cfg,
        search_roots=search_roots,
        join_on=join_on,
        keep_columns=keep_columns,
        required_columns=required_columns,
        base_frame=None,
        base_dir=base_path,
    )
    if table is None or table.empty:
        return {
            "status": "not_ready",
            "table_name": normalized_name,
            "raw_dir": _display_path(raw_path, base_path),
            "output_file": _display_path(resolved_output, base_path),
            "search_roots": [_display_path(path, base_path) for path in search_roots],
            "reason": "source_table_missing",
            "row_count": 0,
        }

    dedupe_on = _normalize_string_list(table_cfg.get("dedupe_on")) or join_on
    dedupe_on = [column for column in dedupe_on if column in table.columns]
    duplicate_rows = int(table.duplicated(subset=dedupe_on).sum()) if dedupe_on else 0
    if dedupe_on:
        table = table.drop_duplicates(subset=dedupe_on, keep="first")

    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(resolved_output, index=False)
    race_id_min: str | None = None
    race_id_max: str | None = None
    race_id_date_start: str | None = None
    race_id_date_end: str | None = None
    if "race_id" in table.columns:
        race_values = table["race_id"].dropna().astype(str)
        if not race_values.empty:
            race_id_min = str(race_values.min())
            race_id_max = str(race_values.max())
            race_id_date_start = _parse_race_id_date_token(race_id_min)
            race_id_date_end = _parse_race_id_date_token(race_id_max)
    return {
        "status": "completed",
        "table_name": normalized_name,
        "raw_dir": _display_path(raw_path, base_path),
        "output_file": _display_path(resolved_output, base_path),
        "search_roots": [_display_path(path, base_path) for path in search_roots],
        "row_count": int(len(table)),
        "column_count": int(len(table.columns)),
        "duplicate_rows_on_key": duplicate_rows,
        "dedupe_on": dedupe_on,
        "columns": [str(column) for column in table.columns],
        "race_id_min": race_id_min,
        "race_id_max": race_id_max,
        "race_id_date_start": race_id_date_start,
        "race_id_date_end": race_id_date_end,
    }
