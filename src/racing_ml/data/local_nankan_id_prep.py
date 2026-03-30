from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import write_csv_file
from racing_ml.data.local_nankan_race_list import discover_local_nankan_race_ids_from_calendar


RACE_TARGET_NAMES = ("race_result", "race_card")


def _normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if pd.isna(value):
            return ""
        if value.is_integer():
            return str(int(value))

    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        return text[:-2]
    return text


def _is_valid_horse_key(value: object) -> bool:
    return bool(_normalize_text(value)) and _normalize_text(value).isdigit()


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_crawl_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("crawl")
    if isinstance(nested, dict):
        return nested
    return config


def _read_seed_frame(seed_path: Path, required_columns: list[str], *, workspace_root: Path) -> pd.DataFrame:
    try:
        frame = pd.read_csv(seed_path, dtype="string", low_memory=False)
    except pd.errors.EmptyDataError:
        raise ValueError(f"seed file is empty: {seed_path}") from None

    missing_columns = [column for column in required_columns if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"seed file missing required columns {missing_columns}: {artifact_display_path(seed_path, workspace_root=workspace_root)}"
        )
    return frame


def _collect_existing_values(path: Path, column_name: str) -> set[str]:
    if not path.exists():
        return set()
    try:
        frame = pd.read_csv(path, usecols=[column_name], dtype="string", low_memory=False)
    except (pd.errors.EmptyDataError, ValueError):
        return set()
    return {
        text
        for text in (_normalize_text(value) for value in frame[column_name].tolist())
        if text
    }


def _read_output_columns(path: Path, columns: list[str]) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        frame = pd.read_csv(
            path,
            usecols=lambda column: str(column).strip() in set(columns),
            dtype="string",
            low_memory=False,
        )
    except (pd.errors.EmptyDataError, ValueError):
        return None
    if frame.empty:
        return None
    return frame


def _collect_completed_values_for_targets(
    *,
    selected_targets: list[str],
    targets: dict[str, Any],
    base_dir: Path,
    column_name: str,
) -> set[str]:
    completed_sets: list[set[str]] = []
    for target_name in selected_targets:
        target_cfg = targets.get(target_name, {})
        output_file = target_cfg.get("output_file")
        if not output_file:
            completed_sets.append(set())
            continue
        completed_sets.append(_collect_existing_values(_resolve_path(output_file, base_dir), column_name))

    if not completed_sets or not all(completed_sets):
        return set()
    return set.intersection(*completed_sets)


def _filter_by_date(frame: pd.DataFrame, date_column: str, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    work = frame.copy()
    work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
    work = work.dropna(subset=[date_column])
    if start_date:
        work = work[work[date_column] >= pd.Timestamp(start_date)]
    if end_date:
        work = work[work[date_column] <= pd.Timestamp(end_date)]
    return work


def _build_race_id_frame(
    seed_frame: pd.DataFrame,
    *,
    race_id_column: str,
    date_column: str,
    selected_targets: list[str],
    targets: dict[str, Any],
    base_dir: Path,
    include_completed: bool,
    limit: int | None,
    date_order: str,
) -> pd.DataFrame:
    work = seed_frame[[race_id_column, date_column]].copy()
    work[race_id_column] = work[race_id_column].map(_normalize_text)
    work = work[work[race_id_column] != ""]
    work = _filter_by_date(work, date_column, None, None)
    work = work.drop_duplicates(subset=[race_id_column], keep="first")

    if not include_completed and selected_targets:
        completed_ids = _collect_completed_values_for_targets(
            selected_targets=selected_targets,
            targets=targets,
            base_dir=base_dir,
            column_name="race_id",
        )
        if completed_ids:
            work = work[~work[race_id_column].isin(completed_ids)]

    descending = str(date_order).strip().lower() == "desc"
    work = work.sort_values([date_column, race_id_column], ascending=[not descending, not descending], kind="stable")
    work[date_column] = work[date_column].dt.strftime("%Y-%m-%d")

    output = work.rename(columns={race_id_column: "race_id", date_column: "date"}).reset_index(drop=True)
    if limit is not None and limit > 0:
        output = output.head(int(limit)).reset_index(drop=True)
    return output


def _build_horse_key_frame(
    seed_frame: pd.DataFrame,
    *,
    horse_key_column: str,
    targets: dict[str, Any],
    base_dir: Path,
    include_completed: bool,
    limit: int | None,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    if horse_key_column in seed_frame.columns:
        for value in seed_frame[horse_key_column].tolist():
            text = _normalize_text(value)
            if not _is_valid_horse_key(text) or text in seen:
                continue
            seen.add(text)
            records.append({"horse_key": text, "source": "seed_file"})

    for target_name in RACE_TARGET_NAMES:
        target_cfg = targets.get(target_name, {})
        output_file = target_cfg.get("output_file")
        if not output_file:
            continue
        frame = _read_output_columns(_resolve_path(output_file, base_dir), ["horse_key"])
        if frame is None or "horse_key" not in frame.columns:
            continue
        for value in frame["horse_key"].tolist():
            text = _normalize_text(value)
            if not _is_valid_horse_key(text) or text in seen:
                continue
            seen.add(text)
            records.append({"horse_key": text, "source": target_name})

    output = pd.DataFrame(records, columns=["horse_key", "source"])

    if not include_completed:
        pedigree_cfg = targets.get("pedigree", {})
        output_file = pedigree_cfg.get("output_file")
        completed_keys = _collect_existing_values(_resolve_path(output_file, base_dir), "horse_key") if output_file else set()
        if completed_keys:
            output = output[~output["horse_key"].isin(completed_keys)].reset_index(drop=True)

    if limit is not None and limit > 0:
        output = output.head(int(limit)).reset_index(drop=True)
    return output


def prepare_local_nankan_ids_from_config(
    crawl_config: dict[str, Any],
    *,
    base_dir: Path,
    seed_file: str | Path | None = None,
    target_filter: str = "all",
    start_date: str | None = None,
    end_date: str | None = None,
    date_order: str = "asc",
    limit: int | None = None,
    include_completed: bool = False,
    race_id_source: str = "seed_file",
) -> dict[str, Any]:
    crawl_cfg = _resolve_crawl_config(crawl_config)
    targets = crawl_cfg.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("crawl.targets must contain at least one target")

    normalized_race_id_source = str(race_id_source).strip().lower() or "seed_file"
    if normalized_race_id_source not in {"seed_file", "race_list"}:
        raise ValueError(f"Unsupported race_id_source: {race_id_source}")

    seed_columns_cfg = crawl_cfg.get("seed_columns") or {}
    race_id_column = str(seed_columns_cfg.get("race_id", "race_id"))
    date_column = str(seed_columns_cfg.get("date", "date"))
    horse_key_column = str(seed_columns_cfg.get("horse_key", "horse_key"))
    configured_seed_file = seed_file or crawl_cfg.get("seed_file")
    seed_path = _resolve_path(str(configured_seed_file), base_dir) if configured_seed_file else None
    race_source_report: dict[str, Any] | None = None
    seed_frame = pd.DataFrame(columns=[race_id_column, date_column, horse_key_column])
    race_source_frame = pd.DataFrame(columns=[race_id_column, date_column])

    if normalized_race_id_source == "race_list":
        if not start_date:
            raise ValueError("race_id_source='race_list' requires start_date")
        completed_ids = set()
        selected_race_targets = [name for name in RACE_TARGET_NAMES if target_filter in {"all", name} and name in targets]
        if not include_completed and selected_race_targets:
            completed_ids = _collect_completed_values_for_targets(
                selected_targets=selected_race_targets,
                targets=targets,
                base_dir=base_dir,
                column_name="race_id",
            )
        discovered_frame, race_source_report = discover_local_nankan_race_ids_from_calendar(
            crawl_config,
            base_dir=base_dir,
            start_date=start_date,
            end_date=end_date or start_date,
            limit=limit,
            date_order=date_order,
            exclude_race_ids=completed_ids,
            require_result_link=target_filter == "race_result",
        )
        if not discovered_frame.empty:
            race_source_frame = discovered_frame.rename(columns={"race_id": race_id_column, "date": date_column})
        if seed_path is not None and seed_path.exists():
            seed_frame = _read_seed_frame(seed_path, [race_id_column, date_column], workspace_root=base_dir)
            seed_frame = _filter_by_date(seed_frame, date_column, start_date, end_date)
    else:
        if seed_path is None:
            raise ValueError("seed_file is required via --seed-file or crawl.seed_file")
        seed_frame = _read_seed_frame(seed_path, [race_id_column, date_column], workspace_root=base_dir)
        seed_frame = _filter_by_date(seed_frame, date_column, start_date, end_date)
        race_source_frame = seed_frame

    selected_race_targets = [name for name in RACE_TARGET_NAMES if target_filter in {"all", name} and name in targets]
    selected_pedigree_targets = ["pedigree"] if target_filter in {"all", "pedigree"} and "pedigree" in targets else []

    summary: dict[str, Any] = {
        "target_filter": target_filter,
        "date_window": {"start": start_date, "end": end_date},
        "date_order": str(date_order),
        "race_id_source": normalized_race_id_source,
        "seed_file": artifact_display_path(seed_path, workspace_root=base_dir) if seed_path is not None else None,
        "reports": [],
    }
    if race_source_report is not None:
        summary["race_id_source_report"] = race_source_report

    if selected_race_targets:
        race_frame = _build_race_id_frame(
            race_source_frame,
            race_id_column=race_id_column,
            date_column=date_column,
            selected_targets=selected_race_targets,
            targets=targets,
            base_dir=base_dir,
            include_completed=include_completed,
            limit=limit,
            date_order=date_order,
        )
        id_paths = {
            _resolve_path(str(targets[target_name].get("id_file")), base_dir)
            for target_name in selected_race_targets
            if str(targets[target_name].get("id_file", "")).strip()
        }
        for path in id_paths:
            write_csv_file(path, race_frame, index=False, label="id output")
        summary["reports"].append(
            {
                "kind": "race_ids",
                "targets": selected_race_targets,
                "row_count": int(len(race_frame)),
                "output_files": [artifact_display_path(path, workspace_root=base_dir) for path in sorted(id_paths)],
                "source": normalized_race_id_source,
            }
        )

    if selected_pedigree_targets:
        horse_key_frame = _build_horse_key_frame(
            seed_frame,
            horse_key_column=horse_key_column,
            targets=targets,
            base_dir=base_dir,
            include_completed=include_completed,
            limit=limit,
        )
        id_paths = {
            _resolve_path(str(targets[target_name].get("id_file")), base_dir)
            for target_name in selected_pedigree_targets
            if str(targets[target_name].get("id_file", "")).strip()
        }
        for path in id_paths:
            write_csv_file(path, horse_key_frame, index=False, label="id output")
        summary["reports"].append(
            {
                "kind": "horse_keys",
                "targets": selected_pedigree_targets,
                "row_count": int(len(horse_key_frame)),
                "output_files": [artifact_display_path(path, workspace_root=base_dir) for path in sorted(id_paths)],
                "source": "seed_file_or_existing_outputs",
            }
        )

    return summary