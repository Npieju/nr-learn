from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from racing_ml.common.artifacts import write_csv_file
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.data.netkeiba_race_list import discover_netkeiba_race_ids_from_race_list


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
    except pd.errors.EmptyDataError:
        return None
    if frame.empty:
        return None
    return frame


def _collect_existing_values(path: Path, column_name: str) -> set[str]:
    frame = _read_output_columns(path, [column_name])
    if frame is None or column_name not in frame.columns:
        return set()
    return {
        text
        for text in (_normalize_text(value) for value in frame[column_name].tolist())
        if text
    }


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


def _select_target_names(target_filter: str | None, targets: dict[str, Any], allowed: tuple[str, ...]) -> list[str]:
    if target_filter in {None, "all"}:
        return [name for name in allowed if name in targets]
    return [target_filter] if target_filter in allowed and target_filter in targets else []


def _normalize_target_names(target_names: list[str] | tuple[str, ...] | None, targets: dict[str, Any]) -> list[str]:
    if not target_names:
        return []

    normalized: list[str] = []
    for target_name in target_names:
        text = str(target_name).strip()
        if text and text in targets and text not in normalized:
            normalized.append(text)
    return normalized


def _filter_by_date(frame: pd.DataFrame, start_date: str | None, end_date: str | None) -> pd.DataFrame:
    if "date" not in frame.columns:
        return frame

    work = frame.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"])
    if start_date:
        work = work[work["date"] >= pd.Timestamp(start_date)]
    if end_date:
        work = work[work["date"] <= pd.Timestamp(end_date)]
    return work


def _build_race_id_frame(
    training_frame: pd.DataFrame,
    *,
    selected_targets: list[str],
    targets: dict[str, Any],
    base_dir: Path,
    pending_only: bool,
    limit: int | None,
    date_order: str,
) -> pd.DataFrame:
    work = training_frame[[column for column in ["race_id", "date"] if column in training_frame.columns]].copy()
    if "race_id" not in work.columns:
        return pd.DataFrame(columns=["race_id", "date"])

    work["race_id"] = work["race_id"].map(_normalize_text)
    work = work[work["race_id"] != ""]
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.drop_duplicates(subset=["race_id"], keep="first")

    completed_ids: set[str] = set()
    if pending_only and selected_targets:
        completed_ids = _collect_completed_values_for_targets(
            selected_targets=selected_targets,
            targets=targets,
            base_dir=base_dir,
            column_name="race_id",
        )

    if completed_ids:
        work = work[~work["race_id"].isin(completed_ids)]

    descending = str(date_order).strip().lower() == "desc"

    if "date" in work.columns:
        work = work.sort_values(["date", "race_id"], ascending=[not descending, not descending], kind="stable")
        work["date"] = work["date"].dt.strftime("%Y-%m-%d")
    else:
        work = work.sort_values(["race_id"], ascending=not descending, kind="stable")

    if limit is not None and limit > 0:
        work = work.head(int(limit))
    return work.reset_index(drop=True)


def _append_unique_keys(records: list[dict[str, str]], seen: set[str], values: list[str], source: str) -> None:
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        records.append({"horse_key": value, "source": source})


def _build_horse_key_frame(
    training_frame: pd.DataFrame,
    *,
    targets: dict[str, Any],
    base_dir: Path,
    pending_only: bool,
    limit: int | None,
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    seen: set[str] = set()

    if "horse_key" in training_frame.columns:
        training_keys = [
            text
            for text in (_normalize_text(value) for value in training_frame["horse_key"].tolist())
            if text
        ]
        _append_unique_keys(records, seen, training_keys, "training_table")

    for target_name in RACE_TARGET_NAMES:
        target_cfg = targets.get(target_name)
        if not isinstance(target_cfg, dict):
            continue
        output_file = target_cfg.get("output_file")
        if not output_file:
            continue
        frame = _read_output_columns(_resolve_path(output_file, base_dir), ["horse_key"])
        if frame is None or "horse_key" not in frame.columns:
            continue
        source_keys = [
            text
            for text in (_normalize_text(value) for value in frame["horse_key"].tolist())
            if text
        ]
        _append_unique_keys(records, seen, source_keys, target_name)

    work = pd.DataFrame(records, columns=["horse_key", "source"])

    if pending_only and not work.empty:
        pedigree_cfg = targets.get("pedigree", {})
        output_file = pedigree_cfg.get("output_file")
        completed_keys = (
            _collect_existing_values(_resolve_path(output_file, base_dir), "horse_key")
            if output_file
            else set()
        )
        if completed_keys:
            work = work[~work["horse_key"].isin(completed_keys)]

    if limit is not None and limit > 0:
        work = work.head(int(limit))
    return work.reset_index(drop=True)


def _write_id_frame(frame: pd.DataFrame, output_path: Path) -> None:
    write_csv_file(output_path, frame, index=False, label="id output")


def prepare_netkeiba_ids_from_config(
    data_config: dict[str, Any],
    crawl_config: dict[str, Any],
    *,
    base_dir: Path,
    target_filter: str | None = None,
    target_names: list[str] | tuple[str, ...] | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    limit: int | None = None,
    include_completed: bool = False,
    training_frame: pd.DataFrame | None = None,
    date_order: str = "asc",
    race_id_source: str = "training_table",
    refresh: bool = False,
    parse_only: bool = False,
) -> dict[str, Any]:
    crawl_cfg = _resolve_crawl_config(crawl_config)
    targets = crawl_cfg.get("targets")
    if not isinstance(targets, dict) or not targets:
        raise ValueError("crawl.targets must contain at least one target")

    normalized_race_id_source = str(race_id_source).strip().lower() or "training_table"
    if normalized_race_id_source not in {"training_table", "race_list"}:
        raise ValueError(f"Unsupported race_id_source: {race_id_source}")

    explicit_targets = _normalize_target_names(target_names, targets)
    selected_race_targets = explicit_targets or _select_target_names(target_filter, targets, RACE_TARGET_NAMES)
    selected_race_targets = [target_name for target_name in selected_race_targets if target_name in RACE_TARGET_NAMES]
    if explicit_targets:
        selected_pedigree_targets = [target_name for target_name in explicit_targets if target_name == "pedigree"]
    else:
        selected_pedigree_targets = _select_target_names(target_filter, targets, ("pedigree",))

    needs_training_frame = bool(selected_pedigree_targets) or (
        bool(selected_race_targets) and normalized_race_id_source == "training_table"
    )
    if needs_training_frame:
        if training_frame is None:
            dataset_cfg = data_config.get("dataset", data_config)
            raw_dir = dataset_cfg.get("raw_dir", "data/raw")
            working_frame = load_training_table(raw_dir, dataset_config=data_config, base_dir=base_dir)
        else:
            working_frame = training_frame.copy()
        working_frame = _filter_by_date(working_frame, start_date, end_date)
    else:
        working_frame = pd.DataFrame()

    summary: dict[str, Any] = {
        "target_filter": target_filter or "all",
        "target_names": explicit_targets,
        "date_window": {"start": start_date, "end": end_date},
        "date_order": str(date_order),
        "race_id_source": normalized_race_id_source,
        "reports": [],
    }

    if selected_race_targets:
        race_source_report: dict[str, Any] | None = None
        if normalized_race_id_source == "race_list":
            if not start_date:
                raise ValueError("race_id_source='race_list' requires start_date")
            completed_ids = set()
            if not include_completed:
                completed_ids = _collect_completed_values_for_targets(
                    selected_targets=selected_race_targets,
                    targets=targets,
                    base_dir=base_dir,
                    column_name="race_id",
                )
            race_frame, race_source_report = discover_netkeiba_race_ids_from_race_list(
                crawl_config,
                base_dir=base_dir,
                start_date=start_date,
                end_date=end_date or start_date,
                limit=limit,
                date_order=date_order,
                exclude_race_ids=completed_ids,
                refresh=refresh,
                parse_only=parse_only,
            )
        else:
            race_frame = _build_race_id_frame(
                working_frame,
                selected_targets=selected_race_targets,
                targets=targets,
                base_dir=base_dir,
                pending_only=not include_completed,
                limit=limit,
                date_order=date_order,
            )

        id_paths = {
            _resolve_path(str(targets[target_name].get("id_file")), base_dir)
            for target_name in selected_race_targets
            if str(targets[target_name].get("id_file", "")).strip()
        }
        for path in id_paths:
            _write_id_frame(race_frame, path)
        report = {
            "kind": "race_ids",
            "targets": selected_race_targets,
            "row_count": int(len(race_frame)),
            "output_files": [str(path) for path in sorted(id_paths)],
            "source": normalized_race_id_source,
        }
        if race_source_report is not None:
            report["source_report"] = race_source_report
        summary["reports"].append(report)

    if selected_pedigree_targets:
        horse_key_frame = _build_horse_key_frame(
            working_frame,
            targets=targets,
            base_dir=base_dir,
            pending_only=not include_completed,
            limit=limit,
        )
        id_paths = {
            _resolve_path(str(targets[target_name].get("id_file")), base_dir)
            for target_name in selected_pedigree_targets
            if str(targets[target_name].get("id_file", "")).strip()
        }
        for path in id_paths:
            _write_id_frame(horse_key_frame, path)
        summary["reports"].append(
            {
                "kind": "horse_keys",
                "targets": selected_pedigree_targets,
                "row_count": int(len(horse_key_frame)),
                "output_files": [str(path) for path in sorted(id_paths)],
            }
        )

    return summary