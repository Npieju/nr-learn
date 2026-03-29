from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from racing_ml.common.artifacts import display_path, utc_now_iso, write_csv_file, write_json


RESULT_REQUIRED_COLUMNS = ["date", "race_id", "horse_id"]
RESULT_KEY_COLUMNS = ["race_id", "horse_id", "horse_key", "owner_name", "breeder_name"]
RESULT_CANONICAL_COLUMNS = [
    "date",
    "race_id",
    "horse_id",
    "horse_key",
    "horse_name",
    "rank",
    "odds",
    "popularity",
    "finish_time",
    "margin",
    "closing_time_3f",
    "passing_order",
    "jockey_id",
    "trainer_id",
    "track",
    "distance",
    "weather",
    "ground_condition",
    "sex",
    "age",
    "carried_weight",
    "weight",
    "weight_change",
    "frame_no",
    "gate_no",
    "owner_name",
    "breeder_name",
    "sire_name",
    "dam_name",
    "damsire_name",
    "is_win",
]

CARD_FILL_COLUMNS = [
    "date",
    "track",
    "distance",
    "weather",
    "ground_condition",
    "odds",
    "popularity",
    "horse_key",
    "frame_no",
    "gate_no",
    "horse_name",
    "sex",
    "age",
    "carried_weight",
    "weight",
    "weight_change",
    "jockey_id",
    "trainer_id",
    "owner_name",
    "breeder_name",
]
PEDIGREE_FILL_COLUMNS = ["sire_name", "dam_name", "damsire_name", "owner_name", "breeder_name"]
SUPPLEMENTAL_OUTPUT_FILENAMES = {
    "local_nankan_race_result_keys": "local_nankan_race_result_keys.csv",
    "local_nankan_race_card": "local_nankan_race_card.csv",
    "local_nankan_pedigree": "local_nankan_pedigree.csv",
}


def _resolve_path(raw_path: str | Path, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return base_dir / path


def _resolve_data_config(config: dict[str, Any]) -> dict[str, Any]:
    nested = config.get("dataset")
    if isinstance(nested, dict):
        return nested
    return config


def _display(path: Path, base_dir: Path) -> str:
    return display_path(path, workspace_root=base_dir)


def _load_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def _missing_required_columns(frame: pd.DataFrame, required_columns: list[str]) -> list[str]:
    return [column for column in required_columns if column not in frame.columns]


def _fill_missing_from_merge(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        merged_column = f"{column}__merge"
        if merged_column not in output.columns:
            continue
        if column not in output.columns:
            output[column] = output[merged_column]
        else:
            output[column] = output[column].where(output[column].notna(), output[merged_column])
            output[column] = output[column].mask(output[column].astype(str).str.strip().isin(["", "nan", "None"]), output[merged_column])
        output = output.drop(columns=[merged_column])
    return output


def _normalize_result_frame(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "finish_position" in output.columns and "rank" not in output.columns:
        output = output.rename(columns={"finish_position": "rank"})
    return output


def _merge_card(frame: pd.DataFrame, card_frame: pd.DataFrame) -> pd.DataFrame:
    if card_frame.empty:
        return frame
    merge_columns = [column for column in ["race_id", "horse_id"] if column in card_frame.columns]
    if merge_columns != ["race_id", "horse_id"]:
        return frame
    working = card_frame.drop_duplicates(subset=merge_columns, keep="last")
    keep_columns = merge_columns + [column for column in CARD_FILL_COLUMNS if column in working.columns]
    merged = frame.merge(working[keep_columns], on=merge_columns, how="left", suffixes=("", "__merge"))
    return _fill_missing_from_merge(merged, CARD_FILL_COLUMNS)


def _merge_pedigree(frame: pd.DataFrame, pedigree_frame: pd.DataFrame) -> pd.DataFrame:
    if pedigree_frame.empty or "horse_key" not in frame.columns or "horse_key" not in pedigree_frame.columns:
        return frame
    working = pedigree_frame.drop_duplicates(subset=["horse_key"], keep="last")
    keep_columns = ["horse_key"] + [column for column in PEDIGREE_FILL_COLUMNS if column in working.columns]
    merged = frame.merge(working[keep_columns], on=["horse_key"], how="left", suffixes=("", "__merge"))
    return _fill_missing_from_merge(merged, PEDIGREE_FILL_COLUMNS)


def _ensure_core_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    if "rank" in output.columns:
        output["rank"] = pd.to_numeric(output["rank"], errors="coerce")
    if "date" in output.columns:
        output["date"] = pd.to_datetime(output["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    for column in [
        "distance",
        "odds",
        "popularity",
        "age",
        "carried_weight",
        "weight",
        "weight_change",
        "frame_no",
        "gate_no",
        "closing_time_3f",
    ]:
        if column in output.columns:
            output[column] = pd.to_numeric(output[column], errors="coerce")
    if "is_win" not in output.columns and "rank" in output.columns:
        output["is_win"] = (output["rank"] == 1).astype("Int64")
    return output


def _build_supplemental_frame(frame: pd.DataFrame, *, columns: list[str], dedupe_on: list[str]) -> pd.DataFrame:
    available_columns = [column for column in columns if column in frame.columns]
    if not available_columns:
        return pd.DataFrame(columns=columns)
    output = frame[available_columns].copy()
    dedupe_subset = [column for column in dedupe_on if column in output.columns]
    if dedupe_subset:
        output = output.drop_duplicates(subset=dedupe_subset, keep="last")
    ordered_columns = available_columns + [column for column in columns if column not in available_columns]
    return output.reindex(columns=ordered_columns)


def _build_manifest(
    *,
    base_dir: Path,
    output_path: Path,
    manifest_path: Path,
    result_path: Path | None,
    card_path: Path | None,
    pedigree_path: Path | None,
    status: str,
    current_phase: str,
    recommended_action: str,
    row_count: int,
    columns: list[str],
    dry_run: bool,
    error_code: str | None = None,
    error_message: str | None = None,
    generated_files: dict[str, str] | None = None,
) -> dict[str, Any]:
    highlights = [
        f"row_count={row_count}",
        f"columns={len(columns)}",
        f"dry_run={dry_run}",
    ]
    if result_path is None:
        highlights.append("race_result source is missing")
    elif card_path is None or pedigree_path is None:
        highlights.append("optional enrichments are partially missing; primary raw can still be materialized")
    if generated_files:
        highlights.append(f"generated_files={len(generated_files)}")

    payload = {
        "started_at": utc_now_iso(),
        "finished_at": utc_now_iso(),
        "status": status,
        "current_phase": current_phase,
        "recommended_action": recommended_action,
        "error_code": error_code,
        "error_message": error_message,
        "output_file": _display(output_path, base_dir),
        "manifest_file": _display(manifest_path, base_dir),
        "source_files": {
            "race_result": _display(result_path, base_dir) if result_path is not None else None,
            "race_card": _display(card_path, base_dir) if card_path is not None else None,
            "pedigree": _display(pedigree_path, base_dir) if pedigree_path is not None else None,
        },
        "row_count": row_count,
        "columns": columns,
        "highlights": highlights,
    }
    if generated_files:
        payload["generated_files"] = generated_files
    return payload


def materialize_local_nankan_primary_from_config(
    data_config: dict[str, Any],
    *,
    base_dir: Path,
    race_result_path: str | Path | None = None,
    race_card_path: str | Path | None = None,
    pedigree_path: str | Path | None = None,
    output_file: str | Path | None = None,
    manifest_file: str | Path = "artifacts/reports/local_nankan_primary_materialize_manifest.json",
    dry_run: bool = False,
) -> dict[str, Any]:
    dataset_cfg = _resolve_data_config(data_config)
    raw_dir = _resolve_path(str(dataset_cfg.get("raw_dir", "data/local_nankan/raw")), base_dir)
    output_path = _resolve_path(str(output_file or (raw_dir / "local_nankan_primary.csv")), base_dir)
    manifest_path = _resolve_path(manifest_file, base_dir)

    result_candidates = [race_result_path] if race_result_path else [
        "data/external/local_nankan/results/local_race_result.csv",
    ]
    card_candidates = [race_card_path] if race_card_path else [
        "data/external/local_nankan/racecard/local_racecard.csv",
    ]
    pedigree_candidates = [pedigree_path] if pedigree_path else [
        "data/external/local_nankan/pedigree/local_pedigree.csv",
    ]

    result_file = _first_existing([_resolve_path(path, base_dir) for path in result_candidates if path])
    card_file = _first_existing([_resolve_path(path, base_dir) for path in card_candidates if path])
    pedigree_file = _first_existing([_resolve_path(path, base_dir) for path in pedigree_candidates if path])

    if result_file is None:
        summary = _build_manifest(
            base_dir=base_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            result_path=None,
            card_path=card_file,
            pedigree_path=pedigree_file,
            status="not_ready",
            current_phase="source_missing",
            recommended_action="populate_external_results",
            row_count=0,
            columns=[],
            dry_run=dry_run,
            error_code="race_result_missing",
            error_message="local_nankan race_result source csv is missing",
        )
        write_json(manifest_path, summary)
        return summary

    result_frame = _normalize_result_frame(_load_csv(result_file))
    missing_result_columns = _missing_required_columns(result_frame, RESULT_REQUIRED_COLUMNS)
    if missing_result_columns:
        summary = _build_manifest(
            base_dir=base_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            result_path=result_file,
            card_path=card_file,
            pedigree_path=pedigree_file,
            status="failed",
            current_phase="validate_result_columns",
            recommended_action="fix_external_result_schema",
            row_count=0,
            columns=list(result_frame.columns),
            dry_run=dry_run,
            error_code="race_result_schema_invalid",
            error_message=f"missing required race_result columns: {missing_result_columns}",
        )
        write_json(manifest_path, summary)
        return summary

    primary_frame = result_frame.copy()
    if card_file is not None:
        primary_frame = _merge_card(primary_frame, _load_csv(card_file))
    if pedigree_file is not None:
        primary_frame = _merge_pedigree(primary_frame, _load_csv(pedigree_file))
    primary_frame = _ensure_core_columns(primary_frame)

    ordered_columns = [column for column in RESULT_CANONICAL_COLUMNS if column in primary_frame.columns]
    remaining_columns = [column for column in primary_frame.columns if column not in ordered_columns]
    primary_frame = primary_frame[ordered_columns + remaining_columns].copy()

    supplemental_frames = {
        "local_nankan_race_result_keys": _build_supplemental_frame(
            primary_frame,
            columns=RESULT_KEY_COLUMNS,
            dedupe_on=["race_id", "horse_id"],
        ),
        "local_nankan_race_card": _build_supplemental_frame(
            primary_frame,
            columns=["race_id", "horse_id"] + CARD_FILL_COLUMNS,
            dedupe_on=["race_id", "horse_id"],
        ),
        "local_nankan_pedigree": _build_supplemental_frame(
            primary_frame,
            columns=["horse_key"] + PEDIGREE_FILL_COLUMNS,
            dedupe_on=["horse_key"],
        ),
    }

    generated_files = {
        name: _display(raw_dir / filename, base_dir)
        for name, filename in SUPPLEMENTAL_OUTPUT_FILENAMES.items()
    }

    if not dry_run:
        write_csv_file(output_path, primary_frame, index=False, label="local_nankan primary raw")
        for name, filename in SUPPLEMENTAL_OUTPUT_FILENAMES.items():
            write_csv_file(
                raw_dir / filename,
                supplemental_frames[name],
                index=False,
                label=f"{name} supplemental raw",
            )

    summary = _build_manifest(
        base_dir=base_dir,
        output_path=output_path,
        manifest_path=manifest_path,
        result_path=result_file,
        card_path=card_file,
        pedigree_path=pedigree_file,
        status="planned" if dry_run else "completed",
        current_phase="planned" if dry_run else "materialized_primary_raw",
        recommended_action="run_local_preflight" if not dry_run else "review_materialize_plan",
        row_count=int(len(primary_frame)),
        columns=[str(column) for column in primary_frame.columns],
        dry_run=dry_run,
        generated_files=generated_files,
    )
    write_json(manifest_path, summary)
    return summary