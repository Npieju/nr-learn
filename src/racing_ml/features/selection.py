from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


DEFAULT_SAFE_EXCLUDE_COLUMNS = {
    "date",
    "race_id",
    "horse_key",
    "rank",
    "is_win",
    "horse_history_key",
    "horse_history_key_source",
    "finish_time",
    "finish_time_sec",
    "closing_time_3f",
    "corner_1_position",
    "corner_2_position",
    "corner_3_position",
    "corner_4_position",
    "race_pace_front3f",
    "race_pace_back3f",
    "course_baseline_time_per_1000m",
    "time_per_1000m",
    "time_margin_sec",
    "time_deviation",
    "owner_name",
    "breeder_name",
    "sire_name",
    "dam_name",
    "damsire_name",
    "horse_track_distance_key",
    "jockey_track_distance_key",
    "trainer_track_distance_key",
    "sire_track_distance_key",
}

DEFAULT_SAFE_EXCLUDE_KEYWORDS = [
    "着順",
    "result",
    "finish_position",
    "タイム",
    "着差",
    "コーナー",
    "corner",
    "上り",
    "lap",
    "passing_order",
    "払戻",
    "配当",
    "payout",
    "確定",
    "賞金",
]

DEFAULT_SAFE_EXCLUDE_SUFFIXES = ("_key",)

DEFAULT_FORCE_CATEGORICAL_COLUMNS = {
    "horse_id",
    "horse_key",
    "jockey_id",
    "trainer_id",
    "track",
    "weather",
    "ground_condition",
    "sex",
}

DEFAULT_FORCE_CATEGORICAL_SUFFIXES = ("_id",)


@dataclass(frozen=True)
class FeatureSelection:
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    excluded_columns: list[str]
    mode: str


def _normalize_string_list(values: Any) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        text = values.strip()
        return [text] if text else []

    normalized: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            normalized.append(text)
    return normalized


def _looks_like_history_feature(column_name: str) -> bool:
    lower_name = column_name.lower()
    return ("last_" in lower_name) or lower_name.startswith("prev_")


def _is_probably_categorical(frame: pd.DataFrame, column_name: str) -> bool:
    if column_name not in frame.columns:
        return False
    if not pd.api.types.is_numeric_dtype(frame[column_name]):
        return True

    lower_name = column_name.lower()
    if lower_name in DEFAULT_FORCE_CATEGORICAL_COLUMNS:
        return True
    return lower_name.endswith(DEFAULT_FORCE_CATEGORICAL_SUFFIXES)


def _should_exclude_column(
    column_name: str,
    excluded_lower_names: set[str],
    excluded_keywords: list[str],
) -> bool:
    lower_name = column_name.lower()
    if lower_name in excluded_lower_names:
        return True

    if _looks_like_history_feature(lower_name):
        return False

    if lower_name.endswith(DEFAULT_SAFE_EXCLUDE_SUFFIXES):
        return True

    return any(keyword in lower_name for keyword in excluded_keywords)


def resolve_feature_selection(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    label_column: str,
) -> FeatureSelection:
    features_cfg = feature_config.get("features", {})
    selection_cfg = feature_config.get("selection", {})
    mode = str(selection_cfg.get("mode", "explicit")).strip().lower() or "explicit"

    base_columns = _normalize_string_list(features_cfg.get("base", []))
    history_columns = _normalize_string_list(features_cfg.get("history", []))
    explicit_includes = _normalize_string_list(features_cfg.get("include", []))
    force_include_columns = _normalize_string_list(selection_cfg.get("force_include_columns", []))

    excluded_columns = {
        *DEFAULT_SAFE_EXCLUDE_COLUMNS,
        label_column,
        *_normalize_string_list(selection_cfg.get("exclude_columns", [])),
    }
    excluded_lower_names = {column.lower() for column in excluded_columns}
    excluded_keywords = [
        keyword.lower()
        for keyword in [
            *DEFAULT_SAFE_EXCLUDE_KEYWORDS,
            *_normalize_string_list(selection_cfg.get("exclude_keywords", [])),
        ]
    ]

    if mode == "explicit":
        candidates = [*base_columns, *history_columns, *explicit_includes]
    elif mode in {"all_safe", "all_usable"}:
        candidates = list(frame.columns)
    else:
        raise ValueError(f"Unsupported feature selection mode: {mode}")

    selected_columns: list[str] = []
    rejected_columns: list[str] = []
    seen: set[str] = set()
    for column in candidates:
        if column in seen or column not in frame.columns:
            continue

        seen.add(column)
        lower_name = column.lower()
        if lower_name in excluded_lower_names:
            rejected_columns.append(column)
            continue

        if mode != "explicit" and _should_exclude_column(column, excluded_lower_names, excluded_keywords):
            rejected_columns.append(column)
            continue

        selected_columns.append(column)

    for column in force_include_columns:
        if column in frame.columns and column not in seen:
            seen.add(column)
            selected_columns.append(column)

    force_categorical_columns = {
        *DEFAULT_FORCE_CATEGORICAL_COLUMNS,
        *_normalize_string_list(selection_cfg.get("force_categorical_columns", [])),
    }
    categorical_columns = [
        column
        for column in selected_columns
        if column in force_categorical_columns or _is_probably_categorical(frame, column)
    ]
    numeric_columns = [column for column in selected_columns if column not in categorical_columns]

    return FeatureSelection(
        feature_columns=selected_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        excluded_columns=sorted(set(rejected_columns)),
        mode=mode,
    )


def prepare_model_input_frame(
    frame: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str] | None = None,
) -> pd.DataFrame:
    if not feature_columns:
        raise ValueError("No feature columns available for model input")

    categorical_set = set(categorical_columns or [])
    work = frame.copy()
    for column in feature_columns:
        if column not in work.columns:
            work[column] = pd.NA

    model_frame = work[feature_columns].copy()
    for column in feature_columns:
        if column in categorical_set:
            model_frame[column] = model_frame[column].astype("string").fillna("__nan__")
        else:
            model_frame[column] = pd.to_numeric(model_frame[column], errors="coerce")

    return model_frame


def resolve_model_feature_selection(
    model: object,
    fallback_selection: FeatureSelection,
) -> FeatureSelection:
    if not isinstance(model, dict):
        return fallback_selection

    model_kind = str(model.get("kind", "")).strip().lower()
    if model_kind not in {"tabular_model", "multi_position_top3", "value_blend_model"}:
        return fallback_selection

    model_feature_columns = [
        str(column)
        for column in model.get("feature_columns", [])
        if str(column).strip()
    ]
    if not model_feature_columns:
        return fallback_selection

    categorical_columns = [
        str(column)
        for column in model.get("categorical_columns", [])
        if str(column).strip() and str(column) in model_feature_columns
    ]
    if not categorical_columns:
        categorical_columns = [
            column for column in fallback_selection.categorical_columns if column in model_feature_columns
        ]

    return FeatureSelection(
        feature_columns=model_feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=[column for column in model_feature_columns if column not in categorical_columns],
        excluded_columns=list(fallback_selection.excluded_columns),
        mode=fallback_selection.mode,
    )


def summarize_feature_coverage(
    frame: pd.DataFrame,
    feature_config: dict[str, Any],
    selection: FeatureSelection,
    coverage_threshold: float = 0.5,
) -> dict[str, Any]:
    selection_cfg = feature_config.get("selection", {})
    configured_threshold = selection_cfg.get("coverage_threshold")
    threshold = float(configured_threshold) if configured_threshold is not None else float(coverage_threshold)
    force_include_columns = _normalize_string_list(selection_cfg.get("force_include_columns"))

    missing_force_include_features: list[str] = []
    empty_force_include_features: list[str] = []
    low_coverage_force_include_features: list[str] = []
    selected_low_coverage_features: list[str] = []

    for column in force_include_columns:
        if column not in frame.columns:
            missing_force_include_features.append(column)
            continue

        non_null_ratio = float(frame[column].notna().mean())
        if non_null_ratio == 0.0:
            empty_force_include_features.append(column)
        elif non_null_ratio < threshold:
            low_coverage_force_include_features.append(column)

    for column in selection.feature_columns:
        if column not in frame.columns:
            continue
        non_null_ratio = float(frame[column].notna().mean())
        if 0.0 < non_null_ratio < threshold:
            selected_low_coverage_features.append(column)

    return {
        "coverage_threshold": threshold,
        "selected_feature_count": int(len(selection.feature_columns)),
        "force_include_total": int(len(force_include_columns)),
        "force_include_present": int(sum(1 for column in force_include_columns if column in frame.columns)),
        "missing_force_include_features": missing_force_include_features,
        "empty_force_include_features": empty_force_include_features,
        "low_coverage_force_include_features": low_coverage_force_include_features,
        "selected_low_coverage_features": selected_low_coverage_features[:20],
    }