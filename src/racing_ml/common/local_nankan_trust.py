from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from racing_ml.data.local_nankan_provenance import build_historical_source_timing_context

DEFAULT_LOCAL_NANKAN_PROVENANCE_MANIFEST = "artifacts/reports/local_nankan_provenance_audit.json"
LEGACY_LOCAL_NANKAN_PROVENANCE_MANIFEST = "artifacts/reports/local_nankan_provenance_audit_issue120_repaired.json"
DEFAULT_LOCAL_NANKAN_SOURCE_TIMING_SUMMARY = "artifacts/reports/local_nankan_source_timing_audit.json"
LEGACY_LOCAL_NANKAN_SOURCE_TIMING_SUMMARY = "artifacts/reports/local_nankan_source_timing_audit_issue121.json"


def _resolve_dataset_config(data_config: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(data_config, dict):
        return {}
    dataset_cfg = data_config.get("dataset")
    if isinstance(dataset_cfg, dict):
        return dataset_cfg
    return data_config


def _is_historical_local_nankan_dataset(data_config: dict[str, Any] | None) -> bool:
    dataset_cfg = _resolve_dataset_config(data_config)
    source_dataset = str(dataset_cfg.get("source_dataset") or "").strip().lower()
    return source_dataset == "local_nankan"


def _resolve_candidate_paths(
    *,
    workspace_root: Path,
    primary_path: str | None,
    fallback_paths: tuple[str, ...],
) -> list[Path]:
    raw_candidates: list[str] = []
    if primary_path:
        raw_candidates.append(primary_path)
    for fallback_path in fallback_paths:
        if fallback_path not in raw_candidates:
            raw_candidates.append(fallback_path)

    resolved_paths: list[Path] = []
    for raw_candidate in raw_candidates:
        candidate = Path(raw_candidate)
        if not candidate.is_absolute():
            candidate = workspace_root / candidate
        resolved_paths.append(candidate)
    return resolved_paths


def _first_existing_path(candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_json_payload(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return None


def _resolve_source_timing_context(
    *,
    workspace_root: Path,
    source_timing_summary_path: str | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    fallback_paths = ()
    if source_timing_summary_path in {None, DEFAULT_LOCAL_NANKAN_SOURCE_TIMING_SUMMARY}:
        fallback_paths = (LEGACY_LOCAL_NANKAN_SOURCE_TIMING_SUMMARY,)
    candidate_paths = _resolve_candidate_paths(
        workspace_root=workspace_root,
        primary_path=source_timing_summary_path or DEFAULT_LOCAL_NANKAN_SOURCE_TIMING_SUMMARY,
        fallback_paths=fallback_paths,
    )
    resolved_path = _first_existing_path(candidate_paths)
    payload = _load_json_payload(resolved_path)
    return build_historical_source_timing_context(payload), resolved_path


def _build_source_timing_suffix(source_timing_context: dict[str, Any] | None) -> str:
    if not isinstance(source_timing_context, dict):
        return ""
    return (
        f"source_timing_status={source_timing_context.get('status')}, "
        f"result_ready_pre_race_rows={int(source_timing_context.get('result_ready_pre_race_rows') or 0)}, "
        f"future_only_pre_race_rows={int(source_timing_context.get('future_only_pre_race_rows') or 0)}. "
    )


def _build_source_timing_block(
    *,
    source_timing_context: dict[str, Any],
    source_timing_summary_path: Path | None,
    data_config_path: str,
    command_name: str,
    profile_name: str | None,
    override_flag: str,
) -> dict[str, Any] | None:
    status = str(source_timing_context.get("status") or "").strip()
    if status == "historical_pre_race_subset_available":
        return None

    recommended_action = str(source_timing_context.get("recommended_action") or "inspect_source_timing_audit")
    result_ready_pre_race_rows = int(source_timing_context.get("result_ready_pre_race_rows") or 0)
    future_only_pre_race_rows = int(source_timing_context.get("future_only_pre_race_rows") or 0)
    profile_fragment = f" profile={profile_name}" if profile_name else ""
    return {
        "error_code": "historical_local_nankan_source_timing_not_ready",
        "error_message": (
            f"{command_name} blocked for historical local Nankan{profile_fragment} data_config={data_config_path} because historical source timing is not ready: "
            f"status={status}, result_ready_pre_race_rows={result_ready_pre_race_rows}, future_only_pre_race_rows={future_only_pre_race_rows}, "
            f"recommended_action={recommended_action}. Historical local Nankan runs are diagnostic-only until #120/#121 are resolved. "
            f"Pass {override_flag} only for an intentional diagnostic run."
        ),
        "data_config_path": data_config_path,
        "recommended_action": recommended_action,
        "source_timing_summary_path": str(source_timing_summary_path) if source_timing_summary_path is not None else None,
        "historical_source_timing_status": status,
        "result_ready_pre_race_rows": result_ready_pre_race_rows,
        "future_only_pre_race_rows": future_only_pre_race_rows,
        "strict_trust_ready": False,
    }


def resolve_local_nankan_trust_block(
    *,
    workspace_root: Path,
    data_config: dict[str, Any] | None,
    data_config_path: str,
    allow_diagnostic_override: bool,
    command_name: str,
    profile_name: str | None = None,
    provenance_manifest_path: str | None = None,
    source_timing_summary_path: str | None = None,
    override_flag: str = "--allow-diagnostic-local-nankan",
) -> dict[str, Any] | None:
    if allow_diagnostic_override:
        return None
    if not _is_historical_local_nankan_dataset(data_config):
        return None

    provenance_fallbacks = ()
    resolved_primary_provenance = provenance_manifest_path or DEFAULT_LOCAL_NANKAN_PROVENANCE_MANIFEST
    if resolved_primary_provenance == DEFAULT_LOCAL_NANKAN_PROVENANCE_MANIFEST:
        provenance_fallbacks = (LEGACY_LOCAL_NANKAN_PROVENANCE_MANIFEST,)
    provenance_candidates = _resolve_candidate_paths(
        workspace_root=workspace_root,
        primary_path=resolved_primary_provenance,
        fallback_paths=provenance_fallbacks,
    )
    manifest_path = _first_existing_path(provenance_candidates)
    source_timing_context, resolved_source_timing_path = _resolve_source_timing_context(
        workspace_root=workspace_root,
        source_timing_summary_path=source_timing_summary_path,
    )

    recommended_action = "inspect_local_nankan_provenance_audit"
    profile_fragment = f" profile={profile_name}" if profile_name else ""
    if manifest_path is None:
        return {
            "error_code": "historical_local_nankan_trust_status_unknown",
            "error_message": (
                f"{command_name} blocked for historical local Nankan{profile_fragment} because strict provenance trust status is unknown: "
                f"missing any of {', '.join(str(path) for path in provenance_candidates)}. Resolve #120/#121 first, or pass {override_flag} for a diagnostic-only run."
            ),
            "data_config_path": data_config_path,
            "recommended_action": recommended_action,
            "provenance_manifest_path": None,
            "searched_provenance_manifest_paths": [str(path) for path in provenance_candidates],
            "pre_race": None,
            "post_race": None,
            "unknown": None,
            "strict_trust_ready": False,
        }

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    readiness = payload.get("readiness") if isinstance(payload.get("readiness"), dict) else {}
    if bool(readiness.get("strict_trust_ready", False)):
        source_timing_block = _build_source_timing_block(
            source_timing_context=source_timing_context,
            source_timing_summary_path=resolved_source_timing_path,
            data_config_path=data_config_path,
            command_name=command_name,
            profile_name=profile_name,
            override_flag=override_flag,
        ) if isinstance(source_timing_context, dict) else None
        if source_timing_block is None:
            return None
        source_timing_block["provenance_manifest_path"] = str(manifest_path)
        return source_timing_block

    provenance_summary = payload.get("provenance_summary") if isinstance(payload.get("provenance_summary"), dict) else {}
    bucket_counts = provenance_summary.get("bucket_counts") if isinstance(provenance_summary.get("bucket_counts"), dict) else {}
    pre_race = int(bucket_counts.get("pre_race") or 0)
    post_race = int(bucket_counts.get("post_race") or 0)
    unknown = int(bucket_counts.get("unknown") or 0)
    recommended_action = str(
        readiness.get("recommended_action")
        or payload.get("recommended_action")
        or recommended_action
    )
    if isinstance(source_timing_context, dict):
        recommended_action = str(source_timing_context.get("recommended_action") or recommended_action)
    source_timing_suffix = _build_source_timing_suffix(source_timing_context)
    return {
        "error_code": "historical_local_nankan_trust_not_ready",
        "error_message": (
            f"{command_name} blocked for historical local Nankan{profile_fragment} data_config={data_config_path} because strict provenance trust is not ready: "
            f"pre_race={pre_race}, post_race={post_race}, unknown={unknown}, recommended_action={recommended_action}. "
            f"{source_timing_suffix}"
            f"Historical local Nankan runs are diagnostic-only until #120/#121 are resolved. Pass {override_flag} only for an intentional diagnostic run."
        ),
        "data_config_path": data_config_path,
        "recommended_action": recommended_action,
        "provenance_manifest_path": str(manifest_path),
        "source_timing_summary_path": str(resolved_source_timing_path) if resolved_source_timing_path is not None else None,
        "historical_source_timing_status": source_timing_context.get("status") if isinstance(source_timing_context, dict) else None,
        "result_ready_pre_race_rows": int(source_timing_context.get("result_ready_pre_race_rows") or 0) if isinstance(source_timing_context, dict) else None,
        "future_only_pre_race_rows": int(source_timing_context.get("future_only_pre_race_rows") or 0) if isinstance(source_timing_context, dict) else None,
        "pre_race": pre_race,
        "post_race": post_race,
        "unknown": unknown,
        "strict_trust_ready": False,
    }


def require_local_nankan_trust_ready(
    *,
    workspace_root: Path,
    data_config: dict[str, Any] | None,
    data_config_path: str,
    allow_diagnostic_override: bool,
    command_name: str,
    profile_name: str | None = None,
    provenance_manifest_path: str | None = None,
    source_timing_summary_path: str | None = None,
    override_flag: str = "--allow-diagnostic-local-nankan",
) -> None:
    blocked = resolve_local_nankan_trust_block(
        workspace_root=workspace_root,
        data_config=data_config,
        data_config_path=data_config_path,
        allow_diagnostic_override=allow_diagnostic_override,
        command_name=command_name,
        profile_name=profile_name,
        provenance_manifest_path=provenance_manifest_path,
        source_timing_summary_path=source_timing_summary_path,
        override_flag=override_flag,
    )
    if blocked is not None:
        raise ValueError(str(blocked.get("error_message") or "historical local Nankan trust is not ready"))