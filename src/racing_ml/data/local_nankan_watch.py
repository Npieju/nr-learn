from __future__ import annotations

from typing import Any


def should_trigger_handoff(probe_summary: dict[str, Any] | None) -> bool:
    if not isinstance(probe_summary, dict):
        return False
    return str(probe_summary.get("status") or "").strip().lower() == "ready"


def build_readiness_watcher_manifest(
    *,
    status: str,
    current_phase: str,
    recommended_action: str,
    attempts: int,
    waited_seconds: int,
    timed_out: bool,
    probe_summary: dict[str, Any] | None,
    handoff_manifest: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "status": str(status),
        "current_phase": str(current_phase),
        "recommended_action": str(recommended_action),
        "attempts": int(attempts),
        "waited_seconds": int(waited_seconds),
        "timed_out": bool(timed_out),
        "probe_summary": probe_summary if isinstance(probe_summary, dict) else {},
    }
    if isinstance(handoff_manifest, dict):
        manifest["handoff_manifest"] = handoff_manifest
    return manifest
