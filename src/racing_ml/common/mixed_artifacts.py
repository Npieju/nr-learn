from __future__ import annotations

from pathlib import Path

from racing_ml.common.artifacts import absolutize_path, read_json


def latest_matching_path(*, workspace_root: Path, pattern: str) -> Path | None:
    matches = sorted(workspace_root.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def prefer_existing_path(*, workspace_root: Path, expected_path: str | Path, fallback_pattern: str) -> Path:
    path = absolutize_path(expected_path, workspace_root)
    if path.exists():
        return path
    fallback = latest_matching_path(workspace_root=workspace_root, pattern=fallback_pattern)
    return fallback if fallback is not None else path


def read_optional_json_path(path: str | Path, *, workspace_root: Path) -> dict[str, object] | None:
    resolved = absolutize_path(path, workspace_root)
    if not resolved.exists():
        return None
    payload = read_json(resolved)
    return payload if isinstance(payload, dict) else None


def revision_prefix_candidates(revision_slug: str) -> list[str]:
    candidates: list[str] = []
    if revision_slug:
        candidates.append(revision_slug)
    head = revision_slug.split("_", 1)[0].strip()
    if head and head not in candidates:
        candidates.append(head)
    return candidates


def resolve_local_snapshot_and_lineage_paths(
    *,
    workspace_root: Path,
    revision_slug: str,
    left_universe: str,
    explicit_snapshot: str | None = None,
    explicit_lineage: str | None = None,
) -> tuple[Path, Path]:
    if explicit_snapshot or explicit_lineage:
        snapshot_path = absolutize_path(explicit_snapshot or f"artifacts/reports/local_public_snapshot_{revision_slug}.json", workspace_root)
        lineage_path = absolutize_path(explicit_lineage or f"artifacts/reports/local_revision_gate_{revision_slug}.json", workspace_root)
        return snapshot_path, lineage_path

    snapshot_candidates: list[Path] = []
    lineage_candidates: list[Path] = []
    seen_snapshots: set[Path] = set()
    seen_lineages: set[Path] = set()

    def add_snapshot(path: Path | None) -> None:
        if path is None:
            return
        resolved = absolutize_path(path, workspace_root)
        if resolved in seen_snapshots:
            return
        seen_snapshots.add(resolved)
        snapshot_candidates.append(resolved)

    def add_lineage(path: Path | None) -> None:
        if path is None:
            return
        resolved = absolutize_path(path, workspace_root)
        if resolved in seen_lineages:
            return
        seen_lineages.add(resolved)
        lineage_candidates.append(resolved)

    for prefix in revision_prefix_candidates(revision_slug):
        add_snapshot(workspace_root / f"artifacts/reports/local_public_snapshot_{prefix}.json")
        add_lineage(workspace_root / f"artifacts/reports/local_revision_gate_{prefix}.json")
        add_snapshot(latest_matching_path(workspace_root=workspace_root, pattern=f"artifacts/reports/local_public_snapshot_{prefix}_*.json"))
        add_lineage(latest_matching_path(workspace_root=workspace_root, pattern=f"artifacts/reports/local_revision_gate_{prefix}_*.json"))

    add_snapshot(latest_matching_path(workspace_root=workspace_root, pattern=f"artifacts/reports/local_public_snapshot_*_{left_universe}_*.json"))
    add_lineage(latest_matching_path(workspace_root=workspace_root, pattern=f"artifacts/reports/local_revision_gate_*_{left_universe}_*.json"))
    add_snapshot(latest_matching_path(workspace_root=workspace_root, pattern="artifacts/reports/local_public_snapshot_*.json"))
    add_lineage(latest_matching_path(workspace_root=workspace_root, pattern="artifacts/reports/local_revision_gate_*.json"))

    snapshot_path = next(
        (candidate for candidate in snapshot_candidates if candidate.exists()),
        snapshot_candidates[0] if snapshot_candidates else workspace_root / f"artifacts/reports/local_public_snapshot_{revision_slug}.json",
    )
    lineage_path = next(
        (candidate for candidate in lineage_candidates if candidate.exists()),
        lineage_candidates[0] if lineage_candidates else workspace_root / f"artifacts/reports/local_revision_gate_{revision_slug}.json",
    )

    if snapshot_path.exists():
        snapshot_payload = read_optional_json_path(snapshot_path, workspace_root=workspace_root)
        if isinstance(snapshot_payload, dict):
            lineage_manifest = snapshot_payload.get("lineage_manifest")
            if isinstance(lineage_manifest, str) and lineage_manifest.strip():
                lineage_path = absolutize_path(lineage_manifest, workspace_root)

    return snapshot_path, lineage_path


def resolve_local_lineage_path(
    *,
    workspace_root: Path,
    revision_slug: str,
    left_universe: str,
    numeric_compare_path: str | Path,
    explicit_manifest: str | None = None,
) -> Path:
    if explicit_manifest:
        return absolutize_path(explicit_manifest, workspace_root)

    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_candidate(path: Path | None) -> None:
        if path is None:
            return
        resolved = absolutize_path(path, workspace_root)
        if resolved in seen:
            return
        seen.add(resolved)
        candidates.append(resolved)

    for prefix in revision_prefix_candidates(revision_slug):
        add_candidate(workspace_root / f"artifacts/reports/local_revision_gate_{prefix}.json")

    numeric_compare_payload = read_optional_json_path(numeric_compare_path, workspace_root=workspace_root)
    if isinstance(numeric_compare_payload, dict):
        artifacts = numeric_compare_payload.get("artifacts") if isinstance(numeric_compare_payload.get("artifacts"), dict) else {}
        readiness_manifest = artifacts.get("readiness_manifest") if isinstance(artifacts, dict) else None
        if isinstance(readiness_manifest, str) and readiness_manifest.strip():
            readiness_payload = read_optional_json_path(readiness_manifest, workspace_root=workspace_root)
            if isinstance(readiness_payload, dict):
                readiness_artifacts = readiness_payload.get("artifacts") if isinstance(readiness_payload.get("artifacts"), dict) else {}
                left_lineage_manifest = readiness_artifacts.get("left_lineage_manifest") if isinstance(readiness_artifacts, dict) else None
                if isinstance(left_lineage_manifest, str) and left_lineage_manifest.strip():
                    add_candidate(absolutize_path(left_lineage_manifest, workspace_root))
                left_public_snapshot = readiness_artifacts.get("left_public_snapshot") if isinstance(readiness_artifacts, dict) else None
                if isinstance(left_public_snapshot, str) and left_public_snapshot.strip():
                    public_snapshot_payload = read_optional_json_path(left_public_snapshot, workspace_root=workspace_root)
                    lineage_manifest = public_snapshot_payload.get("lineage_manifest") if isinstance(public_snapshot_payload, dict) else None
                    if isinstance(lineage_manifest, str) and lineage_manifest.strip():
                        add_candidate(absolutize_path(lineage_manifest, workspace_root))

    for prefix in revision_prefix_candidates(revision_slug):
        add_candidate(latest_matching_path(workspace_root=workspace_root, pattern=f"artifacts/reports/local_revision_gate_{prefix}_*.json"))
    add_candidate(latest_matching_path(workspace_root=workspace_root, pattern=f"artifacts/reports/local_revision_gate_*_{left_universe}_*.json"))
    add_candidate(latest_matching_path(workspace_root=workspace_root, pattern="artifacts/reports/local_revision_gate_*.json"))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0] if candidates else workspace_root / f"artifacts/reports/local_revision_gate_{revision_slug}.json"