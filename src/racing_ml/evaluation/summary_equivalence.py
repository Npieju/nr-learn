from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
import json

DEFAULT_IGNORED_PATHS = {
    "output_files",
    "run_context.artifact_suffix",
    "run_context.model_artifact_suffix",
    "run_context.artifact_manifest",
}


def _prune_paths(value: Any, path: str = "", ignored_paths: set[str] | None = None) -> Any:
    ignored = ignored_paths or set()
    if path in ignored:
        return None

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, child in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            if child_path in ignored:
                continue
            output[str(key)] = _prune_paths(child, child_path, ignored)
        return output

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_prune_paths(child, f"{path}[]", ignored) for child in value]

    return value


def _collect_differences(left: Any, right: Any, path: str = "") -> list[dict[str, Any]]:
    if type(left) is not type(right):
        return [{"path": path or "<root>", "left": repr(left), "right": repr(right), "kind": "type"}]

    if isinstance(left, Mapping):
        diffs: list[dict[str, Any]] = []
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        for key in sorted(left_keys - right_keys):
            diffs.append({"path": f"{path}.{key}" if path else str(key), "left": repr(left[key]), "right": "<missing>", "kind": "left_only"})
        for key in sorted(right_keys - left_keys):
            diffs.append({"path": f"{path}.{key}" if path else str(key), "left": "<missing>", "right": repr(right[key]), "kind": "right_only"})
        for key in sorted(left_keys & right_keys):
            child_path = f"{path}.{key}" if path else str(key)
            diffs.extend(_collect_differences(left[key], right[key], child_path))
        return diffs

    if isinstance(left, Sequence) and not isinstance(left, (str, bytes, bytearray)):
        if len(left) != len(right):
            return [{"path": path or "<root>", "left": f"len={len(left)}", "right": f"len={len(right)}", "kind": "length"}]
        diffs: list[dict[str, Any]] = []
        for index, (left_item, right_item) in enumerate(zip(left, right)):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            diffs.extend(_collect_differences(left_item, right_item, child_path))
        return diffs

    if left != right:
        return [{"path": path or "<root>", "left": repr(left), "right": repr(right), "kind": "value"}]
    return []


def compare_summary_files(
    *,
    left_summary: Path,
    right_summary: Path,
    ignored_paths: set[str] | None = None,
    sample_limit: int = 20,
) -> dict[str, Any]:
    left_obj = json.loads(left_summary.read_text())
    right_obj = json.loads(right_summary.read_text())
    normalized_left = _prune_paths(left_obj, ignored_paths=ignored_paths)
    normalized_right = _prune_paths(right_obj, ignored_paths=ignored_paths)
    diffs = _collect_differences(normalized_left, normalized_right)
    return {
        "left_summary": str(left_summary),
        "right_summary": str(right_summary),
        "ignored_paths": sorted(ignored_paths or set()),
        "exact_equal": len(diffs) == 0,
        "difference_count": int(len(diffs)),
        "difference_samples": diffs[:sample_limit],
    }
