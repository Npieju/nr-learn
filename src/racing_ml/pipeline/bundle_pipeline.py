from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from racing_ml.common.artifacts import build_bundle_manifest, resolve_component_from_config, write_json


@dataclass(frozen=True)
class BundleResult:
    bundle_path: Path
    bundle_payload: dict


def create_model_bundle(
    *,
    bundle_name: str,
    component_configs: dict[str, str],
    primary_component: str | None = None,
    bundle_kind: str = "stack_v1",
    output_path: str | None = None,
    workspace_root: str | Path | None = None,
) -> BundleResult:
    root = Path(workspace_root or Path.cwd()).resolve()
    if not component_configs:
        raise ValueError("At least one component config is required")

    resolved_components = {
        name: resolve_component_from_config(
            workspace_root=root,
            component_name=name,
            config_path=config_path,
        )
        for name, config_path in component_configs.items()
    }

    primary = str(primary_component or next(iter(resolved_components)))
    if primary not in resolved_components:
        raise ValueError(f"Primary component '{primary}' is not present in bundle components")

    bundle_payload = build_bundle_manifest(
        bundle_name=bundle_name,
        bundle_kind=bundle_kind,
        primary_component=primary,
        components=resolved_components,
    )

    if output_path is None:
        bundle_path = root / "artifacts/models" / f"{bundle_name}.bundle.json"
    else:
        bundle_path = Path(output_path)
        if not bundle_path.is_absolute():
            bundle_path = root / bundle_path

    write_json(bundle_path, bundle_payload)
    return BundleResult(bundle_path=bundle_path, bundle_payload=bundle_payload)