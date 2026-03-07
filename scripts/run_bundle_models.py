import argparse
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.pipeline.bundle_pipeline import create_model_bundle


def _parse_components(raw_components: list[str]) -> dict[str, str]:
    components: dict[str, str] = {}
    for raw in raw_components:
        if "=" not in raw:
            raise ValueError(f"Invalid component format: {raw}. Expected name=config_path")
        name, config_path = raw.split("=", 1)
        component_name = name.strip()
        component_config = config_path.strip()
        if not component_name or not component_config:
            raise ValueError(f"Invalid component format: {raw}. Expected name=config_path")
        components[component_name] = component_config
    return components


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-name", default="policy_stack_v1")
    parser.add_argument("--bundle-kind", default="stack_v1")
    parser.add_argument("--primary-component", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument(
        "--component",
        action="append",
        default=[],
        help="Bundle component in the form name=config_path. Repeat for multiple components.",
    )
    args = parser.parse_args()

    try:
        components = _parse_components(args.component)
        result = create_model_bundle(
            bundle_name=args.bundle_name,
            bundle_kind=args.bundle_kind,
            primary_component=args.primary_component,
            component_configs=components,
            output_path=args.output_path,
            workspace_root=ROOT,
        )
        print(f"[bundle] bundle saved: {result.bundle_path}")
        print(f"[bundle] primary component: {result.bundle_payload['primary_component']}")
        print(f"[bundle] components: {list(result.bundle_payload['components'].keys())}")
        return 0
    except KeyboardInterrupt:
        print("[bundle] interrupted by user")
        return 130
    except Exception as error:
        print(f"[bundle] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())