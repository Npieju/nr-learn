import argparse
from pathlib import Path
import sys
import traceback

import joblib

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import build_model_manifest, build_training_report_payload, resolve_output_artifacts, write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.evaluation.policy import PolicyConstraints
from racing_ml.models.value_blend import build_value_blend_bundle, load_component_from_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/model_catboost_value_stack.yaml")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features_catboost_rich.yaml")
    args = parser.parse_args()

    try:
        config = load_yaml(ROOT / args.config)
        progress = ProgressBar(total=4, prefix="[value-stack]", min_interval_sec=0.0)
        progress.start("config loaded")
        components_cfg = config.get("components", {})
        output_cfg = config.get("output", {})
        output_artifacts = resolve_output_artifacts(output_cfg)
        blend_params = dict(config.get("model", {}).get("params", {}))

        win_config_path = components_cfg.get("win")
        alpha_config_path = components_cfg.get("alpha")
        roi_config_path = components_cfg.get("roi")
        time_config_path = components_cfg.get("time")
        if not win_config_path:
            raise ValueError("components.win is required for value stack build")

        with Heartbeat("[value-stack]", "loading component artifacts"):
            win_component = load_component_from_config(workspace_root=ROOT, config_path=win_config_path)
            alpha_component = load_component_from_config(workspace_root=ROOT, config_path=alpha_config_path) if alpha_config_path else None
            roi_component = load_component_from_config(workspace_root=ROOT, config_path=roi_config_path) if roi_config_path else None
            time_component = load_component_from_config(workspace_root=ROOT, config_path=time_config_path) if time_config_path else None
        progress.update(message="component artifacts loaded")

        model_bundle = build_value_blend_bundle(
            win_component=win_component,
            alpha_component=alpha_component,
            roi_component=roi_component,
            time_component=time_component,
            params=blend_params,
        )
        progress.update(message="blend bundle built")

        model_path = ROOT / output_artifacts.model_path
        report_path = ROOT / output_artifacts.report_path
        manifest_path = ROOT / output_artifacts.manifest_path
        model_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with Heartbeat("[value-stack]", "writing stack artifact"):
            joblib.dump(model_bundle, model_path)
        progress.update(message="stack artifact written")

        component_names = list(model_bundle.get("component_metadata", {}).keys())
        metrics = {
            "component_count": float(len(component_names)),
            "feature_count": float(len(model_bundle.get("feature_columns", []))),
            "categorical_feature_count": float(len(model_bundle.get("categorical_columns", []))),
            "alpha_weight": float(blend_params.get("alpha_weight", 0.0)),
            "roi_weight": float(blend_params.get("roi_weight", 0.0)),
            "time_weight": float(blend_params.get("time_weight", 0.0)),
            "market_blend_weight": float(blend_params.get("market_blend_weight", 1.0)),
        }
        run_context = {
            "model_config": str(args.config),
            "data_config": str(args.data_config),
            "feature_config": str(args.feature_config),
            "task": str(config.get("task", "classification")),
            "label_column": str(config.get("label", "is_win")),
            "model_name": "value_blend",
            "artifact_model": output_artifacts.model_path.as_posix(),
            "artifact_report": output_artifacts.report_path.as_posix(),
            "artifact_manifest": output_artifacts.manifest_path.as_posix(),
            "components": component_names,
        }
        policy_constraints = PolicyConstraints.from_config(config.get("evaluation", {})).to_dict()
        with Heartbeat("[value-stack]", "writing report and manifest"):
            report_payload = build_training_report_payload(
                metrics=metrics,
                run_context=run_context,
                leakage_audit=None,
                policy_constraints=policy_constraints,
                extra_metadata={
                    "blend_params": blend_params,
                    "components": model_bundle.get("component_metadata", {}),
                },
            )
            write_json(report_path, report_payload)

            manifest_payload = build_model_manifest(
                workspace_root=ROOT,
                model_config_path=args.config,
                data_config_path=args.data_config,
                feature_config_path=args.feature_config,
                model_path=output_artifacts.model_path,
                report_path=output_artifacts.report_path,
                task=str(config.get("task", "classification")),
                label_column=str(config.get("label", "is_win")),
                model_name="value_blend",
                used_features=list(model_bundle.get("feature_columns", [])),
                categorical_columns=list(model_bundle.get("categorical_columns", [])),
                metrics=metrics,
                run_context=run_context,
                leakage_audit=None,
                policy_constraints=policy_constraints,
                extra_metadata={
                    "blend_params": blend_params,
                    "components": model_bundle.get("component_metadata", {}),
                },
            )
            write_json(manifest_path, manifest_payload)
        progress.complete(message="stack artifacts written")

        print(f"[value-stack] model saved: {model_path}")
        print(f"[value-stack] report saved: {report_path}")
        print(f"[value-stack] manifest saved: {manifest_path}")
        print(f"[value-stack] components: {component_names}")
        print(f"[value-stack] params: {blend_params}")
        return 0
    except KeyboardInterrupt:
        print("[value-stack] interrupted by user")
        return 130
    except Exception as error:
        print(f"[value-stack] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())