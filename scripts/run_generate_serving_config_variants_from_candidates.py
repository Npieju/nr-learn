from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _normalize_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"expected YAML object: {path}")
    return payload


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _slugify(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_")


def _candidate_configs(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    candidates = payload.get("runtime_ready_candidates")
    if not isinstance(candidates, dict) or not candidates:
        raise ValueError("runtime_ready_candidates is missing or empty")
    normalized: dict[str, dict[str, Any]] = {}
    for candidate_name, candidate_payload in candidates.items():
        if not isinstance(candidate_payload, dict):
            continue
        policy = candidate_payload.get("policy")
        if not isinstance(policy, dict) or not policy:
            continue
        normalized[str(candidate_name)] = deepcopy(policy)
    if not normalized:
        raise ValueError("no runtime-ready candidate policies found")
    return normalized


def _kelly_overrides_only(serving_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    overrides = serving_cfg.get("policy_regime_overrides")
    if not isinstance(overrides, list):
        return []
    kept: list[dict[str, Any]] = []
    for override in overrides:
        if not isinstance(override, dict):
            continue
        policy = override.get("policy") if isinstance(override.get("policy"), dict) else override
        strategy_kind = str(policy.get("strategy_kind", "")).strip().lower()
        if strategy_kind == "kelly":
            kept.append(deepcopy(override))
    return kept


def _variant_config(base_config: dict[str, Any], policy: dict[str, Any], *, mode: str) -> dict[str, Any]:
    config = deepcopy(base_config)
    serving_cfg = config.get("serving") if isinstance(config.get("serving"), dict) else {}
    serving_cfg = deepcopy(serving_cfg)
    serving_cfg["policy"] = deepcopy(policy)
    if mode == "single_policy":
        serving_cfg.pop("policy_regime_overrides", None)
    elif mode == "hybrid_keep_kelly":
        kelly_overrides = _kelly_overrides_only(serving_cfg)
        if kelly_overrides:
            serving_cfg["policy_regime_overrides"] = kelly_overrides
        else:
            serving_cfg.pop("policy_regime_overrides", None)
    else:
        raise ValueError(f"unsupported mode: {mode}")
    config["serving"] = serving_cfg
    return config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml",
    )
    parser.add_argument(
        "--candidate-report",
        default="artifacts/reports/generated_serving_candidates_from_mitigation_probe_current_profiles_h1_vs_h2_2024.json",
    )
    parser.add_argument("--output-dir", default="configs")
    parser.add_argument(
        "--output-report",
        default="artifacts/reports/generated_serving_config_variants_from_candidates_current_profiles_h1_vs_h2_2024.json",
    )
    args = parser.parse_args()

    base_config_path = _normalize_path(args.base_config)
    candidate_report_path = _normalize_path(args.candidate_report)
    output_dir = _normalize_path(args.output_dir)
    output_report_path = _normalize_path(args.output_report)

    base_config = _load_yaml(base_config_path)
    payload = _load_json(candidate_report_path)
    candidates = _candidate_configs(payload)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_report_path.parent.mkdir(parents=True, exist_ok=True)

    base_stem = base_config_path.stem
    variant_modes = {
        "single_policy": "",
        "hybrid_keep_kelly": "_hybrid_keep_kelly",
    }
    generated_variants: list[dict[str, Any]] = []
    for candidate_name, policy in candidates.items():
        for mode, suffix in variant_modes.items():
            variant_name = f"{base_stem}_{_slugify(candidate_name)}{suffix}"
            variant_path = output_dir / f"{variant_name}.yaml"
            variant_config = _variant_config(base_config, policy, mode=mode)
            with variant_path.open("w", encoding="utf-8") as file:
                yaml.safe_dump(variant_config, file, sort_keys=False, allow_unicode=False)
            generated_variants.append(
                {
                    "candidate_name": candidate_name,
                    "mode": mode,
                    "config_file": str(variant_path.relative_to(ROOT)),
                    "policy_name": str(policy.get("name", "")),
                    "strategy_kind": str(policy.get("strategy_kind", "")),
                    "evidence_count": int(
                        ((payload.get("runtime_ready_candidates") or {}).get(candidate_name) or {}).get("evidence_count", 0)
                    ),
                }
            )

    report_payload = {
        "base_config": str(base_config_path.relative_to(ROOT)),
        "candidate_report": str(candidate_report_path.relative_to(ROOT)),
        "generated_variants": generated_variants,
        "notes": [
            "These variants keep the base model/components and replace serving.policy with a single runtime-ready portfolio candidate.",
            "single_policy variants remove policy_regime_overrides intentionally so each generated config is a directly loadable single-policy serving probe.",
            "hybrid_keep_kelly variants keep only the existing Kelly month overrides and swap the default portfolio policy for the candidate.",
        ],
    }
    with output_report_path.open("w", encoding="utf-8") as file:
        json.dump(report_payload, file, ensure_ascii=False, indent=2)

    print(f"saved variant report to {output_report_path.relative_to(ROOT)}")
    for variant in generated_variants:
        print(
            f"generated {variant['candidate_name']} mode={variant['mode']} -> {variant['config_file']} policy={variant['policy_name']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())