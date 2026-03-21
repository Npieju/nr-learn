from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import read_json, write_json


def _normalize_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path


def _load_json(path: Path) -> dict[str, Any]:
    payload = read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _parse_signature(signature: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not signature:
        return result
    for part in str(signature).split("|"):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        if value == "None":
            result[key] = None
            continue
        try:
            result[key] = float(value)
            continue
        except ValueError:
            result[key] = value
    return result


def _to_runtime_policy(name: str, signature: str) -> dict[str, Any]:
    parsed = _parse_signature(signature)
    policy: dict[str, Any] = {
        "name": name,
        "strategy_kind": parsed.get("strategy_kind"),
        "blend_weight": parsed.get("blend_weight"),
        "min_prob": parsed.get("min_prob"),
        "odds_min": parsed.get("odds_min"),
        "odds_max": parsed.get("odds_max"),
    }
    if parsed.get("strategy_kind") == "portfolio":
        policy["top_k"] = int(parsed.get("top_k")) if parsed.get("top_k") is not None else 1
        policy["min_expected_value"] = parsed.get("min_expected_value")
    if parsed.get("strategy_kind") == "kelly":
        policy["min_edge"] = parsed.get("min_edge")
        policy["fractional_kelly"] = parsed.get("fractional_kelly")
        policy["max_fraction"] = parsed.get("max_fraction")
    return {key: value for key, value in policy.items() if value is not None}


def _require_dominant_signature(selected_by_stage: dict[str, Counter[str]], stage: str) -> str:
    counter = selected_by_stage.get(stage, Counter())
    if not counter:
        raise ValueError(f"missing selected_signature for required stage: {stage}")
    return counter.most_common(1)[0][0]


def _build_candidates(policy_probe: dict[str, Any]) -> dict[str, Any]:
    rows = policy_probe.get("occurrences") if isinstance(policy_probe.get("occurrences"), list) else []
    stage_counts = Counter(str(row.get("stage") or "unknown") for row in rows if isinstance(row, dict))
    selected_by_stage: dict[str, Counter[str]] = {}
    for stage in stage_counts:
        selected_by_stage[stage] = Counter(
            str(row.get("selected_signature") or "")
            for row in rows
            if isinstance(row, dict) and str(row.get("stage") or "") == stage and str(row.get("selected_signature") or "")
        )

    portfolio_ev_signature = _require_dominant_signature(selected_by_stage, "portfolio_ev_only")
    portfolio_lower_blend_signature = _require_dominant_signature(selected_by_stage, "portfolio_lower_blend")
    kelly_signatures = [signature for signature, _ in selected_by_stage.get("kelly_fallback", Counter()).most_common()]

    runtime_candidates = {
        "portfolio_ev_only": {
            "policy": _to_runtime_policy("runtime_portfolio_ev_only_probe", portfolio_ev_signature),
            "evidence_count": int(stage_counts.get("portfolio_ev_only", 0)),
        },
        "portfolio_lower_blend": {
            "policy": _to_runtime_policy("runtime_portfolio_lower_blend_probe", portfolio_lower_blend_signature),
            "evidence_count": int(stage_counts.get("portfolio_lower_blend", 0)),
        },
    }
    staged_hybrid_spec = {
        "default_policy": _to_runtime_policy("runtime_portfolio_ev_only_probe", portfolio_ev_signature),
        "escalation_policy": _to_runtime_policy("runtime_portfolio_lower_blend_probe", portfolio_lower_blend_signature),
        "kelly_fallback_policies": [
            _to_runtime_policy(f"runtime_kelly_fallback_{index + 1}", signature)
            for index, signature in enumerate(kelly_signatures)
        ],
        "evidence": {
            "stage_counts": dict(stage_counts),
            "kelly_signature_count": int(len(kelly_signatures)),
        },
    }
    limitations = [
        "Current runtime policy overrides are window/date based only; they cannot express fold- or bankroll-state-dependent escalation.",
        "The staged hybrid spec is evidence-backed but not directly runtime-loadable without extending selector logic beyond single policy/date override resolution.",
    ]
    return {
        "target_signature": policy_probe.get("target_signature"),
        "runtime_ready_candidates": runtime_candidates,
        "staged_hybrid_spec": staged_hybrid_spec,
        "limitations": limitations,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy-probe", default="artifacts/reports/wf_threshold_mitigation_policy_probe_current_profiles_h1_vs_h2_2024.json")
    parser.add_argument("--output-json", default="artifacts/reports/generated_serving_candidates_from_mitigation_probe.json")
    parser.add_argument("--output-yaml", default="artifacts/reports/generated_serving_candidates_from_mitigation_probe.yaml")
    args = parser.parse_args()

    policy_probe = _load_json(_normalize_path(args.policy_probe))
    payload = _build_candidates(policy_probe)

    output_json = _normalize_path(args.output_json)
    output_yaml = _normalize_path(args.output_yaml)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    write_json(output_json, payload)
    with output_yaml.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False, allow_unicode=False)

    print(f"saved serving candidate json to {output_json.relative_to(ROOT)}")
    print(f"saved serving candidate yaml to {output_yaml.relative_to(ROOT)}")
    print(f"runtime_ready_candidates={list(payload['runtime_ready_candidates'].keys())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())