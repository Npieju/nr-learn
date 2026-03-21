from __future__ import annotations

from typing import Any

import pandas as pd

from racing_ml.common.regime import resolve_regime_override
from racing_ml.evaluation.policy import blend_prob, compute_market_prob


def _initialize_policy_columns(annotated: pd.DataFrame, *, policy_name: str, strategy_kind: str) -> pd.DataFrame:
    annotated["policy_name"] = str(policy_name)
    annotated["policy_strategy_kind"] = strategy_kind
    annotated["policy_blend_weight"] = pd.NA
    annotated["policy_min_prob"] = pd.NA
    annotated["policy_odds_min"] = pd.NA
    annotated["policy_odds_max"] = pd.NA
    annotated["policy_min_edge"] = pd.NA
    annotated["policy_fractional_kelly"] = pd.NA
    annotated["policy_max_fraction"] = pd.NA
    annotated["policy_top_k"] = pd.NA
    annotated["policy_min_expected_value"] = pd.NA
    annotated["policy_market_prob"] = pd.NA
    annotated["policy_prob"] = pd.NA
    annotated["policy_expected_value"] = pd.NA
    annotated["policy_edge"] = pd.NA
    annotated["policy_selected"] = False
    annotated["policy_selection_rank"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["policy_weight"] = 0.0
    annotated["policy_stage_name"] = pd.NA
    annotated["policy_stage_index"] = pd.Series(pd.NA, index=annotated.index, dtype="Int64")
    annotated["policy_selected_strategy_kind"] = pd.NA
    return annotated


def _annotate_single_runtime_policy(
    frame: pd.DataFrame,
    *,
    odds_col: str | None,
    policy_name: str,
    policy_config: dict[str, Any],
    score_col: str,
) -> pd.DataFrame:
    annotated = frame.copy()
    strategy_kind = str(policy_config.get("strategy_kind", "")).strip().lower()
    blend_weight = float(policy_config.get("blend_weight", 1.0))
    min_prob = float(policy_config.get("min_prob", 0.05))
    odds_min = float(policy_config.get("odds_min", 1.0))
    odds_max = float(policy_config.get("odds_max", 999.0))

    annotated = _initialize_policy_columns(annotated, policy_name=policy_name, strategy_kind=strategy_kind)
    annotated["policy_blend_weight"] = blend_weight
    annotated["policy_min_prob"] = min_prob
    annotated["policy_odds_min"] = odds_min
    annotated["policy_odds_max"] = odds_max

    if odds_col is None or odds_col not in annotated.columns or "race_id" not in annotated.columns:
        return annotated

    annotated[score_col] = pd.to_numeric(annotated[score_col], errors="coerce")
    annotated[odds_col] = pd.to_numeric(annotated[odds_col], errors="coerce")
    annotated["policy_market_prob"] = compute_market_prob(annotated, odds_col=odds_col)
    annotated["policy_prob"] = blend_prob(
        annotated[score_col],
        annotated["policy_market_prob"],
        weight=blend_weight,
    )
    annotated["policy_expected_value"] = annotated["policy_prob"] * annotated[odds_col]
    annotated["policy_edge"] = annotated["policy_expected_value"] - 1.0

    if strategy_kind == "kelly":
        min_edge = float(policy_config.get("min_edge", 0.03))
        fractional_kelly = float(policy_config.get("fractional_kelly", 0.5))
        max_fraction = float(policy_config.get("max_fraction", 0.05))
        annotated["policy_min_edge"] = min_edge
        annotated["policy_fractional_kelly"] = fractional_kelly
        annotated["policy_max_fraction"] = max_fraction

        for _, group in annotated.groupby("race_id", sort=False):
            eligible = group[
                group["policy_prob"].notna()
                & group[odds_col].notna()
                & (group[odds_col] > odds_min)
                & (group[odds_col] <= odds_max)
                & (group["policy_prob"] >= min_prob)
            ].copy()
            if eligible.empty:
                continue

            b = eligible[odds_col] - 1.0
            eligible["policy_raw_kelly"] = ((b * eligible["policy_prob"]) - (1.0 - eligible["policy_prob"])) / b.where(b != 0)
            eligible = eligible[(eligible["policy_edge"] >= min_edge) & (eligible["policy_raw_kelly"] > 0.0)]
            if eligible.empty:
                continue

            pick = eligible.sort_values(["policy_edge", "policy_prob"], ascending=False).iloc[0]
            pick_index = pick.name
            annotated.loc[pick_index, "policy_selected"] = True
            annotated.loc[pick_index, "policy_selection_rank"] = 1
            annotated.loc[pick_index, "policy_weight"] = min(
                max(float(pick["policy_raw_kelly"]) * fractional_kelly, 0.0),
                max_fraction,
            )
            annotated.loc[pick_index, "policy_selected_strategy_kind"] = strategy_kind
    elif strategy_kind == "portfolio":
        top_k = int(policy_config.get("top_k", 1))
        min_expected_value = float(policy_config.get("min_expected_value", 1.0))
        annotated["policy_top_k"] = top_k
        annotated["policy_min_expected_value"] = min_expected_value

        for _, group in annotated.groupby("race_id", sort=False):
            eligible = group[
                group["policy_prob"].notna()
                & group[odds_col].notna()
                & (group[odds_col] > odds_min)
                & (group[odds_col] <= odds_max)
                & (group["policy_prob"] >= min_prob)
                & (group["policy_expected_value"] >= min_expected_value)
            ].copy()
            if eligible.empty:
                continue

            picks = eligible.sort_values(["policy_expected_value", "policy_prob"], ascending=False).head(max(top_k, 1))
            if picks.empty:
                continue

            pick_weight = 1.0 / len(picks)
            for pick_rank, (pick_index, _) in enumerate(picks.iterrows(), start=1):
                annotated.loc[pick_index, "policy_selected"] = True
                annotated.loc[pick_index, "policy_selection_rank"] = pick_rank
                annotated.loc[pick_index, "policy_weight"] = pick_weight
                annotated.loc[pick_index, "policy_selected_strategy_kind"] = strategy_kind

    return annotated


def _annotate_staged_runtime_policy(
    frame: pd.DataFrame,
    *,
    odds_col: str | None,
    policy_name: str,
    policy_config: dict[str, Any],
    score_col: str,
) -> pd.DataFrame:
    annotated = frame.copy()
    annotated = _initialize_policy_columns(annotated, policy_name=policy_name, strategy_kind="staged")

    stages = policy_config.get("stages")
    if not isinstance(stages, list) or not stages:
        return annotated

    stage_results: list[tuple[int, str, dict[str, Any], dict[str, Any], pd.DataFrame]] = []
    for stage_index, stage_cfg in enumerate(stages, start=1):
        if not isinstance(stage_cfg, dict):
            continue
        stage_name = str(stage_cfg.get("name", f"stage_{stage_index}")).strip() or f"stage_{stage_index}"
        stage_policy = stage_cfg.get("policy") if isinstance(stage_cfg.get("policy"), dict) else stage_cfg
        if not isinstance(stage_policy, dict):
            continue
        stage_result = _annotate_single_runtime_policy(
            frame,
            odds_col=odds_col,
            policy_name=stage_name,
            policy_config=stage_policy,
            score_col=score_col,
        )
        stage_results.append((stage_index, stage_name, stage_cfg, stage_policy, stage_result))

    if not stage_results or odds_col is None or "race_id" not in annotated.columns:
        return annotated

    policy_columns = [
        "policy_blend_weight",
        "policy_min_prob",
        "policy_odds_min",
        "policy_odds_max",
        "policy_min_edge",
        "policy_fractional_kelly",
        "policy_max_fraction",
        "policy_top_k",
        "policy_min_expected_value",
        "policy_market_prob",
        "policy_prob",
        "policy_expected_value",
        "policy_edge",
        "policy_selected",
        "policy_selection_rank",
        "policy_weight",
        "policy_selected_strategy_kind",
    ]

    def should_fallback(stage_race: pd.DataFrame, stage_cfg: dict[str, Any]) -> bool:
        fallback_when = stage_cfg.get("fallback_when")
        if not isinstance(fallback_when, dict) or not fallback_when:
            return False

        selected = stage_race[stage_race["policy_selected"].fillna(False).astype(bool)]
        if selected.empty:
            return False

        selected_count = int(len(selected))
        max_expected_value = pd.to_numeric(selected["policy_expected_value"], errors="coerce").max()
        max_prob = pd.to_numeric(selected["policy_prob"], errors="coerce").max()
        max_edge = pd.to_numeric(selected["policy_edge"], errors="coerce").max()

        selected_rows_at_most = fallback_when.get("selected_rows_at_most")
        if selected_rows_at_most is not None and selected_count <= int(selected_rows_at_most):
            return True

        max_expected_value_below = fallback_when.get("max_expected_value_below")
        if max_expected_value_below is not None and pd.notna(max_expected_value) and float(max_expected_value) < float(max_expected_value_below):
            return True

        max_prob_below = fallback_when.get("max_prob_below")
        if max_prob_below is not None and pd.notna(max_prob) and float(max_prob) < float(max_prob_below):
            return True

        max_edge_below = fallback_when.get("max_edge_below")
        if max_edge_below is not None and pd.notna(max_edge) and float(max_edge) < float(max_edge_below):
            return True

        return False

    for race_id, race_group in annotated.groupby("race_id", sort=False):
        race_index = race_group.index
        selected_stage: tuple[int, str, dict[str, Any], dict[str, Any], pd.DataFrame] | None = None
        for stage_index, stage_name, stage_cfg, stage_policy, stage_result in stage_results:
            stage_race = stage_result.loc[race_index]
            if stage_race["policy_selected"].fillna(False).astype(bool).any() and not should_fallback(stage_race, stage_cfg):
                selected_stage = (stage_index, stage_name, stage_cfg, stage_policy, stage_result)
                break

        if selected_stage is None:
            continue

        stage_index, stage_name, _, stage_policy, stage_result = selected_stage
        stage_race = stage_result.loc[race_index]
        for column in policy_columns:
            annotated.loc[race_index, column] = stage_race[column]
        annotated.loc[race_index, "policy_stage_name"] = stage_name
        annotated.loc[race_index, "policy_stage_index"] = stage_index
        annotated.loc[race_index, "policy_strategy_kind"] = "staged"
        selected_mask = stage_race["policy_selected"].fillna(False).astype(bool)
        if selected_mask.any():
            selected_index = stage_race.index[selected_mask]
            annotated.loc[selected_index, "policy_selected_strategy_kind"] = str(stage_policy.get("strategy_kind", "")).strip().lower()

    return annotated


def _merge_runtime_policy(
    base_policy: dict[str, Any] | None,
    override: dict[str, Any] | None,
) -> tuple[str, dict[str, Any]] | None:
    policy: dict[str, Any] = dict(base_policy) if isinstance(base_policy, dict) else {}
    policy_name = str(policy.get("name", "default")).strip() or "default"

    if isinstance(override, dict):
        override_name = str(override.get("name", "")).strip()
        if override_name:
            policy_name = override_name

        override_policy = override.get("policy")
        if isinstance(override_policy, dict):
            policy.update(override_policy)
        else:
            for key, value in override.items():
                if key in {"name", "when"}:
                    continue
                policy[key] = value

    strategy_kind = str(policy.get("strategy_kind", "")).strip().lower()
    if not strategy_kind:
        return None

    policy["strategy_kind"] = strategy_kind
    return policy_name, policy


def resolve_runtime_policy(
    model_config: dict[str, Any],
    *,
    frame: pd.DataFrame,
    date_col: str = "date",
) -> tuple[str, dict[str, Any]] | None:
    serving_cfg = model_config.get("serving", {})
    if not isinstance(serving_cfg, dict):
        return None

    base_policy = serving_cfg.get("policy")
    overrides = serving_cfg.get("policy_regime_overrides", [])
    override = resolve_regime_override(overrides, frame=frame, date_col=date_col)
    return _merge_runtime_policy(base_policy if isinstance(base_policy, dict) else None, override)


def annotate_runtime_policy(
    frame: pd.DataFrame,
    *,
    odds_col: str | None,
    policy_name: str,
    policy_config: dict[str, Any],
    score_col: str = "score",
) -> pd.DataFrame:
    strategy_kind = str(policy_config.get("strategy_kind", "")).strip().lower()
    if strategy_kind == "staged":
        return _annotate_staged_runtime_policy(
            frame,
            odds_col=odds_col,
            policy_name=policy_name,
            policy_config=policy_config,
            score_col=score_col,
        )
    return _annotate_single_runtime_policy(
        frame,
        odds_col=odds_col,
        policy_name=policy_name,
        policy_config=policy_config,
        score_col=score_col,
    )