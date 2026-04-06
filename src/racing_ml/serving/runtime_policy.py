from __future__ import annotations

from typing import Any

import pandas as pd

from racing_ml.common.regime import resolve_regime_override
from racing_ml.evaluation.policy import blend_prob, compute_market_prob


def _append_reject_reason(reject_reasons: dict[Any, list[str]], indexer: pd.Index, reason: str) -> None:
    for index in indexer:
        reject_reasons[index].append(reason)


def _finalize_reject_reason_columns(
    annotated: pd.DataFrame,
    reject_reasons: dict[Any, list[str]],
) -> pd.DataFrame:
    primary_values: list[object] = []
    all_values: list[object] = []
    for index in annotated.index:
        reasons = reject_reasons.get(index, [])
        if reasons:
            primary_values.append(reasons[0])
            all_values.append("|".join(reasons))
        else:
            primary_values.append(pd.NA)
            all_values.append(pd.NA)
    annotated["policy_reject_reason_primary"] = pd.Series(primary_values, index=annotated.index, dtype="object")
    annotated["policy_reject_reasons"] = pd.Series(all_values, index=annotated.index, dtype="object")
    return annotated


def _split_reason_tokens(value: object) -> list[str]:
    if value is None or pd.isna(value):
        return []
    return [token for token in str(value).split("|") if token]


def summarize_policy_diagnostics(frame: pd.DataFrame) -> dict[str, Any]:
    selected_mask = frame["policy_selected"].fillna(False).astype(bool) if "policy_selected" in frame.columns else pd.Series(False, index=frame.index)
    diagnostics_available = any(
        column in frame.columns
        for column in ["policy_reject_reason_primary", "policy_reject_reasons", "policy_stage_fallback_reasons"]
    )
    result: dict[str, Any] = {
        "available": diagnostics_available,
        "policy_selected_rows": int(selected_mask.sum()),
        "policy_selected_races": int(frame.loc[selected_mask, "race_id"].nunique()) if "race_id" in frame.columns and selected_mask.any() else 0,
        "rejected_rows": int((~selected_mask).sum()),
        "unselected_rows_without_reason": 0,
        "likely_blocker_reason": None,
        "primary_reject_reason_counts": {},
        "primary_reject_reason_race_counts": {},
        "primary_reject_reason_examples": {},
        "stage_fallback_reason_counts": {},
        "race_diagnostics": [],
    }
    if frame.empty:
        return result

    unselected = frame.loc[~selected_mask].copy()
    primary_col = (
        unselected["policy_reject_reason_primary"].astype("string")
        if "policy_reject_reason_primary" in unselected.columns
        else pd.Series(pd.NA, index=unselected.index, dtype="string")
    )
    primary_counts_series = primary_col.dropna().value_counts()
    result["primary_reject_reason_counts"] = {str(key): int(value) for key, value in primary_counts_series.to_dict().items()}
    result["unselected_rows_without_reason"] = int((~primary_col.notna()).sum())
    if "race_id" in unselected.columns and not primary_counts_series.empty:
        reason_frame = unselected.loc[primary_col.notna(), ["race_id"]].copy()
        reason_frame["policy_reject_reason_primary"] = primary_col.loc[primary_col.notna()].astype(str)
        result["primary_reject_reason_race_counts"] = {
            str(key): int(value)
            for key, value in reason_frame.groupby("policy_reject_reason_primary")["race_id"].nunique().to_dict().items()
        }

    if not primary_counts_series.empty:
        result["likely_blocker_reason"] = str(primary_counts_series.index[0])
        for reason in primary_counts_series.index[:5]:
            reason_rows = unselected.loc[primary_col == str(reason)].head(3)
            examples: list[dict[str, Any]] = []
            for _, row in reason_rows.iterrows():
                pred_rank = pd.to_numeric(row.get("pred_rank"), errors="coerce")
                expected_value = pd.to_numeric(row.get("expected_value"), errors="coerce")
                examples.append(
                    {
                        "race_id": None if pd.isna(row.get("race_id")) else str(row.get("race_id")),
                        "headline": None if pd.isna(row.get("headline")) else str(row.get("headline")),
                        "horse_name": None if pd.isna(row.get("horse_name")) else str(row.get("horse_name")),
                        "pred_rank": None if pd.isna(pred_rank) else int(pred_rank),
                        "expected_value": None if pd.isna(expected_value) else float(expected_value),
                    }
                )
            result["primary_reject_reason_examples"][str(reason)] = examples

    stage_reason_counts: dict[str, int] = {}
    if "policy_stage_fallback_reasons" in unselected.columns:
        for raw_value in unselected["policy_stage_fallback_reasons"].dropna():
            for token in _split_reason_tokens(raw_value):
                stage_reason_counts[token] = stage_reason_counts.get(token, 0) + 1
    result["stage_fallback_reason_counts"] = dict(sorted(stage_reason_counts.items(), key=lambda item: (-item[1], item[0])))
    if result["likely_blocker_reason"] is None and result["stage_fallback_reason_counts"]:
        result["likely_blocker_reason"] = next(iter(result["stage_fallback_reason_counts"]))

    if "race_id" not in frame.columns:
        return result

    for race_id, race_frame in frame.groupby("race_id", sort=True):
        race_selected_mask = race_frame["policy_selected"].fillna(False).astype(bool) if "policy_selected" in race_frame.columns else pd.Series(False, index=race_frame.index)
        race_primary = (
            race_frame.loc[~race_selected_mask, "policy_reject_reason_primary"].astype("string")
            if "policy_reject_reason_primary" in race_frame.columns
            else pd.Series(pd.NA, index=race_frame.index, dtype="string")
        )
        race_primary_counts = race_primary.dropna().value_counts()
        race_stage_counts: dict[str, int] = {}
        if "policy_stage_fallback_reasons" in race_frame.columns:
            for raw_value in race_frame.loc[~race_selected_mask, "policy_stage_fallback_reasons"].dropna():
                for token in _split_reason_tokens(raw_value):
                    race_stage_counts[token] = race_stage_counts.get(token, 0) + 1
        headline = None
        if "headline" in race_frame.columns:
            for value in race_frame["headline"]:
                if pd.notna(value) and str(value).strip():
                    headline = str(value)
                    break
        result["race_diagnostics"].append(
            {
                "race_id": str(race_id),
                "headline": headline,
                "row_count": int(len(race_frame)),
                "selected_rows": int(race_selected_mask.sum()),
                "top_primary_reject_reasons": [
                    {"reason": str(reason), "rows": int(count)}
                    for reason, count in race_primary_counts.head(3).items()
                ],
                "stage_fallback_reason_counts": dict(sorted(race_stage_counts.items(), key=lambda item: (-item[1], item[0]))[:3]),
            }
        )
    return result


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
    annotated["policy_stage_trace"] = pd.NA
    annotated["policy_stage_fallback_reasons"] = pd.NA
    annotated["policy_selected_strategy_kind"] = pd.NA
    annotated["policy_reject_reason_primary"] = pd.NA
    annotated["policy_reject_reasons"] = pd.NA
    return annotated


def _evaluate_stage_fallback(
    stage_race: pd.DataFrame,
    stage_cfg: dict[str, Any],
    *,
    stage_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    selected = stage_race[stage_race["policy_selected"].fillna(False).astype(bool)]
    if selected.empty:
        return {
            "selected_count": 0,
            "fallback": True,
            "reasons": ["no_selection"],
        }

    fallback_when = stage_cfg.get("fallback_when")
    if not isinstance(fallback_when, dict) or not fallback_when:
        return {
            "selected_count": int(len(selected)),
            "fallback": False,
            "reasons": [],
        }

    selected_count = int(len(selected))
    max_expected_value = pd.to_numeric(selected["policy_expected_value"], errors="coerce").max()
    max_prob = pd.to_numeric(selected["policy_prob"], errors="coerce").max()
    max_edge = pd.to_numeric(selected["policy_edge"], errors="coerce").max()

    reasons: list[str] = []
    selected_rows_at_most = fallback_when.get("selected_rows_at_most")
    if selected_rows_at_most is not None and selected_count <= int(selected_rows_at_most):
        reasons.append("selected_rows_at_most")

    date_selected_rows_at_most = fallback_when.get("date_selected_rows_at_most")
    date_selected_count = None
    if isinstance(stage_context, dict):
        raw_date_selected_count = stage_context.get("date_selected_count")
        if raw_date_selected_count is not None:
            date_selected_count = int(raw_date_selected_count)
    if date_selected_rows_at_most is not None and date_selected_count is not None and date_selected_count <= int(date_selected_rows_at_most):
        reasons.append("date_selected_rows_at_most")

    max_expected_value_below = fallback_when.get("max_expected_value_below")
    if max_expected_value_below is not None and pd.notna(max_expected_value) and float(max_expected_value) < float(max_expected_value_below):
        reasons.append("max_expected_value_below")

    max_prob_below = fallback_when.get("max_prob_below")
    if max_prob_below is not None and pd.notna(max_prob) and float(max_prob) < float(max_prob_below):
        reasons.append("max_prob_below")

    max_edge_below = fallback_when.get("max_edge_below")
    if max_edge_below is not None and pd.notna(max_edge) and float(max_edge) < float(max_edge_below):
        reasons.append("max_edge_below")

    return {
        "selected_count": selected_count,
        "fallback": bool(reasons),
        "reasons": reasons,
    }


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
    reject_reasons: dict[Any, list[str]] = {index: [] for index in annotated.index}
    annotated["policy_blend_weight"] = blend_weight
    annotated["policy_min_prob"] = min_prob
    annotated["policy_odds_min"] = odds_min
    annotated["policy_odds_max"] = odds_max

    if odds_col is None or odds_col not in annotated.columns:
        _append_reject_reason(reject_reasons, annotated.index, "missing_odds_column")
        return _finalize_reject_reason_columns(annotated, reject_reasons)
    if "race_id" not in annotated.columns:
        _append_reject_reason(reject_reasons, annotated.index, "missing_race_id")
        return _finalize_reject_reason_columns(annotated, reject_reasons)

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

    valid_metric_mask = annotated["policy_prob"].notna() & annotated[odds_col].notna()
    _append_reject_reason(reject_reasons, annotated.index[~valid_metric_mask], "missing_prob_or_odds")
    eligible_mask = valid_metric_mask.copy()

    below_min_odds_mask = valid_metric_mask & (annotated[odds_col] <= odds_min)
    _append_reject_reason(reject_reasons, annotated.index[below_min_odds_mask], "odds_at_or_below_min")
    eligible_mask &= ~below_min_odds_mask

    above_max_odds_mask = valid_metric_mask & (annotated[odds_col] > odds_max)
    _append_reject_reason(reject_reasons, annotated.index[above_max_odds_mask], "odds_above_max")
    eligible_mask &= ~above_max_odds_mask

    below_min_prob_mask = valid_metric_mask & (annotated["policy_prob"] < min_prob)
    _append_reject_reason(reject_reasons, annotated.index[below_min_prob_mask], "prob_below_min_prob")
    eligible_mask &= ~below_min_prob_mask

    if strategy_kind == "kelly":
        min_edge = float(policy_config.get("min_edge", 0.03))
        fractional_kelly = float(policy_config.get("fractional_kelly", 0.5))
        max_fraction = float(policy_config.get("max_fraction", 0.05))
        annotated["policy_min_edge"] = min_edge
        annotated["policy_fractional_kelly"] = fractional_kelly
        annotated["policy_max_fraction"] = max_fraction

        b = annotated[odds_col] - 1.0
        raw_kelly = ((b * annotated["policy_prob"]) - (1.0 - annotated["policy_prob"])) / b.where(b != 0)
        edge_below_min_mask = eligible_mask & (annotated["policy_edge"] < min_edge)
        _append_reject_reason(reject_reasons, annotated.index[edge_below_min_mask], "edge_below_min_edge")
        eligible_mask &= ~edge_below_min_mask

        nonpositive_kelly_mask = eligible_mask & ~(raw_kelly > 0.0)
        _append_reject_reason(reject_reasons, annotated.index[nonpositive_kelly_mask], "nonpositive_kelly")
        eligible_mask &= ~nonpositive_kelly_mask

        for _, group in annotated.groupby("race_id", sort=False):
            eligible = group.loc[eligible_mask.loc[group.index]].copy()
            if eligible.empty:
                continue

            eligible = eligible.assign(policy_raw_kelly=raw_kelly.loc[eligible.index])

            pick = eligible.sort_values(["policy_edge", "policy_prob"], ascending=False).iloc[0]
            pick_index = pick.name
            annotated.loc[pick_index, "policy_selected"] = True
            annotated.loc[pick_index, "policy_selection_rank"] = 1
            annotated.loc[pick_index, "policy_weight"] = min(
                max(float(pick["policy_raw_kelly"]) * fractional_kelly, 0.0),
                max_fraction,
            )
            annotated.loc[pick_index, "policy_selected_strategy_kind"] = strategy_kind
            remaining_eligible = eligible.index.difference(pd.Index([pick_index]))
            _append_reject_reason(reject_reasons, remaining_eligible, "not_top_edge_candidate")
    elif strategy_kind == "portfolio":
        top_k = int(policy_config.get("top_k", 1))
        min_expected_value = float(policy_config.get("min_expected_value", 1.0))
        annotated["policy_top_k"] = top_k
        annotated["policy_min_expected_value"] = min_expected_value

        below_min_ev_mask = eligible_mask & (annotated["policy_expected_value"] < min_expected_value)
        _append_reject_reason(reject_reasons, annotated.index[below_min_ev_mask], "expected_value_below_min_expected_value")
        eligible_mask &= ~below_min_ev_mask

        for _, group in annotated.groupby("race_id", sort=False):
            eligible = group.loc[eligible_mask.loc[group.index]].copy()
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

            unselected_eligible = eligible.index.difference(picks.index)
            _append_reject_reason(reject_reasons, unselected_eligible, "ranked_below_top_k")

    return _finalize_reject_reason_columns(annotated, reject_reasons)


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

    stage_results: list[tuple[int, str, dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]]] = []
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
        selected_mask = stage_result["policy_selected"].fillna(False).astype(bool)
        stage_context = {
            "date_selected_count": int(selected_mask.sum()),
        }
        stage_results.append((stage_index, stage_name, stage_cfg, stage_policy, stage_result, stage_context))

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
        "policy_reject_reason_primary",
        "policy_reject_reasons",
        "policy_selected_strategy_kind",
    ]

    for race_id, race_group in annotated.groupby("race_id", sort=False):
        race_index = race_group.index
        selected_stage: tuple[int, str, dict[str, Any], dict[str, Any], pd.DataFrame, dict[str, Any]] | None = None
        stage_trace_parts: list[str] = []
        fallback_reason_parts: list[str] = []
        for stage_index, stage_name, stage_cfg, stage_policy, stage_result, stage_context in stage_results:
            stage_race = stage_result.loc[race_index]
            fallback_state = _evaluate_stage_fallback(stage_race, stage_cfg, stage_context=stage_context)
            reason_text = ",".join(fallback_state["reasons"])
            if fallback_state["reasons"]:
                fallback_reason_parts.extend(f"{stage_name}:{reason}" for reason in fallback_state["reasons"])
            if fallback_state["selected_count"] <= 0:
                stage_trace_parts.append(f"{stage_name}:no_selection")
                continue
            if fallback_state["fallback"]:
                stage_trace_parts.append(f"{stage_name}:fallback({reason_text})")
                continue

            stage_trace_parts.append(f"{stage_name}:selected")
            if stage_race["policy_selected"].fillna(False).astype(bool).any():
                selected_stage = (stage_index, stage_name, stage_cfg, stage_policy, stage_result, stage_context)
                break

        trace_text = " > ".join(stage_trace_parts) if stage_trace_parts else pd.NA
        fallback_reason_text = "|".join(fallback_reason_parts) if fallback_reason_parts else pd.NA
        annotated.loc[race_index, "policy_stage_trace"] = trace_text
        annotated.loc[race_index, "policy_stage_fallback_reasons"] = fallback_reason_text

        if selected_stage is None:
            _, _, _, _, stage_result, _ = stage_results[-1]
            stage_race = stage_result.loc[race_index]
            for column in policy_columns:
                annotated.loc[race_index, column] = stage_race[column]
            continue

        stage_index, stage_name, _, stage_policy, stage_result, _ = selected_stage
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