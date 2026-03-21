from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PolicyConstraints:
    min_bet_ratio: float = 0.05
    min_bets_abs: int = 100
    max_drawdown: float = 0.40
    min_final_bankroll: float = 0.85
    selection_mode: str = "gate_then_roi"

    @classmethod
    def from_config(cls, evaluation_cfg: dict[str, Any] | None = None) -> "PolicyConstraints":
        evaluation_cfg = evaluation_cfg or {}
        policy_cfg = evaluation_cfg.get("policy", {})
        legacy_cfg = evaluation_cfg.get("strategy_constraints", {})

        merged = dict(legacy_cfg)
        merged.update(policy_cfg)

        return cls(
            min_bet_ratio=float(merged.get("min_bet_ratio", 0.05)),
            min_bets_abs=int(merged.get("min_bets_abs", merged.get("min_bets", 100))),
            max_drawdown=float(merged.get("max_drawdown", 0.40)),
            min_final_bankroll=float(merged.get("min_final_bankroll", 0.85)),
            selection_mode=str(merged.get("selection_mode", "gate_then_roi")),
        )

    def min_bets_required(self, n_races: int) -> int:
        return max(int(n_races * self.min_bet_ratio), int(self.min_bets_abs))

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "min_bet_ratio": float(self.min_bet_ratio),
            "min_bets_abs": int(self.min_bets_abs),
            "max_drawdown": float(self.max_drawdown),
            "min_final_bankroll": float(self.min_final_bankroll),
            "selection_mode": str(self.selection_mode),
        }


@dataclass(frozen=True)
class StrategyCandidate:
    strategy: str
    threshold: float | None = None
    odds_min: float = 1.0
    odds_max: float = 80.0
    min_score: float = 0.0

    def to_dict(self) -> dict[str, float | int | str | None]:
        return {
            "strategy": self.strategy,
            "threshold": self.threshold,
            "odds_min": float(self.odds_min),
            "odds_max": float(self.odds_max),
            "min_score": float(self.min_score),
        }


def to_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        converted = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(converted):
            return None
        return float(converted)
    except Exception:
        return None


def compute_market_prob(frame: pd.DataFrame, odds_col: str) -> pd.Series:
    odds = pd.to_numeric(frame[odds_col], errors="coerce")
    implied = 1.0 / odds.replace(0, np.nan)
    denom = implied.groupby(frame["race_id"]).transform("sum")
    market_prob = implied / denom.replace(0, np.nan)
    return market_prob.fillna(0.0)


def blend_prob(model_prob: pd.Series, market_prob: pd.Series, weight: float) -> pd.Series:
    blended = weight * model_prob + (1.0 - weight) * market_prob
    return blended.clip(1e-6, 1 - 1e-6)


def add_market_signals(frame: pd.DataFrame, score_col: str = "score", odds_col: str = "odds") -> pd.DataFrame:
    scored = frame.copy()
    scored[odds_col] = pd.to_numeric(scored[odds_col], errors="coerce")
    scored[score_col] = pd.to_numeric(scored[score_col], errors="coerce")
    scored["expected_value"] = scored[score_col] * scored[odds_col]
    scored["market_prob"] = compute_market_prob(scored, odds_col=odds_col)
    scored["edge"] = scored[score_col] - scored["market_prob"]
    return scored


def generate_flat_strategy_candidates() -> list[StrategyCandidate]:
    candidates = [StrategyCandidate(strategy="top1")]

    for min_score in [0.16, 0.18, 0.20, 0.22, 0.24, 0.27]:
        for odds_min in [1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                candidates.append(
                    StrategyCandidate(
                        strategy="top1_filtered",
                        threshold=float(min_score),
                        odds_min=float(odds_min),
                        odds_max=float(odds_max),
                        min_score=float(min_score),
                    )
                )

    for threshold in [1.00, 1.05, 1.10, 1.20, 1.30, 1.40, 1.60, 1.80, 2.00]:
        for odds_min in [1.2, 1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                candidates.append(
                    StrategyCandidate(
                        strategy="ev",
                        threshold=float(threshold),
                        odds_min=float(odds_min),
                        odds_max=float(odds_max),
                    )
                )

    for threshold in [0.01, 0.02, 0.03, 0.05, 0.08]:
        for odds_min in [1.2, 1.5, 2.0, 3.0]:
            for odds_max in [30.0, 50.0, 80.0]:
                candidates.append(
                    StrategyCandidate(
                        strategy="edge",
                        threshold=float(threshold),
                        odds_min=float(odds_min),
                        odds_max=float(odds_max),
                    )
                )

    return candidates


def _pick_flat_candidate(
    group: pd.DataFrame,
    candidate: StrategyCandidate,
    score_col: str,
    odds_col: str,
) -> pd.Series | None:
    if candidate.strategy == "top1":
        return group.sort_values(score_col, ascending=False).iloc[0]

    if candidate.strategy == "top1_filtered":
        pick = group.sort_values(score_col, ascending=False).iloc[0]
        score_value = pd.to_numeric(pd.Series([pick.get(score_col)]), errors="coerce").iloc[0]
        odds_value = pd.to_numeric(pd.Series([pick.get(odds_col)]), errors="coerce").iloc[0]
        if pd.isna(score_value) or pd.isna(odds_value):
            return None
        if float(score_value) < float(candidate.min_score):
            return None
        if float(odds_value) < float(candidate.odds_min) or float(odds_value) > float(candidate.odds_max):
            return None
        return pick

    if candidate.strategy == "ev":
        candidates = group[
            (group["expected_value"] >= float(candidate.threshold or 0.0))
            & (group[odds_col] >= float(candidate.odds_min))
            & (group[odds_col] <= float(candidate.odds_max))
        ]
        if candidates.empty:
            return None
        return candidates.sort_values("expected_value", ascending=False).iloc[0]

    if candidate.strategy == "edge":
        candidates = group[
            (group["edge"] >= float(candidate.threshold or 0.0))
            & (group[odds_col] >= float(candidate.odds_min))
            & (group[odds_col] <= float(candidate.odds_max))
        ]
        if candidates.empty:
            return None
        return candidates.sort_values("edge", ascending=False).iloc[0]

    raise ValueError(f"Unsupported strategy mode: {candidate.strategy}")


def simulate_flat_strategy(
    frame: pd.DataFrame,
    candidate: StrategyCandidate,
    score_col: str = "score",
    odds_col: str = "odds",
    stake_per_bet: float = 1.0,
    initial_bankroll: float | None = None,
) -> dict[str, float | int | None]:
    if odds_col not in frame.columns or "rank" not in frame.columns:
        return {
            "roi": None,
            "bets": 0,
            "hit_rate": None,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "profit": None,
        }

    n_races = int(frame["race_id"].nunique())
    starting_bankroll = float(initial_bankroll) if initial_bankroll is not None else float(max(n_races, 1) * stake_per_bet)
    bankroll = float(starting_bankroll)
    peak_bankroll = float(starting_bankroll)
    max_drawdown = 0.0
    total_bet = 0.0
    total_return = 0.0
    hits = 0
    bets = 0

    for _, group in frame.groupby("race_id"):
        pick = _pick_flat_candidate(group, candidate, score_col=score_col, odds_col=odds_col)
        if pick is None:
            continue

        stake = float(stake_per_bet)
        bets += 1
        total_bet += stake
        bankroll -= stake

        rank = to_float(pick.get("rank"))
        odds = to_float(pick.get(odds_col))
        payout = 0.0
        if rank is not None and int(rank) == 1 and odds is not None and odds > 0:
            hits += 1
            payout = stake * odds
        total_return += payout
        bankroll += payout

        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, drawdown)

    if total_bet == 0:
        return {
            "roi": None,
            "bets": 0,
            "hit_rate": None,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "profit": 0.0,
        }

    roi = float(total_return / total_bet)
    hit_rate = float(hits / bets) if bets > 0 else None
    final_bankroll = float(bankroll / starting_bankroll) if starting_bankroll > 0 else None
    return {
        "roi": roi,
        "bets": int(bets),
        "hit_rate": hit_rate,
        "final_bankroll": final_bankroll,
        "max_drawdown": float(max_drawdown),
        "profit": float(total_return - total_bet),
    }


def evaluate_fixed_stake_summary(
    frame: pd.DataFrame,
    odds_col: str | None,
    score_col: str = "score",
    stake: float = 100.0,
) -> dict[str, float | int | None]:
    if odds_col is None:
        return {
            "top1_roi": None,
            "ev_top1_roi": None,
            "ev_threshold_1_0_roi": None,
            "ev_threshold_1_0_bets": 0,
            "ev_threshold_1_2_roi": None,
            "ev_threshold_1_2_bets": 0,
        }

    scored = add_market_signals(frame, score_col=score_col, odds_col=odds_col)
    top1 = simulate_flat_strategy(
        scored,
        StrategyCandidate(strategy="top1"),
        score_col=score_col,
        odds_col=odds_col,
        stake_per_bet=stake,
    )
    ev_top1 = simulate_flat_strategy(
        scored,
        StrategyCandidate(strategy="ev", threshold=-1e9, odds_min=0.0, odds_max=1e9),
        score_col=score_col,
        odds_col=odds_col,
        stake_per_bet=stake,
    )
    ev_threshold_1_0 = simulate_flat_strategy(
        scored,
        StrategyCandidate(strategy="ev", threshold=1.0, odds_min=0.0, odds_max=1e9),
        score_col=score_col,
        odds_col=odds_col,
        stake_per_bet=stake,
    )
    ev_threshold_1_2 = simulate_flat_strategy(
        scored,
        StrategyCandidate(strategy="ev", threshold=1.2, odds_min=0.0, odds_max=1e9),
        score_col=score_col,
        odds_col=odds_col,
        stake_per_bet=stake,
    )

    return {
        "top1_roi": top1.get("roi"),
        "ev_top1_roi": ev_top1.get("roi"),
        "ev_threshold_1_0_roi": ev_threshold_1_0.get("roi"),
        "ev_threshold_1_0_bets": int(ev_threshold_1_0.get("bets") or 0),
        "ev_threshold_1_2_roi": ev_threshold_1_2.get("roi"),
        "ev_threshold_1_2_bets": int(ev_threshold_1_2.get("bets") or 0),
    }


def evaluate_candidate_gate(
    *,
    bets: int,
    max_drawdown: float | None,
    final_bankroll: float | None,
    constraints: PolicyConstraints,
    n_races: int,
) -> dict[str, Any]:
    min_bets_required = constraints.min_bets_required(n_races)
    drawdown_value = float(max_drawdown) if max_drawdown is not None else 1.0
    bankroll_value = float(final_bankroll) if final_bankroll is not None else 0.0

    gate_failures: list[str] = []
    if int(bets) < min_bets_required:
        gate_failures.append("min_bets")
    if drawdown_value > constraints.max_drawdown:
        gate_failures.append("max_drawdown")
    if bankroll_value < constraints.min_final_bankroll:
        gate_failures.append("min_final_bankroll")

    return {
        "min_bets_required": int(min_bets_required),
        "n_races": int(n_races),
        "gate_failures": gate_failures,
        "failed_min_bets": bool("min_bets" in gate_failures),
        "failed_max_drawdown": bool("max_drawdown" in gate_failures),
        "failed_min_final_bankroll": bool("min_final_bankroll" in gate_failures),
        "is_feasible": bool(not gate_failures),
    }


def apply_selection_mode(
    base_score: float,
    gate_result: dict[str, Any],
    constraints: PolicyConstraints,
    *,
    bets: int,
    max_drawdown: float | None,
    final_bankroll: float | None,
) -> float | None:
    selection_mode = constraints.selection_mode.strip().lower()
    if selection_mode in {"gate_then_roi", "risk_first"}:
        return base_score if gate_result.get("is_feasible") else None

    score = float(base_score)
    if int(bets) < int(gate_result.get("min_bets_required", 0)):
        score -= 1.0
    if max_drawdown is not None and max_drawdown > constraints.max_drawdown:
        score -= 0.10 * float(max_drawdown - constraints.max_drawdown)
    if final_bankroll is not None and final_bankroll < constraints.min_final_bankroll:
        score -= 0.20 * float(constraints.min_final_bankroll - final_bankroll)
    return score


def evaluate_flat_strategy_catalog(
    frame: pd.DataFrame,
    constraints: PolicyConstraints,
    score_col: str = "score",
    odds_col: str = "odds",
    stake_per_bet: float = 1.0,
    initial_bankroll: float | None = None,
    candidates: Iterable[StrategyCandidate] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n_races = int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0
    candidate_rows: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None
    best_score = float("-inf")
    fallback_row: dict[str, Any] | None = None
    fallback_score = float("-inf")

    if candidates is None:
        candidates = generate_flat_strategy_candidates()

    for candidate in candidates:
        metrics = simulate_flat_strategy(
            frame,
            candidate,
            score_col=score_col,
            odds_col=odds_col,
            stake_per_bet=stake_per_bet,
            initial_bankroll=initial_bankroll,
        )
        row = {**candidate.to_dict(), **metrics}
        gate_result = evaluate_candidate_gate(
            bets=int(metrics.get("bets") or 0),
            max_drawdown=to_float(metrics.get("max_drawdown")),
            final_bankroll=to_float(metrics.get("final_bankroll")),
            constraints=constraints,
            n_races=n_races,
        )
        row.update(gate_result)
        row["constraints_max_drawdown"] = float(constraints.max_drawdown)
        row["constraints_min_final_bankroll"] = float(constraints.min_final_bankroll)
        row["selection_mode"] = constraints.selection_mode

        roi = to_float(row.get("roi"))
        if roi is None:
            row["selection_score"] = None
            candidate_rows.append(row)
            continue

        fallback_candidate_score = float(roi)
        if fallback_candidate_score > fallback_score:
            fallback_score = fallback_candidate_score
            fallback_row = row

        selection_score = apply_selection_mode(
            fallback_candidate_score,
            gate_result,
            constraints,
            bets=int(row.get("bets") or 0),
            max_drawdown=to_float(row.get("max_drawdown")),
            final_bankroll=to_float(row.get("final_bankroll")),
        )
        row["selection_score"] = selection_score
        candidate_rows.append(row)

        if selection_score is not None and selection_score > best_score:
            best_score = float(selection_score)
            best_row = row

    feasible_count = int(sum(int(bool(row.get("is_feasible"))) for row in candidate_rows))
    selected_via = "constraints"
    if best_row is None:
        best_row = fallback_row or {
            "strategy": "top1",
            "threshold": None,
            "roi": None,
            "bets": 0,
            "hit_rate": None,
            "final_bankroll": 1.0,
            "max_drawdown": 0.0,
            "selection_score": None,
            "is_feasible": False,
            "gate_failures": ["no_valid_candidates"],
            "min_bets_required": constraints.min_bets_required(n_races),
            "n_races": n_races,
        }
        selected_via = "fallback_no_feasible"
    elif not bool(best_row.get("is_feasible")):
        selected_via = "fallback_no_feasible"

    best = dict(best_row)
    best["feasible_candidate_count"] = feasible_count
    best["selected_via"] = selected_via
    return best, candidate_rows


def ev_top1_roi_from_prob(
    frame: pd.DataFrame,
    prob_col: str,
    odds_col: str | None,
    stake: float = 100.0,
) -> float | None:
    if odds_col is None:
        return None
    tmp = frame.copy()
    tmp[prob_col] = pd.to_numeric(tmp[prob_col], errors="coerce")
    tmp = add_market_signals(tmp, score_col=prob_col, odds_col=odds_col)
    metrics = simulate_flat_strategy(
        tmp,
        StrategyCandidate(strategy="ev", threshold=-1e9, odds_min=0.0, odds_max=1e9),
        score_col=prob_col,
        odds_col=odds_col,
        stake_per_bet=stake,
    )
    return to_float(metrics.get("roi"))


def simulate_fractional_kelly(
    frame: pd.DataFrame,
    prob_col: str,
    odds_col: str | None,
    min_edge: float = 0.03,
    min_prob: float = 0.05,
    fractional_kelly: float = 0.5,
    max_fraction: float = 0.05,
    odds_min: float = 1.0,
    odds_max: float = 999.0,
    initial_bankroll: float = 1.0,
) -> dict[str, float | int | None]:
    if odds_col is None or "rank" not in frame.columns:
        return {
            "kelly_roi": None,
            "kelly_bets": 0,
            "kelly_hit_rate": None,
            "kelly_final_bankroll": initial_bankroll,
            "kelly_max_drawdown": None,
        }

    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    total_bet = 0.0
    total_return = 0.0
    hits = 0
    bets = 0

    for _, group in frame.groupby("race_id"):
        work = group.copy()
        work["prob"] = pd.to_numeric(work[prob_col], errors="coerce")
        work["odds_num"] = pd.to_numeric(work[odds_col], errors="coerce")
        work = work.dropna(subset=["prob", "odds_num"])
        work = work[(work["odds_num"] > odds_min) & (work["odds_num"] <= odds_max)]
        if work.empty:
            continue

        work["edge"] = work["prob"] * work["odds_num"] - 1.0
        pick = work.sort_values("edge", ascending=False).iloc[0]

        prob = float(pick["prob"])
        odds = float(pick["odds_num"])
        edge = float(pick["edge"])
        if edge < min_edge or prob < min_prob:
            continue

        b = odds - 1.0
        raw_kelly = ((b * prob) - (1.0 - prob)) / b
        if raw_kelly <= 0:
            continue
        frac = min(max(raw_kelly * fractional_kelly, 0.0), max_fraction)
        stake = bankroll * frac
        if stake <= 0:
            continue

        bets += 1
        total_bet += stake

        rank = to_float(pick.get("rank"))
        payout = 0.0
        if rank is not None and int(rank) == 1:
            hits += 1
            payout = stake * odds
        total_return += payout
        bankroll = bankroll - stake + payout
        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, drawdown)

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (hits / bets) if bets > 0 else None
    return {
        "kelly_roi": float(roi) if roi is not None else None,
        "kelly_bets": int(bets),
        "kelly_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "kelly_final_bankroll": float(bankroll),
        "kelly_max_drawdown": float(max_drawdown),
    }


def simulate_ev_portfolio(
    frame: pd.DataFrame,
    prob_col: str,
    odds_col: str | None,
    top_k: int = 2,
    min_prob: float = 0.05,
    min_expected_value: float = 1.0,
    odds_min: float = 1.0,
    odds_max: float = 50.0,
    stake_per_race: float = 1.0,
    initial_bankroll: float | None = None,
) -> dict[str, float | int | None]:
    if odds_col is None or "rank" not in frame.columns:
        return {
            "portfolio_roi": None,
            "portfolio_bets": 0,
            "portfolio_hit_rate": None,
            "portfolio_avg_synthetic_odds": None,
            "portfolio_final_bankroll": 1.0,
            "portfolio_max_drawdown": None,
        }

    n_races = int(frame["race_id"].nunique()) if "race_id" in frame.columns else 0
    starting_bankroll = float(initial_bankroll) if initial_bankroll is not None else float(max(n_races, 1) * stake_per_race)
    bankroll = float(starting_bankroll)
    peak_bankroll = float(starting_bankroll)
    max_drawdown = 0.0
    total_bet = 0.0
    total_return = 0.0
    race_bets = 0
    race_hits = 0
    synthetic_odds_hits: list[float] = []

    for _, group in frame.groupby("race_id"):
        work = group.copy()
        work["prob"] = pd.to_numeric(work[prob_col], errors="coerce")
        work["odds_num"] = pd.to_numeric(work[odds_col], errors="coerce")
        work = work.dropna(subset=["prob", "odds_num"])
        work = work[(work["odds_num"] > odds_min) & (work["odds_num"] <= odds_max)]
        if work.empty:
            continue

        work["expected_value"] = work["prob"] * work["odds_num"]
        work = work[(work["prob"] >= min_prob) & (work["expected_value"] >= min_expected_value)]
        if work.empty:
            continue

        picks = work.sort_values("expected_value", ascending=False).head(max(top_k, 1))
        if picks.empty:
            continue

        race_bets += 1
        stake_each = stake_per_race / len(picks)
        total_bet += stake_per_race

        payout = 0.0
        for _, pick in picks.iterrows():
            rank = to_float(pick.get("rank"))
            if rank is not None and int(rank) == 1:
                payout += float(stake_each * float(pick["odds_num"]))

        if payout > 0:
            race_hits += 1
            synthetic_odds_hits.append(float(payout / stake_per_race))
        total_return += payout
        bankroll = bankroll - stake_per_race + payout
        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, drawdown)

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (race_hits / race_bets) if race_bets > 0 else None
    avg_synth_odds = float(np.mean(synthetic_odds_hits)) if synthetic_odds_hits else None
    return {
        "portfolio_roi": float(roi) if roi is not None else None,
        "portfolio_bets": int(race_bets),
        "portfolio_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "portfolio_avg_synthetic_odds": avg_synth_odds,
        "portfolio_final_bankroll": float(bankroll / starting_bankroll) if starting_bankroll > 0 else None,
        "portfolio_max_drawdown": float(max_drawdown),
    }


def _no_bet_policy_metrics(initial_bankroll: float = 1.0) -> dict[str, float | int | None]:
    bankroll = float(initial_bankroll)
    return {
        "kelly_roi": None,
        "kelly_bets": 0,
        "kelly_hit_rate": None,
        "kelly_final_bankroll": bankroll,
        "kelly_max_drawdown": 0.0,
        "portfolio_roi": None,
        "portfolio_bets": 0,
        "portfolio_hit_rate": None,
        "portfolio_avg_synthetic_odds": None,
        "portfolio_final_bankroll": bankroll,
        "portfolio_max_drawdown": 0.0,
    }


def simulate_annotated_runtime_policy(
    frame: pd.DataFrame,
    odds_col: str | None,
    *,
    selection_col: str = "policy_selected",
    weight_col: str = "policy_weight",
    strategy_col: str = "policy_selected_strategy_kind",
    initial_bankroll: float = 1.0,
) -> dict[str, float | int | None]:
    if odds_col is None or "rank" not in frame.columns or "race_id" not in frame.columns:
        return {
            "policy_roi": None,
            "policy_bets": 0,
            "policy_hit_rate": None,
            "policy_final_bankroll": initial_bankroll,
            "policy_max_drawdown": 0.0,
            "policy_avg_synthetic_odds": None,
        }

    bankroll = float(initial_bankroll)
    peak_bankroll = float(initial_bankroll)
    max_drawdown = 0.0
    total_bet = 0.0
    total_return = 0.0
    race_bets = 0
    race_hits = 0
    synthetic_odds_hits: list[float] = []

    for _, group in frame.groupby("race_id", sort=False):
        selected = group[group[selection_col].fillna(False).astype(bool)].copy()
        if selected.empty:
            continue

        strategy_values = selected[strategy_col].dropna().astype(str).str.strip().str.lower().unique().tolist()
        strategy_kind = strategy_values[0] if strategy_values else "portfolio"
        payout = 0.0

        if strategy_kind == "kelly":
            pick = selected.sort_values(weight_col, ascending=False).iloc[0]
            weight = to_float(pick.get(weight_col)) or 0.0
            stake = bankroll * max(weight, 0.0)
            if stake <= 0:
                continue

            race_bets += 1
            total_bet += stake
            rank = to_float(pick.get("rank"))
            odds = to_float(pick.get(odds_col))
            if rank is not None and int(rank) == 1 and odds is not None and odds > 0:
                payout = stake * odds
                race_hits += 1
        else:
            stake = 1.0
            race_bets += 1
            total_bet += stake
            for _, pick in selected.iterrows():
                rank = to_float(pick.get("rank"))
                odds = to_float(pick.get(odds_col))
                weight = to_float(pick.get(weight_col)) or 0.0
                if rank is not None and int(rank) == 1 and odds is not None and odds > 0:
                    payout += stake * weight * odds
            if payout > 0:
                race_hits += 1
                synthetic_odds_hits.append(float(payout / stake))

        total_return += payout
        bankroll = bankroll - stake + payout
        peak_bankroll = max(peak_bankroll, bankroll)
        if peak_bankroll > 0:
            drawdown = max((peak_bankroll - bankroll) / peak_bankroll, 0.0)
            max_drawdown = max(max_drawdown, drawdown)

    roi = (total_return / total_bet) if total_bet > 0 else None
    hit_rate = (race_hits / race_bets) if race_bets > 0 else None
    avg_synth_odds = float(np.mean(synthetic_odds_hits)) if synthetic_odds_hits else None
    return {
        "policy_roi": float(roi) if roi is not None else None,
        "policy_bets": int(race_bets),
        "policy_hit_rate": float(hit_rate) if hit_rate is not None else None,
        "policy_final_bankroll": float(bankroll),
        "policy_max_drawdown": float(max_drawdown),
        "policy_avg_synthetic_odds": avg_synth_odds,
    }


def run_policy_strategy(frame: pd.DataFrame, prob_col: str, odds_col: str, params: dict[str, float | str]) -> dict[str, float | int | None]:
    strategy_kind = str(params.get("strategy_kind", "kelly"))
    if strategy_kind == "no_bet":
        return _no_bet_policy_metrics(initial_bankroll=float(params.get("initial_bankroll", 1.0)))

    if strategy_kind == "portfolio":
        return simulate_ev_portfolio(
            frame=frame,
            prob_col=prob_col,
            odds_col=odds_col,
            top_k=int(params.get("top_k", 2)),
            min_prob=float(params.get("min_prob", 0.05)),
            min_expected_value=float(params.get("min_expected_value", 1.0)),
            odds_min=float(params.get("odds_min", 1.0)),
            odds_max=float(params.get("odds_max", 50.0)),
            stake_per_race=1.0,
            initial_bankroll=(float(params["initial_bankroll"]) if "initial_bankroll" in params else None),
        )

    return simulate_fractional_kelly(
        frame=frame,
        prob_col=prob_col,
        odds_col=odds_col,
        min_edge=float(params.get("min_edge", 0.03)),
        min_prob=float(params.get("min_prob", 0.05)),
        fractional_kelly=float(params.get("fractional_kelly", 0.5)),
        max_fraction=float(params.get("max_fraction", 0.05)),
        odds_min=float(params.get("odds_min", 1.0)),
        odds_max=float(params.get("odds_max", 50.0)),
        initial_bankroll=float(params.get("initial_bankroll", 1.0)),
    )