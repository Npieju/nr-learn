from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import resolve_output_artifacts
from racing_ml.common.config import load_yaml
from racing_ml.data.dataset_loader import load_training_table_for_feature_build
from racing_ml.evaluation.policy import (
    PolicyConstraints,
    blend_prob,
    compute_market_prob,
    evaluate_candidate_gate,
    run_policy_strategy,
)
from racing_ml.evaluation.scoring import (
    generate_prediction_outputs,
    prepare_scored_frame,
    resolve_odds_column,
)
from racing_ml.evaluation.stability import build_stability_guardrail
from racing_ml.evaluation.walk_forward import build_nested_wf_slices, fit_isotonic
from racing_ml.features.builder import build_features
from racing_ml.features.selection import (
    prepare_model_input_frame,
    resolve_feature_selection,
    resolve_model_feature_selection,
)


def _log(message: str) -> None:
    print(f"[wf-liquidity-probe] {message}", flush=True)


def _parse_float_list(value: str) -> list[float]:
    return [float(token.strip()) for token in str(value).split(",") if str(token).strip()]


def _parse_int_list(value: str) -> list[int]:
    return [int(token.strip()) for token in str(value).split(",") if str(token).strip()]


def _filter_frame_by_date_window(
    frame: pd.DataFrame,
    *,
    start_date: str | None,
    end_date: str | None,
) -> pd.DataFrame:
    if not start_date and not end_date:
        return frame

    date_series = pd.to_datetime(frame["date"], errors="coerce")
    mask = date_series.notna()
    if start_date:
        mask &= date_series >= pd.Timestamp(start_date)
    if end_date:
        mask &= date_series <= pd.Timestamp(end_date)
    filtered = frame.loc[mask].copy()
    if filtered.empty:
        raise RuntimeError(
            f"No rows found in date window: {start_date or '-inf'} .. {end_date or '+inf'}"
        )
    return filtered


def _sanitize_output_slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "model"


def _derive_output_slug(config_path: str, model_path: Path) -> str:
    candidates = [model_path.stem, Path(config_path).stem]
    for candidate in candidates:
        text = str(candidate).strip()
        if not text:
            continue
        if text.endswith("_model"):
            text = text[: -len("_model")]
        if text.startswith("model_"):
            text = text[len("model_") :]
        slug = _sanitize_output_slug(text)
        if slug:
            return slug
    return "model"


def _derive_date_window_slug(start_date: str | None, end_date: str | None) -> str:
    if not start_date and not end_date:
        return ""
    start_token = _sanitize_output_slug((start_date or "start_auto").replace("-", ""))
    end_token = _sanitize_output_slug((end_date or "end_auto").replace("-", ""))
    return f"_{start_token}_{end_token}"


def _derive_fold_slug(folds: set[int]) -> str:
    if not folds:
        return ""
    tokens = "_".join(f"f{fold}" for fold in sorted(folds))
    return f"_{tokens}"


def _serialize_float(value: object) -> float | None:
    if value is None:
        return None
    return float(value)


def _evaluate_portfolio_grid(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    *,
    odds_col: str,
    constraints: PolicyConstraints,
    blend_weights: list[float],
    min_probabilities: list[float],
    min_expected_values: list[float],
    odds_min: float,
    odds_max: float,
    top_k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid_races = int(valid_df["race_id"].nunique())
    test_races = int(test_df["race_id"].nunique()) if test_df is not None else 0
    for blend_weight in blend_weights:
        valid_work = valid_df.copy()
        valid_work["blend_prob"] = blend_prob(valid_work["iso_prob"], valid_work["market_prob"], blend_weight)
        test_work = None
        if test_df is not None:
            test_work = test_df.copy()
            test_work["blend_prob"] = blend_prob(test_work["iso_prob"], test_work["market_prob"], blend_weight)
        for min_prob in min_probabilities:
            for min_ev in min_expected_values:
                params = {
                    "strategy_kind": "portfolio",
                    "blend_weight": float(blend_weight),
                    "min_prob": float(min_prob),
                    "odds_min": float(odds_min),
                    "odds_max": float(odds_max),
                    "top_k": int(top_k),
                    "min_expected_value": float(min_ev),
                }
                valid_metrics = run_policy_strategy(valid_work, prob_col="blend_prob", odds_col=odds_col, params=params)
                valid_gate = evaluate_candidate_gate(
                    bets=int(valid_metrics.get("portfolio_bets") or 0),
                    max_drawdown=_serialize_float(valid_metrics.get("portfolio_max_drawdown")),
                    final_bankroll=_serialize_float(valid_metrics.get("portfolio_final_bankroll")),
                    constraints=constraints,
                    n_races=valid_races,
                )
                test_metrics: dict[str, object] = {}
                test_gate: dict[str, object] = {"is_feasible": None, "gate_failures": []}
                if test_work is not None:
                    test_metrics = run_policy_strategy(test_work, prob_col="blend_prob", odds_col=odds_col, params=params)
                    test_gate = evaluate_candidate_gate(
                        bets=int(test_metrics.get("portfolio_bets") or 0),
                        max_drawdown=_serialize_float(test_metrics.get("portfolio_max_drawdown")),
                        final_bankroll=_serialize_float(test_metrics.get("portfolio_final_bankroll")),
                        constraints=constraints,
                        n_races=test_races,
                    )
                valid_roi = _serialize_float(valid_metrics.get("portfolio_roi"))
                test_roi = _serialize_float(test_metrics.get("portfolio_roi"))
                rows.append(
                    {
                        "strategy": "portfolio",
                        "blend_weight": float(blend_weight),
                        "min_prob": float(min_prob),
                        "threshold": float(min_ev),
                        "bets": int(valid_metrics.get("portfolio_bets") or 0),
                        "roi": valid_roi,
                        "hit_rate": _serialize_float(valid_metrics.get("portfolio_hit_rate")),
                        "final_bankroll": _serialize_float(valid_metrics.get("portfolio_final_bankroll")),
                        "max_drawdown": _serialize_float(valid_metrics.get("portfolio_max_drawdown")),
                        "feasible": bool(valid_gate.get("is_feasible")),
                        "gate_failures": list(valid_gate.get("gate_failures", [])),
                        "test_bets": int(test_metrics.get("portfolio_bets") or 0),
                        "test_roi": test_roi,
                        "test_hit_rate": _serialize_float(test_metrics.get("portfolio_hit_rate")),
                        "test_final_bankroll": _serialize_float(test_metrics.get("portfolio_final_bankroll")),
                        "test_max_drawdown": _serialize_float(test_metrics.get("portfolio_max_drawdown")),
                        "test_feasible": None if test_gate.get("is_feasible") is None else bool(test_gate.get("is_feasible")),
                        "test_gate_failures": list(test_gate.get("gate_failures", [])),
                        "roi_gap": None if valid_roi is None or test_roi is None else float(test_roi - valid_roi),
                    }
                )
    return rows


def _evaluate_kelly_grid(
    valid_df: pd.DataFrame,
    test_df: pd.DataFrame | None,
    *,
    odds_col: str,
    constraints: PolicyConstraints,
    blend_weights: list[float],
    min_probabilities: list[float],
    min_edges: list[float],
    odds_min: float,
    odds_max: float,
    fractional_kelly: float,
    max_fraction: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    valid_races = int(valid_df["race_id"].nunique())
    test_races = int(test_df["race_id"].nunique()) if test_df is not None else 0
    for blend_weight in blend_weights:
        valid_work = valid_df.copy()
        valid_work["blend_prob"] = blend_prob(valid_work["iso_prob"], valid_work["market_prob"], blend_weight)
        test_work = None
        if test_df is not None:
            test_work = test_df.copy()
            test_work["blend_prob"] = blend_prob(test_work["iso_prob"], test_work["market_prob"], blend_weight)
        for min_prob in min_probabilities:
            for min_edge in min_edges:
                params = {
                    "strategy_kind": "kelly",
                    "blend_weight": float(blend_weight),
                    "min_edge": float(min_edge),
                    "min_prob": float(min_prob),
                    "fractional_kelly": float(fractional_kelly),
                    "max_fraction": float(max_fraction),
                    "odds_min": float(odds_min),
                    "odds_max": float(odds_max),
                }
                valid_metrics = run_policy_strategy(valid_work, prob_col="blend_prob", odds_col=odds_col, params=params)
                valid_gate = evaluate_candidate_gate(
                    bets=int(valid_metrics.get("kelly_bets") or 0),
                    max_drawdown=_serialize_float(valid_metrics.get("kelly_max_drawdown")),
                    final_bankroll=_serialize_float(valid_metrics.get("kelly_final_bankroll")),
                    constraints=constraints,
                    n_races=valid_races,
                )
                test_metrics: dict[str, object] = {}
                test_gate: dict[str, object] = {"is_feasible": None, "gate_failures": []}
                if test_work is not None:
                    test_metrics = run_policy_strategy(test_work, prob_col="blend_prob", odds_col=odds_col, params=params)
                    test_gate = evaluate_candidate_gate(
                        bets=int(test_metrics.get("kelly_bets") or 0),
                        max_drawdown=_serialize_float(test_metrics.get("kelly_max_drawdown")),
                        final_bankroll=_serialize_float(test_metrics.get("kelly_final_bankroll")),
                        constraints=constraints,
                        n_races=test_races,
                    )
                valid_roi = _serialize_float(valid_metrics.get("kelly_roi"))
                test_roi = _serialize_float(test_metrics.get("kelly_roi"))
                rows.append(
                    {
                        "strategy": "kelly",
                        "blend_weight": float(blend_weight),
                        "min_prob": float(min_prob),
                        "threshold": float(min_edge),
                        "bets": int(valid_metrics.get("kelly_bets") or 0),
                        "roi": valid_roi,
                        "hit_rate": _serialize_float(valid_metrics.get("kelly_hit_rate")),
                        "final_bankroll": _serialize_float(valid_metrics.get("kelly_final_bankroll")),
                        "max_drawdown": _serialize_float(valid_metrics.get("kelly_max_drawdown")),
                        "feasible": bool(valid_gate.get("is_feasible")),
                        "gate_failures": list(valid_gate.get("gate_failures", [])),
                        "test_bets": int(test_metrics.get("kelly_bets") or 0),
                        "test_roi": test_roi,
                        "test_hit_rate": _serialize_float(test_metrics.get("kelly_hit_rate")),
                        "test_final_bankroll": _serialize_float(test_metrics.get("kelly_final_bankroll")),
                        "test_max_drawdown": _serialize_float(test_metrics.get("kelly_max_drawdown")),
                        "test_feasible": None if test_gate.get("is_feasible") is None else bool(test_gate.get("is_feasible")),
                        "test_gate_failures": list(test_gate.get("gate_failures", [])),
                        "roi_gap": None if valid_roi is None or test_roi is None else float(test_roi - valid_roi),
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012.yaml",
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument(
        "--feature-config",
        default="configs/features_catboost_rich_high_coverage_diag.yaml",
    )
    parser.add_argument("--pre-feature-max-rows", type=int, default=None)
    parser.add_argument("--start-date", default="2024-01-01")
    parser.add_argument("--end-date", default="2024-09-30")
    parser.add_argument("--folds", default="4,5")
    parser.add_argument("--blend-weights", default="0.8,0.6")
    parser.add_argument("--portfolio-min-probs", default="0.05,0.04,0.03")
    parser.add_argument("--portfolio-min-evs", default="1.0,0.98,0.95")
    parser.add_argument("--kelly-min-probs", default="0.05,0.04,0.03")
    parser.add_argument("--kelly-min-edges", default="0.01,0.005,0.0")
    parser.add_argument("--top-k", type=int, default=1)
    parser.add_argument("--odds-min", type=float, default=1.0)
    parser.add_argument("--odds-max", type=float, default=25.0)
    parser.add_argument("--fractional-kelly", type=float, default=0.25)
    parser.add_argument("--max-fraction", type=float, default=0.02)
    args = parser.parse_args()

    model_cfg = load_yaml(ROOT / args.config)
    data_cfg = load_yaml(ROOT / args.data_config)
    feature_cfg = load_yaml(ROOT / args.feature_config)
    label_col = str(model_cfg.get("label", "is_win"))
    constraints = PolicyConstraints.from_config(model_cfg.get("evaluation", {}))

    _log("loading training table")
    load_result = load_training_table_for_feature_build(
        data_cfg.get("dataset", {}).get("raw_dir", "data/raw"),
        pre_feature_max_rows=int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
        dataset_config=data_cfg,
        base_dir=ROOT,
    )
    frame = load_result.frame
    loaded_rows = load_result.loaded_rows
    pre_feature_rows = load_result.pre_feature_rows
    data_load_strategy = load_result.data_load_strategy
    primary_source_rows_total = load_result.primary_source_rows_total
    load_message = f"training table loaded rows={loaded_rows:,} strategy={data_load_strategy}"
    if primary_source_rows_total is not None:
        load_message += f" primary_source_rows_total={primary_source_rows_total:,}"
    _log(load_message)

    _log(f"pre-feature slice ready rows={pre_feature_rows:,} loaded_rows={loaded_rows:,}")

    _log("building features")
    frame = build_features(frame)
    frame = _filter_frame_by_date_window(
        frame,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    _log(
        "date filter applied "
        f"rows={len(frame):,} start={args.start_date or '-inf'} end={args.end_date or '+inf'}"
    )

    output_artifacts = resolve_output_artifacts(model_cfg.get("output", {}))
    model_path = ROOT / output_artifacts.model_path
    _log(f"loading model bundle {model_path.name}")
    model = joblib.load(model_path)
    fallback_selection = resolve_feature_selection(frame, feature_cfg, label_column=label_col)
    feature_selection = resolve_model_feature_selection(model, fallback_selection)
    x_eval = prepare_model_input_frame(
        frame,
        feature_selection.feature_columns,
        feature_selection.categorical_columns,
    )
    odds_col = resolve_odds_column(frame)
    if odds_col is None:
        raise RuntimeError("Odds column is required for liquidity probe")

    _log(
        "running inference "
        f"features={len(feature_selection.feature_columns)} categorical={len(feature_selection.categorical_columns)}"
    )
    outputs = generate_prediction_outputs(model, x_eval, race_ids=frame["race_id"])
    pred = prepare_scored_frame(frame, outputs.score, odds_col=odds_col, score_col="score")
    _log(f"inference complete rows={len(pred):,} races={pred['race_id'].nunique():,}")

    _log("building nested walk-forward slices")
    nested_slices = build_nested_wf_slices(
        pred,
        date_col="date",
        n_folds=5,
        valid_ratio=0.15,
        test_ratio=0.15,
        min_train_rows=1000,
        min_valid_rows=500,
        min_test_rows=500,
    )

    selected_folds = set(_parse_int_list(args.folds))
    blend_weights = _parse_float_list(args.blend_weights)
    portfolio_min_probs = _parse_float_list(args.portfolio_min_probs)
    portfolio_min_evs = _parse_float_list(args.portfolio_min_evs)
    kelly_min_probs = _parse_float_list(args.kelly_min_probs)
    kelly_min_edges = _parse_float_list(args.kelly_min_edges)
    _log(
        f"selected folds={sorted(selected_folds)} portfolio_grid={len(blend_weights) * len(portfolio_min_probs) * len(portfolio_min_evs)} "
        f"kelly_grid={len(blend_weights) * len(kelly_min_probs) * len(kelly_min_edges)}"
    )

    report_rows: list[dict[str, object]] = []
    fold_summaries: list[dict[str, object]] = []
    for fold_index, (train_df, valid_df, test_df) in enumerate(nested_slices, start=1):
        if fold_index not in selected_folds:
            continue

        _log(
            f"analyzing fold={fold_index} train_rows={len(train_df):,} valid_rows={len(valid_df):,} test_rows={len(test_df):,}"
        )

        valid = valid_df.copy()
        valid["iso_prob"] = fit_isotonic(
            train_df["score"].to_numpy(),
            train_df[label_col].astype(int).to_numpy(),
            valid["score"].to_numpy(),
        )
        valid["market_prob"] = compute_market_prob(valid, odds_col=odds_col)
        test = test_df.copy()
        test["iso_prob"] = fit_isotonic(
            train_df["score"].to_numpy(),
            train_df[label_col].astype(int).to_numpy(),
            test["score"].to_numpy(),
        )
        test["market_prob"] = compute_market_prob(test, odds_col=odds_col)

        fold_rows = _evaluate_portfolio_grid(
            valid,
            test,
            odds_col=odds_col,
            constraints=constraints,
            blend_weights=blend_weights,
            min_probabilities=portfolio_min_probs,
            min_expected_values=portfolio_min_evs,
            odds_min=args.odds_min,
            odds_max=args.odds_max,
            top_k=args.top_k,
        )
        fold_rows.extend(
            _evaluate_kelly_grid(
                valid,
                test,
                odds_col=odds_col,
                constraints=constraints,
                blend_weights=blend_weights,
                min_probabilities=kelly_min_probs,
                min_edges=kelly_min_edges,
                odds_min=args.odds_min,
                odds_max=args.odds_max,
                fractional_kelly=args.fractional_kelly,
                max_fraction=args.max_fraction,
            )
        )

        for row in fold_rows:
            row["fold"] = int(fold_index)
            row["valid_dates"] = [
                str(pd.to_datetime(valid_df["date"], errors="coerce").min().date()),
                str(pd.to_datetime(valid_df["date"], errors="coerce").max().date()),
            ]
            row["valid_races"] = int(valid_df["race_id"].nunique())
            row["test_dates"] = [
                str(pd.to_datetime(test_df["date"], errors="coerce").min().date()),
                str(pd.to_datetime(test_df["date"], errors="coerce").max().date()),
            ]
            row["test_races"] = int(test_df["race_id"].nunique())
        report_rows.extend(fold_rows)

        fold_frame = pd.DataFrame(fold_rows)
        ranked = fold_frame.sort_values(
            ["feasible", "bets", "roi"],
            ascending=[False, False, False],
        )
        fold_summaries.append(
            {
                "fold": int(fold_index),
                "valid_dates": [
                    str(pd.to_datetime(valid_df["date"], errors="coerce").min().date()),
                    str(pd.to_datetime(valid_df["date"], errors="coerce").max().date()),
                ],
                "test_dates": [
                    str(pd.to_datetime(test_df["date"], errors="coerce").min().date()),
                    str(pd.to_datetime(test_df["date"], errors="coerce").max().date()),
                ],
                "valid_races": int(valid_df["race_id"].nunique()),
                "valid_stability_guardrail": build_stability_guardrail(frame=valid_df),
                "test_stability_guardrail": build_stability_guardrail(frame=test_df),
                "top_candidates": ranked.head(10)
                .assign(gate_failures=lambda df: df["gate_failures"].apply(list))
                .to_dict("records"),
            }
        )
        fold_summaries[-1]["valid_stability_assessment"] = fold_summaries[-1]["valid_stability_guardrail"]["assessment"]
        fold_summaries[-1]["test_stability_assessment"] = fold_summaries[-1]["test_stability_guardrail"]["assessment"]
        _log(f"fold={fold_index} complete candidates={len(fold_rows)}")

    report_dir = ROOT / "artifacts" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    output_slug = _derive_output_slug(args.config, model_path)
    date_slug = _derive_date_window_slug(args.start_date, args.end_date)
    fold_slug = _derive_fold_slug(selected_folds)
    summary_path = report_dir / f"wf_liquidity_probe_{output_slug}{date_slug}{fold_slug}.json"
    detail_path = report_dir / f"wf_liquidity_probe_{output_slug}{date_slug}{fold_slug}.csv"
    stability_guardrail = build_stability_guardrail(frame=pred)
    if stability_guardrail["assessment"] != "representative":
        _log(
            "stability guardrail="
            f"{stability_guardrail['assessment']}: "
            f"{'; '.join(stability_guardrail.get('warnings', [])[:2])}"
        )

    payload = {
        "run_context": {
            "config": args.config,
            "data_config": args.data_config,
            "feature_config": args.feature_config,
            "loaded_rows": loaded_rows,
            "data_load_strategy": data_load_strategy,
            "primary_source_rows_total": int(primary_source_rows_total) if primary_source_rows_total is not None else None,
            "pre_feature_max_rows": int(args.pre_feature_max_rows) if args.pre_feature_max_rows is not None else None,
            "pre_feature_rows": pre_feature_rows,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "folds": sorted(selected_folds),
            "blend_weights": blend_weights,
            "portfolio_min_probs": portfolio_min_probs,
            "portfolio_min_evs": portfolio_min_evs,
            "kelly_min_probs": kelly_min_probs,
            "kelly_min_edges": kelly_min_edges,
            "top_k": int(args.top_k),
            "odds_min": float(args.odds_min),
            "odds_max": float(args.odds_max),
            "fractional_kelly": float(args.fractional_kelly),
            "max_fraction": float(args.max_fraction),
        },
        "policy_constraints": constraints.to_dict(),
        "stability_assessment": stability_guardrail["assessment"],
        "stability_guardrail": stability_guardrail,
        "folds": fold_summaries,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    detail_frame = pd.DataFrame(report_rows)
    if not detail_frame.empty:
        detail_frame = detail_frame.copy()
        detail_frame["gate_failures"] = detail_frame["gate_failures"].apply(
            lambda values: ",".join(values) if isinstance(values, list) else str(values or "")
        )
        detail_frame["test_gate_failures"] = detail_frame["test_gate_failures"].apply(
            lambda values: ",".join(values) if isinstance(values, list) else str(values or "")
        )
        detail_frame.to_csv(detail_path, index=False)
    else:
        pd.DataFrame().to_csv(detail_path, index=False)

    _log(f"summary saved: {summary_path}")
    _log(f"detail saved: {detail_path}")
    for fold_summary in fold_summaries:
        print(json.dumps(fold_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())