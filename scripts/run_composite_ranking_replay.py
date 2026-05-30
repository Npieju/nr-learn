from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

import numpy as np
import pandas as pd

from racing_ml.common.artifacts import display_path, ensure_output_file_path, utc_now_iso, write_json


PRESETS: dict[str, dict[str, float]] = {
    "balanced": {"ev": 0.45, "prob": 0.35, "rank": 0.15, "odds_penalty": 0.05, "pop_penalty": 0.05},
    "prob_heavy": {"ev": 0.25, "prob": 0.50, "rank": 0.20, "odds_penalty": 0.05, "pop_penalty": 0.05},
    "ev_capped": {"ev": 0.35, "prob": 0.35, "rank": 0.20, "odds_penalty": 0.10, "pop_penalty": 0.10},
}


def _resolve_files(pattern: str) -> list[Path]:
    files = sorted(ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _read_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = display_path(path, workspace_root=ROOT)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _date_series(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        date = pd.to_datetime(frame["date"], errors="coerce").dt.date.astype("string")
        if date.notna().any():
            return date.fillna("unknown")
    source = frame["__source_file"].astype(str).str.extract(r"predictions_(\d{8})", expand=False)
    return source.fillna("unknown")


def _bucket_series(frame: pd.DataFrame) -> pd.Series:
    popularity = _numeric(frame, "popularity")
    bucket = pd.Series("unknown", index=frame.index, dtype="object")
    bucket.loc[popularity == 1] = "favorite"
    bucket.loc[popularity == 2] = "second_favorite"
    bucket.loc[popularity == 3] = "third_favorite"
    bucket.loc[(popularity >= 4) & (popularity <= 6)] = "pop_4_6"
    bucket.loc[(popularity >= 7) & (popularity <= 10)] = "pop_7_10"
    bucket.loc[popularity >= 11] = "pop_11_plus"
    return bucket


def _scale_high(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    rank = values.rank(method="average", pct=True)
    return rank.fillna(0.0)


def _scale_low(series: pd.Series) -> pd.Series:
    return 1.0 - _scale_high(series)


def _composite_score(frame: pd.DataFrame, preset: str) -> pd.Series:
    weights = PRESETS[preset]
    policy_ev = _numeric(frame, "policy_expected_value")
    prob = _numeric(frame, "policy_prob")
    score = _numeric(frame, "score")
    pred_rank = _numeric(frame, "pred_rank")
    odds = _numeric(frame, "odds")
    popularity = _numeric(frame, "popularity")
    ev_component = _scale_high(policy_ev.clip(upper=1.25))
    prob_component = _scale_high(prob.fillna(score))
    rank_component = _scale_low(pred_rank)
    odds_penalty = _scale_high(odds)
    pop_penalty = _scale_high(popularity)
    return (
        weights["ev"] * ev_component
        + weights["prob"] * prob_component
        + weights["rank"] * rank_component
        - weights["odds_penalty"] * odds_penalty
        - weights["pop_penalty"] * pop_penalty
    )


def _roi(frame: pd.DataFrame, selected: pd.Series) -> float | None:
    if not bool(selected.any()):
        return None
    odds = _numeric(frame.loc[selected], "odds")
    rank = _numeric(frame.loc[selected], "rank")
    return float(odds.where(rank == 1, 0.0).mean())


def _hit_rate(frame: pd.DataFrame, selected: pd.Series) -> float | None:
    if not bool(selected.any()):
        return None
    return float((_numeric(frame.loc[selected], "rank") == 1).mean())


def _select(
    frame: pd.DataFrame,
    *,
    preset: str,
    budget_per_date: int,
    min_policy_ev: float,
    max_popularity: int,
    odds_min: float,
    odds_max: float,
) -> pd.Series:
    work = frame.copy()
    work["__date_key"] = _date_series(work)
    work["__composite_score"] = _composite_score(work, preset)
    policy_ev = _numeric(work, "policy_expected_value")
    popularity = _numeric(work, "popularity")
    odds = _numeric(work, "odds")
    eligible = (
        policy_ev.notna()
        & (policy_ev >= min_policy_ev)
        & odds.notna()
        & (odds > odds_min)
        & (odds <= odds_max)
        & popularity.notna()
        & (popularity <= max_popularity)
    )
    selected = pd.Series(False, index=frame.index)
    for _, group in work.loc[eligible].groupby("__date_key", sort=False):
        picks = group.sort_values(["__composite_score", "policy_expected_value", "policy_prob"], ascending=False).head(
            max(budget_per_date, 0)
        )
        selected.loc[picks.index] = True
    return selected


def _summary(frame: pd.DataFrame, selected: pd.Series, *, label: str, scenario: str) -> dict[str, Any]:
    return {
        "label": label,
        "scenario": scenario,
        "selected_rows": int(selected.sum()),
        "selected_races": int(frame.loc[selected, "race_id"].nunique()) if selected.any() else 0,
        "selected_roi": _roi(frame, selected),
        "hit_rate": _hit_rate(frame, selected),
        "mean_odds": None if not selected.any() else float(_numeric(frame.loc[selected], "odds").mean()),
        "mean_policy_ev": None if not selected.any() else float(_numeric(frame.loc[selected], "policy_expected_value").mean()),
    }


def _bucket_rows(frame: pd.DataFrame, selected: pd.Series, *, label: str, scenario: str) -> list[dict[str, Any]]:
    rows = []
    bucket = _bucket_series(frame)
    for bucket_name, index in bucket.groupby(bucket).groups.items():
        mask = pd.Series(False, index=frame.index)
        mask.loc[index] = True
        bucket_selected = selected & mask
        rows.append(
            {
                "label": label,
                "scenario": scenario,
                "bucket": str(bucket_name),
                "rows": int(mask.sum()),
                "selected_rows": int(bucket_selected.sum()),
                "selected_roi": _roi(frame, bucket_selected),
                "hit_rate": _hit_rate(frame, bucket_selected),
                "mean_odds": None if not bucket_selected.any() else float(_numeric(frame.loc[bucket_selected], "odds").mean()),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay composite budgeted ranking selections.")
    parser.add_argument("--prediction-glob", action="append", required=True, help="label=glob")
    parser.add_argument("--presets", default="balanced,prob_heavy,ev_capped")
    parser.add_argument("--budgets-per-date", default="2,3,4")
    parser.add_argument("--min-policy-ev", type=float, default=0.95)
    parser.add_argument("--max-popularity", type=int, default=6)
    parser.add_argument("--odds-min", type=float, default=1.0)
    parser.add_argument("--odds-max", type=float, default=25.0)
    parser.add_argument("--output-json", default="artifacts/reports/composite_ranking_replay.json")
    parser.add_argument("--output-csv", default="artifacts/reports/composite_ranking_replay_by_bucket.csv")
    args = parser.parse_args()

    presets = [token.strip() for token in args.presets.split(",") if token.strip()]
    unknown = [preset for preset in presets if preset not in PRESETS]
    if unknown:
        raise ValueError(f"Unknown presets: {unknown}")
    budgets = [int(token.strip()) for token in args.budgets_per_date.split(",") if token.strip()]
    summaries: list[dict[str, Any]] = []
    bucket_rows: list[dict[str, Any]] = []
    inputs: dict[str, list[str]] = {}

    for spec in args.prediction_glob:
        if "=" not in spec:
            raise ValueError(f"--prediction-glob must be label=glob, got: {spec}")
        label, pattern = spec.split("=", 1)
        label = label.strip()
        paths = _resolve_files(pattern.strip())
        inputs[label] = [display_path(path, workspace_root=ROOT) for path in paths]
        frame = _read_files(paths)
        baseline_selected = frame["policy_selected"].fillna(False).astype(bool)
        summaries.append(_summary(frame, baseline_selected, label=label, scenario="baseline_policy"))
        bucket_rows.extend(_bucket_rows(frame, baseline_selected, label=label, scenario="baseline_policy"))
        for preset in presets:
            for budget in budgets:
                scenario = f"{preset}_budget_{budget}"
                selected = _select(
                    frame,
                    preset=preset,
                    budget_per_date=budget,
                    min_policy_ev=args.min_policy_ev,
                    max_popularity=args.max_popularity,
                    odds_min=args.odds_min,
                    odds_max=args.odds_max,
                )
                summaries.append(_summary(frame, selected, label=label, scenario=scenario))
                bucket_rows.extend(_bucket_rows(frame, selected, label=label, scenario=scenario))

    output_json = ensure_output_file_path(args.output_json, label="output-json", workspace_root=ROOT)
    output_csv = ensure_output_file_path(args.output_csv, label="output-csv", workspace_root=ROOT)
    write_json(
        output_json,
        {
            "created_at": utc_now_iso(),
            "presets": {name: PRESETS[name] for name in presets},
            "budgets_per_date": budgets,
            "min_policy_ev": args.min_policy_ev,
            "max_popularity": args.max_popularity,
            "odds_min": args.odds_min,
            "odds_max": args.odds_max,
            "inputs": inputs,
            "summaries": summaries,
            "bucket_csv": display_path(output_csv, workspace_root=ROOT),
        },
    )
    pd.DataFrame(bucket_rows).to_csv(output_csv, index=False)
    print(f"wrote {display_path(output_json, workspace_root=ROOT)}")
    print(f"wrote {display_path(output_csv, workspace_root=ROOT)}")


if __name__ == "__main__":
    main()
