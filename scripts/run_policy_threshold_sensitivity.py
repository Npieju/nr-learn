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
from racing_ml.serving.runtime_policy import annotate_runtime_policy


def _resolve_files(pattern: str) -> list[Path]:
    files = sorted(ROOT.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched: {pattern}")
    return files


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _read_prediction_files(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for path in paths:
        frame = pd.read_csv(path)
        frame["__source_file"] = display_path(path, workspace_root=ROOT)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def _extract_policy_config(frame: pd.DataFrame) -> tuple[str, dict[str, Any]]:
    strategy = str(frame.get("policy_strategy_kind", pd.Series(["portfolio"])).dropna().iloc[0]).strip().lower()
    policy_name = str(frame.get("policy_name", pd.Series(["threshold_sensitivity"])).dropna().iloc[0]).strip()
    config: dict[str, Any] = {
        "strategy_kind": strategy,
        "blend_weight": float(_numeric(frame, "policy_blend_weight").dropna().iloc[0]),
        "min_prob": float(_numeric(frame, "policy_min_prob").dropna().iloc[0]),
        "odds_min": float(_numeric(frame, "policy_odds_min").dropna().iloc[0]),
        "odds_max": float(_numeric(frame, "policy_odds_max").dropna().iloc[0]),
    }
    if strategy == "kelly":
        config.update(
            {
                "min_edge": float(_numeric(frame, "policy_min_edge").dropna().iloc[0]),
                "fractional_kelly": float(_numeric(frame, "policy_fractional_kelly").dropna().iloc[0]),
                "max_fraction": float(_numeric(frame, "policy_max_fraction").dropna().iloc[0]),
            }
        )
    elif strategy == "portfolio":
        config.update(
            {
                "top_k": int(_numeric(frame, "policy_top_k").dropna().iloc[0]),
                "min_expected_value": float(_numeric(frame, "policy_min_expected_value").dropna().iloc[0]),
            }
        )
    else:
        raise ValueError(f"Unsupported policy_strategy_kind: {strategy}")
    return policy_name or "threshold_sensitivity", config


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


def _selected_roi(frame: pd.DataFrame, selected: pd.Series) -> float | None:
    if not bool(selected.any()):
        return None
    odds = _numeric(frame.loc[selected], "odds")
    rank = _numeric(frame.loc[selected], "rank")
    return float(odds.where(rank == 1, 0.0).mean())


def _reject_reason_counts(frame: pd.DataFrame, selected: pd.Series) -> dict[str, int]:
    if "policy_reject_reason_primary" not in frame.columns:
        return {}
    reasons = frame.loc[~selected, "policy_reject_reason_primary"].dropna().astype(str)
    return {str(key): int(value) for key, value in reasons.value_counts().head(10).to_dict().items()}


def _summarize(
    frame: pd.DataFrame,
    *,
    label: str,
    scenario: str,
    baseline_selected: pd.Series | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    selected = frame["policy_selected"].fillna(False).astype(bool)
    summary = {
        "label": label,
        "scenario": scenario,
        "rows": int(len(frame)),
        "races": int(frame["race_id"].nunique()),
        "selected_rows": int(selected.sum()),
        "selected_races": int(frame.loc[selected, "race_id"].nunique()) if selected.any() else 0,
        "selected_roi": _selected_roi(frame, selected),
        "mean_policy_prob": float(_numeric(frame, "policy_prob").mean()),
        "mean_policy_expected_value": float(_numeric(frame, "policy_expected_value").mean()),
        "top_reject_reason_counts": _reject_reason_counts(frame, selected),
    }
    if baseline_selected is not None:
        baseline_selected = baseline_selected.reindex(frame.index).fillna(False).astype(bool)
        summary["selected_added_rows"] = int((selected & ~baseline_selected).sum())
        summary["selected_removed_rows"] = int((~selected & baseline_selected).sum())
        summary["selected_flip_rows"] = int((selected != baseline_selected).sum())

    bucket_rows: list[dict[str, Any]] = []
    bucket = _bucket_series(frame)
    for bucket_name, index in bucket.groupby(bucket).groups.items():
        mask = pd.Series(False, index=frame.index)
        mask.loc[index] = True
        group_selected = selected & mask
        row: dict[str, Any] = {
            "label": label,
            "scenario": scenario,
            "bucket": str(bucket_name),
            "rows": int(mask.sum()),
            "selected_rows": int(group_selected.sum()),
            "selected_roi": _selected_roi(frame, group_selected),
            "mean_policy_prob": float(_numeric(frame.loc[mask], "policy_prob").mean()),
            "mean_policy_expected_value": float(_numeric(frame.loc[mask], "policy_expected_value").mean()),
        }
        if baseline_selected is not None:
            baseline_bucket_selected = baseline_selected & mask
            row["selected_added_rows"] = int((group_selected & ~baseline_bucket_selected).sum())
            row["selected_removed_rows"] = int((~group_selected & baseline_bucket_selected).sum())
            row["selected_flip_rows"] = int((group_selected != baseline_bucket_selected).sum())
        bucket_rows.append(row)
    return summary, bucket_rows


def _replay_with_score(frame: pd.DataFrame, score: pd.Series, *, policy_name: str, policy_config: dict[str, Any], score_col: str) -> pd.DataFrame:
    replay = frame.copy()
    replay[score_col] = score.clip(1e-6, 1.0 - 1e-6)
    replay = annotate_runtime_policy(
        replay,
        odds_col="odds",
        policy_name=policy_name,
        policy_config=policy_config,
        score_col=score_col,
    )
    replay["expected_value"] = _numeric(replay, score_col) * _numeric(replay, "odds")
    replay["score"] = _numeric(replay, score_col)
    replay["pred_rank"] = replay.groupby("race_id")["score"].rank(method="first", ascending=False).astype("Int64")
    replay["ev_rank"] = replay.groupby("race_id")["expected_value"].rank(method="first", ascending=False).astype("Int64")
    return replay


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay runtime policy under bounded score perturbations.")
    parser.add_argument("--prediction-glob", action="append", required=True, help="label=glob")
    parser.add_argument("--score-col", default="score")
    parser.add_argument("--multipliers", default="0.95,0.98,1.00,1.02,1.05")
    parser.add_argument("--output-json", default="artifacts/reports/policy_threshold_sensitivity.json")
    parser.add_argument("--output-csv", default="artifacts/reports/policy_threshold_sensitivity_by_bucket.csv")
    args = parser.parse_args()

    multipliers = [float(token.strip()) for token in args.multipliers.split(",") if token.strip()]
    if 1.0 not in multipliers:
        multipliers.append(1.0)
        multipliers = sorted(multipliers)

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
        frame = _read_prediction_files(paths)
        policy_name, policy_config = _extract_policy_config(frame)
        base_score = _numeric(frame, args.score_col)
        baseline = _replay_with_score(
            frame,
            base_score,
            policy_name=f"{policy_name}_sensitivity_base",
            policy_config=policy_config,
            score_col=args.score_col,
        )
        baseline_selected = baseline["policy_selected"].fillna(False).astype(bool)

        summary, rows = _summarize(baseline, label=label, scenario="multiplier_1.000", baseline_selected=None)
        summaries.append(summary)
        bucket_rows.extend(rows)

        for multiplier in multipliers:
            if abs(multiplier - 1.0) < 1e-12:
                continue
            scenario = f"multiplier_{multiplier:.3f}"
            replay = _replay_with_score(
                frame,
                base_score * multiplier,
                policy_name=f"{policy_name}_sensitivity_{multiplier:.3f}",
                policy_config=policy_config,
                score_col=args.score_col,
            )
            summary, rows = _summarize(replay, label=label, scenario=scenario, baseline_selected=baseline_selected)
            summaries.append(summary)
            bucket_rows.extend(rows)

    output_json = ensure_output_file_path(args.output_json, label="output-json", workspace_root=ROOT)
    output_csv = ensure_output_file_path(args.output_csv, label="output-csv", workspace_root=ROOT)
    report = {
        "created_at": utc_now_iso(),
        "score_col": args.score_col,
        "multipliers": multipliers,
        "inputs": inputs,
        "summaries": summaries,
        "bucket_csv": display_path(output_csv, workspace_root=ROOT),
    }
    write_json(output_json, report)
    pd.DataFrame(bucket_rows).to_csv(output_csv, index=False)
    print(f"wrote {display_path(output_json, workspace_root=ROOT)}")
    print(f"wrote {display_path(output_csv, workspace_root=ROOT)}")


if __name__ == "__main__":
    main()
