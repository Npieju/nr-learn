from __future__ import annotations

import argparse
from collections import defaultdict
import difflib
import json
from pathlib import Path
import traceback
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


MONTH_LABELS = {
    1: "jan",
    2: "feb",
    3: "mar",
    4: "apr",
    5: "may",
    6: "june",
    7: "july",
    8: "aug",
    9: "sep",
    10: "oct",
    11: "nov",
    12: "dec",
}


def _load_summary(summary_file: str) -> dict[str, Any]:
    summary_path = Path(summary_file)
    if not summary_path.is_absolute():
        summary_path = ROOT / summary_path
    with summary_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _load_yaml_file(path_value: str) -> tuple[Path, dict[str, Any]]:
    file_path = Path(path_value)
    if not file_path.is_absolute():
        file_path = ROOT / file_path
    with file_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return file_path, data or {}


def _serving_yaml_text(serving: dict[str, Any]) -> str:
    return yaml.safe_dump({"serving": serving}, allow_unicode=True, sort_keys=False).strip()


def _compare_serving_blocks(
    generated_serving: dict[str, Any],
    config_serving: dict[str, Any] | None,
) -> tuple[bool, str]:
    config_serving = config_serving if isinstance(config_serving, dict) else {}
    if generated_serving == config_serving:
        return True, ""

    generated_text = _serving_yaml_text(generated_serving).splitlines()
    config_text = _serving_yaml_text(config_serving).splitlines()
    diff = "\n".join(
        difflib.unified_diff(
            config_text,
            generated_text,
            fromfile="config:serving",
            tofile="generated:serving",
            lineterm="",
        )
    )
    return False, diff


def _require_fold_field(row: dict[str, Any], key: str, mode: str) -> Any:
    value = row.get(key)
    if value is None:
        raise ValueError(
            f"Fold metadata field '{key}' is missing for mapping mode '{mode}'. "
            "Rerun scripts/run_evaluate.py after the fold-window metadata update."
        )
    return value


def _months_from_range(start_date: str, end_date: str) -> list[int]:
    start = pd.Timestamp(start_date).replace(day=1)
    end = pd.Timestamp(end_date).replace(day=1)
    months: list[int] = []
    current = start
    while current <= end:
        months.append(int(current.month))
        if int(current.month) == 12:
            current = current.replace(year=int(current.year) + 1, month=1)
        else:
            current = current.replace(month=int(current.month) + 1)
    return months


def _build_test_partition_windows(folds: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    ordered_rows = sorted(
        folds,
        key=lambda row: (
            str(_require_fold_field(row, "test_start_date", "test_partition_window")),
            int(row.get("fold", 0) or 0),
        ),
    )
    partition_windows: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(ordered_rows):
        fold = int(row.get("fold", 0) or 0)
        start = pd.Timestamp(_require_fold_field(row, "test_start_date", "test_partition_window"))
        raw_end = pd.Timestamp(_require_fold_field(row, "test_end_date", "test_partition_window"))
        clipped_end = raw_end
        if index + 1 < len(ordered_rows):
            next_start = pd.Timestamp(_require_fold_field(ordered_rows[index + 1], "test_start_date", "test_partition_window"))
            clipped_end = min(raw_end, next_start - pd.Timedelta(days=1))
        if clipped_end < start:
            raise ValueError(
                f"Invalid partition window for fold {fold}: start={start.date()} clipped_end={clipped_end.date()}"
            )
        partition_windows[fold] = {
            "start_on_or_after": str(start.date()),
            "end_on_or_before": str(clipped_end.date()),
        }
    return partition_windows


def _when_payload(
    row: dict[str, Any],
    mode: str,
    *,
    partition_windows: dict[int, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if mode == "valid_end_month":
        return {"end_month_in": [int(_require_fold_field(row, "valid_end_month", mode))]}
    if mode == "test_start_month":
        return {"end_month_in": [int(_require_fold_field(row, "test_start_month", mode))]}
    if mode == "test_end_month":
        return {"end_month_in": [int(_require_fold_field(row, "test_end_month", mode))]}
    if mode == "test_months":
        return {
            "end_month_in": _months_from_range(
                str(_require_fold_field(row, "test_start_date", mode)),
                str(_require_fold_field(row, "test_end_date", mode)),
            )
        }
    if mode == "test_date_window":
        return {
            "start_on_or_after": str(_require_fold_field(row, "test_start_date", mode)),
            "end_on_or_before": str(_require_fold_field(row, "test_end_date", mode)),
        }
    if mode == "test_partition_window":
        fold = int(_require_fold_field(row, "fold", mode))
        if not isinstance(partition_windows, dict) or fold not in partition_windows:
            raise ValueError(f"Partition window is missing for fold {fold}")
        return dict(partition_windows[fold])
    raise ValueError(f"Unsupported mapping mode: {mode}")


def _policy_from_fold_row(row: dict[str, Any]) -> dict[str, Any]:
    strategy_kind = str(row.get("strategy_kind", "")).strip().lower()
    if not strategy_kind:
        raise ValueError("Fold row is missing strategy_kind")

    policy: dict[str, Any] = {
        "strategy_kind": strategy_kind,
        "blend_weight": float(row.get("blend_weight", 0.0)),
    }
    if strategy_kind != "no_bet":
        min_prob = row.get("min_prob")
        odds_min = row.get("odds_min")
        odds_max = row.get("odds_max")
        if min_prob is not None:
            policy["min_prob"] = float(min_prob)
        if odds_min is not None:
            policy["odds_min"] = float(odds_min)
        if odds_max is not None:
            policy["odds_max"] = float(odds_max)

    if strategy_kind == "kelly":
        policy["min_edge"] = float(row.get("min_edge", 0.03))
        policy["fractional_kelly"] = float(row.get("fractional_kelly", 0.5))
        policy["max_fraction"] = float(row.get("max_fraction", 0.05))
    elif strategy_kind == "portfolio":
        policy["top_k"] = int(row.get("top_k", 1))
        policy["min_expected_value"] = float(row.get("min_expected_value", 1.0))
    return policy


def _policy_signature(policy: dict[str, Any]) -> str:
    return json.dumps(policy, sort_keys=True, ensure_ascii=True)


def _month_label(month: int) -> str:
    return MONTH_LABELS.get(int(month), f"m{int(month):02d}")


def _policy_name(
    policy: dict[str, Any],
    *,
    months: list[int] | None = None,
    fold: int | None = None,
    is_default: bool = False,
) -> str:
    strategy_kind = str(policy.get("strategy_kind", "policy")).strip().lower() or "policy"
    if months:
        if len(months) == 1:
            return f"{_month_label(months[0])}_runtime_{strategy_kind}"
        return f"{'_'.join(_month_label(month) for month in months)}_runtime_{strategy_kind}"
    if fold is not None and not is_default:
        return f"fold{int(fold)}_runtime_{strategy_kind}"
    return f"default_runtime_{strategy_kind}"


def _merge_month_overrides(
    rows: list[dict[str, Any]],
    *,
    mode: str,
    key_builder: Any,
) -> list[dict[str, Any]]:
    grouped_months: dict[Any, set[int]] = defaultdict(set)
    representatives: dict[Any, dict[str, Any]] = {}
    for row in rows:
        when = _when_payload(row, mode)
        months = when.get("end_month_in", [])
        key = key_builder(row)
        grouped_months[key].update(int(month) for month in months)
        representatives[key] = row

    merged: list[dict[str, Any]] = []
    for key, months in grouped_months.items():
        merged.append({
            "row": representatives[key],
            "months": sorted(months),
        })
    return merged


def _select_default_policy_row(
    folds: list[dict[str, Any]],
    default_policy_source: str,
) -> dict[str, Any]:
    if default_policy_source == "last_fold":
        return folds[-1]

    for row in reversed(folds):
        if str(row.get("strategy_kind", "")).strip().lower() != "no_bet":
            return row
    return folds[-1]


def export_serving_from_summary(
    summary: dict[str, Any],
    *,
    policy_when_source: str,
    score_when_source: str,
    default_policy_source: str,
) -> dict[str, Any]:
    folds = summary.get("wf_nested_folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("Summary does not contain wf_nested_folds")

    partition_windows = None
    if policy_when_source == "test_partition_window" or score_when_source == "test_partition_window":
        partition_windows = _build_test_partition_windows(folds)

    score_sources = summary.get("score_sources") if isinstance(summary.get("score_sources"), dict) else {}
    default_policy_row = _select_default_policy_row(folds, default_policy_source)
    default_policy = _policy_from_fold_row(default_policy_row)
    default_months = None
    if policy_when_source in {"valid_end_month", "test_start_month", "test_end_month", "test_months"}:
        default_months = _when_payload(default_policy_row, policy_when_source, partition_windows=partition_windows).get("end_month_in", [])

    serving: dict[str, Any] = {
        "policy": {
            "name": _policy_name(default_policy, months=default_months, is_default=True),
            **default_policy,
        }
    }

    score_override_rows = []
    for row in folds:
        score_source = str(row.get("score_source", "default")).strip() or "default"
        if score_source == "default":
            continue
        source_info = score_sources.get(score_source)
        if not isinstance(source_info, dict):
            continue
        model_config = str(source_info.get("model_config", "")).strip()
        if not model_config:
            continue
        score_override_rows.append(
            {
                "fold": int(row.get("fold", 0) or 0),
                "score_source": score_source,
                "model_config": model_config,
                **row,
            }
        )

    if score_override_rows:
        if score_when_source in {"valid_end_month", "test_start_month", "test_end_month", "test_months"}:
            merged_score_rows = _merge_month_overrides(
                score_override_rows,
                mode=score_when_source,
                key_builder=lambda row: (row["score_source"], row["model_config"]),
            )
            score_regime_overrides = [
                {
                    "name": str(item["row"]["score_source"]),
                    "model_config": str(item["row"]["model_config"]),
                    "when": {"end_month_in": item["months"]},
                }
                for item in sorted(merged_score_rows, key=lambda item: item["months"])
            ]
        else:
            score_regime_overrides = []
            for row in score_override_rows:
                score_regime_overrides.append(
                    {
                        "name": f"{row['score_source']}_fold{int(row['fold'])}",
                        "model_config": str(row["model_config"]),
                        "when": _when_payload(row, score_when_source, partition_windows=partition_windows),
                    }
                )
        serving["score_regime_overrides"] = score_regime_overrides

    policy_override_rows = []
    default_signature = _policy_signature(default_policy)
    for row in folds:
        strategy_kind = str(row.get("strategy_kind", "")).strip().lower()
        if strategy_kind == "no_bet":
            continue
        policy = _policy_from_fold_row(row)
        if _policy_signature(policy) == default_signature:
            continue
        policy_override_rows.append(
            {
                "fold": int(row.get("fold", 0) or 0),
                "policy": policy,
                **row,
            }
        )

    if policy_override_rows:
        if policy_when_source in {"valid_end_month", "test_start_month", "test_end_month", "test_months"}:
            merged_policy_rows = _merge_month_overrides(
                policy_override_rows,
                mode=policy_when_source,
                key_builder=lambda row: _policy_signature(row["policy"]),
            )
            policy_regime_overrides = []
            for item in sorted(merged_policy_rows, key=lambda item: item["months"]):
                months = item["months"]
                row = item["row"]
                policy = dict(row["policy"])
                policy_regime_overrides.append(
                    {
                        "name": _policy_name(policy, months=months),
                        "when": {"end_month_in": months},
                        "policy": policy,
                    }
                )
        else:
            policy_regime_overrides = []
            for row in policy_override_rows:
                policy = dict(row["policy"])
                policy_regime_overrides.append(
                    {
                        "name": _policy_name(policy, fold=int(row["fold"])),
                        "when": _when_payload(row, policy_when_source, partition_windows=partition_windows),
                        "policy": policy,
                    }
                )
        serving["policy_regime_overrides"] = policy_regime_overrides

    return {"serving": serving}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-file", required=True)
    parser.add_argument(
        "--policy-when-source",
        choices=["valid_end_month", "test_start_month", "test_end_month", "test_months", "test_date_window", "test_partition_window"],
        default="valid_end_month",
    )
    parser.add_argument(
        "--score-when-source",
        choices=["valid_end_month", "test_start_month", "test_end_month", "test_months", "test_date_window", "test_partition_window"],
        default=None,
    )
    parser.add_argument(
        "--default-policy-source",
        choices=["last_fold", "last_non_no_bet"],
        default="last_non_no_bet",
    )
    parser.add_argument("--config-file", default=None)
    parser.add_argument("--check-config-serving", action="store_true")
    parser.add_argument("--sync-config-serving", action="store_true")
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--format", choices=["yaml", "json"], default="yaml")
    args = parser.parse_args()

    try:
        summary = _load_summary(args.summary_file)
        score_when_source = args.score_when_source or args.policy_when_source
        payload = export_serving_from_summary(
            summary,
            policy_when_source=args.policy_when_source,
            score_when_source=score_when_source,
            default_policy_source=args.default_policy_source,
        )
        generated_serving = payload["serving"]

        config_path: Path | None = None
        config_data: dict[str, Any] | None = None
        if args.config_file:
            config_path, config_data = _load_yaml_file(args.config_file)

        if args.check_config_serving:
            if config_data is None:
                raise ValueError("--check-config-serving requires --config-file")
            is_match, diff_text = _compare_serving_blocks(generated_serving, config_data.get("serving"))
            if is_match:
                print(f"[export-serving] config serving matches generated block: {config_path}")
            else:
                print(f"[export-serving] config serving differs from generated block: {config_path}")
                if diff_text:
                    print(diff_text)
                return 1

        if args.sync_config_serving:
            if config_data is None or config_path is None:
                raise ValueError("--sync-config-serving requires --config-file")
            config_data["serving"] = generated_serving
            config_path.write_text(
                yaml.safe_dump(config_data, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            print(f"[export-serving] config serving synced: {config_path}")

        if args.format == "json":
            output_text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            output_text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)

        if args.output_file:
            output_path = Path(args.output_file)
            if not output_path.is_absolute():
                output_path = ROOT / output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(output_text, encoding="utf-8")
            print(f"[export-serving] output saved: {output_path}")

        print(output_text)
        return 0
    except KeyboardInterrupt:
        print("[export-serving] interrupted by user")
        return 130
    except Exception as error:
        print(f"[export-serving] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())