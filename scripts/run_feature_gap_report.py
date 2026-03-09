import argparse
import csv
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import write_json
from racing_ml.common.config import load_yaml
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.features.builder import build_features
from racing_ml.features.selection import resolve_feature_selection


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[feature-gap {now}] {message}", flush=True)


def _normalize_string_list(values: object) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, Path)):
        text = str(values).strip()
        return [text] if text else []

    output: list[str] = []
    for value in values:
        text = str(value).strip()
        if text:
            output.append(text)
    return output


def _derive_output_slug(feature_config_path: str) -> str:
    slug = Path(feature_config_path).stem.strip().lower().replace("-", "_")
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug or "features"


def _collect_template_columns(template_dataset_cfg: dict[str, object]) -> list[str]:
    columns: set[str] = set()
    for key in ["append_tables", "supplemental_tables"]:
        tables = template_dataset_cfg.get(key, [])
        if not isinstance(tables, list):
            continue
        for table in tables:
            if not isinstance(table, dict):
                continue
            for list_key in ["required_columns", "join_on", "keep_columns", "dedupe_on"]:
                columns.update(_normalize_string_list(table.get(list_key)))
            column_aliases = table.get("column_aliases", {})
            if isinstance(column_aliases, dict):
                columns.update(str(column).strip() for column in column_aliases.keys() if str(column).strip())
    return sorted(columns)


def _build_column_rows(frame, columns: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for column in columns:
        present = column in frame.columns
        series = frame[column] if present else None
        rows.append(
            {
                "column": column,
                "present": bool(present),
                "dtype": str(series.dtype) if present else None,
                "non_null_ratio": round(float(series.notna().mean()), 6) if present else None,
                "nunique": int(series.nunique(dropna=True)) if present else None,
            }
        )
    return rows


def _build_feature_rows(frame, selected_columns: list[str], force_include_columns: list[str]) -> list[dict[str, object]]:
    feature_names = sorted(set(selected_columns) | set(force_include_columns))
    rows: list[dict[str, object]] = []
    for column in feature_names:
        present = column in frame.columns
        series = frame[column] if present else None
        non_null_ratio = float(series.notna().mean()) if present else None
        if not present:
            status = "missing"
        elif non_null_ratio == 0.0:
            status = "empty"
        elif non_null_ratio < 0.5:
            status = "low_coverage"
        else:
            status = "ok"
        rows.append(
            {
                "feature": column,
                "selected": bool(column in selected_columns),
                "force_included": bool(column in force_include_columns),
                "present": bool(present),
                "dtype": str(series.dtype) if present else None,
                "non_null_ratio": round(non_null_ratio, 6) if non_null_ratio is not None else None,
                "nunique": int(series.nunique(dropna=True)) if present else None,
                "status": status,
            }
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--feature-config", default="configs/features_catboost_fundamental_enriched.yaml")
    parser.add_argument("--model-config", default="configs/model_catboost_fundamental_enriched.yaml")
    parser.add_argument("--template-config", default="configs/data_netkeiba_template.yaml")
    parser.add_argument("--max-rows", type=int, default=100000)
    parser.add_argument("--coverage-threshold", type=float, default=0.5)
    parser.add_argument("--summary-output", default="")
    parser.add_argument("--feature-output", default="")
    parser.add_argument("--raw-output", default="")
    args = parser.parse_args()

    try:
        progress = ProgressBar(total=5, prefix="[feature-gap]", logger=log_progress, min_interval_sec=0.0)
        progress.start("starting feature gap report")

        data_cfg = load_yaml(ROOT / args.config)
        feature_cfg = load_yaml(ROOT / args.feature_config)
        model_cfg = load_yaml(ROOT / args.model_config)
        template_cfg = load_yaml(ROOT / args.template_config)
        progress.update(message="configs loaded")

        dataset_cfg = data_cfg.get("dataset", {})
        raw_dir = dataset_cfg.get("raw_dir", "data/raw")
        with Heartbeat("[feature-gap]", "loading training table", logger=log_progress):
            frame = load_training_table(raw_dir, dataset_config=dataset_cfg, base_dir=ROOT)
        if args.max_rows and len(frame) > args.max_rows:
            frame = frame.tail(int(args.max_rows)).reset_index(drop=True)
        progress.update(message=f"training slice ready rows={len(frame):,}")

        with Heartbeat("[feature-gap]", "building features", logger=log_progress):
            feature_frame = build_features(frame)
        progress.update(message=f"features built columns={len(feature_frame.columns):,}")

        label_column = str(model_cfg.get("label", "is_win"))
        selection = resolve_feature_selection(feature_frame, feature_cfg, label_column=label_column)
        selection_cfg = feature_cfg.get("selection", {})
        force_include_columns = _normalize_string_list(selection_cfg.get("force_include_columns"))
        template_columns = _collect_template_columns(template_cfg.get("dataset", {}))
        raw_rows = _build_column_rows(frame, template_columns)
        feature_rows = _build_feature_rows(feature_frame, selection.feature_columns, force_include_columns)

        coverage_threshold = float(args.coverage_threshold)
        priority_missing_raw_columns = [
            row["column"]
            for row in raw_rows
            if not row["present"] and row["column"] not in {"date", "race_id", "horse_id"}
        ]
        missing_force_include_features = [
            row["feature"]
            for row in feature_rows
            if row["force_included"] and row["status"] == "missing"
        ]
        low_coverage_force_include_features = [
            row["feature"]
            for row in feature_rows
            if row["force_included"] and row["present"] and (row["non_null_ratio"] or 0.0) < coverage_threshold
        ]
        empty_force_include_features = [
            row["feature"]
            for row in feature_rows
            if row["force_included"] and row["status"] == "empty"
        ]

        report = {
            "run_context": {
                "data_config": args.config,
                "feature_config": args.feature_config,
                "model_config": args.model_config,
                "template_config": args.template_config,
                "max_rows": int(args.max_rows),
                "coverage_threshold": coverage_threshold,
                "rows_evaluated": int(len(frame)),
                "feature_columns_total": int(len(feature_frame.columns)),
                "selected_feature_count": int(len(selection.feature_columns)),
                "categorical_feature_count": int(len(selection.categorical_columns)),
            },
            "summary": {
                "template_columns_total": int(len(template_columns)),
                "template_columns_present": int(sum(1 for row in raw_rows if row["present"])),
                "priority_missing_raw_columns": priority_missing_raw_columns,
                "force_include_total": int(len(force_include_columns)),
                "missing_force_include_features": missing_force_include_features,
                "empty_force_include_features": empty_force_include_features,
                "low_coverage_force_include_features": low_coverage_force_include_features,
            },
            "raw_columns": raw_rows,
            "feature_coverage": feature_rows,
        }
        progress.update(message="coverage analysis complete")

        slug = _derive_output_slug(args.feature_config)
        summary_output = ROOT / (args.summary_output or f"artifacts/reports/feature_gap_summary_{slug}.json")
        feature_output = ROOT / (args.feature_output or f"artifacts/reports/feature_gap_feature_coverage_{slug}.csv")
        raw_output = ROOT / (args.raw_output or f"artifacts/reports/feature_gap_raw_column_coverage_{slug}.csv")

        with Heartbeat("[feature-gap]", "writing report files", logger=log_progress):
            write_json(summary_output, report)
            _write_csv(
                feature_output,
                feature_rows,
                ["feature", "selected", "force_included", "present", "dtype", "non_null_ratio", "nunique", "status"],
            )
            _write_csv(
                raw_output,
                raw_rows,
                ["column", "present", "dtype", "non_null_ratio", "nunique"],
            )
        progress.complete(message="report files written")

        print(f"[feature-gap] priority_missing_raw_columns={priority_missing_raw_columns}")
        print(f"[feature-gap] missing_force_include_features={missing_force_include_features}")
        print(f"[feature-gap] low_coverage_force_include_features={low_coverage_force_include_features}")
        print(f"[feature-gap] summary saved: {summary_output}")
        print(f"[feature-gap] feature csv saved: {feature_output}")
        print(f"[feature-gap] raw column csv saved: {raw_output}")
        return 0
    except KeyboardInterrupt:
        print("[feature-gap] interrupted by user")
        return 130
    except Exception as error:
        print(f"[feature-gap] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())