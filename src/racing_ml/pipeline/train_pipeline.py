from pathlib import Path

from racing_ml.common.config import load_yaml
from racing_ml.data.dataset_loader import load_training_table
from racing_ml.features.builder import build_features
from racing_ml.models.trainer import train_and_evaluate


def run_train(model_config_path: str, data_config_path: str, feature_config_path: str) -> None:
    model_path = Path(model_config_path)
    data_path = Path(data_config_path)
    feature_path = Path(feature_config_path)

    model_config = load_yaml(model_path)
    data_config = load_yaml(data_path)
    feature_config = load_yaml(feature_path)

    dataset_cfg = data_config.get("dataset", {})
    split_cfg = data_config.get("split", {})

    raw_dir = dataset_cfg.get("raw_dir", "data/raw")
    frame = load_training_table(raw_dir)
    frame = build_features(frame)

    features_cfg = feature_config.get("features", {})
    feature_columns = features_cfg.get("base", []) + features_cfg.get("history", [])

    label_column = model_config.get("label", "is_win")
    model_cfg = model_config.get("model", {})
    output_cfg = model_config.get("output", {})
    model_name = model_cfg.get("name", "lightgbm")
    model_params = model_cfg.get("params", {})
    device_type = str(model_params.get("device_type", "cpu")).strip().lower() or "cpu"
    allow_fallback = bool(model_config.get("training", {}).get("allow_fallback_model", False))

    print(f"[train] model: {model_name}")
    print(f"[train] device_type: {device_type}")
    print(f"[train] allow_fallback_model: {allow_fallback}")

    result = train_and_evaluate(
        frame=frame,
        feature_columns=feature_columns,
        label_column=label_column,
        model_name=model_name,
        model_params=model_params,
        train_end=split_cfg.get("train_end", "2022-12-31"),
        valid_start=split_cfg.get("valid_start", "2023-01-01"),
        valid_end=split_cfg.get("valid_end", "2023-12-31"),
        max_train_rows=model_config.get("training", {}).get("max_train_rows"),
        max_valid_rows=model_config.get("training", {}).get("max_valid_rows"),
        allow_fallback=allow_fallback,
        model_dir=output_cfg.get("model_dir", "artifacts/models"),
        report_dir=output_cfg.get("report_dir", "artifacts/reports"),
    )

    print(f"[train] model saved: {result.model_path}")
    print(f"[train] report saved: {result.report_path}")
    print(f"[train] metrics: {result.metrics}")
    print(f"[train] used features: {result.used_features}")
