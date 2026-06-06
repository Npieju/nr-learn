from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import ensure_output_file_path, utc_now_iso, write_json
from racing_ml.common.probability import diagnose_race_probabilities


def _resolve_path(path_text: str) -> Path:
    path = Path(path_text)
    return path if path.is_absolute() else ROOT / path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-file", required=True)
    parser.add_argument("--probability-column", action="append", dest="probability_columns")
    parser.add_argument("--race-id-column", default="race_id")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument("--output", default="artifacts/reports/race_probability_audit.json")
    parser.add_argument("--require-contract", action="store_true")
    args = parser.parse_args()

    predictions_path = _resolve_path(args.predictions_file)
    output_path = _resolve_path(args.output)
    ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
    frame = pd.read_csv(predictions_path, low_memory=False)

    requested_columns = args.probability_columns or [
        column
        for column in ["score", "policy_prob", "policy_market_prob"]
        if column in frame.columns
    ]
    if not requested_columns:
        raise ValueError("No probability columns were supplied or discovered")

    diagnostics: dict[str, object] = {}
    for column in requested_columns:
        diagnostics[column] = diagnose_race_probabilities(
            frame,
            column,
            race_id_col=args.race_id_column,
            tolerance=args.tolerance,
        ).to_dict()

    failed_columns = [
        column
        for column, diagnostic in diagnostics.items()
        if not bool(diagnostic.get("probability_contract_ok"))
    ]
    payload = {
        "generated_at": utc_now_iso(),
        "predictions_file": args.predictions_file,
        "race_id_column": args.race_id_column,
        "probability_columns": requested_columns,
        "tolerance": float(args.tolerance),
        "status": "failed" if failed_columns else "pass",
        "failed_columns": failed_columns,
        "diagnostics": diagnostics,
    }
    write_json(output_path, payload)
    print(
        f"[race-probability-audit] status={payload['status']} "
        f"failed_columns={','.join(failed_columns) if failed_columns else 'none'} output={args.output}",
        flush=True,
    )
    return 1 if args.require_contract and failed_columns else 0


if __name__ == "__main__":
    raise SystemExit(main())
