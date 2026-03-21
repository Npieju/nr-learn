from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-profile-compare {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def _normalize_dates(date_values: list[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for value in date_values:
        normalized_value = str(value).strip()
        if not normalized_value or normalized_value in seen:
            continue
        seen.add(normalized_value)
        normalized.append(normalized_value)
    if not normalized:
        raise ValueError("At least one --date is required")
    return normalized


def _default_window_label(dates: list[str]) -> str:
    compact_dates = [date_value.replace("-", "") for date_value in dates]
    if len(compact_dates) == 1:
        return compact_dates[0]
    return f"{compact_dates[0]}_{compact_dates[-1]}_{len(compact_dates)}d"


def _artifact_suffix(profile_name: str, window_label: str) -> str:
    return f"{profile_name}_{window_label}"


def _run_command(command: list[str], *, label: str) -> None:
    log_progress(f"running {label}: {' '.join(command)}")
    subprocess.run(command, cwd=ROOT, check=True)


def _smoke_command(
    *,
    profile_name: str,
    dates: list[str],
    prediction_backend: str,
    artifact_suffix: str,
    output_file: Path,
) -> list[str]:
    command = [
        sys.executable,
        "scripts/run_serving_smoke.py",
        "--profile",
        profile_name,
        "--prediction-backend",
        prediction_backend,
        "--artifact-suffix",
        artifact_suffix,
        "--output-file",
        _display_path(output_file),
    ]
    for date_value in dates:
        command.extend(["--date", date_value])
    return command


def _compare_command(
    *,
    left_summary: Path,
    right_summary: Path,
    left_label: str,
    right_label: str,
    output_json: Path,
    output_csv: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_serving_smoke_compare.py",
        "--left-summary",
        _display_path(left_summary),
        "--right-summary",
        _display_path(right_summary),
        "--left-label",
        left_label,
        "--right-label",
        right_label,
        "--output-json",
        _display_path(output_json),
        "--output-csv",
        _display_path(output_csv),
    ]


def _bankroll_sweep_command(
    *,
    left_summary: Path,
    right_summary: Path,
    left_label: str,
    right_label: str,
    bankroll_floor_values: str,
    initial_bankroll: float,
    output_json: Path,
    output_csv: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_serving_stateful_bankroll_sweep.py",
        "--summary-files",
        _display_path(left_summary),
        _display_path(right_summary),
        "--labels",
        f"{left_label},{right_label}",
        "--bankroll-floor-values",
        bankroll_floor_values,
        "--initial-bankroll",
        str(initial_bankroll),
        "--output-json",
        _display_path(output_json),
        "--output-csv",
        _display_path(output_csv),
    ]


def _dashboard_command(
    *,
    manifest_file: Path,
    output_summary: Path,
    output_chart: Path,
    output_csv: Path,
) -> list[str]:
    return [
        sys.executable,
        "scripts/run_serving_compare_dashboard.py",
        "--manifest-file",
        _display_path(manifest_file),
        "--output-summary",
        _display_path(output_summary),
        "--output-chart",
        _display_path(output_chart),
        "--output-csv",
        _display_path(output_csv),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list-profiles", action="store_true")
    parser.add_argument("--left-profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--right-profile", choices=sorted(MODEL_RUN_PROFILES), default=None)
    parser.add_argument("--date", action="append", default=None)
    parser.add_argument("--window-label", default=None)
    parser.add_argument("--prediction-backend", choices=["fresh", "replay-existing"], default="replay-existing")
    parser.add_argument("--left-artifact-suffix", default=None)
    parser.add_argument("--right-artifact-suffix", default=None)
    parser.add_argument("--left-summary-output", default=None)
    parser.add_argument("--right-summary-output", default=None)
    parser.add_argument("--compare-json-output", default=None)
    parser.add_argument("--compare-csv-output", default=None)
    parser.add_argument("--run-bankroll-sweep", action="store_true")
    parser.add_argument("--bankroll-floor-values", default="1.01,1.0,0.99,0.98,0.97,0.96,0.95,0.94,0.92,0.90")
    parser.add_argument("--initial-bankroll", type=float, default=1.0)
    parser.add_argument("--bankroll-json-output", default=None)
    parser.add_argument("--bankroll-csv-output", default=None)
    parser.add_argument("--run-dashboard", action="store_true")
    parser.add_argument("--dashboard-summary-output", default=None)
    parser.add_argument("--dashboard-chart-output", default=None)
    parser.add_argument("--dashboard-csv-output", default=None)
    parser.add_argument("--manifest-output", default=None)
    args = parser.parse_args()

    if args.list_profiles:
        print(format_model_run_profiles())
        return 0

    if not args.left_profile or not args.right_profile:
        raise ValueError("--left-profile and --right-profile are required unless --list-profiles is used")

    dates = _normalize_dates(list(args.date or []))
    window_label = str(args.window_label or _default_window_label(dates)).strip()
    if not window_label:
        raise ValueError("window label must not be empty")

    left_suffix = str(args.left_artifact_suffix or _artifact_suffix(args.left_profile, window_label)).strip()
    right_suffix = str(args.right_artifact_suffix or _artifact_suffix(args.right_profile, window_label)).strip()
    if not left_suffix or not right_suffix:
        raise ValueError("artifact suffix must not be empty")

    report_dir = ROOT / "artifacts" / "reports"
    left_summary_output = _resolve_path(args.left_summary_output) if args.left_summary_output else report_dir / f"serving_smoke_{left_suffix}.json"
    right_summary_output = _resolve_path(args.right_summary_output) if args.right_summary_output else report_dir / f"serving_smoke_{right_suffix}.json"
    compare_json_output = _resolve_path(args.compare_json_output) if args.compare_json_output else report_dir / f"serving_smoke_compare_{left_suffix}_vs_{right_suffix}.json"
    compare_csv_output = _resolve_path(args.compare_csv_output) if args.compare_csv_output else report_dir / f"serving_smoke_compare_{left_suffix}_vs_{right_suffix}.csv"
    bankroll_json_output = _resolve_path(args.bankroll_json_output) if args.bankroll_json_output else report_dir / f"serving_stateful_bankroll_sweep_{left_suffix}_vs_{right_suffix}.json"
    bankroll_csv_output = _resolve_path(args.bankroll_csv_output) if args.bankroll_csv_output else report_dir / f"serving_stateful_bankroll_sweep_{left_suffix}_vs_{right_suffix}.csv"
    manifest_output = _resolve_path(args.manifest_output) if args.manifest_output else report_dir / f"serving_smoke_profile_compare_{left_suffix}_vs_{right_suffix}.json"
    dashboard_dir = report_dir / "dashboard"
    dashboard_stem = f"serving_compare_dashboard_{left_suffix}_vs_{right_suffix}"
    dashboard_summary_output = _resolve_path(args.dashboard_summary_output) if args.dashboard_summary_output else dashboard_dir / f"{dashboard_stem}.json"
    dashboard_chart_output = _resolve_path(args.dashboard_chart_output) if args.dashboard_chart_output else dashboard_dir / f"{dashboard_stem}.png"
    dashboard_csv_output = _resolve_path(args.dashboard_csv_output) if args.dashboard_csv_output else dashboard_dir / f"{dashboard_stem}.csv"

    log_progress(
        "resolved compare run "
        f"left={args.left_profile} right={args.right_profile} dates={len(dates)} "
        f"backend={args.prediction_backend} window_label={window_label}"
    )

    _run_command(
        _smoke_command(
            profile_name=args.left_profile,
            dates=dates,
            prediction_backend=args.prediction_backend,
            artifact_suffix=left_suffix,
            output_file=left_summary_output,
        ),
        label=f"left smoke {args.left_profile}",
    )
    _run_command(
        _smoke_command(
            profile_name=args.right_profile,
            dates=dates,
            prediction_backend=args.prediction_backend,
            artifact_suffix=right_suffix,
            output_file=right_summary_output,
        ),
        label=f"right smoke {args.right_profile}",
    )
    _run_command(
        _compare_command(
            left_summary=left_summary_output,
            right_summary=right_summary_output,
            left_label=args.left_profile,
            right_label=args.right_profile,
            output_json=compare_json_output,
            output_csv=compare_csv_output,
        ),
        label="smoke compare",
    )
    if args.run_bankroll_sweep:
        _run_command(
            _bankroll_sweep_command(
                left_summary=left_summary_output,
                right_summary=right_summary_output,
                left_label=args.left_profile,
                right_label=args.right_profile,
                bankroll_floor_values=str(args.bankroll_floor_values),
                initial_bankroll=float(args.initial_bankroll),
                output_json=bankroll_json_output,
                output_csv=bankroll_csv_output,
            ),
            label="bankroll sweep",
        )

    manifest_payload = {
        "created_at": utc_now_iso(),
        "window_label": window_label,
        "dates": dates,
        "prediction_backend": args.prediction_backend,
        "left": {
            "profile": args.left_profile,
            "artifact_suffix": left_suffix,
            "summary_file": _display_path(left_summary_output),
        },
        "right": {
            "profile": args.right_profile,
            "artifact_suffix": right_suffix,
            "summary_file": _display_path(right_summary_output),
        },
        "outputs": {
            "compare_json": _display_path(compare_json_output),
            "compare_csv": _display_path(compare_csv_output),
            "bankroll_sweep_json": _display_path(bankroll_json_output) if args.run_bankroll_sweep else None,
            "bankroll_sweep_csv": _display_path(bankroll_csv_output) if args.run_bankroll_sweep else None,
            "dashboard_summary": _display_path(dashboard_summary_output) if args.run_dashboard else None,
            "dashboard_chart": _display_path(dashboard_chart_output) if args.run_dashboard else None,
            "dashboard_csv": _display_path(dashboard_csv_output) if args.run_dashboard else None,
        },
    }
    write_json(manifest_output, manifest_payload)
    if args.run_dashboard:
        _run_command(
            _dashboard_command(
                manifest_file=manifest_output,
                output_summary=dashboard_summary_output,
                output_chart=dashboard_chart_output,
                output_csv=dashboard_csv_output,
            ),
            label="compare dashboard",
        )
    print(f"[serving-profile-compare] manifest saved: {_display_path(manifest_output)}", flush=True)
    print(f"[serving-profile-compare] compare json: {_display_path(compare_json_output)}", flush=True)
    print(f"[serving-profile-compare] compare csv: {_display_path(compare_csv_output)}", flush=True)
    if args.run_bankroll_sweep:
        print(f"[serving-profile-compare] bankroll sweep json: {_display_path(bankroll_json_output)}", flush=True)
        print(f"[serving-profile-compare] bankroll sweep csv: {_display_path(bankroll_csv_output)}", flush=True)
    if args.run_dashboard:
        print(f"[serving-profile-compare] dashboard summary: {_display_path(dashboard_summary_output)}", flush=True)
        print(f"[serving-profile-compare] dashboard chart: {_display_path(dashboard_chart_output)}", flush=True)
        print(f"[serving-profile-compare] dashboard csv: {_display_path(dashboard_csv_output)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())