from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar
from racing_ml.common.model_profiles import MODEL_RUN_PROFILES, format_model_run_profiles


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[serving-profile-compare {now}] {message}", flush=True)


def _resolve_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return ROOT / path


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


def _run_command_result(command: list[str], *, label: str) -> subprocess.CompletedProcess[str]:
    log_progress(f"running {label}: {' '.join(command)}")
    return subprocess.run(command, cwd=ROOT, check=False, text=True, capture_output=False)


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
        artifact_display_path(output_file, workspace_root=ROOT),
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
        artifact_display_path(left_summary, workspace_root=ROOT),
        "--right-summary",
        artifact_display_path(right_summary, workspace_root=ROOT),
        "--left-label",
        left_label,
        "--right-label",
        right_label,
        "--output-json",
        artifact_display_path(output_json, workspace_root=ROOT),
        "--output-csv",
        artifact_display_path(output_csv, workspace_root=ROOT),
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
        artifact_display_path(left_summary, workspace_root=ROOT),
        artifact_display_path(right_summary, workspace_root=ROOT),
        "--labels",
        f"{left_label},{right_label}",
        "--bankroll-floor-values",
        bankroll_floor_values,
        "--initial-bankroll",
        str(initial_bankroll),
        "--output-json",
        artifact_display_path(output_json, workspace_root=ROOT),
        "--output-csv",
        artifact_display_path(output_csv, workspace_root=ROOT),
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
        artifact_display_path(manifest_file, workspace_root=ROOT),
        "--output-summary",
        artifact_display_path(output_summary, workspace_root=ROOT),
        "--output-chart",
        artifact_display_path(output_chart, workspace_root=ROOT),
        "--output-csv",
        artifact_display_path(output_csv, workspace_root=ROOT),
    ]


def _build_manifest_payload(
    *,
    started_at: str,
    window_label: str,
    dates: list[str],
    prediction_backend: str,
    status: str,
    decision: str,
    left_profile: str,
    right_profile: str,
    left_suffix: str,
    right_suffix: str,
    left_summary_output: Path,
    right_summary_output: Path,
    compare_json_output: Path,
    compare_csv_output: Path,
    bankroll_json_output: Path,
    bankroll_csv_output: Path,
    dashboard_summary_output: Path,
    dashboard_chart_output: Path,
    dashboard_csv_output: Path,
    run_bankroll_sweep: bool,
    run_dashboard: bool,
    executed_steps: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "created_at": utc_now_iso(),
        "started_at": started_at,
        "status": status,
        "decision": decision,
        "window_label": window_label,
        "dates": dates,
        "prediction_backend": prediction_backend,
        "left": {
            "profile": left_profile,
            "artifact_suffix": left_suffix,
            "summary_file": artifact_display_path(left_summary_output, workspace_root=ROOT),
        },
        "right": {
            "profile": right_profile,
            "artifact_suffix": right_suffix,
            "summary_file": artifact_display_path(right_summary_output, workspace_root=ROOT),
        },
        "steps": executed_steps,
        "outputs": {
            "compare_json": artifact_display_path(compare_json_output, workspace_root=ROOT),
            "compare_csv": artifact_display_path(compare_csv_output, workspace_root=ROOT),
            "bankroll_sweep_json": artifact_display_path(bankroll_json_output, workspace_root=ROOT) if run_bankroll_sweep else None,
            "bankroll_sweep_csv": artifact_display_path(bankroll_csv_output, workspace_root=ROOT) if run_bankroll_sweep else None,
            "dashboard_summary": artifact_display_path(dashboard_summary_output, workspace_root=ROOT) if run_dashboard else None,
            "dashboard_chart": artifact_display_path(dashboard_chart_output, workspace_root=ROOT) if run_dashboard else None,
            "dashboard_csv": artifact_display_path(dashboard_csv_output, workspace_root=ROOT) if run_dashboard else None,
        },
    }


def _write_manifest(manifest_output: Path, payload: dict[str, object], *, label: str) -> None:
    with Heartbeat("[serving-profile-compare]", label, logger=log_progress):
        write_json(manifest_output, payload)


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

    try:
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

        artifact_ensure_output_file_path(left_summary_output, label="left summary output", workspace_root=ROOT)
        artifact_ensure_output_file_path(right_summary_output, label="right summary output", workspace_root=ROOT)
        artifact_ensure_output_file_path(compare_json_output, label="compare json output", workspace_root=ROOT)
        artifact_ensure_output_file_path(compare_csv_output, label="compare csv output", workspace_root=ROOT)
        artifact_ensure_output_file_path(manifest_output, label="manifest output", workspace_root=ROOT)
        if args.run_bankroll_sweep:
            artifact_ensure_output_file_path(bankroll_json_output, label="bankroll json output", workspace_root=ROOT)
            artifact_ensure_output_file_path(bankroll_csv_output, label="bankroll csv output", workspace_root=ROOT)
        if args.run_dashboard:
            artifact_ensure_output_file_path(dashboard_summary_output, label="dashboard summary output", workspace_root=ROOT)
            artifact_ensure_output_file_path(dashboard_chart_output, label="dashboard chart output", workspace_root=ROOT)
            artifact_ensure_output_file_path(dashboard_csv_output, label="dashboard csv output", workspace_root=ROOT)

        log_progress(
            "resolved compare run "
            f"left={args.left_profile} right={args.right_profile} dates={len(dates)} "
            f"backend={args.prediction_backend} window_label={window_label}"
        )
        total_steps = 4 + int(args.run_bankroll_sweep) + int(args.run_dashboard)
        progress = ProgressBar(total=total_steps, prefix="[serving-profile-compare]", logger=log_progress, min_interval_sec=0.0)
        progress.start(message="compare plan resolved")

        started_at = utc_now_iso()
        executed_steps: list[dict[str, object]] = []
        status = "completed"
        decision = "ready"

        left_smoke_command = _smoke_command(
            profile_name=args.left_profile,
            dates=dates,
            prediction_backend=args.prediction_backend,
            artifact_suffix=left_suffix,
            output_file=left_summary_output,
        )
        right_smoke_command = _smoke_command(
            profile_name=args.right_profile,
            dates=dates,
            prediction_backend=args.prediction_backend,
            artifact_suffix=right_suffix,
            output_file=right_summary_output,
        )
        compare_command = _compare_command(
            left_summary=left_summary_output,
            right_summary=right_summary_output,
            left_label=args.left_profile,
            right_label=args.right_profile,
            output_json=compare_json_output,
            output_csv=compare_csv_output,
        )
        bankroll_command = _bankroll_sweep_command(
            left_summary=left_summary_output,
            right_summary=right_summary_output,
            left_label=args.left_profile,
            right_label=args.right_profile,
            bankroll_floor_values=str(args.bankroll_floor_values),
            initial_bankroll=float(args.initial_bankroll),
            output_json=bankroll_json_output,
            output_csv=bankroll_csv_output,
        )
        dashboard_command = _dashboard_command(
            manifest_file=manifest_output,
            output_summary=dashboard_summary_output,
            output_chart=dashboard_chart_output,
            output_csv=dashboard_csv_output,
        )

        with Heartbeat("[serving-profile-compare]", f"left smoke {args.left_profile}", logger=log_progress):
            left_result = _run_command_result(left_smoke_command, label=f"left smoke {args.left_profile}")
        executed_steps.append(
            {
                "name": "left_smoke",
                "command": left_smoke_command,
                "status": "completed" if left_result.returncode == 0 else "failed",
                "return_code": int(left_result.returncode),
            }
        )
        if left_result.returncode != 0:
            status = "failed"
            decision = "error"
            manifest_payload = _build_manifest_payload(
                started_at=started_at,
                window_label=window_label,
                dates=dates,
                prediction_backend=args.prediction_backend,
                status=status,
                decision=decision,
                left_profile=args.left_profile,
                right_profile=args.right_profile,
                left_suffix=left_suffix,
                right_suffix=right_suffix,
                left_summary_output=left_summary_output,
                right_summary_output=right_summary_output,
                compare_json_output=compare_json_output,
                compare_csv_output=compare_csv_output,
                bankroll_json_output=bankroll_json_output,
                bankroll_csv_output=bankroll_csv_output,
                dashboard_summary_output=dashboard_summary_output,
                dashboard_chart_output=dashboard_chart_output,
                dashboard_csv_output=dashboard_csv_output,
                run_bankroll_sweep=args.run_bankroll_sweep,
                run_dashboard=args.run_dashboard,
                executed_steps=executed_steps,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing failed compare manifest")
            print(f"[serving-profile-compare] manifest saved: {artifact_display_path(manifest_output, workspace_root=ROOT)}", flush=True)
            print("[serving-profile-compare] decision: error", flush=True)
            return int(left_result.returncode) or 1
        progress.update(message=f"left summary ready {left_suffix}")

        with Heartbeat("[serving-profile-compare]", f"right smoke {args.right_profile}", logger=log_progress):
            right_result = _run_command_result(right_smoke_command, label=f"right smoke {args.right_profile}")
        executed_steps.append(
            {
                "name": "right_smoke",
                "command": right_smoke_command,
                "status": "completed" if right_result.returncode == 0 else "failed",
                "return_code": int(right_result.returncode),
            }
        )
        if right_result.returncode != 0:
            status = "failed"
            decision = "error"
            manifest_payload = _build_manifest_payload(
                started_at=started_at,
                window_label=window_label,
                dates=dates,
                prediction_backend=args.prediction_backend,
                status=status,
                decision=decision,
                left_profile=args.left_profile,
                right_profile=args.right_profile,
                left_suffix=left_suffix,
                right_suffix=right_suffix,
                left_summary_output=left_summary_output,
                right_summary_output=right_summary_output,
                compare_json_output=compare_json_output,
                compare_csv_output=compare_csv_output,
                bankroll_json_output=bankroll_json_output,
                bankroll_csv_output=bankroll_csv_output,
                dashboard_summary_output=dashboard_summary_output,
                dashboard_chart_output=dashboard_chart_output,
                dashboard_csv_output=dashboard_csv_output,
                run_bankroll_sweep=args.run_bankroll_sweep,
                run_dashboard=args.run_dashboard,
                executed_steps=executed_steps,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing failed compare manifest")
            print(f"[serving-profile-compare] manifest saved: {artifact_display_path(manifest_output, workspace_root=ROOT)}", flush=True)
            print("[serving-profile-compare] decision: error", flush=True)
            return int(right_result.returncode) or 1
        progress.update(message=f"right summary ready {right_suffix}")

        with Heartbeat("[serving-profile-compare]", "building smoke comparison", logger=log_progress):
            compare_result = _run_command_result(compare_command, label="smoke compare")
        executed_steps.append(
            {
                "name": "compare",
                "command": compare_command,
                "status": "completed" if compare_result.returncode == 0 else "failed",
                "return_code": int(compare_result.returncode),
            }
        )
        if compare_result.returncode != 0:
            status = "failed"
            decision = "error"
            manifest_payload = _build_manifest_payload(
                started_at=started_at,
                window_label=window_label,
                dates=dates,
                prediction_backend=args.prediction_backend,
                status=status,
                decision=decision,
                left_profile=args.left_profile,
                right_profile=args.right_profile,
                left_suffix=left_suffix,
                right_suffix=right_suffix,
                left_summary_output=left_summary_output,
                right_summary_output=right_summary_output,
                compare_json_output=compare_json_output,
                compare_csv_output=compare_csv_output,
                bankroll_json_output=bankroll_json_output,
                bankroll_csv_output=bankroll_csv_output,
                dashboard_summary_output=dashboard_summary_output,
                dashboard_chart_output=dashboard_chart_output,
                dashboard_csv_output=dashboard_csv_output,
                run_bankroll_sweep=args.run_bankroll_sweep,
                run_dashboard=args.run_dashboard,
                executed_steps=executed_steps,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing failed compare manifest")
            print(f"[serving-profile-compare] manifest saved: {artifact_display_path(manifest_output, workspace_root=ROOT)}", flush=True)
            print("[serving-profile-compare] decision: error", flush=True)
            return int(compare_result.returncode) or 1
        progress.update(message="comparison outputs ready")

        if args.run_bankroll_sweep:
            with Heartbeat("[serving-profile-compare]", "running bankroll sweep", logger=log_progress):
                bankroll_result = _run_command_result(bankroll_command, label="bankroll sweep")
            executed_steps.append(
                {
                    "name": "bankroll_sweep",
                    "command": bankroll_command,
                    "status": "completed" if bankroll_result.returncode == 0 else "failed",
                    "return_code": int(bankroll_result.returncode),
                }
            )
            if bankroll_result.returncode != 0:
                status = "failed"
                decision = "error"
                manifest_payload = _build_manifest_payload(
                    started_at=started_at,
                    window_label=window_label,
                    dates=dates,
                    prediction_backend=args.prediction_backend,
                    status=status,
                    decision=decision,
                    left_profile=args.left_profile,
                    right_profile=args.right_profile,
                    left_suffix=left_suffix,
                    right_suffix=right_suffix,
                    left_summary_output=left_summary_output,
                    right_summary_output=right_summary_output,
                    compare_json_output=compare_json_output,
                    compare_csv_output=compare_csv_output,
                    bankroll_json_output=bankroll_json_output,
                    bankroll_csv_output=bankroll_csv_output,
                    dashboard_summary_output=dashboard_summary_output,
                    dashboard_chart_output=dashboard_chart_output,
                    dashboard_csv_output=dashboard_csv_output,
                    run_bankroll_sweep=args.run_bankroll_sweep,
                    run_dashboard=args.run_dashboard,
                    executed_steps=executed_steps,
                )
                _write_manifest(manifest_output, manifest_payload, label="writing failed compare manifest")
                print(f"[serving-profile-compare] manifest saved: {artifact_display_path(manifest_output, workspace_root=ROOT)}", flush=True)
                print("[serving-profile-compare] decision: error", flush=True)
                return int(bankroll_result.returncode) or 1
            progress.update(message="bankroll sweep ready")

        if args.run_dashboard:
            manifest_payload = _build_manifest_payload(
                started_at=started_at,
                window_label=window_label,
                dates=dates,
                prediction_backend=args.prediction_backend,
                status=status,
                decision=decision,
                left_profile=args.left_profile,
                right_profile=args.right_profile,
                left_suffix=left_suffix,
                right_suffix=right_suffix,
                left_summary_output=left_summary_output,
                right_summary_output=right_summary_output,
                compare_json_output=compare_json_output,
                compare_csv_output=compare_csv_output,
                bankroll_json_output=bankroll_json_output,
                bankroll_csv_output=bankroll_csv_output,
                dashboard_summary_output=dashboard_summary_output,
                dashboard_chart_output=dashboard_chart_output,
                dashboard_csv_output=dashboard_csv_output,
                run_bankroll_sweep=args.run_bankroll_sweep,
                run_dashboard=args.run_dashboard,
                executed_steps=executed_steps,
            )
            _write_manifest(manifest_output, manifest_payload, label="writing compare manifest")
            progress.update(message="manifest written")
            with Heartbeat("[serving-profile-compare]", "building compare dashboard", logger=log_progress):
                dashboard_result = _run_command_result(dashboard_command, label="compare dashboard")
            executed_steps.append(
                {
                    "name": "dashboard",
                    "command": dashboard_command,
                    "status": "completed" if dashboard_result.returncode == 0 else "failed",
                    "return_code": int(dashboard_result.returncode),
                }
            )
            if dashboard_result.returncode != 0:
                status = "failed"
                decision = "error"
                manifest_payload = _build_manifest_payload(
                    started_at=started_at,
                    window_label=window_label,
                    dates=dates,
                    prediction_backend=args.prediction_backend,
                    status=status,
                    decision=decision,
                    left_profile=args.left_profile,
                    right_profile=args.right_profile,
                    left_suffix=left_suffix,
                    right_suffix=right_suffix,
                    left_summary_output=left_summary_output,
                    right_summary_output=right_summary_output,
                    compare_json_output=compare_json_output,
                    compare_csv_output=compare_csv_output,
                    bankroll_json_output=bankroll_json_output,
                    bankroll_csv_output=bankroll_csv_output,
                    dashboard_summary_output=dashboard_summary_output,
                    dashboard_chart_output=dashboard_chart_output,
                    dashboard_csv_output=dashboard_csv_output,
                    run_bankroll_sweep=args.run_bankroll_sweep,
                    run_dashboard=args.run_dashboard,
                    executed_steps=executed_steps,
                )
                _write_manifest(manifest_output, manifest_payload, label="writing failed compare manifest")
                print(f"[serving-profile-compare] manifest saved: {artifact_display_path(manifest_output, workspace_root=ROOT)}", flush=True)
                print("[serving-profile-compare] decision: error", flush=True)
                return int(dashboard_result.returncode) or 1
            progress.update(message="dashboard ready")

        manifest_payload = _build_manifest_payload(
            started_at=started_at,
            window_label=window_label,
            dates=dates,
            prediction_backend=args.prediction_backend,
            status=status,
            decision=decision,
            left_profile=args.left_profile,
            right_profile=args.right_profile,
            left_suffix=left_suffix,
            right_suffix=right_suffix,
            left_summary_output=left_summary_output,
            right_summary_output=right_summary_output,
            compare_json_output=compare_json_output,
            compare_csv_output=compare_csv_output,
            bankroll_json_output=bankroll_json_output,
            bankroll_csv_output=bankroll_csv_output,
            dashboard_summary_output=dashboard_summary_output,
            dashboard_chart_output=dashboard_chart_output,
            dashboard_csv_output=dashboard_csv_output,
            run_bankroll_sweep=args.run_bankroll_sweep,
            run_dashboard=args.run_dashboard,
            executed_steps=executed_steps,
        )
        _write_manifest(manifest_output, manifest_payload, label="writing compare manifest")
        if not args.run_dashboard:
            progress.update(message="manifest written")

        print(f"[serving-profile-compare] manifest saved: {artifact_display_path(manifest_output, workspace_root=ROOT)}", flush=True)
        print(f"[serving-profile-compare] compare json: {artifact_display_path(compare_json_output, workspace_root=ROOT)}", flush=True)
        print(f"[serving-profile-compare] compare csv: {artifact_display_path(compare_csv_output, workspace_root=ROOT)}", flush=True)
        if args.run_bankroll_sweep:
            print(f"[serving-profile-compare] bankroll sweep json: {artifact_display_path(bankroll_json_output, workspace_root=ROOT)}", flush=True)
            print(f"[serving-profile-compare] bankroll sweep csv: {artifact_display_path(bankroll_csv_output, workspace_root=ROOT)}", flush=True)
        if args.run_dashboard:
            print(f"[serving-profile-compare] dashboard summary: {artifact_display_path(dashboard_summary_output, workspace_root=ROOT)}", flush=True)
            print(f"[serving-profile-compare] dashboard chart: {artifact_display_path(dashboard_chart_output, workspace_root=ROOT)}", flush=True)
            print(f"[serving-profile-compare] dashboard csv: {artifact_display_path(dashboard_csv_output, workspace_root=ROOT)}", flush=True)
        print(f"[serving-profile-compare] decision: {decision}", flush=True)
        progress.complete(message="profile compare completed")
        return 0
    except KeyboardInterrupt:
        print("[serving-profile-compare] interrupted by user", flush=True)
        return 130
    except (ValueError, FileNotFoundError, IsADirectoryError) as error:
        print(f"[serving-profile-compare] failed: {error}", flush=True)
        return 1
    except Exception as error:
        print(f"[serving-profile-compare] failed: {error}", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
