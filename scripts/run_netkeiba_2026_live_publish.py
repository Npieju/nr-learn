from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from racing_ml.common.artifacts import display_path as artifact_display_path
from racing_ml.common.artifacts import ensure_output_file_path as artifact_ensure_output_file_path
from racing_ml.common.artifacts import utc_now_iso, write_json
from racing_ml.common.progress import Heartbeat, ProgressBar


DEFAULT_PROFILE = "current_recommended_serving_2025_latest"
DEFAULT_LIVE_HANDOFF_SCRIPT = "scripts/run_netkeiba_2026_live_handoff.py"
DEFAULT_HANDOFF_MANIFEST = "artifacts/reports/netkeiba_2026_live_handoff_manifest.json"
DEFAULT_PAGES_SCRIPT = "scripts/publishing/run_build_jra_live_pages.py"
DEFAULT_PAGES_OUTPUT_DIR = "pages"
DEFAULT_OUTPUT = "artifacts/reports/netkeiba_2026_live_publish_manifest.json"


def log_progress(message: str) -> None:
    now = time.strftime("%H:%M:%S")
    print(f"[netkeiba-2026-live-publish {now}] {message}", flush=True)


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path)
    return path if path.is_absolute() else (ROOT / path)


def _read_json_dict(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _run_command(*, label: str, command: list[str]) -> int:
    print(f"[netkeiba-2026-live-publish] running {label}: {shlex.join(command)}", flush=True)
    with Heartbeat("[netkeiba-2026-live-publish]", f"{label} child command", logger=log_progress):
        result = subprocess.run(command, cwd=ROOT, check=False)
    return int(result.returncode)


def _capture_command(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def _git_status_lines(*paths: str) -> list[str]:
    command = ["git", "status", "--porcelain", "--untracked-files=all"]
    if paths:
        command.extend(["--", *paths])
    result = _capture_command(command)
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git status failed").strip())
    return [line.rstrip() for line in result.stdout.splitlines() if line.strip()]


def _git_status_paths(*paths: str) -> list[str]:
    status_lines = _git_status_lines(*paths)
    changed: list[str] = []
    for line in status_lines:
        payload = line[3:] if len(line) > 3 else line
        if " -> " in payload:
            payload = payload.split(" -> ", 1)[1]
        changed.append(payload.strip())
    return changed


def _git_head_sha() -> str | None:
    result = _capture_command(["git", "rev-parse", "HEAD"])
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _git_remote_url(remote_name: str) -> str | None:
    result = _capture_command(["git", "remote", "get-url", remote_name])
    if result.returncode != 0:
        return None
    text = result.stdout.strip()
    return text or None


def _derive_pages_url(*, remote_url: str | None, target_date: str) -> str | None:
    if not remote_url:
        return None
    normalized = remote_url.strip()
    if normalized.startswith("git@github.com:"):
        normalized = normalized.replace("git@github.com:", "https://github.com/", 1)
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    if not normalized.startswith("https://github.com/"):
        return None
    repo_slug = normalized.removeprefix("https://github.com/").strip("/")
    parts = repo_slug.split("/")
    if len(parts) != 2:
        return None
    owner, repo = parts
    return f"https://{owner.lower()}.github.io/{repo}/jra-live/{target_date}/"


def _prediction_path_from_handoff_manifest(*, manifest: dict[str, Any], race_date: str) -> Path | None:
    if str(manifest.get("race_date") or "") != str(race_date):
        return None
    prediction_file = manifest.get("live_prediction_file")
    if not prediction_file:
        return None
    path = _resolve_path(str(prediction_file))
    return path if path.exists() else None


def _latest_prediction_for_race_date(race_date: str) -> Path | None:
    date_tag = race_date.replace("-", "")
    predictions_dir = ROOT / "artifacts" / "predictions"
    candidates = sorted(predictions_dir.glob(f"predictions_{date_tag}_jra_live.csv"))
    return candidates[-1] if candidates else None


def _resolve_prediction_path(
    *,
    race_date: str,
    explicit_prediction_file: str | None,
    handoff_manifest: dict[str, Any],
) -> Path:
    if explicit_prediction_file:
        path = _resolve_path(explicit_prediction_file)
        if not path.exists():
            raise FileNotFoundError(f"predictions file not found: {artifact_display_path(path, workspace_root=ROOT)}")
        return path
    manifest_path = _prediction_path_from_handoff_manifest(manifest=handoff_manifest, race_date=race_date)
    if manifest_path is not None:
        return manifest_path
    latest_path = _latest_prediction_for_race_date(race_date)
    if latest_path is not None:
        return latest_path
    raise FileNotFoundError(f"no live prediction csv found for race_date={race_date}")


def _derive_live_paths(prediction_path: Path) -> dict[str, Path]:
    return {
        "prediction_file": prediction_path,
        "summary_file": prediction_path.with_suffix(".summary.json"),
        "live_summary_file": prediction_path.with_suffix(".live.json"),
        "report_file": prediction_path.with_suffix(".report.md"),
    }


def _live_odds_timestamp_for_prediction(path: Path | None) -> str | None:
    if path is None:
        return None
    return _read_json_dict(path.with_suffix(".live.json")).get("odds_official_datetime_max")


def _odds_provenance(
    *,
    odds_refresh_requested: bool,
    refresh_live_crawl_requested: bool,
    before_timestamp: str | None,
    after_timestamp: str | None,
) -> dict[str, Any]:
    if odds_refresh_requested or refresh_live_crawl_requested:
        source = "direct_refresh"
    elif before_timestamp and after_timestamp and before_timestamp != after_timestamp:
        source = "updated_during_handoff"
    elif after_timestamp:
        source = "reused_live_cache"
    else:
        source = "unknown"
    return {
        "source": source,
        "odds_refresh_requested": bool(odds_refresh_requested),
        "refresh_live_crawl_requested": bool(refresh_live_crawl_requested),
        "pre_handoff_odds_official_datetime_max": before_timestamp,
        "post_handoff_odds_official_datetime_max": after_timestamp,
        "timestamp_changed": bool(before_timestamp and after_timestamp and before_timestamp != after_timestamp),
    }


def _build_handoff_command(args: argparse.Namespace) -> list[str]:
    command = [
        str(args.python_executable),
        str(_resolve_path(args.live_handoff_script)),
        "--profile",
        args.profile,
        "--race-date",
        args.race_date,
        "--history-lag-days",
        str(args.history_lag_days),
        "--poll-interval-seconds",
        str(args.poll_interval_seconds),
        "--wrapper-manifest-output",
        args.handoff_manifest,
    ]
    if args.headline_contains:
        command.extend(["--headline-contains", args.headline_contains])
    if args.limit is not None:
        command.extend(["--limit", str(args.limit)])
    if args.wait_for_ready:
        command.append("--wait-for-ready")
    if args.max_wait_seconds > 0:
        command.extend(["--max-wait-seconds", str(args.max_wait_seconds)])
    if args.force:
        command.append("--force")
    if args.refresh_live_crawl:
        command.append("--refresh-live-crawl")
    return command


def _build_pages_command(
    *,
    python_executable: str,
    pages_script: str,
    prediction_path: Path,
    summary_file: str | None,
    live_summary_file: str | None,
    output_dir: str,
) -> list[str]:
    command = [
        str(python_executable),
        str(_resolve_path(pages_script)),
        "--predictions-file",
        str(prediction_path),
        "--output-dir",
        output_dir,
    ]
    if summary_file:
        command.extend(["--summary-file", summary_file])
    if live_summary_file:
        command.extend(["--live-summary-file", live_summary_file])
    return command


def _git_add_and_commit(
    *,
    target_paths: list[str],
    commit_message: str,
) -> dict[str, Any]:
    add_result = _capture_command(["git", "add", "--", *target_paths])
    if add_result.returncode != 0:
        raise RuntimeError((add_result.stderr or add_result.stdout or "git add failed").strip())

    commit_result = _capture_command(["git", "commit", "-m", commit_message])
    if commit_result.returncode != 0:
        raise RuntimeError((commit_result.stderr or commit_result.stdout or "git commit failed").strip())
    return {
        "status": "committed",
        "commit_message": commit_message,
        "commit_sha": _git_head_sha(),
        "stdout": commit_result.stdout.strip(),
    }


def _git_push(*, remote_name: str, branch_name: str) -> dict[str, Any]:
    result = _capture_command(["git", "push", remote_name, branch_name])
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "git push failed").strip())
    return {
        "status": "pushed",
        "remote": remote_name,
        "branch": branch_name,
        "stdout": result.stdout.strip(),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-executable", default=sys.executable)
    parser.add_argument("--mode", choices=["full", "rerun_only", "publish_only"], default="full")
    parser.add_argument("--race-date", required=True)
    parser.add_argument("--headline-contains", default=None)
    parser.add_argument("--profile", default=DEFAULT_PROFILE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--history-lag-days", type=int, default=1)
    parser.add_argument("--wait-for-ready", action="store_true")
    parser.add_argument("--max-wait-seconds", type=int, default=0)
    parser.add_argument("--poll-interval-seconds", type=int, default=300)
    parser.add_argument("--odds-refresh", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--refresh-live-crawl", action="store_true")
    parser.add_argument("--live-handoff-script", default=DEFAULT_LIVE_HANDOFF_SCRIPT)
    parser.add_argument("--handoff-manifest", default=DEFAULT_HANDOFF_MANIFEST)
    parser.add_argument("--pages-script", default=DEFAULT_PAGES_SCRIPT)
    parser.add_argument("--predictions-file", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--live-summary-file", default=None)
    parser.add_argument("--pages-output-dir", default=DEFAULT_PAGES_OUTPUT_DIR)
    parser.add_argument("--git-commit", action="store_true")
    parser.add_argument("--git-push", action="store_true")
    parser.add_argument("--git-remote", default="origin")
    parser.add_argument("--git-branch", default="main")
    parser.add_argument("--git-commit-message", default=None)
    parser.add_argument("--allow-dirty-pages-worktree", action="store_true")
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.odds_refresh:
        args.force = True
        args.refresh_live_crawl = True
    if args.git_push:
        args.git_commit = True

    output_path = _resolve_path(args.output)
    artifact_ensure_output_file_path(output_path, label="output", workspace_root=ROOT)
    handoff_manifest_path = _resolve_path(args.handoff_manifest)

    progress = ProgressBar(total=4, prefix="[netkeiba-2026-live-publish]", logger=log_progress, min_interval_sec=0.0)
    run_payload: dict[str, Any] = {
        "status": "running",
        "current_phase": "starting",
        "recommended_action": "wait_for_completion",
        "race_date": args.race_date,
        "mode": args.mode,
        "odds_refresh": bool(args.odds_refresh),
        "dry_run": bool(args.dry_run),
        "started_at": utc_now_iso(),
        "handoff": {},
        "pages": {},
        "git": {},
    }

    try:
        progress.start(message=f"preparing mode={args.mode} race_date={args.race_date}")
        preexisting_pages_dirty = _git_status_paths(args.pages_output_dir)
        run_payload["git"]["preexisting_pages_dirty_paths"] = preexisting_pages_dirty
        progress.update(current=1, message=f"preexisting_pages_dirty={len(preexisting_pages_dirty)}")

        handoff_manifest: dict[str, Any] = _read_json_dict(handoff_manifest_path)
        pre_handoff_prediction_path = _prediction_path_from_handoff_manifest(
            manifest=handoff_manifest,
            race_date=args.race_date,
        )
        pre_handoff_odds_timestamp = _live_odds_timestamp_for_prediction(pre_handoff_prediction_path)
        run_payload["handoff"]["pre_handoff_prediction_file"] = (
            artifact_display_path(pre_handoff_prediction_path, workspace_root=ROOT)
            if pre_handoff_prediction_path is not None
            else None
        )
        run_payload["handoff"]["pre_handoff_odds_official_datetime_max"] = pre_handoff_odds_timestamp
        prediction_path: Path | None = None

        if args.mode in {"full", "rerun_only"}:
            handoff_command = _build_handoff_command(args)
            run_payload["handoff"]["command"] = handoff_command
            if args.dry_run:
                run_payload["handoff"]["status"] = "planned"
            else:
                handoff_exit = _run_command(label="live_handoff", command=handoff_command)
                handoff_manifest = _read_json_dict(handoff_manifest_path)
                run_payload["handoff"]["status"] = "completed" if handoff_exit == 0 else "failed"
                run_payload["handoff"]["exit_code"] = handoff_exit
                run_payload["handoff"]["manifest"] = artifact_display_path(handoff_manifest_path, workspace_root=ROOT)
                if handoff_exit != 0:
                    run_payload["status"] = "failed"
                    run_payload["current_phase"] = "live_handoff_failed"
                    run_payload["recommended_action"] = "inspect_live_handoff_manifest"
                    run_payload["finished_at"] = utc_now_iso()
                    write_json(output_path, run_payload)
                    progress.complete(message=f"handoff failed output={artifact_display_path(output_path, workspace_root=ROOT)}")
                    return 1
            progress.update(current=2, message="live handoff step ready")
        else:
            progress.update(current=2, message="live handoff skipped for publish_only")

        if args.mode == "rerun_only":
            if args.dry_run:
                run_payload["status"] = "dry_run"
                run_payload["current_phase"] = "plan_ready"
                run_payload["recommended_action"] = "run_live_handoff_then_build_pages"
            else:
                prediction_path = _resolve_prediction_path(
                    race_date=args.race_date,
                    explicit_prediction_file=args.predictions_file,
                    handoff_manifest=handoff_manifest,
                )
                live_paths = _derive_live_paths(prediction_path)
                live_summary_payload = _read_json_dict(live_paths["live_summary_file"])
                run_payload["handoff"]["prediction_file"] = artifact_display_path(live_paths["prediction_file"], workspace_root=ROOT)
                run_payload["handoff"]["report_file"] = artifact_display_path(live_paths["report_file"], workspace_root=ROOT)
                run_payload["handoff"]["odds_official_datetime_max"] = live_summary_payload.get("odds_official_datetime_max")
                run_payload["handoff"]["odds_provenance"] = _odds_provenance(
                    odds_refresh_requested=bool(args.odds_refresh),
                    refresh_live_crawl_requested=bool(args.refresh_live_crawl),
                    before_timestamp=pre_handoff_odds_timestamp,
                    after_timestamp=live_summary_payload.get("odds_official_datetime_max"),
                )
                run_payload["handoff"]["policy_selected_rows"] = _read_json_dict(live_paths["summary_file"]).get("policy_selected_rows")
                run_payload["status"] = "completed"
                run_payload["current_phase"] = "live_rerun_completed"
                run_payload["recommended_action"] = "build_pages_or_review_outputs"
            run_payload["finished_at"] = utc_now_iso()
            write_json(output_path, run_payload)
            progress.complete(message=f"rerun_only output={artifact_display_path(output_path, workspace_root=ROOT)}")
            return 0

        prediction_path = _resolve_prediction_path(
            race_date=args.race_date,
            explicit_prediction_file=args.predictions_file,
            handoff_manifest=handoff_manifest,
        )
        live_paths = _derive_live_paths(prediction_path)
        live_summary_payload = _read_json_dict(live_paths["live_summary_file"])
        summary_payload = _read_json_dict(live_paths["summary_file"])
        target_date = str(live_summary_payload.get("target_date") or args.race_date)
        target_page = _resolve_path(args.pages_output_dir) / "jra-live" / target_date / "index.html"
        target_data_file = _resolve_path(args.pages_output_dir) / "jra-live" / target_date / "data.json"
        pages_command = _build_pages_command(
            python_executable=args.python_executable,
            pages_script=args.pages_script,
            prediction_path=prediction_path,
            summary_file=args.summary_file,
            live_summary_file=args.live_summary_file,
            output_dir=args.pages_output_dir,
        )
        run_payload["pages"]["command"] = pages_command
        run_payload["pages"]["target_page"] = artifact_display_path(target_page, workspace_root=ROOT)
        run_payload["pages"]["data_file"] = artifact_display_path(target_data_file, workspace_root=ROOT)

        if args.dry_run:
            run_payload["pages"]["status"] = "planned"
            progress.update(current=3, message="pages build planned")
        else:
            if preexisting_pages_dirty and args.git_commit and not args.allow_dirty_pages_worktree:
                raise RuntimeError(
                    "pages/ に事前差分があります。auto commit を使うなら先に整理するか --allow-dirty-pages-worktree を付けてください"
                )
            pages_exit = _run_command(label="build_pages", command=pages_command)
            run_payload["pages"]["status"] = "completed" if pages_exit == 0 else "failed"
            run_payload["pages"]["exit_code"] = pages_exit
            if pages_exit != 0:
                run_payload["status"] = "failed"
                run_payload["current_phase"] = "pages_build_failed"
                run_payload["recommended_action"] = "inspect_pages_build_failure"
                run_payload["finished_at"] = utc_now_iso()
                write_json(output_path, run_payload)
                progress.complete(message=f"pages build failed output={artifact_display_path(output_path, workspace_root=ROOT)}")
                return 1
            progress.update(current=3, message=f"pages built target_page={artifact_display_path(target_page, workspace_root=ROOT)}")

        run_payload["handoff"]["prediction_file"] = artifact_display_path(live_paths["prediction_file"], workspace_root=ROOT)
        run_payload["handoff"]["report_file"] = artifact_display_path(live_paths["report_file"], workspace_root=ROOT)
        run_payload["handoff"]["odds_official_datetime_max"] = live_summary_payload.get("odds_official_datetime_max")
        run_payload["handoff"]["odds_provenance"] = _odds_provenance(
            odds_refresh_requested=bool(args.odds_refresh),
            refresh_live_crawl_requested=bool(args.refresh_live_crawl),
            before_timestamp=pre_handoff_odds_timestamp,
            after_timestamp=live_summary_payload.get("odds_official_datetime_max"),
        )
        run_payload["handoff"]["policy_selected_rows"] = summary_payload.get("policy_selected_rows")
        run_payload["pages"]["page_url"] = _derive_pages_url(
            remote_url=_git_remote_url(args.git_remote),
            target_date=target_date,
        )

        if args.git_commit:
            if args.dry_run:
                run_payload["git"]["status"] = "planned"
            else:
                changed_pages_paths = _git_status_paths(args.pages_output_dir)
                run_payload["git"]["changed_pages_paths"] = changed_pages_paths
                if not changed_pages_paths:
                    run_payload["git"]["status"] = "no_changes"
                else:
                    commit_message = args.git_commit_message or f"pages: refresh jra live report for {args.race_date}"
                    commit_info = _git_add_and_commit(
                        target_paths=[args.pages_output_dir],
                        commit_message=commit_message,
                    )
                    run_payload["git"].update(commit_info)
                    if args.git_push:
                        push_info = _git_push(remote_name=args.git_remote, branch_name=args.git_branch)
                        run_payload["git"]["push"] = push_info
            progress.update(current=4, message=f"git_status={run_payload['git'].get('status')}")
        else:
            run_payload["git"]["status"] = "skipped"
            progress.update(current=4, message="git step skipped")

        if args.dry_run:
            run_payload["status"] = "dry_run"
            run_payload["current_phase"] = "plan_ready"
            run_payload["recommended_action"] = "run_planned_commands"
        else:
            run_payload["status"] = "completed"
            run_payload["current_phase"] = "pages_ready"
            run_payload["recommended_action"] = "review_or_wait_for_publish_workflow"
        run_payload["finished_at"] = utc_now_iso()
        run_payload["source_head_sha"] = _git_head_sha()
        run_payload["highlights"] = [
            f"mode={args.mode}",
            f"odds_refresh={args.odds_refresh}",
            f"odds_provenance={run_payload['handoff'].get('odds_provenance', {}).get('source')}",
            f"prediction_file={run_payload['handoff'].get('prediction_file')}",
            f"odds_official_datetime_max={run_payload['handoff'].get('odds_official_datetime_max')}",
            f"policy_selected_rows={run_payload['handoff'].get('policy_selected_rows')}",
            f"target_page={run_payload['pages'].get('target_page')}",
            f"git_status={run_payload['git'].get('status')}",
        ]
        write_json(output_path, run_payload)
        progress.complete(message=f"manifest ready output={artifact_display_path(output_path, workspace_root=ROOT)}")
        return 0
    except KeyboardInterrupt:
        print("[netkeiba-2026-live-publish] interrupted by user")
        return 130
    except Exception as error:
        run_payload["status"] = "failed"
        run_payload["current_phase"] = "failed"
        run_payload["recommended_action"] = "inspect_live_publish_manifest"
        run_payload["error"] = str(error)
        run_payload["finished_at"] = utc_now_iso()
        write_json(output_path, run_payload)
        print(f"[netkeiba-2026-live-publish] failed: {error}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
