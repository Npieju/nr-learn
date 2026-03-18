#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="$ROOT/artifacts/logs"
STATUS_FILE="$LOG_DIR/meeting_full_train_high_coverage_current.status"
TAIL_LINES=60

while [[ "$#" -gt 0 ]]; do
  case "$1" in
    --tail-lines)
      TAIL_LINES="${2:-60}"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

echo "[meeting-status] now=$(date '+%Y-%m-%d %H:%M:%S %z')"

if [[ -f "$STATUS_FILE" ]]; then
  echo "[meeting-status] status_file=$STATUS_FILE"
  sed 's/^/[meeting-status] /' "$STATUS_FILE"
else
  echo "[meeting-status] status_file=missing"
fi

echo "[meeting-status] running_processes_begin"
ps -ef | grep -E 'run_meeting_full_train_high_coverage.sh|scripts/run_train.py --config configs/model_lightgbm_roi_high_coverage|scripts/run_evaluate.py --config .*high_coverage_meeting_|scripts/run_wf_liquidity_probe.py --config .*high_coverage_meeting_' | grep -v grep || true
echo "[meeting-status] running_processes_end"

latest_log="$(ls -t "$LOG_DIR"/meeting_full_train_high_coverage_*.log 2>/dev/null | head -n 1 || true)"
if [[ -z "$latest_log" ]]; then
  echo "[meeting-status] latest_log=missing"
  exit 0
fi

echo "[meeting-status] latest_log=$latest_log"
last_ts="$(grep -oE '\[meeting-run [0-9-]{10} [0-9:]{8}\]' "$latest_log" | tail -n 1 | sed -E 's/^\[meeting-run ([0-9-]{10} [0-9:]{8})\]$/\1/' || true)"
if [[ -n "$last_ts" ]]; then
  now_epoch=$(date +%s)
  last_epoch=$(date -d "$last_ts" +%s)
  delta=$((now_epoch - last_epoch))
  echo "[meeting-status] last_meeting_ts=$last_ts"
  echo "[meeting-status] idle_seconds=$delta"
  echo "[meeting-status] idle_hms=$((delta/3600))h$(((delta%3600)/60))m$((delta%60))s"
else
  echo "[meeting-status] last_meeting_ts=unknown"
fi

echo "[meeting-status] latest_log_tail_begin"
tail -n "$TAIL_LINES" "$latest_log" || true
echo "[meeting-status] latest_log_tail_end"
