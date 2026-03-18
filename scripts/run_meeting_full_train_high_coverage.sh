#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
LOG_DIR="$ROOT/artifacts/logs"
RUNTIME_CFG_DIR="$ROOT/artifacts/runtime_configs"
STATUS_FILE="$LOG_DIR/meeting_full_train_high_coverage_current.status"
mkdir -p "$LOG_DIR"
mkdir -p "$RUNTIME_CFG_DIR"
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="$LOG_DIR/meeting_full_train_high_coverage_${RUN_TS}.log"
HEARTBEAT_INTERVAL_SEC="${HEARTBEAT_INTERVAL_SEC:-60}"

export MALLOC_ARENA_MAX=2

exec > >(tee -a "$LOG_FILE") 2>&1

log() {
  printf '[meeting-run %s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*"
}

update_status() {
  local state="$1"
  shift
  local detail="$*"
  cat > "$STATUS_FILE" <<EOF
timestamp=$(date '+%Y-%m-%d %H:%M:%S')
run_ts=$RUN_TS
log_file=$LOG_FILE
state=$state
detail=$detail
EOF
}

show_previous_log_snapshot() {
  local prev_log=""
  prev_log="$(ls -t "$LOG_DIR"/meeting_full_train_high_coverage_*.log 2>/dev/null | grep -v -F "$LOG_FILE" | head -n 1 || true)"
  if [[ -z "$prev_log" ]]; then
    log "preflight previous_log=none"
    return 0
  fi

  local last_ts=""
  last_ts="$(grep -oE '\[meeting-run [0-9-]{10} [0-9:]{8}\]' "$prev_log" | tail -n 1 | sed -E 's/^\[meeting-run ([0-9-]{10} [0-9:]{8})\]$/\1/' || true)"
  if [[ -n "$last_ts" ]]; then
    local now_epoch=0
    local last_epoch=0
    local delta=0
    now_epoch=$(date +%s)
    last_epoch=$(date -d "$last_ts" +%s)
    delta=$((now_epoch - last_epoch))
    log "preflight previous_log=$prev_log last_ts=$last_ts idle=${delta}s"
  else
    log "preflight previous_log=$prev_log last_ts=unknown"
  fi

  log "preflight previous_log_tail_begin"
  tail -n 30 "$prev_log" || true
  log "preflight previous_log_tail_end"
}

run_monitored_command() {
  local step_name="$1"
  shift
  local start_epoch=0
  local now_epoch=0
  local elapsed=0

  log "START ${step_name}"
  update_status "running" "$step_name"
  "$@" &
  local cmd_pid=$!
  start_epoch=$(date +%s)

  while kill -0 "$cmd_pid" 2>/dev/null; do
    sleep "$HEARTBEAT_INTERVAL_SEC"
    if kill -0 "$cmd_pid" 2>/dev/null; then
      now_epoch=$(date +%s)
      elapsed=$((now_epoch - start_epoch))
      log "HEARTBEAT ${step_name} elapsed=${elapsed}s pid=${cmd_pid}"
      update_status "running" "${step_name} elapsed=${elapsed}s pid=${cmd_pid}"
    fi
  done

  wait "$cmd_pid"
  local exit_code=$?
  now_epoch=$(date +%s)
  elapsed=$((now_epoch - start_epoch))
  if [[ "$exit_code" -ne 0 ]]; then
    log "FAIL ${step_name} exit_code=${exit_code} elapsed=${elapsed}s"
    update_status "failed" "${step_name} exit_code=${exit_code} elapsed=${elapsed}s"
    return "$exit_code"
  fi

  log "DONE ${step_name} elapsed=${elapsed}s"
  update_status "running" "completed ${step_name} elapsed=${elapsed}s"
}

run_step() {
  local step_name="$1"
  shift
  run_monitored_command "$step_name" "$@"
}

train_component_with_fallback() {
  local step_name="$1"
  local feature_config="$2"
  shift 2

  local config_path=""
  local exit_code=1
  SELECTED_COMPONENT_CONFIG=""
  for config_path in "$@"; do
    if run_monitored_command "${step_name} config=${config_path}" "$PY" scripts/run_train.py \
      --config "$config_path" \
      --data-config configs/data.yaml \
      --feature-config "$feature_config"; then
      SELECTED_COMPONENT_CONFIG="$config_path"
      return 0
    else
      exit_code=$?
      log "WAIT ${step_name} before next fallback"
      sleep 10
    fi
  done

  return "$exit_code"
}

write_runtime_stack_config() {
  local base_config="$1"
  local output_config="$2"
  local win_config="$3"
  local roi_config="$4"
  local model_file="$5"
  local report_file="$6"
  local manifest_file="$7"

  sed \
    -e "s|^  win: \".*\"$|  win: \"${win_config}\"|" \
    -e "s|^  roi: \".*\"$|  roi: \"${roi_config}\"|" \
    -e "s|^  model_file: \".*\"$|  model_file: \"${model_file}\"|" \
    -e "s|^  report_file: \".*\"$|  report_file: \"${report_file}\"|" \
    -e "s|^  manifest_file: \".*\"$|  manifest_file: \"${manifest_file}\"|" \
    "$base_config" > "$output_config"
}

cd "$ROOT"

on_exit() {
  local exit_code=$?
  if [[ "$exit_code" -eq 0 ]]; then
    update_status "completed" "meeting batch finished successfully"
  else
    update_status "failed" "meeting batch failed exit_code=${exit_code}"
    log "SCRIPT_EXIT exit_code=${exit_code}"
  fi
}
trap on_exit EXIT

update_status "starting" "initializing meeting batch"

log "root=$ROOT"
log "log_file=$LOG_FILE"
log "goal=roi-focused staged retrain + stack rebuild + recent nested evaluation"
log "heartbeat_interval_sec=$HEARTBEAT_INTERVAL_SEC"
show_previous_log_snapshot

# CatBoost win retrain exceeded host memory at full, 900k, and unique 700k caps.
# Reuse the current high-coverage win artifact and spend meeting time on ROI expansion instead.
WIN_COMPONENT_CONFIG="configs/model_catboost_win_high_coverage_diag.yaml"
log "reuse_win_component=${WIN_COMPONENT_CONFIG}"

train_component_with_fallback \
  "train lightgbm roi" \
  configs/features_catboost_rich_high_coverage_diag.yaml \
  configs/model_lightgbm_roi_high_coverage_full.yaml \
  configs/model_lightgbm_roi_high_coverage_fullsafe.yaml \
  configs/model_lightgbm_roi_high_coverage_fullfallback.yaml
ROI_COMPONENT_CONFIG="$SELECTED_COMPONENT_CONFIG"

log "selected_win_config=${WIN_COMPONENT_CONFIG}"
log "selected_roi_config=${ROI_COMPONENT_CONFIG}"

ROI012_STACK_CONFIG="$RUNTIME_CFG_DIR/model_catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_roi012.yaml"
LIQUIDITY_STACK_CONFIG="$RUNTIME_CFG_DIR/model_catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_liquidity.yaml"

write_runtime_stack_config \
  "$ROOT/configs/model_catboost_value_stack_lgbm_roi_high_coverage_full_tune_roi012.yaml" \
  "$ROI012_STACK_CONFIG" \
  "$WIN_COMPONENT_CONFIG" \
  "$ROI_COMPONENT_CONFIG" \
  "catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_roi012_model.joblib" \
  "train_metrics_catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_roi012.json" \
  "catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_roi012_model.manifest.json"

write_runtime_stack_config \
  "$ROOT/configs/model_catboost_value_stack_lgbm_roi_high_coverage_full_tune_roi012_liquidity.yaml" \
  "$LIQUIDITY_STACK_CONFIG" \
  "$WIN_COMPONENT_CONFIG" \
  "$ROI_COMPONENT_CONFIG" \
  "catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_liquidity_model.joblib" \
  "train_metrics_catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_liquidity.json" \
  "catboost_value_stack_lgbm_roi_high_coverage_meeting_${RUN_TS}_liquidity_model.manifest.json"

log "runtime_roi012_stack_config=${ROI012_STACK_CONFIG}"
log "runtime_liquidity_stack_config=${LIQUIDITY_STACK_CONFIG}"

run_step \
  "build stack roi012" \
  "$PY" scripts/run_build_value_stack.py \
  --config "$ROI012_STACK_CONFIG" \
  --data-config configs/data.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml

run_step \
  "build stack liquidity" \
  "$PY" scripts/run_build_value_stack.py \
  --config "$LIQUIDITY_STACK_CONFIG" \
  --data-config configs/data.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml

run_step \
  "evaluate stack roi012 recent nested" \
  "$PY" scripts/run_evaluate.py \
  --config "$ROI012_STACK_CONFIG" \
  --data-config configs/data.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-09-30 \
  --wf-mode full \
  --wf-scheme nested

run_step \
  "evaluate stack liquidity recent nested" \
  "$PY" scripts/run_evaluate.py \
  --config "$LIQUIDITY_STACK_CONFIG" \
  --data-config configs/data.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-09-30 \
  --wf-mode full \
  --wf-scheme nested

run_step \
  "probe liquidity fold1 recent" \
  "$PY" scripts/run_wf_liquidity_probe.py \
  --config "$LIQUIDITY_STACK_CONFIG" \
  --data-config configs/data.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --start-date 2024-01-01 \
  --end-date 2024-09-30 \
  --folds 1 \
  --blend-weights 0.8 \
  --portfolio-min-probs 0.05,0.04,0.03 \
  --portfolio-min-evs 1.0,0.98,0.95 \
  --kelly-min-probs 0.05,0.04,0.03 \
  --kelly-min-edges 0.01,0.005,0.0

log "artifacts_expected=artifacts/models/*meeting_${RUN_TS}* artifacts/reports/evaluation_summary_*meeting_${RUN_TS}_*"
log "completed successfully"