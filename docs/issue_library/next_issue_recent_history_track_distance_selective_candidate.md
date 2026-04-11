# Next Issue: Recent History Track-Distance Selective Candidate

## Summary

`recent form / history` family は current JRA baseline の土台だが、現行 high-coverage line では `horse_last_3_avg_rank` と `horse_last_5_win_rate` に寄っており、course-conditioned な recent history はまだ selective child として切り分けていない。

builder には次の 2 列が既にある。

- `horse_track_distance_last_3_avg_rank`
- `horse_track_distance_last_5_win_rate`

これらは `fundamental_enriched` 系では force include されていたが、current high-coverage rich baseline では actual used set に入っていない。したがって次の history 仮説は broad rerun ではなく、この track-distance pair を narrow selective child として切るのが妥当である。

## Objective

current JRA high-coverage line に対して `horse_track_distance_last_3_avg_rank` と `horse_track_distance_last_5_win_rate` を narrow add-on し、`recent form / history` family の selective child として formal compare に載せる価値があるかを first read で判定する。

## Hypothesis

if horse-level recent history のうち track-distance conditioned pair だけを selective に追加する, then broad history widening をせずに、current baseline に対して course-conditioned recent-form signal を上積みできる可能性がある。

## Ready-To-Use Issue Draft

### Title

`[experiment] Recent history track-distance selective child read`

### Universe

`JRA`

### Category

`Feature`

## Current Read

- `feature_family_ranking.md` では `recent form / history` family は Tier B
- current high-coverage baseline は
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`
  を history core として使っている
- builder には track-distance conditioned pair が既にある
- `configs/features_catboost_fundamental_enriched.yaml` では
  - `horse_track_distance_last_3_avg_rank`
  - `horse_track_distance_last_5_win_rate`
  を force include していた
- ただし current high-coverage rich baseline ではこの pair は未採用で、独立仮説としてまだ読んでいない

## Candidate Definition

keep current JRA high-coverage core:

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `horse_surface_switch`
- `race_class_score`
- `horse_class_change`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`
- `owner_last_50_win_rate`

add selective pair:

- `horse_track_distance_last_3_avg_rank`
- `horse_track_distance_last_5_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml`

## In-Scope Surface

- `configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml`
- `configs/model_catboost_win_high_coverage_diag.yaml`
- `configs/model_lightgbm_roi_high_coverage_diag.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml`
- feature-gap / coverage read
- selective config 定義
- no-op risk の事前確認
- JRA true component retrain に進めるかの execution decision

## Non-Goals

- broad history family rerun
- new builder implementation
- policy family work
- NAR work
- actual-date role split の再議論

## Success Metrics

- track-distance pair の両方が `present=True`、`selected=True`、`non_null_ratio >= 0.90` を満たす
- selective child として actual used set に乗る見込みを説明できる
- no-op risk が低いと判断できる
- true component retrain に進めるか止めるかを 1 issue で確定できる

## Validation Plan

1. feature-gap / coverage を読み、2 本が `present=True` かつ low-coverage でないことを確認する
2. selected / force-include / non-null ratio を確認する
3. no-op risk が低ければ true component retrain に進む
4. formal compare では
   - `auc`
   - `top1_roi`
   - `ev_top1_roi`
   - nested WF shape
   - held-out formal `weighted_roi`
   - `bets / races / bet_rate`
   を baseline と比較する

## Validation Commands

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_feature_gap_report.py \
  --config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml
  --model-config configs/model_catboost_win_high_coverage_diag.yaml \
  --template-config configs/data_netkeiba_template.yaml \
  --summary-output artifacts/reports/feature_gap_summary_recent_history_track_distance_selective_<revision>.json \
  --feature-output artifacts/reports/feature_gap_feature_coverage_recent_history_track_distance_selective_<revision>.csv \
  --raw-output artifacts/reports/feature_gap_raw_column_coverage_recent_history_track_distance_selective_<revision>.csv

/workspaces/nr-learn/.venv/bin/python scripts/run_train.py \
  --config configs/model_catboost_win_high_coverage_diag.yaml \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml \
  --artifact-suffix r<revision>

/workspaces/nr-learn/.venv/bin/python scripts/run_train.py \
  --config configs/model_lightgbm_roi_high_coverage_diag.yaml \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml \
  --artifact-suffix r<revision>

/workspaces/nr-learn/.venv/bin/python scripts/run_build_value_stack.py \
  --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml \
  --artifact-suffix r<revision>

/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml \
  --revision r<revision> \
  --skip-train \
  --train-artifact-suffix r<revision> \
  --evaluate-model-artifact-suffix r<revision> \
  --evaluate-pre-feature-max-rows 300000 \
  --evaluate-max-rows 200000 \
  --evaluate-wf-mode full \
  --evaluate-wf-scheme nested
```

## Expected Artifacts

- `artifacts/reports/feature_gap_summary_recent_history_track_distance_selective_<revision>.json`
- `artifacts/reports/feature_gap_feature_coverage_recent_history_track_distance_selective_<revision>.csv`
- `artifacts/reports/feature_gap_raw_column_coverage_recent_history_track_distance_selective_<revision>.csv`
- `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_<revision>.json`
- `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_<revision>.json`
- `artifacts/reports/evaluation_summary_*_<revision>.json`
- `artifacts/reports/revision_gate_<revision>.json`
- `artifacts/reports/promotion_gate_<revision>.json`

## Stop Condition

- track-distance pair のどちらかが missing または low coverage
- selected set に乗らず no-op が濃厚
- Tier B history child として独立性が弱い

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
  - `artifacts/reports/feature_gap_summary_recent_history_track_distance_selective_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_recent_history_track_distance_selective_v1.csv`
  - `artifacts/reports/feature_gap_raw_column_coverage_recent_history_track_distance_selective_v1.csv`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `empty_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `template_columns_present=32/32`
  - `force_include_total=36`
  - `selected_feature_count=109`
  - `categorical_feature_count=37`
- focal coverage:
  - `horse_track_distance_last_3_avg_rank`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=1.0`
    - `status=ok`
  - `horse_track_distance_last_5_win_rate`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=1.0`
    - `status=ok`

interpretation:

- track-distance pair は current high-coverage line 上で clean に build / select される
- no-op でも low-coverage でもない
- `recent form / history` family の narrow child として true component retrain に進めてよい

## Execution Note

- compare は `value_blend` family の base stack config を使い、component train と stack build を先に済ませたうえで `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix r<revision>` を使う。
- suffixed stack config を直接 `revision_gate` に渡すのではなく、base stack config と feature config の組で compare する。
- next acceptance point は true component retrain と formal compare の完了であり、actual-date role split はその後の別 issue に切る。
