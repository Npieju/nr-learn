# Next Issue: Pruning Stage 5 Calendar Recent-History Gate-Frame-Course Track-Weather Class-Rest-Surface JT-ID Bundle Audit

## Summary

current staged pruning read はここまでで 5 つに分岐した。

- stage-1 `calendar + recent-history`:
  - supported
- stage-2 `calendar + recent-history + race-condition/dispatch context`:
  - hold
- stage-2 `calendar + recent-history + gate/frame/course core`:
  - supported
- stage-3 `calendar + recent-history + gate/frame/course core + track/weather/surface context`:
  - supported
- stage-4 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core`:
  - supported, but support margin is thinner

したがって next measurable hypothesis は、supported stage-4 line を起点に smallest remaining fifth block を 1 つだけ追加しても support が維持されるかを narrow に測ることにある。

今回の候補は `jockey / trainer ID core` 2 列である。

- supported base line:
  - `race_year`
  - `race_month`
  - `race_dayofweek`
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`
  - `gate_ratio`
  - `frame_ratio`
  - `course_gate_bucket_last_100_win_rate`
  - `course_gate_bucket_last_100_avg_rank`
  - `track`
  - `weather`
  - `ground_condition`
  - `馬場状態2`
  - `芝・ダート区分`
  - `芝・ダート区分2`
  - `右左回り・直線区分`
  - `内・外・襷区分`
  - `horse_days_since_last_race`
  - `horse_days_since_last_race_log1p`
  - `horse_is_short_turnaround`
  - `horse_is_long_layoff`
  - `horse_weight_change`
  - `horse_weight_change_abs`
  - `horse_distance_change`
  - `horse_distance_change_abs`
  - `horse_surface_switch`
  - `horse_surface_switch_short_turnaround`
  - `horse_surface_switch_long_layoff`
  - `race_class_score`
  - `horse_last_class_score`
  - `horse_class_change`
  - `horse_is_class_up`
  - `horse_is_class_down`
  - `horse_class_up_short_turnaround`
  - `horse_class_down_short_turnaround`
  - `horse_class_up_long_layoff`
  - `horse_class_down_long_layoff`
- candidate add-on block:
  - `jockey_id`
  - `trainer_id`

## Objective

supported stage-4 line に `jockey / trainer ID core` 2 列を加えた stage-5 bundle を formal compare し、branch-sensitive staged simplification が fifth block まで伸ばせるかを判定する。

## Hypothesis

if `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core` に `jockey_id`, `trainer_id` を追加除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then current staged simplification は supported fifth block まで進められる。

## Current Read

- `docs/issue_library/next_issue_pruning_stage4_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_bundle.md` は supported fourth block として完了した
- `docs/issue_library/next_issue_jockey_trainer_id_core_ablation_audit.md` は individual audit で formal `pass / promote` と actual-date equivalence まで完了した
- `docs/issue_library/next_issue_owner_signal_ablation_audit.md` は historical keep / hold reference であり、next block 候補からは外す

したがって next measurable hypothesis は、「thinner margin の supported branch の上でも smallest remaining categorical identity block なら still viable か」である。

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `race_year`
- `race_month`
- `race_dayofweek`
- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`
- `track`
- `weather`
- `ground_condition`
- `馬場状態2`
- `芝・ダート区分`
- `芝・ダート区分2`
- `右左回り・直線区分`
- `内・外・襷区分`
- `horse_days_since_last_race`
- `horse_days_since_last_race_log1p`
- `horse_is_short_turnaround`
- `horse_is_long_layoff`
- `horse_weight_change`
- `horse_weight_change_abs`
- `horse_distance_change`
- `horse_distance_change_abs`
- `horse_surface_switch`
- `horse_surface_switch_short_turnaround`
- `horse_surface_switch_long_layoff`
- `race_class_score`
- `horse_last_class_score`
- `horse_class_change`
- `horse_is_class_up`
- `horse_is_class_down`
- `horse_class_up_short_turnaround`
- `horse_class_down_short_turnaround`
- `horse_class_up_long_layoff`
- `horse_class_down_long_layoff`
- `jockey_id`
- `trainer_id`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id.yaml`

## In Scope

- stage-5 bundle 39 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- owner signal の再審
- baseline config の即時 rewrite
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 39 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、少なくとも stage-4 supported line との差を説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-5 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. formal compare
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 39 columns が十分に抜けず no-op が濃厚
- formal support が stage-4 supported line より明確に悪化し、fifth-block としての説明価値が薄い
- actual-date で September または December に regression が出る

## Notes

- この issue は supported stage-4 line の上に fifth block を足せるかを見る execution source である
- 現時点では baseline rewrite の承認を意味しない

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
  - `artifacts/reports/feature_gap_summary_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.csv`
  - `artifacts/reports/feature_gap_raw_column_coverage_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.csv`
- log:
  - `artifacts/logs/feature_gap_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.log`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `empty_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=70`
  - `categorical_feature_count=27`
  - `force_include_total=5`

read:

- stage-5 bundle config は clean に buildable
- target 39 columns を同時除外しても selected feature set は `70` まで縮み、no-op ではない
- low coverage / missing force include blocker は見えていない
- したがって next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
- metrics:
  - `auc=0.8395652657680374`
  - `logloss=0.20287809560154038`
  - `best_iteration=479`
  - `feature_count=70`
  - `categorical_feature_count=27`

read:

- win component は clean に完走した
- manifest 上も target 39 columns は actual used set に残っていない
- leakage audit でも suspicious feature は検出されなかった

### ROI Component

- artifact:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
  - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
- metrics:
  - `top1_roi=0.9270911722141819`
  - `best_iteration=117`
  - `feature_count=70`
  - `categorical_feature_count=27`

read:

- ROI component も clean に完走した
- ROI 側でも target 39 columns は actual used set に残っていない
- leakage audit でも blocker は見えていない

### Stack Build

- artifact:
  - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
  - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
- metrics:
  - `component_count=2`

read:

- stack も clean に bundle できた
- stage-5 bundle は end-to-end で no-op ではなく、formal compare へ進める状態になった

## Formal Compare

- revision:
  - `r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1_wf_full_nested.json`
  - `auc=0.8419342099107424`
  - `top1_roi=0.8033716915995397`
  - `ev_top1_roi=0.7088952819332566`
  - `wf_nested_test_roi_weighted=1.0510135135135137`
  - `wf_nested_test_bets_total=296`
  - `stability_assessment=representative`
- WF feasibility:
  - `artifacts/reports/wf_feasibility_diag_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
  - `fold_count=5`
  - `feasible_fold_count=3`
  - `dominant_failure_reason=min_bets`
  - `min_bets_required_range=947..995`
- promotion gate:
  - `artifacts/reports/promotion_gate_r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1.json`
  - `status=pass`
  - `decision=promote`

read:

- stage-5 は stage-4 より formal support が改善し、`3/5 feasible folds` に戻った
- `jockey_id / trainer_id` を fifth block として足しても formal `pass / promote` は維持された
- よって supported branch は stage-5 まで延長できる

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1` を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_base_vs_sep25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_sep25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_base_vs_sep25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_base_vs_sep25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
  - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_base_vs_dec25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_dec25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_base_vs_dec25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_base_vs_dec25_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
  - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core` は current staged simplification の supported fifth block と読める
- stage-4 の一時的な薄化は見えたが、smallest remaining identity block では support が `3/5 feasible folds` に戻った