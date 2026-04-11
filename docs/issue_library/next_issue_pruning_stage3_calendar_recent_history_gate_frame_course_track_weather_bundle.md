# Next Issue: Pruning Stage 3 Calendar Recent-History Gate-Frame-Course Track-Weather Bundle Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

current staged pruning read はここまでで 3 つに分岐した。

- stage-1 `calendar + recent-history`:
  - supported
- stage-2 `calendar + recent-history + race-condition/dispatch context`:
  - hold
- stage-2 `calendar + recent-history + gate/frame/course core`:
  - supported

したがって next measurable hypothesis は、supported second-block line を起点に third block を 1 つだけ追加しても support が維持されるかを narrow に測ることにある。

今回の候補は `track / weather / surface context` 8 列である。

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
- candidate add-on block:
  - `track`
  - `weather`
  - `ground_condition`
  - `馬場状態2`
  - `芝・ダート区分`
  - `芝・ダート区分2`
  - `右左回り・直線区分`
  - `内・外・襷区分`

## Objective

supported stage-2 line に `track / weather / surface context` 8 列を加えた stage-3 bundle を formal compare し、context-sensitive pruning が third block まで伸ばせるかを判定する。

## Hypothesis

if `calendar + recent-history + gate/frame/course core` に `track`, `weather`, `ground_condition`, `馬場状態2`, `芝・ダート区分`, `芝・ダート区分2`, `右左回り・直線区分`, `内・外・襷区分` を追加除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then current staged simplification は supported third block まで進められる。

## Current Read

- `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_gate_frame_course_bundle.md` は supported second block として完了した
- `docs/issue_library/next_issue_track_weather_surface_context_ablation_audit.md` は individual audit で formal `pass / promote` と actual-date equivalence まで完了した
- `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md` は context block の一例として `hold` まで完了した

したがって next measurable hypothesis は、「supported な structural second block の上なら、別の supported context block を third block として足しても still viable か」である。

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

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather.yaml`

## In Scope

- stage-3 bundle 17 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- race-condition/dispatch block の再実行
- baseline config の即時 rewrite
- owner signal keep decision の再審
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 17 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、少なくとも stage-2 supported line との差を説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-3 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. formal compare
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 17 columns が十分に抜けず no-op が濃厚
- formal support が stage-2 supported line より明確に悪化し、third-block としての説明価値が薄い
- actual-date で September または December に regression が出る

## Notes

- この issue は supported second-block line の上に third block を足せるかを見る execution source である
- 現時点では baseline rewrite の承認を意味しない

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
  - `artifacts/reports/feature_gap_summary_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.csv`
  - `artifacts/reports/feature_gap_raw_column_coverage_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.csv`
- log:
  - `artifacts/logs/feature_gap_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.log`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `empty_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=92`
  - `categorical_feature_count=29`
  - `force_include_total=25`

read:

- stage-3 bundle config は clean に buildable
- target 17 columns を同時除外しても selected feature set は `92` まで縮み、no-op ではない
- low coverage / missing force include blocker は見えていない
- したがって next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
- metrics:
  - `auc=0.8398774194159776`
  - `logloss=0.202773648708527`
  - `best_iteration=521`
  - `feature_count=92`
  - `categorical_feature_count=29`

read:

- win component は clean に完走した
- manifest 上も target 17 columns は actual used set に残っていない
- leakage audit でも suspicious feature は検出されなかった

### ROI Component

- artifact:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
  - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
- metrics:
  - `top1_roi=0.8691751085383498`
  - `best_iteration=113`
  - `feature_count=92`
  - `categorical_feature_count=29`

read:

- ROI component も clean に完走した
- ROI 側でも target 17 columns は actual used set に残っていない
- leakage audit でも blocker は見えていない

### Stack Build

- artifact:
  - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
  - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
- metrics:
  - `component_count=2`

read:

- stack も clean に bundle できた
- stage-3 bundle は end-to-end で no-op ではなく、formal compare へ進める状態になった

## Formal Compare

- revision:
  - `r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1_wf_full_nested.json`
  - `auc=0.8422893707697219`
  - `top1_roi=0.8068584579976985`
  - `ev_top1_roi=0.6806214039125431`
  - `wf_nested_test_roi_weighted=0.8109090909090909`
  - `wf_nested_test_bets_total=935`
  - `stability_assessment=representative`
- WF feasibility:
  - `artifacts/reports/wf_feasibility_diag_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
  - `fold_count=5`
  - `feasible_fold_count=3`
  - `dominant_failure_reason=min_bets`
  - `min_bets_required_range=947..995`
- promotion gate:
  - `artifacts/reports/promotion_gate_r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1.json`
  - `status=pass`
  - `decision=promote`

read:

- supported second-block line の `3/5 feasible folds` は維持された
- `track/weather/surface context` を third block として足しても formal `pass / promote` は崩れなかった
- よって current staged simplification read は、supported structural line の上では third block まで伸ばせる可能性がある

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1` を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_base_vs_sep25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_sep25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_base_vs_sep25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_base_vs_sep25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
  - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_base_vs_dec25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_dec25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_base_vs_dec25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_base_vs_dec25_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
  - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `calendar + recent-history + gate/frame/course core + track/weather/surface context` は current staged simplification の supported third block と読める
- current staged reading は単純な ceiling ではなく、supported branch 上で段階的に block を追加できる branch-sensitive path と読むのが妥当である