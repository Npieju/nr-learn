# Next Issue: Pruning Stage 2 Calendar Recent-History Gate-Frame-Course Bundle Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

stage-1 `calendar context + recent-history core` bundle は `r20260410_pruning_stage1_calendar_recent_history_v1` で formal `pass / promote` と actual-date Sep/Dec equivalence まで完了した。

一方、stage-2 として `race-condition / dispatch context` を足した line は actual-date 同値を保ちながらも `0/5 feasible folds` で `hold` に戻った。

したがって next measurable hypothesis は、「stage-1 に足す second block が悪かったのか、それとも stage-1 を超えた時点で support が崩れるのか」を narrow に切り分けることにある。

今回の候補は `gate / frame / course core` 4 列である。

- stage-1 bundle:
  - `race_year`
  - `race_month`
  - `race_dayofweek`
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`
- candidate add-on block:
  - `gate_ratio`
  - `frame_ratio`
  - `course_gate_bucket_last_100_win_rate`
  - `course_gate_bucket_last_100_avg_rank`

## Objective

stage-1 bundle に `gate / frame / course core` 4 列を加えた alternative stage-2 bundle を formal compare し、`race-condition / dispatch context` 以外の second block なら defendability を維持できるかを判定する。

## Hypothesis

if stage-1 に `gate_ratio`, `frame_ratio`, `course_gate_bucket_last_100_win_rate`, `course_gate_bucket_last_100_avg_rank` を追加除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then current staged simplification boundary は「stage-1まで」ではなく「context block の選び方次第で second block まで進める」である。

## Current Read

- `docs/issue_library/next_issue_pruning_stage1_calendar_recent_history_bundle.md` は staged simplification の first executable block として supported まで完了した
- `docs/issue_library/next_issue_gate_frame_course_core_ablation_audit.md` は individual audit で formal `pass / promote` と actual-date equivalence まで完了した
- `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md` は second block alternative として formal `hold` まで完了した

したがって next measurable hypothesis は、「stage-1 supported block に lower-risk structural block を 1 つだけ足しても support が維持されるか」である。

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

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage2_calendar_recent_history_gate_frame_course.yaml`

## In Scope

- stage-2 bundle 9 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- full pruning package の再実行
- race-condition/dispatch block の再実行
- baseline config の即時 rewrite
- owner signal keep decision の再審
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 9 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、少なくとも stage-1 / stage-2-race-condition の差を説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-2 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. formal compare
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 9 columns が十分に抜けず no-op が濃厚
- formal support が stage-1 より明確に悪化し、alternative second-block としての説明価値が薄い
- actual-date で September または December に regression が出る

## Notes

- この issue は stage-1 support と stage-2 race-condition hold のあいだを切り分ける alternative second-block execution source である
- 現時点では baseline rewrite の承認を意味しない

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
  - `artifacts/reports/feature_gap_summary_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_pruning_stage2_calendar_recent_history_gate_frame_course_v1.csv`
  - `artifacts/reports/feature_gap_raw_column_coverage_pruning_stage2_calendar_recent_history_gate_frame_course_v1.csv`
- log:
  - `artifacts/logs/feature_gap_pruning_stage2_calendar_recent_history_gate_frame_course_v1.log`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `empty_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=100`
  - `categorical_feature_count=37`
  - `force_include_total=25`

read:

- alternative stage-2 bundle config は clean に buildable
- target 9 columns を同時除外しても selected feature set は `100` まで縮み、no-op ではない
- low coverage / missing force include blocker は見えていない
- したがって next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
- metrics:
  - `auc=0.8396215721904369`
  - `logloss=0.2027988430335601`
  - `best_iteration=536`
  - `feature_count=100`
  - `categorical_feature_count=37`

read:

- win component は clean に完走した
- manifest 上も target 9 columns は actual used set に残っていない
- leakage audit でも suspicious feature は検出されなかった

### ROI Component

- artifact:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
  - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
- metrics:
  - `top1_roi=0.8885962373371915`
  - `best_iteration=110`
  - `feature_count=100`
  - `categorical_feature_count=37`

read:

- ROI component も clean に完走した
- ROI 側でも target 9 columns は actual used set に残っていない
- leakage audit でも blocker は見えていない

### Stack Build

- artifact:
  - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
  - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
- metrics:
  - `component_count=2`

read:

- stack も clean に bundle できた
- alternative stage-2 bundle は end-to-end で no-op ではなく、formal compare へ進める状態になった

## Formal Compare

- revision:
  - `r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1_wf_full_nested.json`
  - `auc=0.8420815082342571`
  - `top1_roi=0.8077445339470656`
  - `ev_top1_roi=0.7973993095512083`
  - `wf_nested_test_roi_weighted=0.9506373117033606`
  - `wf_nested_test_bets_total=863`
  - `stability_assessment=representative`
- WF feasibility:
  - `artifacts/reports/wf_feasibility_diag_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
  - `fold_count=5`
  - `feasible_fold_count=3`
  - `dominant_failure_reason=min_bets`
  - `min_bets_required_range=947..995`
- promotion gate:
  - `artifacts/reports/promotion_gate_r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1.json`
  - `status=pass`
  - `decision=promote`

read:

- stage-1 の `3/5 feasible folds` は維持された
- race-condition / dispatch context を足した stage-2 と違い、gate/frame/course core を足した alternative stage-2 は `pass / promote` を通過した
- よって current staged simplification boundary は「stage-1 が上限」ではなく、「second block の選び方で support が分岐する」と読むのが正しい

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1` を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_pruning_stage2_calendar_recent_history_gate_frame_course_base_vs_sep25_pruning_stage2_calendar_recent_history_gate_frame_course_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_sep25_pruning_stage2_calendar_recent_history_gate_frame_course_base_vs_sep25_pruning_stage2_calendar_recent_history_gate_frame_course_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_pruning_stage2_calendar_recent_history_gate_frame_course_base_vs_sep25_pruning_stage2_calendar_recent_history_gate_frame_course_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
  - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_pruning_stage2_calendar_recent_history_gate_frame_course_base_vs_dec25_pruning_stage2_calendar_recent_history_gate_frame_course_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_dec25_pruning_stage2_calendar_recent_history_gate_frame_course_base_vs_dec25_pruning_stage2_calendar_recent_history_gate_frame_course_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_pruning_stage2_calendar_recent_history_gate_frame_course_base_vs_dec25_pruning_stage2_calendar_recent_history_gate_frame_course_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
  - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `calendar + recent-history + gate/frame/course core` は current staged simplification の supported second block と読める
- 同日の `race-condition / dispatch context` second block が `hold` だったことから、current staged reading は「second block の可否は block-sensitive」である