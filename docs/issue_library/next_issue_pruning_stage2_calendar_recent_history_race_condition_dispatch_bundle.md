# Next Issue: Pruning Stage 2 Calendar Recent-History Race-Condition Bundle Audit

## Summary

stage-1 `calendar context + recent-history core` bundle は `r20260410_pruning_stage1_calendar_recent_history_v1` で formal `pass / promote` と actual-date Sep/Dec equivalence まで完了した。

次に問うべきは、ここへ `race-condition / dispatch context` 6 列を追加しても staged simplification の defendability が保たれるかである。

- stage-1 bundle:
  - `race_year`
  - `race_month`
  - `race_dayofweek`
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`
- candidate add-on block:
  - `競争条件`
  - `リステッド・重賞競走`
  - `障害区分`
  - `発走時刻`
  - `sex`
  - `東西・外国・地方区分`

one-shot pruning bundle は `0/5 feasible folds` で `hold` に留まったため、current staged path の価値は「どの追加 block までなら support が維持されるか」を narrow に積み上げて読むことにある。

## Objective

stage-1 bundle に race-condition / dispatch context 6 列を加えた stage-2 bundle を formal compare し、第二段階の staged simplification block が still viable か、それとも stage-1 から追加した categorical context が support を崩すかを判定する。

## Hypothesis

if stage-1 で支持された 5 列 bundle に `競争条件`, `リステッド・重賞競走`, `障害区分`, `発走時刻`, `sex`, `東西・外国・地方区分` を追加除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then staged simplification は second block まで安全に拡張できる。

## Current Read

- `docs/issue_library/next_issue_pruning_stage1_calendar_recent_history_bundle.md` は staged simplification の first executable block として supported まで完了した
- `docs/issue_library/next_issue_race_condition_dispatch_context_ablation_audit.md` は individual audit で formal `pass / promote` と actual-date equivalence まで完了した
- `docs/issue_library/next_issue_pruning_bundle_ablation_audit.md` は one-shot bundle を formal `hold` で閉じた
- stage-2 first read `artifacts/reports/feature_gap_summary_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json` では `priority_missing_raw_columns=[]`, `missing_force_include_features=[]`, `empty_force_include_features=[]`, `low_coverage_force_include_features=[]`, `selected_feature_count=98`, `categorical_feature_count=31` を確認した

したがって next measurable hypothesis は、「stage-1 supported block に structural risk の低い categorical context block を 1 つだけ足しても support が維持されるか」である。

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `race_year`
- `race_month`
- `race_dayofweek`
- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `競争条件`
- `リステッド・重賞競走`
- `障害区分`
- `発走時刻`
- `sex`
- `東西・外国・地方区分`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage2_calendar_recent_history_race_condition_dispatch.yaml`

## In Scope

- stage-2 bundle 11 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- full pruning package の再実行
- track/weather/surface context の同時追加
- baseline config の即時 rewrite
- owner signal keep decision の再審
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 11 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、少なくとも stage-1 からの deterioration が許容範囲かを説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-2 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. formal compare
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 11 columns が十分に抜けず no-op が濃厚
- formal support が stage-1 より明確に悪化し、追加 block の説明価値が薄い
- actual-date で September または December に regression が出る

## Notes

- この issue は stage-1 support を踏まえた second staged block の execution source である
- 現時点では baseline rewrite の承認を意味しない
- current status は `first read complete / heavy retrain pending`

## Acceptance Points

### Win Component

- artifact:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
- metrics:
  - `auc=0.8397916557003996`
  - `logloss=0.20282625196956816`
  - `best_iteration=526`
  - `feature_count=98`
  - `categorical_feature_count=31`

read:

- win component は clean に完走した
- manifest 上も target 11 columns は actual used set に残っていない
- leakage audit でも suspicious feature は検出されなかった

### ROI Component

- artifact:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
  - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
- metrics:
  - `top1_roi=0.8864254703328502`
  - `best_iteration=77`
  - `feature_count=98`
  - `categorical_feature_count=31`

read:

- ROI component も clean に完走した
- ROI 側でも target 11 columns は actual used set に残っていない
- leakage audit でも blocker は見えていない

### Stack Build

- artifact:
  - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
  - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
- metrics:
  - `component_count=2`

read:

- stack も clean に bundle できた
- stage-2 bundle は end-to-end で no-op ではなく、formal compare へ進める状態になった

## Formal Compare

- revision:
  - `r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1_wf_full_nested.json`
  - `auc=0.8422432477925081`
  - `top1_roi=0.8084810126582278`
  - `ev_top1_roi=0.7635097813578826`
  - `wf_nested_test_roi_weighted=1.0147154471544715`
  - `wf_nested_test_bets_total=1230`
  - `stability_assessment=representative`
- WF feasibility:
  - `artifacts/reports/wf_feasibility_diag_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
  - `fold_count=5`
  - `feasible_fold_count=0`
  - `dominant_failure_reason=min_bets`
  - `min_bets_required_range=947..995`
- promotion gate:
  - `artifacts/reports/promotion_gate_r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1.json`
  - `status=block`
  - `decision=hold`

read:

- top-line evaluate 自体は baseline 近辺を維持した
- ただし stage-1 の `3/5 feasible folds` と違い、race-condition / dispatch context を足した stage-2 は `0/5 feasible folds` へ戻った
- best fallback candidates は各 fold で positive ROI を示すが、`177..684 bets` に留まり ratio-bound `min_bets` を満たせない
- current gate 設定では stage-2 block は supported staged simplification とは読めない

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1` を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_pruning_stage2_calendar_recent_history_race_condition_dispatch_base_vs_sep25_pruning_stage2_calendar_recent_history_race_condition_dispatch_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_sep25_pruning_stage2_calendar_recent_history_race_condition_dispatch_base_vs_sep25_pruning_stage2_calendar_recent_history_race_condition_dispatch_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_pruning_stage2_calendar_recent_history_race_condition_dispatch_base_vs_sep25_pruning_stage2_calendar_recent_history_race_condition_dispatch_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
  - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_pruning_stage2_calendar_recent_history_race_condition_dispatch_base_vs_dec25_pruning_stage2_calendar_recent_history_race_condition_dispatch_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_dec25_pruning_stage2_calendar_recent_history_race_condition_dispatch_base_vs_dec25_pruning_stage2_calendar_recent_history_race_condition_dispatch_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_pruning_stage2_calendar_recent_history_race_condition_dispatch_base_vs_dec25_pruning_stage2_calendar_recent_history_race_condition_dispatch_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
  - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1` は actual-date replay では broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- それでも WF feasibility は `0/5 feasible folds` で、promotion gate は `status=block`, `decision=hold` だった
- したがって staged simplification の defendable boundary は current reading では stage-1 で止まり、race-condition / dispatch context を second block として足すのはまだ早い
- この結果は one-shot bundle hold と同じ failure class を、より narrow な second staged block で再現した artifact として retain する