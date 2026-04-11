# Next Issue: Race Condition Dispatch Context Ablation Audit

## Summary

`base race context` family では calendar context 3 列と track / weather / surface-layout context 8 列を pruning candidate と判断できたが、race-condition / dispatch categorical subgroup はまだ formal に監査していない。

current high-coverage baseline では、次の categorical context 群が still remaining conditioning 土台として残っている。

- `競争条件`
- `リステッド・重賞競走`
- `障害区分`
- `発走時刻`
- `sex`
- `東西・外国・地方区分`

これらは broad base race context の残り subgroup であり、current serving family の必須 core かどうかはまだ artifact ベースで閉じていない。

## Objective

上記 6 列を current JRA high-coverage baseline から外した selective ablation を formal compare し、race-condition / dispatch context が current serving family の必須 core か、それとも pruning candidate かを判定する。

## Hypothesis

if race-condition / dispatch context 6 列を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then this subgroup は current serving family の必須 core ではなく pruning candidate とみなせる。

## Current Read

- `feature_family_ranking.md` では `base race context` は Tier B
- calendar context 3 列は pruning candidate judgment まで完了した
- track / weather / surface-layout context 8 列も pruning candidate judgment まで完了した
- したがって next narrow JRA hypothesis は、base race context に残る race-condition / dispatch subgroup の keep vs prune judgment を閉じることになる

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only the 6 categorical context columns listed above.

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_race_condition_dispatch_context_ablation.yaml`

## In Scope

- race-condition / dispatch context 6 列だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow

## Non-Goals

- calendar context 3 列の再実行
- track / weather / surface-layout context 8 列の再実行
- base race context family 全体の broad redesign
- policy rewrite
- NAR work

## Success Metrics

- ablation config が clean に buildable
- win / ROI component の actual used feature set から focal 6 列が消える見込みを説明できる
- formal compare に進めるか、first read で止めるかを 1 issue で確定できる

## Validation Plan

1. feature-gap / coverage を読み、ablation config 自体が clean に buildable であることを確認する
2. selected feature count と focal 6 列の扱いを確認する
3. no-op risk が低ければ true component retrain に進む
4. formal compare では
   - `auc`
   - `top1_roi`
   - `ev_top1_roi`
   - nested WF shape
   - held-out formal `weighted_roi`
   - `bets / races / bet_rate`
   を baseline と比較する

## Stop Condition

- ablation config が clean に buildable でない
- selected set 上 no-op が濃厚で analysis value が薄い
- base race context の残り subgroup を外して読む意味が弱いと判明する

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
   - `artifacts/reports/feature_gap_summary_race_condition_dispatch_context_ablation_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_race_condition_dispatch_context_ablation_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_race_condition_dispatch_context_ablation_v1.csv`
- log:
   - `artifacts/logs/feature_gap_race_condition_dispatch_context_ablation_v1.log`
- summary:
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `empty_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `selected_feature_count=103`
   - `categorical_feature_count=31`
   - `force_include_total=34`

interpretation:

- ablation config 自体は clean に buildable
- focal 6 列を除いた状態でも selected feature set は `103`、categorical feature set は `31` まで縮み、current baseline 比で no-op ではない
- したがって next step は JRA true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
   - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_race_condition_dispatch_context_ablation_v1.json`
   - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_race_condition_dispatch_context_ablation_v1.json`
- metrics:
   - `auc=0.8396613060398579`
   - `logloss=0.20280944560707406`
   - `best_iteration=529`
   - `feature_count=103`
   - `categorical_feature_count=31`

read:

- win component は clean に完走した
- manifest には race-condition/dispatch context 6 列が残っていない
- race-condition/dispatch context ablation は win side で no-op ではない

### ROI Component

- artifact:
   - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_race_condition_dispatch_context_ablation_v1.json`
   - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_race_condition_dispatch_context_ablation_v1.json`
- metrics:
   - `top1_roi=1.0128219971056431`
   - `best_iteration=128`
   - `feature_count=103`
   - `categorical_feature_count=31`

read:

- ROI component も clean に完走した
- ROI manifest にも focal 6 列は残っていない
- ablation candidate は component 2 本とも true retrain が成立した

### Stack Build

- artifact:
   - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_race_condition_dispatch_context_ablation_v1.json`
   - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_race_condition_dispatch_context_ablation_v1.json`
- metrics:
   - `component_count=2`
   - `feature_count=103`
   - `categorical_feature_count=31`

read:

- stack も clean に bundle できた
- stack manifest にも focal 6 列は残っておらず、race-condition/dispatch context ablation は end-to-end で no-op ではない

## Formal Compare

- revision:
   - `r20260410_race_condition_dispatch_context_ablation_v1`
- evaluation summary:
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_race_condition_dispatch_context_ablation_v1_wf_full_nested.json`
   - `auc=0.8377720856205183`
   - `top1_roi=0.8002901353965184`
   - `ev_top1_roi=0.7449295385465599`
   - `wf_nested_test_roi_weighted=1.0027554535017225`
   - `wf_nested_test_bets_total=871`
   - `stability_assessment=representative`
- promotion gate:
   - `artifacts/reports/promotion_gate_r20260410_race_condition_dispatch_context_ablation_v1.json`
   - `status=pass`
   - `decision=promote`
   - `weighted_roi=1.0027554535017225`
   - `bets_total=871`
   - `feasible_fold_count=2`
- revision gate:
   - `artifacts/reports/revision_gate_r20260410_race_condition_dispatch_context_ablation_v1.json`
   - `status=pass`
   - `decision=promote`

read:

- evaluation top-line は baseline 近辺を維持した
- formal support は `feasible_fold_count=2/5` と薄めだが、`weighted_roi=1.0027554535017225` で current gate 設定では `pass / promote` を通過した
- current baseline core family の keep vs prune judgment を閉じるには、actual-date role split を追加して operational delta を確認する必要がある

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_race_condition_dispatch_context_ablation_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_sep25_race_condition_dispatch_context_base_vs_sep25_race_condition_dispatch_context_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_sep25_race_condition_dispatch_context_base_vs_sep25_race_condition_dispatch_context_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_race_condition_dispatch_context_base_vs_sep25_race_condition_dispatch_context_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
   - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_dec25_race_condition_dispatch_context_base_vs_dec25_race_condition_dispatch_context_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_dec25_race_condition_dispatch_context_base_vs_dec25_race_condition_dispatch_context_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_race_condition_dispatch_context_base_vs_dec25_race_condition_dispatch_context_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
   - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_race_condition_dispatch_context_ablation_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `競争条件`, `リステッド・重賞競走`, `障害区分`, `発走時刻`, `sex`, `東西・外国・地方区分` は current JRA high-coverage serving family の必須 core とは言えず、pruning candidate judgment は完了した
- baseline config への実反映は human review 前提だが、この issue の hypothesis は artifact ベースで支持された