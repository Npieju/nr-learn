# Next Issue: Class Rest Surface Core Ablation Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

`class / rest / surface` family は current JRA high-coverage baseline の最上位 core family として残っている。

現在 baseline 側に残っている family は次の 20 列である。

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

一方で、この family については interaction child と actual-date role split までは読んだが、current serving family の core として本当に必須かどうかはまだ formal に監査していない。

calendar context 3 列、gate/frame/course core 4 列、recent-history core 2 列、combo core 4 列は pruning candidate と判断できたため、次の narrow JRA hypothesis は baseline core に残る `class / rest / surface` family 全体の ablation に切るのが自然である。

## Objective

上記 20 列を current JRA high-coverage baseline から外した selective ablation を formal compare し、`class / rest / surface` family が current serving family の必須 core か、それとも pruning candidate かを判定する。

## Hypothesis

if `class / rest / surface` family の core 20 列を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then this family は current serving family の必須 core ではなく pruning candidate とみなせる。

## Current Read

- `feature_family_ranking.md` では `class / rest / surface change` は Tier A / 最上位
- conditional selective child は formal `pass / promote` まで到達したが、actual-date role は serving default に上がらなかった
- current high-coverage baseline config では class/rest/surface family が force include として最も厚く残っている
- add-on side の child hypothesis は一巡したため、次は baseline family 全体の必須性監査に進むべきである

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only the class/rest/surface family 20 列 listed above.

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_class_rest_surface_core_ablation.yaml`

## In Scope

- class/rest/surface core 20 列だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow

## Non-Goals

- conditional child の再実行
- class/rest/surface family の broad redesign
- policy rewrite
- NAR work

## Success Metrics

- ablation config が clean に buildable
- win / ROI component の actual used feature set から focal 20 列が消える見込みを説明できる
- formal compare に進めるか、first read で止めるかを 1 issue で確定できる

## Validation Plan

1. feature-gap / coverage を読み、ablation config 自体が clean に buildable であることを確認する
2. selected feature count と focal 20 列の扱いを確認する
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
- feature family の core としての寄与を外して読む意味が弱いと判明する

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
   - `artifacts/reports/feature_gap_summary_class_rest_surface_core_ablation_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_class_rest_surface_core_ablation_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_class_rest_surface_core_ablation_v1.csv`
- log:
   - `artifacts/logs/feature_gap_class_rest_surface_core_ablation_v1.log`
- summary:
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `empty_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `selected_feature_count=89`
   - `categorical_feature_count=37`
   - `force_include_total=14`

interpretation:

- ablation config 自体は clean に buildable
- focal 20 列を除いた状態でも selected feature set は `89` まで縮み、current baseline 比で no-op ではない
- したがって next step は JRA true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
   - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_class_rest_surface_core_ablation_v1.json`
   - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_class_rest_surface_core_ablation_v1.json`
- metrics:
   - `auc=0.8394804126333921`
   - `logloss=0.20291890140795762`
   - `best_iteration=508`
   - `feature_count=89`
   - `categorical_feature_count=37`

read:

- win component は clean に完走した
- manifest には class/rest/surface core 20 列が残っていない
- class/rest/surface core ablation は win side で no-op ではない

### ROI Component

- artifact:
   - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_class_rest_surface_core_ablation_v1.json`
   - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_class_rest_surface_core_ablation_v1.json`
- metrics:
   - `top1_roi=0.9661649782923295`
   - `best_iteration=76`
   - `feature_count=89`
   - `categorical_feature_count=37`

read:

- ROI component も clean に完走した
- ROI manifest にも focal 20 列は残っていない
- ablation candidate は component 2 本とも true retrain が成立した

### Stack Build

- artifact:
   - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_class_rest_surface_core_ablation_v1.json`
   - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_class_rest_surface_core_ablation_v1.json`
- metrics:
   - `component_count=2`
   - `feature_count=89`
   - `categorical_feature_count=37`

read:

- stack も clean に bundle できた
- stack manifest にも focal 20 列は残っておらず、class/rest/surface core ablation は end-to-end で no-op ではない

## Formal Compare

- revision:
   - `r20260410_class_rest_surface_core_ablation_v1`
- evaluation summary:
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_class_rest_surface_core_ablation_v1_wf_full_nested.json`
   - `auc=0.837477380377901`
   - `top1_roi=0.7977549046697983`
   - `ev_top1_roi=0.7081790549875656`
   - `wf_nested_test_roi_weighted=0.9825688073394497`
   - `wf_nested_test_bets_total=654`
   - `stability_assessment=representative`
- promotion gate:
   - `artifacts/reports/promotion_gate_r20260410_class_rest_surface_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`
   - `weighted_roi=0.9825688073394497`
   - `bets_total=654`
   - `feasible_fold_count=2`
- revision gate:
   - `artifacts/reports/revision_gate_r20260410_class_rest_surface_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`

read:

- evaluation の top-line は強くなく、formal support も `2/5` feasible folds と薄い
- それでも current gate 設定では promotion gate は `pass / promote` まで到達した
- current baseline core family の keep vs prune judgment を閉じるには、actual-date role split を追加して operational delta を確認する必要がある

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_class_rest_surface_core_ablation_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_sep25_class_rest_surface_core_base_vs_sep25_class_rest_surface_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_sep25_class_rest_surface_core_base_vs_sep25_class_rest_surface_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_class_rest_surface_core_base_vs_sep25_class_rest_surface_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
   - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_dec25_class_rest_surface_core_base_vs_dec25_class_rest_surface_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_dec25_class_rest_surface_core_base_vs_dec25_class_rest_surface_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_class_rest_surface_core_base_vs_dec25_class_rest_surface_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
   - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_class_rest_surface_core_ablation_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって class/rest/surface core 20 列は current JRA high-coverage serving family の必須 core とは言えず、pruning candidate judgment は完了した
- baseline config への実反映は human review 前提だが、この issue の hypothesis は artifact ベースで支持された