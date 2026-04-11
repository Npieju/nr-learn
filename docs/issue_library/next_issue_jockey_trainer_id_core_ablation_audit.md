# Next Issue: Jockey Trainer ID Core Ablation Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

current JRA high-coverage baseline では、base race context の主要 categorical subgroup は calendar、track/weather/surface-layout、race-condition/dispatch と順に pruning candidate judgment まで閉じた。

その結果、baseline に残る高影響 categorical identity core は実質的に次の 2 列である。

- `jockey_id`
- `trainer_id`

これらは raw identity として強い conditioning source になり得る一方、current serving family の必須 core かどうかはまだ formal に監査していない。

## Objective

`jockey_id`, `trainer_id` を current JRA high-coverage baseline から外した selective ablation を formal compare し、categorical ID core が current serving family の必須 core か、それとも pruning candidate かを判定する。

## Hypothesis

if `jockey_id`, `trainer_id` を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then categorical ID core は current serving family の必須 core ではなく pruning candidate とみなせる。

## Current Read

- `jockey / trainer / combo` の rate/rank core 4 列は pruning candidate judgment まで完了した
- base race context の major categorical subgroup も pruning candidate judgment まで完了した
- それでも current baseline には `jockey_id`, `trainer_id` が force categorical として残っている
- add-on child ではなく baseline core keep/prune judgment を続けるなら、next narrow hypothesis は ID core 2 列の ablation に切るのが自然である

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `jockey_id`
- `trainer_id`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_id_core_ablation.yaml`

## In Scope

- categorical ID core 2 列だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow

## Non-Goals

- jockey/trainer rate/rank core 4 列の再実行
- combo child hypothesis の再実行
- policy rewrite
- NAR work

## Success Metrics

- ablation config が clean に buildable
- win / ROI component の actual used feature set から focal 2 列が消える見込みを説明できる
- formal compare に進めるか、first read で止めるかを 1 issue で確定できる

## Validation Plan

1. feature-gap / coverage を読み、ablation config 自体が clean に buildable であることを確認する
2. selected feature count と focal 2 列の扱いを確認する
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
- categorical ID core を外して読む意味が弱いと判明する

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
   - `artifacts/reports/feature_gap_summary_jockey_trainer_id_core_ablation_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_jockey_trainer_id_core_ablation_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_jockey_trainer_id_core_ablation_v1.csv`
- log:
   - `artifacts/logs/feature_gap_jockey_trainer_id_core_ablation_v1.log`
- summary:
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `empty_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `selected_feature_count=107`
   - `categorical_feature_count=35`
   - `force_include_total=34`

interpretation:

- ablation config 自体は clean に buildable
- focal 2 列を除いた状態でも selected feature set は `107`、categorical feature set は `35` まで縮み、current baseline 比で no-op ではない
- したがって next step は JRA true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
   - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_jockey_trainer_id_core_ablation_v1.json`
   - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_jockey_trainer_id_core_ablation_v1.json`
- metrics:
   - `auc=0.8395835619406387`
   - `logloss=0.2028705884997268`
   - `best_iteration=382`
   - `feature_count=107`
   - `categorical_feature_count=35`

read:

- win component は clean に完走した
- manifest には `jockey_id`, `trainer_id` が残っていない
- jockey/trainer ID core ablation は win side で no-op ではない

### ROI Component

- artifact:
   - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_jockey_trainer_id_core_ablation_v1.json`
   - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_jockey_trainer_id_core_ablation_v1.json`
- metrics:
   - `top1_roi=0.9031548480463086`
   - `best_iteration=104`
   - `feature_count=107`
   - `categorical_feature_count=35`

read:

- ROI component も clean に完走した
- ROI manifest にも focal 2 列は残っていない
- ablation candidate は component 2 本とも true retrain が成立した

### Stack Build

- artifact:
   - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_jockey_trainer_id_core_ablation_v1.json`
   - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_jockey_trainer_id_core_ablation_v1.json`
- metrics:
   - `component_count=2`
   - `feature_count=107`
   - `categorical_feature_count=35`

read:

- stack も clean に bundle できた
- stack manifest にも focal 2 列は残っておらず、jockey/trainer ID core ablation は end-to-end で no-op ではない

## Formal Compare

- revision:
   - `r20260410_jockey_trainer_id_core_ablation_v1`
- evaluation summary:
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_jockey_trainer_id_core_ablation_v1_wf_full_nested.json`
   - `auc=0.8374255422726626`
   - `top1_roi=0.7978585244542691`
   - `ev_top1_roi=0.7478170765404808`
   - `wf_nested_test_roi_weighted=1.0293216630196937`
   - `wf_nested_test_bets_total=914`
   - `stability_assessment=representative`
- promotion gate:
   - `artifacts/reports/promotion_gate_r20260410_jockey_trainer_id_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`
   - `weighted_roi=1.0293216630196937`
   - `bets_total=914`
   - `feasible_fold_count=2`
- revision gate:
   - `artifacts/reports/revision_gate_r20260410_jockey_trainer_id_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`

read:

- evaluation top-line は baseline 近辺を維持した
- formal support は `feasible_fold_count=2/5` と薄めだが、`weighted_roi=1.0293216630196937` で current gate 設定では `pass / promote` を通過した
- current baseline core family の keep vs prune judgment を閉じるには、actual-date role split を追加して operational delta を確認する必要がある

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_jockey_trainer_id_core_ablation_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_sep25_jt_id_core_base_vs_sep25_jt_id_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_sep25_jt_id_core_base_vs_sep25_jt_id_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_jt_id_core_base_vs_sep25_jt_id_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
   - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_dec25_jt_id_core_base_vs_dec25_jt_id_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_dec25_jt_id_core_base_vs_dec25_jt_id_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_jt_id_core_base_vs_dec25_jt_id_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
   - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260410_jockey_trainer_id_core_ablation_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `jockey_id`, `trainer_id` は current JRA high-coverage serving family の必須 core とは言えず、pruning candidate judgment は完了した
- baseline config への実反映は human review 前提だが、この issue の hypothesis は artifact ベースで支持された