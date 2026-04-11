# Next Issue: Jockey Trainer Combo Core Ablation Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

`jockey / trainer / combo` family は current JRA high-coverage baseline の core として残っている。

- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`

一方で、この family については style-distance / closing-time / track-distance の add-on child と actual-date role split までは読んだが、current serving family の core として本当に必須かどうかはまだ formal に監査していない。

calendar context 3 列、gate/frame/course core 4 列、recent-history core 2 列は pruning candidate と判断できたため、次の narrow JRA hypothesis は baseline core に残る `jockey / trainer / combo` 4 列の ablation に切るのが自然である。

## Objective

`jockey_last_30_win_rate`, `trainer_last_30_win_rate`, `jockey_trainer_combo_last_50_win_rate`, `jockey_trainer_combo_last_50_avg_rank` を current JRA high-coverage baseline から外した selective ablation を formal compare し、`jockey / trainer / combo` core が current serving family の必須土台か、それとも pruning candidate かを判定する。

## Hypothesis

if `jockey / trainer / combo` core 4 列を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then this family は current serving family の必須 core ではなく pruning candidate とみなせる。

## Current Read

- `feature_family_ranking.md` では `jockey / trainer / combo` family は Tier A
- style-distance, closing-time, track-distance の child line は formal `pass / promote` まで到達したが、actual-date role は serving default に上がらなかった
- current high-coverage baseline config では combo core として 4 列が force include に固定されている
- add-on side の child hypothesis は一巡したため、次は baseline core 4 列の必須性監査に進むべきである

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_core_ablation.yaml`

## In Scope

- combo core 4 列だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow

## Non-Goals

- style-distance / closing-time / track-distance child の再実行
- `jockey / trainer / combo` family の broad redesign
- policy rewrite
- NAR work

## Success Metrics

- ablation config が clean に buildable
- win / ROI component の actual used feature set から focal 4 列が消える見込みを説明できる
- formal compare に進めるか、first read で止めるかを 1 issue で確定できる

## Validation Plan

1. feature-gap / coverage を読み、ablation config 自体が clean に buildable であることを確認する
2. selected feature count と focal 4 列の扱いを確認する
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
   - `artifacts/reports/feature_gap_summary_jockey_trainer_combo_core_ablation_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_jockey_trainer_combo_core_ablation_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_jockey_trainer_combo_core_ablation_v1.csv`
- log:
   - `artifacts/logs/feature_gap_jockey_trainer_combo_core_ablation_v1.log`
- summary:
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `empty_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `selected_feature_count=105`
   - `categorical_feature_count=37`
   - `force_include_total=30`

interpretation:

- ablation config 自体は clean に buildable
- focal 4 列を除いた状態でも selected feature set は `105` まで縮み、current baseline 比で no-op ではない
- したがって next step は JRA true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
   - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260408_jockey_trainer_combo_core_ablation_v1.json`
   - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260408_jockey_trainer_combo_core_ablation_v1.json`
- metrics:
   - `auc=0.8397196612693565`
   - `logloss=0.2028606204060377`
   - `best_iteration=549`
   - `feature_count=105`
   - `categorical_feature_count=37`

read:

- win component は clean に完走した
- manifest には `jockey_last_30_win_rate`, `trainer_last_30_win_rate`, `jockey_trainer_combo_last_50_win_rate`, `jockey_trainer_combo_last_50_avg_rank` が残っていない
- combo core ablation は win side で no-op ではない

### ROI Component

- artifact:
   - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260408_jockey_trainer_combo_core_ablation_v1.json`
   - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260408_jockey_trainer_combo_core_ablation_v1.json`
- metrics:
   - `top1_roi=0.805643994211288`
   - `best_iteration=126`
   - `feature_count=105`
   - `categorical_feature_count=37`

read:

- ROI component も clean に完走した
- ROI manifest にも focal 4 列は残っていない
- ablation candidate は component 2 本とも true retrain が成立した

### Stack Build

- artifact:
   - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260408_jockey_trainer_combo_core_ablation_v1.json`
   - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260408_jockey_trainer_combo_core_ablation_v1.json`
- metrics:
   - `feature_count=105`
   - `categorical_feature_count=37`

read:

- stack も clean に bundle できた
- stack manifest にも focal 4 列は残っておらず、combo core ablation は end-to-end で no-op ではない

## Formal Compare

- revision:
   - `r20260408_jockey_trainer_combo_core_ablation_v1`
- evaluation summary:
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260408_jockey_trainer_combo_core_ablation_v1_wf_full_nested.json`
   - `auc=0.8379842049304117`
   - `top1_roi=0.8031085935341254`
   - `ev_top1_roi=0.7597057198121028`
   - `wf_nested_test_roi_weighted=1.004608788853162`
   - `wf_nested_test_bets_total=933`
   - `stability_assessment=representative`
- promotion gate:
   - `artifacts/reports/promotion_gate_r20260408_jockey_trainer_combo_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`
   - `weighted_roi=1.004608788853162`
   - `bets_total=933`
   - `feasible_fold_count=2`
- revision gate:
   - `artifacts/reports/revision_gate_r20260408_jockey_trainer_combo_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`

read:

- evaluation の top-line は強くないが、formal promotion gate は `pass / promote` まで到達した
- feasible folds は `2/5` と薄く、support は recent-history core ablation より弱い
- それでも current baseline core family の keep vs prune judgment を閉じるには、actual-date role split を追加して operational delta を確認する必要がある

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260408_jockey_trainer_combo_core_ablation_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_sep25_combo_core_base_vs_sep25_combo_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_sep25_combo_core_base_vs_sep25_combo_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_combo_core_base_vs_sep25_combo_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
   - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_dec25_combo_core_base_vs_dec25_combo_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_dec25_combo_core_base_vs_dec25_combo_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_combo_core_base_vs_dec25_combo_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
   - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260408_jockey_trainer_combo_core_ablation_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `jockey_last_30_win_rate`, `trainer_last_30_win_rate`, `jockey_trainer_combo_last_50_win_rate`, `jockey_trainer_combo_last_50_avg_rank` は current JRA high-coverage serving family の必須 core とは言えず、pruning candidate judgment は完了した
- baseline config への実反映は human review 前提だが、この issue の hypothesis は artifact ベースで支持された