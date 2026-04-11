# Next Issue: Recent History Core Ablation Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

`recent form / history` family は current JRA high-coverage baseline の土台として残っている。

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`

一方で、この family については track-distance child と actual-date role split までは読んだが、current serving family の core として本当に必須かどうかはまだ formal に監査していない。

calendar context 3 列は pruning candidate、gate/frame/course core 4 列も pruning candidate と判断できたため、次の narrow JRA hypothesis は baseline core に残る `recent form / history` 2 列の ablation に切るのが自然である。

## Objective

`horse_last_3_avg_rank`, `horse_last_5_win_rate` を current JRA high-coverage baseline から外した selective ablation を formal compare し、`recent form / history` core が current serving family の必須土台か、それとも pruning candidate かを判定する。

## Hypothesis

if `horse_last_3_avg_rank`, `horse_last_5_win_rate` を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then `recent form / history` core は current serving family の必須要素ではなく pruning candidate とみなせる。

## Current Read

- `feature_family_ranking.md` では `recent form / history` family は Tier B
- `#96` の track-distance child は formal `pass / promote` まで到達したが、`#97` の actual-date role split では serving default には上がらなかった
- current high-coverage baseline config では history core として `horse_last_3_avg_rank`, `horse_last_5_win_rate` が force include に固定されている
- add-on side の child hypothesis は一巡したため、次は baseline core 2 列の必須性監査に進むべきである

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_recent_history_core_ablation.yaml`

## In Scope

- recent-history core 2 列だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow

## Non-Goals

- `horse_track_distance_last_3_avg_rank` / `horse_track_distance_last_5_win_rate` child の再実行
- recent-history family の broad redesign
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
- feature family の core としての寄与を外して読む意味が弱いと判明する

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
   - `artifacts/reports/feature_gap_summary_recent_history_core_ablation_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_recent_history_core_ablation_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_recent_history_core_ablation_v1.csv`
- log:
   - `artifacts/logs/feature_gap_recent_history_core_ablation_v1.log`
- summary:
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `empty_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `selected_feature_count=107`
   - `categorical_feature_count=37`
   - `force_include_total=32`

interpretation:

- ablation config 自体は clean に buildable
- focal 2 列を除いた状態でも selected feature set は `107` まで縮み、current baseline 比で no-op ではない
- したがって next step は JRA true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
   - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260408_recent_history_core_ablation_v1.json`
   - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260408_recent_history_core_ablation_v1.json`
- metrics:
   - `auc=0.83959982231444`
   - `logloss=0.20285637765955158`
   - `best_iteration=546`
   - `feature_count=107`
   - `categorical_feature_count=37`

read:

- win component は clean に完走した
- manifest には `horse_last_3_avg_rank`, `horse_last_5_win_rate` が残っていない
- recent-history core ablation は win side で no-op ではない

### ROI Component

- artifact:
   - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260408_recent_history_core_ablation_v1.json`
   - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260408_recent_history_core_ablation_v1.json`
- metrics:
   - `top1_roi=0.8522720694645436`
   - `best_iteration=104`
   - `feature_count=107`
   - `categorical_feature_count=37`

read:

- ROI component も clean に完走した
- ROI manifest にも focal 2 列は残っていない
- ablation candidate は component 2 本とも true retrain が成立した

### Stack Build

- artifact:
   - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260408_recent_history_core_ablation_v1.json`
   - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260408_recent_history_core_ablation_v1.json`
- metrics:
   - `feature_count=107`
   - `categorical_feature_count=37`

read:

- stack も clean に bundle できた
- stack manifest にも focal 2 列は残っておらず、recent-history core ablation は end-to-end で no-op ではない

## Formal Compare

- revision:
   - `r20260408_recent_history_core_ablation_v1`
- evaluation summary:
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260408_recent_history_core_ablation_v1_wf_full_nested.json`
   - `auc=0.8377999131013059`
   - `top1_roi=0.8008358662613981`
   - `ev_top1_roi=0.7759118541033435`
   - `wf_nested_test_roi_weighted=1.0468756201775806`
   - `wf_nested_test_bets_total=1336`
   - `stability_assessment=representative`
- promotion gate:
   - `artifacts/reports/promotion_gate_r20260408_recent_history_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`
   - `weighted_roi=1.0468756201775806`
   - `bets_total=1336`
   - `feasible_fold_count=5`
- revision gate:
   - `artifacts/reports/revision_gate_r20260408_recent_history_core_ablation_v1.json`
   - `status=pass`
   - `decision=promote`

read:

- evaluation の top-line は極端に強くはないが、formal promotion gate は `pass / promote` まで到達した
- feasible folds が `5/5` まで揃っており、support はむしろ strong である
- したがって current baseline core family の keep vs prune judgment を閉じるには、actual-date role split を追加して operational delta を確認する必要がある

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260408_recent_history_core_ablation_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_sep25_recent_hist_core_base_vs_sep25_recent_hist_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_sep25_recent_hist_core_base_vs_sep25_recent_hist_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_recent_hist_core_base_vs_sep25_recent_hist_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
   - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
   - `artifacts/reports/serving_smoke_profile_compare_dec25_recent_hist_core_base_vs_dec25_recent_hist_core_cand.json`
- compare summary:
   - `artifacts/reports/serving_smoke_compare_dec25_recent_hist_core_base_vs_dec25_recent_hist_core_cand.json`
- bankroll sweep:
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_recent_hist_core_base_vs_dec25_recent_hist_core_cand.json`
- result:
   - `shared_ok_dates=8`
   - `differing_score_source_dates=[]`
   - `differing_policy_dates=[]`
   - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
   - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260408_recent_history_core_ablation_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `horse_last_3_avg_rank`, `horse_last_5_win_rate` は current JRA high-coverage serving family の必須 core とは言えず、pruning candidate judgment は完了した
- baseline config への実反映は human review 前提だが、この issue の hypothesis は artifact ベースで支持された