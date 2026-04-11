# Next Issue: Gate Frame Course Core Ablation Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

`gate / frame / course bucket` family は current JRA high-coverage baseline に残り続けている。

- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`

一方で、この family については add-on child と actual-date role split までは読んだが、current serving family の core として本当に必須かどうかはまだ formal に監査していない。

calendar context 3 列は pruning candidate、owner signal は keep 判定まで終わったため、次の narrow JRA hypothesis は current baseline core に残る `gate / frame / course` 4 列の ablation に切るのが自然である。

## Objective

`gate_ratio`, `frame_ratio`, `course_gate_bucket_last_100_win_rate`, `course_gate_bucket_last_100_avg_rank` を current JRA high-coverage baseline から外した selective ablation を formal compare し、`gate / frame / course bucket` family が current serving family の必須 core か、それとも pruning candidate かを判定する。

## Hypothesis

if `gate / frame / course bucket` core 4 列を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then this family は current serving family の必須 core ではなく pruning candidate とみなせる。

## Current Read

- `feature_family_ranking.md` では `gate / frame / course bucket bias` は Tier A
- `#79` の regime extension child `r20260403_gate_frame_course_regime_extension_v1` は formal `pass / promote` まで到達した
- ただし `#80` の actual-date role split では September downside control はあったが December control window で baseline に劣後し、`analysis-first promoted candidate` に留まった
- したがって add-on side の family replay は一段落しており、次は current baseline core 4 列の必須性監査に進むべきである

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_gate_frame_course_core_ablation.yaml`

## In Scope

- gate/frame/course core 4 列だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow

## Non-Goals

- `course_baseline_race_pace_balance_3f` child の再実行
- gate/frame/course family の broad redesign
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

- feature-gap:
   - `artifacts/reports/feature_gap_summary_gate_frame_course_core_ablation_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_gate_frame_course_core_ablation_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_gate_frame_course_core_ablation_v1.csv`
- log:
   - `artifacts/logs/feature_gap_gate_frame_course_core_ablation_v1.log`
- summary:
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `empty_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `selected_feature_count=105`
   - `categorical_feature_count=37`
   - `force_include_total=30`
- interpretation:
   - ablation config 自体は clean に buildable
   - focal 4 列を除いた状態でも selected feature set は `105` まで縮み、current baseline 比で no-op ではない
   - add-on side の replay が一巡した後の core family audit として成立している
   - したがって next step は JRA true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
   - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260408_gate_frame_course_core_ablation_v1.json`
   - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260408_gate_frame_course_core_ablation_v1.json`
- metrics:
   - `auc=0.8396801810817666`
   - `logloss=0.20280775286088068`
   - `best_iteration=548`
   - `feature_count=105`
   - `categorical_feature_count=37`

read:

- win component は clean に完走した
- manifest 上も focal 4 列は残っておらず、core ablation は no-op ではない

### ROI Component

- artifact:
   - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260408_gate_frame_course_core_ablation_v1.json`
   - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260408_gate_frame_course_core_ablation_v1.json`
- metrics:
   - `top1_roi=0.9133429811866859`
   - `best_iteration=138`
   - `feature_count=105`
   - `categorical_feature_count=37`

read:

- ROI component も clean に完走した
- win / ROI の両 component で gate/frame/course core 4 列を外した line が成立している

### Stack Build

- artifact:
   - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260408_gate_frame_course_core_ablation_v1.json`
   - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260408_gate_frame_course_core_ablation_v1.json`
- metrics:
   - `component_count=2`
   - `feature_count=105`
   - `categorical_feature_count=37`

read:

- `value_blend` stack も revision suffix 付き artifact で clean に build できた

### Formal Compare

- artifact:
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260408_gate_frame_course_core_ablation_v1_wf_full_nested.json`
   - `artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_wf_full_nested.json`
   - `artifacts/reports/promotion_gate_r20260408_gate_frame_course_core_ablation_v1.json`
   - `artifacts/reports/revision_gate_r20260408_gate_frame_course_core_ablation_v1.json`
- evaluation:
   - `auc=0.837932981224192`
   - `logloss=0.20377264580735457`
   - `top1_roi=0.8057474440453164`
   - `ev_top1_roi=0.7682163581099751`
   - `wf_nested_test_roi_weighted=0.9823850685689185`
   - `wf_nested_test_bets_total=1482`
   - `stability_assessment=representative`
- WF feasibility:
   - `fold_count=5`
   - `feasible_fold_count=2`
   - `dominant_failure_reason=min_bets`
- promotion:
   - `status=pass`
   - `decision=promote`
   - `weighted_roi=1.0230031948881788`
   - `bets_total=939`
   - `feasible_fold_count=2`

read:

- evaluation の weighted nested read 自体は `0.9824` と強くはないが、formal promotion gate は `pass / promote` まで到達した
- feasible folds は `2/5` と薄めで、dominant failure は一貫して `min_bets` だった
- この時点では「gate/frame/course core 4 列を外しても formal candidate はまだ成立する」までは言える
- ただし current baseline core family の keep vs prune judgment を閉じるには、actual-date role split を追加して operational delta を確認する必要がある

## Current Decision

- `r20260408_gate_frame_course_core_ablation_v1` は formal `pass / promote` まで完了した
- よって gate/frame/course core 4 列は current baseline から外した瞬間に即 reject される必須 core ではない可能性が高い
- 一方で feasible folds は `2/5` と厚くなく、evaluation top-line も強いとは言いにくい
- したがって next step は baseline 対 challenger の actual-date role split を行い、September difficult window と December control window で operational keep vs prune judgment を固定する

### Actual-Date Role Split

- artifact:
   - `artifacts/reports/serving_smoke_profile_compare_sep25_gfcc_core_base_vs_sep25_gfcc_core_cand.json`
   - `artifacts/reports/serving_smoke_compare_sep25_gfcc_core_base_vs_sep25_gfcc_core_cand.json`
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_gfcc_core_base_vs_sep25_gfcc_core_cand.json`
   - `artifacts/reports/serving_smoke_profile_compare_dec25_gfcc_core_base_vs_dec25_gfcc_core_cand.json`
   - `artifacts/reports/serving_smoke_compare_dec25_gfcc_core_base_vs_dec25_gfcc_core_cand.json`
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_gfcc_core_base_vs_dec25_gfcc_core_cand.json`
- broad September 2025:
   - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
   - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`
- December control 2025:
   - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
   - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

read:

- September difficult window でも December control window でも `bets / total net / pure bankroll` は完全同値だった
- differing score source / differing policy dates も空で、operational path は baseline と変わらない

## Decision Summary

- `r20260408_gate_frame_course_core_ablation_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `gate_ratio`, `frame_ratio`, `course_gate_bucket_last_100_win_rate`, `course_gate_bucket_last_100_avg_rank` は current JRA high-coverage serving family の必須 core とは言えず、pruning candidate judgment は完了した
- baseline config への実反映は human review 前提だが、この issue の hypothesis は artifact ベースで支持された