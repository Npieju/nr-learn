# Next Issue: Jockey Trainer Combo Style-Distance Candidate

Historical note:

- この hypothesis は `#108` として formal `pass / promote` まで完了している。
- current active issue は `#115` の actual-date role split であり、この文書は historical issue source / artifact reference として使う。

## Summary

`jockey / trainer / combo` family は Tier A 次順位で、current strong family に一貫して残っている。

一方で、builder には current baseline config でまだ使っていない richer な high-coverage side がすでに入っている。

- style extension:
  - `jockey_last_30_avg_corner_gain_2_to_4`
  - `trainer_last_30_avg_corner_gain_2_to_4`
  - `jockey_last_30_avg_closing_time_3f`
  - `trainer_last_30_avg_closing_time_3f`
- track-distance extension:
  - `jockey_track_distance_last_50_win_rate`
  - `jockey_track_distance_last_50_avg_rank`
  - `trainer_track_distance_last_50_win_rate`
  - `trainer_track_distance_last_50_avg_rank`

したがって first candidate は、新しい builder 実装を足す前に、この 8 本を narrow に追加する `style + track-distance` extension とする。

## Objective

current promoted line の next family として、`jockey / trainer / combo` family を regime-aware に広げつつ、coverage と support を壊さず formal candidate を 1 本試す。

## Current Read

- `feature_family_ranking.md` では Tier A 第二候補
- current high-coverage baseline に残っている core は:
  - `jockey_last_30_win_rate`
  - `trainer_last_30_win_rate`
  - `jockey_trainer_combo_last_50_win_rate`
  - `jockey_trainer_combo_last_50_avg_rank`
- builder には richer side として style / track-distance history がすでにある
- `fundamental_enriched_no_lineage` では、この richer side 8 本も force include 済みである

## Candidate Definition

keep current core:

- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`

add narrow extension:

- `jockey_last_30_avg_corner_gain_2_to_4`
- `trainer_last_30_avg_corner_gain_2_to_4`
- `jockey_last_30_avg_closing_time_3f`
- `trainer_last_30_avg_closing_time_3f`
- `jockey_track_distance_last_50_win_rate`
- `jockey_track_distance_last_50_avg_rank`
- `trainer_track_distance_last_50_win_rate`
- `trainer_track_distance_last_50_avg_rank`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_regime_extension.yaml`

## In Scope

- `jockey / trainer / combo` richer side 8 本の selective child
- feature-gap / coverage read
- true component retrain flow
- formal compare

## Non-Goals

- 新しい builder 実装の追加
- broad policy redesign
- serving role split の先回り
- NAR readiness / benchmark work

## Execution Standard

前回の feature family と同じ true component retrain flow を使う。

1. win component retrain
2. roi component retrain
3. stack rebuild
4. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`

## Success Criteria

- current promoted line に対して formal compare できる candidate が 1 本立つ
- support が極端に崩れない
- `jockey / trainer / combo` family の richer side が alpha source かどうかを family 単位で判断できる

## Validation Plan

1. feature-gap / coverage read
2. win component retrain
3. ROI component retrain
4. stack rebuild
5. `run_revision_gate.py --skip-train` で formal compare

## Stop Condition

- focal 8 本の coverage が低く、selective child 自体が buildable でない
- actual used set に乗らず no-op と判定される
- evaluation / formal のどちらかで support が崩れ、family の next child として残す根拠が弱い

## First Read

- issue:
  - `#108`
- config:
  - `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_regime_extension.yaml`
- feature gap summary:
  - `artifacts/reports/feature_gap_summary_jockey_trainer_combo_style_distance_selective_v1.json`
- feature coverage csv:
  - `artifacts/reports/feature_gap_feature_coverage_jockey_trainer_combo_style_distance_selective_v1.csv`

summary:

- `priority_missing_raw_columns=[]`
- `missing_force_include_features=[]`
- `empty_force_include_features=[]`
- `low_coverage_force_include_features=[]`
- `template_columns_present=32/32`
- `force_include_total=42`
- `selected_feature_count=109`
- `categorical_feature_count=37`

focal 8 features:

- `jockey_last_30_avg_corner_gain_2_to_4`: `selected=True`, `present=True`, `non_null_ratio=0.98734`
- `trainer_last_30_avg_corner_gain_2_to_4`: `selected=True`, `present=True`, `non_null_ratio=0.98817`
- `jockey_last_30_avg_closing_time_3f`: `selected=True`, `present=True`, `non_null_ratio=0.99629`
- `trainer_last_30_avg_closing_time_3f`: `selected=True`, `present=True`, `non_null_ratio=0.99501`
- `jockey_track_distance_last_50_win_rate`: `selected=True`, `present=True`, `non_null_ratio=1.0`
- `jockey_track_distance_last_50_avg_rank`: `selected=True`, `present=True`, `non_null_ratio=1.0`
- `trainer_track_distance_last_50_win_rate`: `selected=True`, `present=True`, `non_null_ratio=1.0`
- `trainer_track_distance_last_50_avg_rank`: `selected=True`, `present=True`, `non_null_ratio=1.0`

current read:

- selective child は buildable
- no-op risk は低い
- 次段は true component retrain

## First Acceptance Point

- win component:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
- train metrics:
  - `auc=0.8397296191222723`
  - `logloss=0.2027975460526154`
  - `best_iteration=531`
- selected features:
  - `109`
- focal 8 features は actual used set に全て入った

current judgment:

- selective child は no-op ではない
- 次段は ROI component retrain

## Second Acceptance Point

- roi component:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
  - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
- train metrics:
  - `top1_roi=0.9386975397973951`
  - `best_iteration=66`
- selected features:
  - `109`
- focal 8 features は ROI actual used set に全て入った

stack configs:

- `configs/model_catboost_win_high_coverage_diag_jockey_trainer_combo_style_distance_selective_v1.yaml`
- `configs/model_lightgbm_roi_high_coverage_diag_jockey_trainer_combo_style_distance_selective_v1.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_jockey_trainer_combo_style_distance_selective_v1.yaml`

current judgment:

- selective child は win / ROI の両 component で no-op ではない
- stack build は完了した
- 次段は formal compare

## Final Read

- revision:
  - `r20260404_jockey_trainer_combo_style_distance_selective_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260404_jockey_trainer_combo_style_distance_selective_v1_r20260404_jockey_trainer_combo_style_distance_selective_v1_wf_full_nested.json`
  - `auc=0.8377559190837222`
  - `top1_roi=0.8022312793589389`
  - `ev_top1_roi=0.7669936446532192`
  - `wf_nested_test_roi_weighted=1.168363624344555`
  - `wf_nested_test_bets_total=966`
- promotion / formal:
  - `artifacts/reports/promotion_gate_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
  - `status=pass`
  - `decision=promote`
  - held-out formal `weighted_roi=1.023657866236539`
  - `formal_benchmark_bets_total=1040`
  - `formal_benchmark_feasible_fold_count=4`

current judgment:

- `style + track-distance` selective child は formal candidate として成立した
- `jockey / trainer / combo` family の richer side は no-op ではなく、support も極端には崩れていない
- 次段は actual-date role split を別 issue として切り、operational role を固定する

next execution source:

- `docs/issue_library/next_issue_jockey_trainer_combo_style_distance_actual_date_role_split.md`
