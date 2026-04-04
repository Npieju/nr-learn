# Next Issue: Jockey Trainer Combo Style-Distance Candidate

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
- `low_coverage_force_include_features=[]`

focal 8 features:

- `jockey_last_30_avg_corner_gain_2_to_4`: `selected=True`, `present=True`, `non_null_ratio=0.92175`
- `trainer_last_30_avg_corner_gain_2_to_4`: `selected=True`, `present=True`, `non_null_ratio=0.92134`
- `jockey_last_30_avg_closing_time_3f`: `selected=True`, `present=True`, `non_null_ratio=0.92171`
- `trainer_last_30_avg_closing_time_3f`: `selected=True`, `present=True`, `non_null_ratio=0.92133`
- `jockey_track_distance_last_50_win_rate`: `selected=True`, `present=True`, `non_null_ratio=0.92386`
- `jockey_track_distance_last_50_avg_rank`: `selected=True`, `present=True`, `non_null_ratio=0.92386`
- `trainer_track_distance_last_50_win_rate`: `selected=True`, `present=True`, `non_null_ratio=0.92386`
- `trainer_track_distance_last_50_avg_rank`: `selected=True`, `present=True`, `non_null_ratio=0.92386`

current read:

- selective child は buildable
- no-op risk は低い
- 次段は true component retrain

## First Acceptance Point

- win component:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260404_jockey_trainer_combo_style_distance_selective_v1.json`
- train metrics:
  - `auc=0.8345232157261651`
  - `logloss=0.20221578593508155`
  - `best_iteration=548`
- selected features:
  - `109`
- focal 8 features は actual used set に全て入った

current judgment:

- selective child は no-op ではない
- 次段は ROI component retrain
