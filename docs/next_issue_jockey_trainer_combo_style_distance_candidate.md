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
