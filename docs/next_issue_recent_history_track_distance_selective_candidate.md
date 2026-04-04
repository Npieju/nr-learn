# Next Issue: Recent History Track-Distance Selective Candidate

## Summary

`recent form / history` family は current JRA baseline の土台だが、現行 high-coverage line では `horse_last_3_avg_rank` と `horse_last_5_win_rate` に寄っており、course-conditioned な recent history はまだ selective child として切り分けていない。

builder には次の 2 列が既にある。

- `horse_track_distance_last_3_avg_rank`
- `horse_track_distance_last_5_win_rate`

これらは `fundamental_enriched` 系では force include されていたが、current high-coverage rich baseline では actual used set に入っていない。したがって次の history 仮説は broad rerun ではなく、この track-distance pair を narrow selective child として切るのが妥当である。

## Objective

current JRA high-coverage line に対して `horse_track_distance_last_3_avg_rank` と `horse_track_distance_last_5_win_rate` を narrow add-on し、`recent form / history` family の selective child として formal compare に載せる価値があるかを first read で判定する。

## Hypothesis

if horse-level recent history のうち track-distance conditioned pair だけを selective に追加する, then broad history widening をせずに、current baseline に対して course-conditioned recent-form signal を上積みできる可能性がある。

## Current Read

- `feature_family_ranking.md` では `recent form / history` family は Tier B
- current high-coverage baseline は
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`
  を history core として使っている
- builder には track-distance conditioned pair が既にある
- `configs/features_catboost_fundamental_enriched.yaml` では
  - `horse_track_distance_last_3_avg_rank`
  - `horse_track_distance_last_5_win_rate`
  を force include していた
- ただし current high-coverage rich baseline ではこの pair は未採用で、独立仮説としてまだ読んでいない

## Candidate Definition

keep current JRA high-coverage core:

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `horse_surface_switch`
- `race_class_score`
- `horse_class_change`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`
- `owner_last_50_win_rate`

add selective pair:

- `horse_track_distance_last_3_avg_rank`
- `horse_track_distance_last_5_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_recent_history_track_distance_selective.yaml`

## In Scope

- feature-gap / coverage read
- selective config 定義
- no-op risk の事前確認
- true component retrain に進めるかの execution decision

## Non-Goals

- broad history family rerun
- new builder implementation
- policy family work
- NAR work
- actual-date role split の再議論

## Success Criteria

- track-distance pair の coverage / selection risk を独立に読める
- selective child として actual used set に乗る見込みを説明できる
- true component retrain に進めるか止めるかを 1 issue で確定できる

## Validation Plan

1. feature-gap / coverage を読み、2 本が `present=True` かつ low-coverage でないことを確認する
2. selected / force-include / non-null ratio を確認する
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

- track-distance pair のどちらかが missing または low coverage
- selected set に乗らず no-op が濃厚
- Tier B history child として独立性が弱い

## First Read

- feature-gap:
  - `artifacts/reports/feature_gap_summary_recent_history_track_distance_selective_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_recent_history_track_distance_selective_v1.csv`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=109`
  - `categorical_feature_count=37`
- focal coverage:
  - `horse_track_distance_last_3_avg_rank`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.92386`
    - `status=ok`
  - `horse_track_distance_last_5_win_rate`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.92386`
    - `status=ok`

interpretation:

- track-distance pair は current high-coverage line 上で clean に build / select される
- no-op でも low-coverage でもない
- `recent form / history` family の narrow child として true component retrain に進めてよい
