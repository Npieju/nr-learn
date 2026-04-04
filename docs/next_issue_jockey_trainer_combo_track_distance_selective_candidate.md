# Next Issue: Jockey Trainer Combo Track-Distance Selective Candidate

## Summary

`jockey / trainer / combo` family は Tier A で、broad child と `closing_time_3f` child までは formal `pass / promote` を通した。

ただし broad child は actual-date で family anchor にならず、`closing_time_3f` child も role は `analysis-first promoted candidate` に留まった。

この family でまだ独立に切り分けていない高 coverage side は、`track-distance` quartet である。

## Objective

`jockey_track_distance_last_50_*` と `trainer_track_distance_last_50_*` を current JRA high-coverage line に narrow add-on し、`jockey / trainer / combo` family の third child として formal compare に載せる価値があるかを判定する。

## Hypothesis

if `track-distance` quartet を selective に追加する, then `closing_time_3f` child より regime specificity を上げつつ、broad child より redundancy を抑えた candidate を作れる。

## Current Read

- `feature_family_ranking.md` では `jockey / trainer / combo` family は Tier A
- broad `style + track-distance` child は formal `pass / promote` だが actual-date role は弱かった
- `closing_time_3f` pair child は formal `pass / promote` だが serving role までは上がらなかった
- 未切り分けの高 coverage side は `track-distance` quartet である
- builder には既に存在し、新しい実装なしで selective child を作れる

## Candidate Definition

keep current JRA high-coverage core:

- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`

add selective quartet:

- `jockey_track_distance_last_50_win_rate`
- `jockey_track_distance_last_50_avg_rank`
- `trainer_track_distance_last_50_win_rate`
- `trainer_track_distance_last_50_avg_rank`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_track_distance_selective.yaml`

## In Scope

- feature-gap / coverage read
- true component retrain ready な selective config
- formal compare へ進めるかどうかの execution decision

## Non-Goals

- broad `style + track-distance` child の再実行
- `closing_time_3f` child の role 再議論
- new builder implementation
- NAR work

## Success Criteria

- quartet の coverage / selection risk を独立に読める
- narrow child として true component retrain に載せる価値があるか判断できる
- `jockey / trainer / combo` family の next child hypothesis が 1 本に固定される

## Validation Plan

1. feature-gap / coverage を読み、4 本が `present=True` かつ low-coverage でないことを確認する
2. win / roi component retrain に進める場合は true component flow を使う
3. formal compare では
   - `auc`
   - `top1_roi`
   - `ev_top1_roi`
   - nested WF shape
   - held-out formal `weighted_roi`
   - `bets / races / bet_rate`
   を baseline と比較する

## Stop Condition

- quartet のいずれかが low coverage か missing
- selected set に乗らず no-op の可能性が高い
- existing child と比べて独立仮説として弱い

## First Read

- feature-gap:
  - `artifacts/reports/feature_gap_summary_jockey_trainer_combo_track_distance_selective_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_jockey_trainer_combo_track_distance_selective_v1.csv`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=109`
  - `categorical_feature_count=37`
- focal coverage:
  - `jockey_track_distance_last_50_win_rate`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.92386`
  - `jockey_track_distance_last_50_avg_rank`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.92386`
  - `trainer_track_distance_last_50_win_rate`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.92386`
  - `trainer_track_distance_last_50_avg_rank`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.92386`

interpretation:

- track-distance quartet は high coverage で clean に build / select される
- no-op でも low-coverage でもない
- third child として true component retrain に進めてよい
