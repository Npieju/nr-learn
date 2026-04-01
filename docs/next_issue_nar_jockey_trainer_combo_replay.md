# Next Issue: NAR Jockey-Trainer Combo Replay

## Summary

`#63` で buildable な class/rest replay features 3 本を actual candidate に載せる selection fix は成功したが、formal compare では baseline を下回った。したがって次の replay family は、NAR raw/source で既に coverage があり、baseline でも signal を持っている `jockey / trainer` 周辺へ移す。

## Objective

local Nankan baseline に対して、JRA で一定の formal strength を持っていた `jockey / trainer / combo` family を narrow replay し、NAR でも support を維持したまま ROI family を改善できるかを測る。

## Hypothesis

if local Nankan baseline に `jockey / trainer / combo` の regime-aware replay features を narrow に追加する, then class/rest replay より高い coverage のまま support を落とさず、formal benchmark で baseline と比較可能な challenger を作れる。

## Current Read

- baseline narrow formal:
  - `auc=0.8775353752835744`
  - `formal_benchmark_weighted_roi=3.6903437891931246`
  - `formal_benchmark_bets_total=3525`
  - `wf_feasible_fold_count=3`
  - `bet_rate=12.16%` on `28997` test races
- class/rest replay v2:
  - `auc=0.8737311965910113`
  - `formal_benchmark_weighted_roi=1.011422845691383`
  - `formal_benchmark_bets_total=499`
  - `wf_feasible_fold_count=1`
  - race-denominator bet rate `499 / 9819 = 5.08%`
- first gap read for this family:
  - selected candidate features before narrowing were `21`
  - buildable and `selected=True` with `non_null_ratio > 0.91`:
    - `jockey_trainer_combo_last_50_win_rate`
    - `jockey_trainer_combo_last_50_avg_rank`
    - `jockey_last_30_avg_closing_time_3f`
    - `trainer_last_30_avg_closing_time_3f`
    - `jockey_track_distance_last_50_win_rate`
    - `jockey_track_distance_last_50_avg_rank`
    - `trainer_track_distance_last_50_win_rate`
    - `trainer_track_distance_last_50_avg_rank`
  - not built:
    - `jockey_last_30_avg_corner_gain_2_to_4`
    - `trainer_last_30_avg_corner_gain_2_to_4`
  - therefore the actual candidate is narrowed to the 8 buildable features above

## In Scope

- local Nankan feature config の narrow challenger 1 本
- `jockey / trainer / combo` family の replay
- matching-tuple local revision gate
- denominator-first formal read

## Non-Goals

- class/surface raw schema 拡張
- broad family widening
- serving role split の再設計

## Success Metrics

- selected features に new replay features が実際に入る
- baseline narrow と比較して support を大きく落とさない
- `bets / races / bet_rate` を明示した formal compare ができる

## Stop Condition

- no-op replay になる
- formal bet rate が baseline から大きく低下する
- weighted ROI / feasible folds が baseline を明確に下回る
