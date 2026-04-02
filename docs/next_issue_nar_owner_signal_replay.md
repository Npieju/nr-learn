# Next Issue: NAR Owner Signal Replay

## Summary

`#66` の gate/frame replay は no-op ではなかったが、baseline narrow を上回れず `reject` で閉じた。次の family は ranking の次順位である owner signal を pedigree から切り離して単独評価する。

## Objective

local Nankan baseline に対して owner signal family を narrow add-on として replay し、support を壊さず denominator-first formal benchmark を改善できるかを測る。

## Hypothesis

if local Nankan baseline に owner signal の buildable high-coverage features を narrow に追加する, then lineage-heavy family より低リスクに、current promoted combo line と比較可能な challenger を作れる。

## Current Read

- current promoted NAR line:
  - jockey/trainer/combo replay
  - `formal_benchmark_weighted_roi=4.324481148818757`
  - `formal_benchmark_bets_total=3725`
  - `bet_rate=12.85%` on `28997` test races
- rejected gate/frame replay:
  - `auc=0.8760517067265055`
  - `top1_roi=0.832151950300438`
  - `ev_top1_roi=1.2202668296160506`
  - nested WF `3/3 no_bet`
- ranking read:
  - `feature_family_ranking.md` では owner signal は Tier C
  - lineage / pace より先に owner を単独で切り出すのが妥当
- first gap read:
  - `feature_gap_summary_local_nankan_owner_signal_replay_v1.json`
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - actual candidate は single add-on:
    - `owner_last_50_win_rate`
  - selected feature count は baseline `13` から `14` へ増えた
  - `owner_last_50_win_rate` は `selected=True`, `present=True`, `non_null_ratio=0.918175`
  - therefore first actual candidate は owner single-feature replay とする

## In Scope

- local Nankan owner family の feature-gap read
- narrow candidate config 1 本
- matching-tuple local revision gate
- denominator-first formal compare against current promoted NAR line

## Non-Goals

- pedigree / lineage heavy family の広い追加
- pace / corner family の primary replay
- CatBoost / ensemble line への移行

## Success Metrics

- owner replay features が actual selected set に入る
- formal `bets / races / bet_rate` が current promoted line と同程度の order を保つ
- weighted ROI / feasible folds で current promoted line を clean challenge できる

## Validation

- feature gap summary / coverage CSV
- `python scripts/run_local_revision_gate.py ...`
- final read with:
  - `auc`
  - `ev_top1_roi`
  - `formal_benchmark_weighted_roi`
  - `formal_benchmark_bets_total`
  - `bet_rate`

## Stop Condition

- no-op replay
- support collapse versus current promoted line
- owner family が low coverage / lineage proxy に寄りすぎる

## Current First Candidate

first candidate は `owner_last_50_win_rate` single-feature replay とする。

理由:

- feature gap 上で buildable / selected が確認できた owner family はこの 1 本だった
- baseline `13 -> 14 features` なので no-op ではない
- lineage-heavy family に広げる前に、owner 単独の寄与を denominator-first で切り分ける方が defensible
