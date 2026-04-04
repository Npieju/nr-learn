# Next Issue: Recent History Track-Distance Actual-Date Role Split

## Summary

`#96` の selective child `r20260404_recent_history_track_distance_selective_v1` は formal `pass / promote` に到達した。

- evaluation
  - `auc=0.8449560456255405`
  - `top1_roi=0.458794459525301`
  - `ev_top1_roi=0.832437430649735`
  - nested WF `kelly / no_bet / no_bet`
  - `wf_nested_test_roi_weighted=0.5974478647406058`
  - `wf_nested_test_bets_total=192`
- formal
  - `weighted_roi=2.1315595528260776`
  - `bets_total=1814`
  - `feasible_fold_count=2`

ただし、JRA の serving role は formal `pass / promote` だけでは決めない。次に必要なのは、current operational default と difficult / control windows の actual-date compare を読み、この line を

- serving contender
- analysis-first promoted candidate
- compare reference

のどこに置くかを固定することである。

## Objective

`r20260404_recent_history_track_distance_selective_v1` を current operational default と actual-date windows で比較し、operational role を固定する。

## Hypothesis

if `r20260404_recent_history_track_distance_selective_v1` が September difficult window で baseline を de-risk しつつ December control を壊さない, then this line can be considered a serving contender rather than an analysis-first promoted candidate.

## Current Read

- formal `pass / promote`
- evaluation:
  - `auc=0.8449560456255405`
  - `top1_roi=0.458794459525301`
  - `ev_top1_roi=0.832437430649735`
  - nested WF `kelly / no_bet / no_bet`
  - `wf_nested_test_bets_total=192`
- formal:
  - `weighted_roi=2.1315595528260776`
  - `bets_total=1814`
  - `feasible_fold_count=2`

## In Scope

- broad September actual-date compare
- December control actual-date compare
- `bets / total net / pure bankroll` compare
- operational wording fix

## Non-Goals

- feature widening
- new component retrain
- policy redesign
- NAR work

## Success Criteria

- role が 1 行で固定される
- compare は `bets / total net / pure bankroll` で並ぶ
- next serving queue に置くか、analysis-first に留めるかが決まる

## Validation Plan

1. broad September actual-date compare
2. December control actual-date compare
3. 必要なら late-September narrow window compare
4. docs / queue / issue thread に decision summary を固定

## Stop Condition

- actual-date compare で current operational default に一貫して劣後
- control window を壊し、serving contender を主張できない

