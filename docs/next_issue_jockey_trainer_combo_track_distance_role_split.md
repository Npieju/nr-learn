# Next Issue: Jockey Trainer Combo Track-Distance Role Split

## Summary

`#98` の third child `r20260404_jockey_trainer_combo_track_distance_selective_v1` は formal `pass / promote` に到達した。

ただし current read は、formal top-line に対して evaluation shape が弱い。次に必要なのは actual-date compare であり、この line を

- serving contender
- analysis-first promoted candidate
- compare reference

のどこに置くかを固定することである。

## Objective

`r20260404_jockey_trainer_combo_track_distance_selective_v1` を current operational default と actual-date windows で比較し、operational role を固定する。

## Hypothesis

if `r20260404_jockey_trainer_combo_track_distance_selective_v1` が September difficult window で baseline を de-risk しつつ December control を壊さない, then this line can be considered a serving contender rather than a compare-only formal candidate.

## Current Read

- formal `pass / promote`
- evaluation:
  - `auc=0.8449560456255405`
  - `top1_roi=0.458794459525301`
  - `ev_top1_roi=0.832437430649735`
  - nested WF `kelly / no_bet / no_bet`
  - `wf_nested_test_roi_weighted=0.5974478647406058`
  - `wf_nested_test_bets_total=192`
- formal:
  - `weighted_roi=2.1315595528260776`
  - `bets_total=1814`
  - `feasible_fold_count=2`
  - `metric_source_counts={test: 2}`

## In Scope

- broad September actual-date compare
- December control actual-date compare
- `bets / total net / pure bankroll` compare
- operational wording fix

## Non-Goals

- feature widening
- component retrain rerun
- policy redesign
- NAR work

## Success Criteria

- role が 1 行で固定される
- compare は `bets / total net / pure bankroll` で並ぶ
- serving contender か `analysis-first` か `compare reference` かが決まる

## Validation Plan

1. broad September actual-date compare
2. December control actual-date compare
3. 必要なら late-September narrow compare
4. docs / queue / issue thread に decision summary を固定

## Stop Condition

- actual-date compare で baseline に一貫して劣後
- difficult window 改善が弱く、formal uplift だけで role を主張する形になる

## Actual-Date Read

- broad September:
  - baseline `32 bets / -27.3 / pure bankroll 0.2958869306148325`
  - candidate `28 bets / -28.0 / pure bankroll 0.30568644833662867`
- December control:
  - baseline `45 bets / +21.8 / pure bankroll 1.6711564921099862`
  - candidate `121 bets / +2.1999999999999957 / pure bankroll 0.4147571870121275`

## Decision Summary

- `r20260404_jockey_trainer_combo_track_distance_selective_v1` は formal `pass / promote`
- ただし operational role は `compare reference`
- broad September では baseline に対して near-flat で、明確な difficult-window de-risk を示せない
- December control では exposure が `45 -> 121` に増え、`total net` と `pure bankroll` の両方で大きく劣後する
- したがって serving default / seasonal fallback / analysis-first promoted candidate には上げない
