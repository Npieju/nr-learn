# Next Issue: Jockey Trainer Combo Style-Distance Actual-Date Role Split

## Summary

`#108` の selective child `r20260404_jockey_trainer_combo_style_distance_selective_v1` は formal `pass / promote` に到達した。

- evaluation
  - `auc=0.8377559190837222`
  - `top1_roi=0.8022312793589389`
  - `ev_top1_roi=0.7669936446532192`
  - `wf_nested_test_roi_weighted=1.168363624344555`
  - `wf_nested_test_bets_total=966`
- formal
  - `weighted_roi=1.023657866236539`
  - `bets_total=1040`
  - `feasible_fold_count=4`

ただし、JRA の serving role は formal `pass / promote` だけでは決めない。次に必要なのは current operational default と difficult / control windows の actual-date compare を読み、この line を

- serving contender
- analysis-first promoted candidate
- compare reference

のどこに置くかを固定することである。

## Objective

`r20260404_jockey_trainer_combo_style_distance_selective_v1` を current operational default と actual-date windows で比較し、operational role を固定する。

## Hypothesis

if `r20260404_jockey_trainer_combo_style_distance_selective_v1` が September difficult window で baseline を de-risk しつつ December control を壊さない, then this line can be considered a serving contender rather than an analysis-first promoted candidate.

## Current Read

- formal `pass / promote`
- evaluation:
  - `auc=0.8377559190837222`
  - `top1_roi=0.8022312793589389`
  - `ev_top1_roi=0.7669936446532192`
  - `wf_nested_test_roi_weighted=1.168363624344555`
  - `wf_nested_test_bets_total=966`
- formal:
  - `weighted_roi=1.023657866236539`
  - `bets_total=1040`
  - `feasible_fold_count=4`
- wf read:
  - dominant failure reason は `min_bets`
  - feasible folds は `4/5`
  - binding source は全 fold で `ratio`

## In Scope

- broad September actual-date compare
- December control actual-date compare
- 必要なら late-September narrow compare
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
- serving contender か `analysis-first promoted candidate` か `compare reference` かが決まる

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
  - baseline `current_recommended_serving_2025_latest`
    - `33 bets`
    - `total net = -20.0`
    - `pure bankroll = 0.3931722898269604`
  - candidate `r20260404_jockey_trainer_combo_style_distance_selective_v1`
    - `4 bets`
    - `total net = -4.0`
    - `pure bankroll = 0.841869212962963`
- December control:
  - baseline `current_recommended_serving_2025_latest`
    - `17 bets`
    - `total net = -5.199999999999999`
    - `pure bankroll = 0.7886889523160848`
  - candidate `r20260404_jockey_trainer_combo_style_distance_selective_v1`
    - `8 bets`
    - `total net = -8.0`
    - `pure bankroll = 0.7623626543209877`

## Decision Summary

- `r20260404_jockey_trainer_combo_style_distance_selective_v1` は formal `pass / promote`
- operational role は `analysis-first promoted candidate`
- broad September difficult window では baseline より strong downside control を示す
- ただし December control window では baseline の positive carry を上回れず、`total net` と `pure bankroll` の両方で劣後する
- したがって serving default / seasonal fallback には上げない