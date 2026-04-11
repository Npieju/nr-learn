# Next Issue: Class-Rest-Surface Conditional Actual-Date Role Split

Historical note:

- この draftは `#95` 相当の actual-date role split decision まで完了している。
- conditional selective line の operational role は確定済みであり、この文書は historical issue source / role-split reference として使う。

## Summary

`#94` の selective child `r20260404_class_rest_surface_conditional_selective_v1` は formal `pass / promote` まで通った。

- `auc=0.8426169492248933`
- `top1_roi=0.8082087836284203`
- `ev_top1_roi=0.8213612324672338`
- `wf_nested_test_roi_weighted=0.7632385120350111`
- `wf_nested_test_bets_total=457`
- held-out formal `weighted_roi=1.2311149102465346`
- `formal_benchmark_bets_total=9806`
- `formal_benchmark_feasible_fold_count=3`

ただし、serving default に寄せるには actual-date role が未読である。

## Objective

`class / rest / surface` conditional selective line の actual-date role を baseline と比較し、serving default contender か `analysis-first promoted candidate` かを切り分ける。

## Hypothesis

if `r20260404_class_rest_surface_conditional_selective_v1` が formal だけでなく actual-date windows でも baseline 比で安定した downside control か upside を示す, then this family can advance from promoted candidate to serving-role contender.

## In Scope

- `r20260404_class_rest_surface_conditional_selective_v1`
- current recommended serving baseline compare
- September / December / latest windows の actual-date compare
- role split conclusion

## Non-Goals

- new feature family
- policy rewrite
- NAR work

## Success Criteria

- actual-date compare で role が 1 つに絞れる
- serving default contender か `analysis-first promoted candidate` かを明文化できる

## Validation Plan

- compare dashboard / serving smoke artifact を baseline と並べる
- September, December, latest windows の net, bets, pure bankroll を読む
- operational role を fixed wording でまとめる

## Stop Condition

- actual-date read が mixed で role を 1 本に決められない
- baseline より明確に悪化して serving role を主張できない

## Current Read

### Broad September 2025

- baseline `current_recommended_serving_2025_latest`
  - `32 bets`
  - `total net = -27.3`
  - `pure bankroll = 0.2958869306148325`
- candidate `r20260404_class_rest_surface_conditional_selective_v1`
  - `5 bets`
  - `total net = -3.6`
  - `pure bankroll = 0.8808888460219477`

Read:

- September difficult window では candidate が baseline より明確に上
- `right_minus_left_total_policy_net = +23.7`
- `pure bankroll delta = +0.5850019154071152`
- ただし bets は `32 -> 5` まで縮むため、December / latest の control read なしには serving default へは上げない

Artifacts:

- `artifacts/reports/serving_smoke_compare_sep25_crs_cond_base_vs_sep25_crs_cond_cand.json`
- `artifacts/reports/serving_stateful_bankroll_sweep_sep25_crs_cond_base_vs_sep25_crs_cond_cand.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_sep25_crs_cond_base_vs_sep25_crs_cond_cand.json`

### December 2025 Control

- baseline `current_recommended_serving_2025_latest`
  - `45 bets`
  - `total net = +21.8`
  - `pure bankroll = 1.6711564921099862`
- candidate `r20260404_class_rest_surface_conditional_selective_v1`
  - `30 bets`
  - `total net = -9.3`
  - `pure bankroll = 0.7513642270122226`

Read:

- December control window では candidate が baseline に明確に劣後
- `right_minus_left_total_policy_net = -31.1`
- `pure bankroll delta = -0.9197922650977636`
- September downside control は確認できたが、control window を loss に反転させるため serving default contender には上げない

Artifacts:

- `artifacts/reports/serving_smoke_compare_dec25_crs_cond_base_vs_dec25_crs_cond_cand.json`
- `artifacts/reports/serving_stateful_bankroll_sweep_dec25_crs_cond_base_vs_dec25_crs_cond_cand.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_dec25_crs_cond_base_vs_dec25_crs_cond_cand.json`

## Decision

`r20260404_class_rest_surface_conditional_selective_v1` の operational role は `analysis-first promoted candidate` に固定する。

Reason:

- broad September difficult window では baseline より明確に良い
- ただし December control window では baseline の positive carry を壊す
- よって serving default には上げず、difficult-regime compare reference として扱う
