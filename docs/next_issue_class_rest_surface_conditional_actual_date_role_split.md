# Next Issue: Class-Rest-Surface Conditional Actual-Date Role Split

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
