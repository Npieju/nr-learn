# Next Issue: Pace Closing-Fit Actual-Date Role Split

## Summary

`#83` の first child `r20260403_pace_closing_fit_selective_v1` は formal `pass / promote` まで通った。

- `auc=0.8411224406691354`
- `ev_top1_roi=0.6212479889680533`
- held-out formal `weighted_roi=1.0307760253096685`
- `formal_benchmark_bets_total=938`
- `bets / races / bet_rate = 938 / 6244 = 15.02%`

ただし、serving default に寄せるには actual-date role が未読である。

## Objective

`pace / closing-fit` selective line の actual-date role を baseline と比較し、serving default contender か `analysis-first promoted candidate` かを切り分ける。

## Hypothesis

if `r20260403_pace_closing_fit_selective_v1` が formal だけでなく actual-date windows でも baseline 比で安定した downside control か upside を示す, then this family can advance from promoted candidate to serving-role contender.

## In Scope

- `r20260403_pace_closing_fit_selective_v1`
- current recommended serving baseline compare
- September / December / latest windows の actual-date compare
- role split conclusion

## Non-Goals

- new feature family
- policy rewrite
- NAR work

## Success Criteria

- actual-date compare で role が 1 つに絞れる
- serving default 候補か `analysis-first promoted candidate` かを明文化できる

## Validation Plan

- compare dashboard / serving smoke artifact を baseline と並べる
- September, December, latest windows の net, bets, bankroll を読む
- operational role を fixed wording でまとめる

## Stop Condition

- actual-date read が mixed で role を 1 本に決められない
- baseline より明確に悪化して serving role を主張できない

## Actual-Date Read

Broad September difficult window は candidate が baseline より明確に良い。

- baseline
  - `32 bets`
  - `total net = -27.3`
  - `pure bankroll = 0.2958869306148325`
- candidate `r20260403_pace_closing_fit_selective_v1`
  - `3 bets`
  - `total net = -3.0`
  - `pure bankroll = 0.892891589506173`

December control window は逆に baseline が明確に良い。

- baseline
  - `45 bets`
  - `total net = +21.8`
  - `pure bankroll = 1.6711564921099862`
- candidate `r20260403_pace_closing_fit_selective_v1`
  - `0 bets`
  - `total net = 0.0`
  - `pure bankroll = 1.0`

## Decision Summary

`r20260403_pace_closing_fit_selective_v1` は formal `pass / promote` だが、actual-date role は mixed ではなく明確に split している。

- September difficult window では baseline より defensive
- December control window では baseline の upside を再現できない

したがって operational role は `analysis-first promoted candidate` に据え置く。serving default には上げない。

次の execution source は、`current_sep_guard_candidate` に対する September fallback contender comparison である。
