# Next Issue: Pace Closing-Fit September Fallback Compare

## Summary

`#84` で `r20260403_pace_closing_fit_selective_v1` は serving default ではなく `analysis-first promoted candidate` と確定した。

- broad September difficult window では baseline より良い
- December control window では baseline より明確に悪い

残る自然な論点は、`current_sep_guard_candidate` に対する September fallback contender になれるかどうかである。

## Objective

`r20260403_pace_closing_fit_selective_v1` を `current_sep_guard_candidate` と actual-date September windows で比較し、September seasonal fallback contender に上げる価値があるかを判定する。

## Hypothesis

if `r20260403_pace_closing_fit_selective_v1` が broad September と late-September の actual-date compare で `current_sep_guard_candidate` を pure bankroll と net で上回る, then this line can be considered a September fallback contender.

## In Scope

- `current_sep_guard_candidate`
- `r20260403_pace_closing_fit_selective_v1`
- broad September compare
- late-September compare
- fallback role decision

## Non-Goals

- serving default replacement
- new feature family
- policy rewrite
- NAR work

## Success Criteria

- September difficult windows で `sep_guard` と `pace-closing-fit` の優劣が 1 本に決まる
- seasonal fallback contender か compare reference かを固定できる

## Validation Plan

- broad September 8 日 compare
- late-September compare
- `bets / total net / pure bankroll` を比較
- operational wording を `seasonal fallback contender` か `compare reference` に固定

## Stop Condition

- broad September と late-September が逆方向で、role を 1 本に決められない
- `sep_guard` より一貫して弱く、fallback contender を主張できない

## Actual-Date Read

Broad September difficult window:

- `current_sep_guard_candidate_2025_latest`
  - `9 bets`
  - `total net = -4.3`
  - `pure bankroll = 0.9995842542264094`
- `r20260403_pace_closing_fit_selective_v1`
  - `3 bets`
  - `total net = -3.0`
  - `pure bankroll = 0.892891589506173`

Late-September 5-day window:

- `current_sep_guard_candidate_2025_latest`
  - `6 bets`
  - `total net = -6.0`
  - `pure bankroll = 0.9857901270555528`
- `r20260403_pace_closing_fit_selective_v1`
  - `2 bets`
  - `total net = -2.0`
  - `pure bankroll = 0.9184027777777779`

## Decision Summary

`r20260403_pace_closing_fit_selective_v1` は September difficult windows で baseline より defensive だが、September fallback role では `current_sep_guard_candidate_2025_latest` を更新しない。

- broad September では `pace-closing-fit` の net は小さいが、pure bankroll は `sep_guard` が上
- late-September でも同じで、`sep_guard` が pure bankroll で上

したがって operational role は変わらない。

- `current_sep_guard_candidate_2025_latest` を September seasonal fallback のまま据え置く
- `r20260403_pace_closing_fit_selective_v1` は `analysis-first promoted candidate` に留める
