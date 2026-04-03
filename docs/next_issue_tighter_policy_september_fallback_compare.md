# Next Issue: Tighter Policy September Fallback Compare

## Summary

`current_tighter_policy_search_candidate_2025_latest` は formal support が強いが、現状の seasonal ordering では third defensive option に留まっている。

- formal support は `pass / promote`
- September difficult window では baseline より defensive
- ただし current secondary fallback は `current_sep_guard_candidate`

残る論点は、tighter policy が September seasonal fallback ordering で `sep_guard` を更新できるかどうかである。

## Objective

`current_tighter_policy_search_candidate_2025_latest` を `current_sep_guard_candidate` と actual-date September windows で直接比較し、secondary fallback に上げる価値があるかを判定する。

## Hypothesis

if `current_tighter_policy_search_candidate_2025_latest` が broad September と late-September の actual-date compare で `current_sep_guard_candidate` を pure bankroll と total net で一貫して上回る, then tighter policy can be promoted from third defensive option to second seasonal fallback.

## In Scope

- `current_tighter_policy_search_candidate_2025_latest`
- `current_sep_guard_candidate_2025_latest`
- broad September difficult window compare
- late-September compare
- seasonal fallback ordering decision

## Non-Goals

- baseline default replacement
- new feature family
- formal threshold frontier rerun
- NAR work

## Success Criteria

- September difficult windows で `sep_guard` と `tighter policy` の優劣が 1 本に決まる
- `second seasonal fallback` か `third defensive option` かを fixed wording で決められる

## Validation Plan

- broad September 8 日 compare
- late-September 5 日 compare
- `bets`, `total net`, `pure bankroll` を比較
- 必要なら bankroll sweep の best result も確認する

## Stop Condition

- broad September と late-September が逆方向で、role を 1 本に決められない
- `sep_guard` より一貫して弱く、secondary fallback を主張できない
