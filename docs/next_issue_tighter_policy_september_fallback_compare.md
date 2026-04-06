# Next Issue: Tighter Policy September Fallback Compare

Historical note:

- この draft は `#89` として seasonal fallback ordering decision まで完了している。
- `current_sep_guard_candidate` を second seasonal fallback に据え置く判断は確定済みであり、この文書は historical decision reference として使う。

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

## Actual-Date Read

### Broad September Difficult Window

- `current_sep_guard_candidate_2025_latest`
  - `9 bets`
  - `total net = -4.3`
  - `pure bankroll = 0.9995842542264094`
- `current_tighter_policy_search_candidate_2025_latest`
  - `9 bets`
  - `total net = -4.3`
  - `pure bankroll = 0.8394653592417107`
- read:
  - `bets` と `total net` は同値
  - `pure bankroll` は tighter policy が `-0.16011889498469867` 劣後

artifact:

- `artifacts/reports/dashboard/serving_compare_dashboard_current_sep_guard_candidate_2025_latest_sep25_guard_vs_tighter_vs_current_tighter_policy_search_candidate_2025_latest_sep25_guard_vs_tighter.json`

### Late-September 5-Day Window

- `current_sep_guard_candidate_2025_latest`
  - `6 bets`
  - `total net = -6.0`
  - `pure bankroll = 0.9857901270555528`
- `current_tighter_policy_search_candidate_2025_latest`
  - `6 bets`
  - `total net = -6.0`
  - `pure bankroll = 0.7701189959490742`
- read:
  - `bets` と `total net` は同値
  - `pure bankroll` は tighter policy が `-0.21567113110647862` 劣後

artifact:

- `artifacts/reports/dashboard/serving_compare_dashboard_current_sep_guard_candidate_2025_latest_late25_guard_vs_tighter_vs_current_tighter_policy_search_candidate_2025_latest_late25_guard_vs_tighter.json`

## Decision Summary

`current_tighter_policy_search_candidate_2025_latest` は secondary fallback に上げない。

- broad September と late-September の両方で `bets` と `total net` は `sep_guard` と同値
- そのうえで `pure bankroll` は両 window で `sep_guard` が明確に上
- したがって seasonal fallback ordering は変えない

Conclusion

- first seasonal alias remains `current_long_horizon_serving_2025_latest`
- second seasonal fallback remains `current_sep_guard_candidate`
- `current_tighter_policy_search_candidate_2025_latest` stays third defensive option
