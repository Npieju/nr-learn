# Next Issue: Gate Frame Course September Fallback Compare

## Summary

`r20260403_gate_frame_course_regime_extension_v1` は formal `pass / promote` まで通り、actual-date role split では次が確定した。

- September difficult window は baseline より明確に改善
- December control window は baseline より明確に悪化
- したがって broad serving default ではなく `analysis-first promoted candidate`

残論点は、この line が September seasonal fallback 候補として既存の `current_sep_guard_candidate` を脅かすかどうかである。

## Objective

`r20260403_gate_frame_course_regime_extension_v1` を September difficult regime で `current_sep_guard_candidate` と比較し、seasonal fallback contender に進める価値があるかを判定する。

## Hypothesis

if `r20260403_gate_frame_course_regime_extension_v1` が September difficult windows で `current_sep_guard_candidate` と同等以上の downside control と bankroll を示す, then this family can advance from analysis-first promoted candidate to seasonal-fallback contender.

## In Scope

- `r20260403_gate_frame_course_regime_extension_v1`
- `current_sep_guard_candidate`
- September difficult windows の actual-date compare
- role wording: contender か compare reference か

## Non-Goals

- broad serving default promotion
- new feature family
- policy rewrite
- NAR work

## Success Criteria

- September difficult windows で `current_sep_guard_candidate` と比較した role を 1 つに絞れる
- contender に進めるか、reference のまま据え置くかを fixed wording で決められる

## Validation Plan

- late-September 5-day と broad September difficult window の compare を並べる
- `bets`, `total net`, `pure bankroll`, `bankroll sweep` を比較する
- `current_sep_guard_candidate` に勝てないなら fallback 候補には上げない

## Stop Condition

- `current_sep_guard_candidate` 比で優位が見えない
- September window の読みが mixed で fallback 候補を主張できない

## Actual-Date Read

### Broad September Difficult Window

- compare artifact:
  - `artifacts/reports/dashboard/serving_compare_dashboard_sep25_guard_vs_sep25_gfc.json`
- `current_sep_guard_candidate_2025_latest`
  - `9 bets`
  - `total net = -4.3`
  - `pure bankroll = 0.9995842542264094`
- `r20260403_gate_frame_course_regime_extension_v1`
  - `5 bets`
  - `total net = -3.6`
  - `pure bankroll = 0.8808888460219477`
- read:
  - candidate は net だけ `+0.7` 改善
  - ただし pure bankroll は `-0.11869540820446167` 劣後
  - bankroll sweep の best result は `current_sep_guard_candidate_2025_latest` 単独採用

### Late-September 5-Day Window

- compare artifact:
  - `artifacts/reports/dashboard/serving_compare_dashboard_late25_guard_vs_late25_gfc.json`
- `current_sep_guard_candidate_2025_latest`
  - `6 bets`
  - `total net = -6.0`
  - `pure bankroll = 0.9857901270555528`
- `r20260403_gate_frame_course_regime_extension_v1`
  - `3 bets`
  - `total net = -1.6`
  - `pure bankroll = 0.9319444444444444`
- read:
  - candidate は net だけ `+4.4` 改善
  - ただし pure bankroll は `-0.05384568261110845` 劣後
  - bankroll sweep best result は floor `1.0` で mixed path
  - stage use counts は `sep_guard=3`, `gfc=2`
  - pure stage winner は依然 `current_sep_guard_candidate_2025_latest`

## Decision

`r20260403_gate_frame_course_regime_extension_v1` は September fallback contender に上げない。

理由:

- broad September でも late-September でも pure bankroll で `current_sep_guard_candidate` に負ける
- candidate の改善は `total net` 側に限られ、drawdown / bankroll preservation を崩す
- mixed bankroll sweep は参考にはなるが、fallback の primary role を移す根拠には足りない

固定 wording:

- `current_sep_guard_candidate` は September seasonal fallback のまま据え置く
- `r20260403_gate_frame_course_regime_extension_v1` は `analysis-first promoted candidate / compare reference` に留める
