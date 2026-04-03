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
