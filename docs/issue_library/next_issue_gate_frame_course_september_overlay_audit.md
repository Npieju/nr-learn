# Next Issue: Gate Frame Course September Overlay Audit

## Summary

`#81` で `r20260403_gate_frame_course_regime_extension_v1` は `current_sep_guard_candidate` を September fallback contender として置き換えられないと確定した。

- broad September:
  - sep guard `9 bets / net -4.3 / pure bankroll 0.9995842542264094`
  - gate-frame-course candidate `5 bets / net -3.6 / pure bankroll 0.8808888460219477`
- late-September:
  - sep guard `6 bets / net -6.0 / pure bankroll 0.9857901270555528`
  - gate-frame-course candidate `3 bets / net -1.6 / pure bankroll 0.9319444444444444`

一方で late-September 5-day compare では、bankroll sweep best result が pure sep guard ではなく mixed path だった。

- best hybrid final bankroll: `1.0132697207948398`
- pure sep guard final bankroll: `0.9857901270555528`
- stage use counts:
  - `current_sep_guard_candidate_2025_latest = 3`
  - `current_recommended_serving_2025_latest + r20260403_gate_frame_course_regime_extension_v1 = 2`

次に問うべきは、candidate 単独の置換価値ではなく、September difficult regime での narrow overlay / complementarity が実在するかである。

## Objective

`current_sep_guard_candidate` を primary fallback に維持したまま、`r20260403_gate_frame_course_regime_extension_v1` を September overlay として部分採用する価値があるかを判定する。

## Hypothesis

if gate-frame-course candidate の有効日は late-September の一部に限定され、pure sep guard より良い hybrid bankroll path を再現できる, then this line can advance from compare reference to September overlay candidate without replacing the primary seasonal fallback.

## In Scope

- `current_sep_guard_candidate_2025_latest`
- `current_recommended_serving_2025_latest`
- `r20260403_gate_frame_course_regime_extension_v1`
- September difficult windows の compare / bankroll sweep artifacts
- overlay role wording: compare reference か, overlay candidate か

## Non-Goals

- broad serving default promotion
- September fallback primary の置換
- new feature family
- NAR work

## Success Criteria

- candidate 単独ではなく overlay としての役割を 1 つに固定できる
- broad September と late-September の両方で、overlay hypothesis の根拠または否定根拠を artifact で示せる
- next step が `explicit overlay profile` 実装か `close as compare reference` の二択に絞れる

## Validation Plan

- `#81` の broad September / late-September compare と bankroll sweep を再読する
- mixed path が late-September 固有の偶然か、repeatable な complementarity かを判定する
- 必要なら最小の spot rerun で September subset compare を追加する

## Stop Condition

- mixed path の優位が late-September 5-day の局所読みにしか見えない
- broad September で overlay value を説明できない
- explicit overlay profile を作るより compare reference のままの方が defensible

## Actual-Date Read

### Broad September Difficult Window

- artifact:
  - `artifacts/reports/dashboard/serving_compare_dashboard_sep25_guard_vs_sep25_gfc.json`
- pure stage:
  - sep guard `pure bankroll = 0.9995842542264094`
  - gate-frame-course candidate `pure bankroll = 0.8808888460219477`
- bankroll sweep:
  - best result は `current_sep_guard_candidate_2025_latest` 単独採用
  - `stage_use_counts = {sep_guard: 8, gfc: 0}`

### Late-September 5-Day Window

- artifact:
  - `artifacts/reports/dashboard/serving_compare_dashboard_late25_guard_vs_late25_gfc.json`
- pure stage:
  - sep guard `pure bankroll = 0.9857901270555528`
  - gate-frame-course candidate `pure bankroll = 0.9319444444444444`
- bankroll sweep:
  - best result `final_bankroll = 1.0132697207948398`
  - `stage_use_counts = {sep_guard: 3, gfc: 2}`
  - mixed path は `2025-09-20`, `2025-09-21` だけ candidate を採用

## Decision

overlay hypothesis は close する。

理由:

- broad September では overlay value が出ていない
- pure stage でも mixed sweep でも sep guard が primary であり続ける
- mixed improvement は late-September 5-day の局所 path に限られ、explicit overlay profile を作る根拠としては弱い

固定 wording:

- `current_sep_guard_candidate` は September seasonal fallback のまま据え置く
- `r20260403_gate_frame_course_regime_extension_v1` は `analysis-first promoted candidate / compare reference` に留める
- September overlay candidate には進めない
