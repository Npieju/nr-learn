# Next Issue: NAR Pre-Race Capture Window Expansion

## Summary

`#101` で strict `pre_race_only` subset の materialization、result-ready primary materialization、benchmark handoff まで実装した。

一方で current live capture は次に留まっている。

- `pre_race_only_rows=281`
- `pre_race_only_races=24`
- dates:
  - `2026-04-06`
  - `2026-04-07`
- `result_ready_races=0`

したがって benchmark rerun を result arrival だけに依存させず、strict `pre_race` row 自体の capture window を広げて、将来の labeled sample pool を増やす corrective が次段となる。

## Objective

local Nankan の strict `pre_race` capture window を拡張し、single upcoming slice ではなく継続的に labeled `pre_race` races が溜まる運用単位を作る。

## Hypothesis

if pre-race recrawl をより広い date window / cadence で回し、strict `pre_race` row を継続的に保存できる, then `#101` の result-ready rerun は sporadic ではなく benchmarkable な sample accumulation として回せる。

## In Scope

- local Nankan race-list discovery cadence
- pre-race recrawl window の date range / overwrite policy
- provenance summary の日次 accumulation read
- capture coverage summary の artifact 化

## Non-Goals

- JRA baseline work
- NAR feature family 実験
- promotion gate redesign
- full historical recrawl

## Success Metrics

- strict `pre_race` capture race count を single 24 races から継続的に積み増せる
- capture coverage を date / races / rows で artifact 化できる
- result-ready rerun 前の operator read が `await_result_arrival` ではなく `capturing_pre_race_pool` として運用可能になる

## Validation Plan

1. current discovery horizon を再確認する
2. wider date window / repeated recrawl の最小 execution source を定義する
3. coverage summary artifact を出す
4. `pre_race` races の accumulation が増えるかを small-scope で確認する

## First Read

current source horizon は `2026-04-30` まで広げても `24 races` のまま頭打ちだった。

- race-list discovery:
  - `2026-04-04 .. 2026-04-30`
  - `rows=24`
  - dates:
    - `2026-04-06`
    - `2026-04-07`
- strict `pre_race` pool:
  - `rows=281`
  - `races=24`
  - date coverage:
    - `2026-04-06: 136`
    - `2026-04-07: 145`

meaning:

- current source で即時に広げられる upcoming horizon は 2 dates / 24 races まで
- したがって capture window expansion の本丸は date range を広げること自体ではなく
  - repeated recrawl cadence
  - 日次 coverage artifact
  - result-ready への accumulation handoff
  の設計になる

next cut:

- repeated recrawl を回したときの coverage accumulation artifact を追加する
- operator が `capturing_pre_race_pool` を日次で読める summary を作る

## Stop Condition

- source 側で upcoming race discovery が 24 races 以上に広がらない
- repeated recrawl でも strict `pre_race` row が有意に増えない
- その場合は capture expansion を止め、result arrival ベースの `#101` rerun を primary path に据え置く
