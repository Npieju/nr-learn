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

## Second Cut

repeated recrawl cadence の operator read を固定するため、strict `pre_race` pool の coverage artifact を追加した。

- script:
  - [run_local_nankan_pre_race_capture_coverage.py](/workspaces/nr-learn/scripts/run_local_nankan_pre_race_capture_coverage.py)
- helper:
  - [local_nankan_provenance.py](/workspaces/nr-learn/src/racing_ml/data/local_nankan_provenance.py)
- tests:
  - [test_local_nankan_provenance.py](/workspaces/nr-learn/tests/test_local_nankan_provenance.py)

出力:

- [local_nankan_pre_race_capture_coverage_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json)
- [local_nankan_pre_race_capture_date_coverage.csv](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_date_coverage.csv)
- [nar_pre_race_capture_coverage_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_capture_coverage_smoke.log)

confirmed read:

- `status=capturing`
- `current_phase=capturing_pre_race_pool`
- `pre_race_only_rows=281`
- `pre_race_only_races=24`
- `result_ready_races=0`
- `pending_result_races=24`
- date coverage:
  - `2026-04-06: 136 rows / 12 races`
  - `2026-04-07: 145 rows / 12 races`
- baseline compare:
  - `delta_pre_race_only_rows=0`
  - `delta_pre_race_only_races=0`
  - `added_dates=[]`

meaning:

- `#102` の success metric のうち
  - capture coverage の artifact 化
  - operator read を `capturing_pre_race_pool` に上げる
  は満たした
- 一方で pool 自体はまだ `2 dates / 24 races / 281 rows` に留まっており、window expansion の本題は未解決

next cut:

- repeated recrawl cadence 自体を bounded loop / snapshot output として実装する
- recrawl のたびに coverage summary を残し、pool growth が本当に起きるか small-scope で確認する

## Stop Condition

- source 側で upcoming race discovery が 24 races 以上に広がらない
- repeated recrawl でも strict `pre_race` row が有意に増えない
- その場合は capture expansion を止め、result arrival ベースの `#101` rerun を primary path に据え置く
