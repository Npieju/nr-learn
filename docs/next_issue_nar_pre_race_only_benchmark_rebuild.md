# Next Issue: NAR Pre-Race-Only Benchmark Rebuild

## Summary

`#100` で local Nankan collector に provenance persistence を入れ、small-scope live recrawl で strict `pre_race` row の実在を確認した。

- live recrawl:
  - `2026-04-06 / 2026-04-07`
  - `24 races`
- provenance audit:
  - `pre_race_only_rows=281`
  - `post_race_rows=0`
  - `unknown_rows=731941`

したがって次の corrective は、strict `pre_race_only` subset を benchmark-ready な raw / primary へ materialize し、market-aware NAR line を provenance-defensible な universe で読み直すことにある。

## Objective

strict `pre_race_only` provenance bucket だけを使った local Nankan benchmark-ready subset を構築し、current market-aware line を backfilled benchmark ではなく provenance-defensible benchmark として再評価できる導線を作る。

## Hypothesis

if strict `pre_race_only` subset だけで local Nankan raw / primary / evaluation input を再構築できる, then NAR market-aware line は provenance 不足ではなく sample/support の問題として読み直せる。

## Current Read

- `#100` third cut で future / upcoming window の strict `pre_race` row は取得できた
- current backfilled raw は provenance strict filter で落とせる
- ただし現状の benchmark run は `unknown` を大量に含む historical raw を前提にしている
- したがって次に必要なのは collector corrective ではなく subset rebuild と benchmark rerun である

## In Scope

- `src/racing_ml/data/local_nankan_provenance.py`
- `scripts/run_local_nankan_provenance_audit.py`
- local Nankan raw / primary materialization 導線
- strict `pre_race_only` subset CSV の生成
- subset を使った small-scope benchmark / sanity rerun

## Non-Goals

- full historical timestamped recrawl
- JRA baseline work
- NAR feature family の新規 replay
- promotion gate redesign

## Success Metrics

- strict `pre_race_only` subset を raw / primary 相当の入力として再利用できる
- `unknown` / backfilled row を混ぜない benchmark-ready subset が 1 本できる
- small-scope rerun で market-aware / no-market の比較が provenance-defensible に読める
- 次の NAR benchmark read を `backfilled benchmark` ではなく `pre-race-only benchmark` として扱える

## Validation Plan

1. strict `pre_race_only` subset materialization 導線を作る
2. subset row count / race count / date range を検証する
3. subset を使った small-scope market-aware rerun を 1 本回す
4. 必要なら no-market rerun も 1 本合わせて、market dependency を provenance-defensible な universe で再確認する

## Stop Condition

- strict `pre_race_only` subset が benchmark run に必要な最小 row/race を満たさない
- downstream scripts が subset universe を前提に簡潔に再利用できず、full recrawl なしでは意味のある rerun が組めない
- その場合は benchmark rebuild を止め、provenance-aware data collection の継続を優先する
