# Next Issue: Netkeiba Race-Card Load Residual

Historical note:

- この draft は `#29` として exact-safe residual cut の landing まで完了している。
- netkeiba race-card load residual は completed runtime reference として扱い、この文書は historical issue source / artifact reference として使う。

## Summary

`#28` で exact-safe source-range skip を landing し、current mainline の supplemental residual はかなり縮んだ。

accepted results:

- `laptime load 0.1168s -> 0.0045s`
- `corner_passing_order load 0.2325s -> 0.0040s`
- `_merge_supplemental_tables`: `3.2791s - 3.4708s -> 1.0970s`
- reduced smoke: `loading training table 0m15s -> 0m14s`, total `0m25s -> 0m24s`
- summary equivalence: `exact_equal=true`

current supplemental table timing after that cut:

- `netkeiba_race_result_keys`: `load 0.2407s`, `merge 0.0126s`
- `netkeiba_race_card`: `load 0.5575s`, `merge 0.1230s`
- `netkeiba_pedigree`: `load 0.0471s`, `merge 0.0090s`

したがって next exact-safe candidate は broad supplemental revisit ではなく `netkeiba_race_card` load residual の再訪である。

## Objective

current baseline path に直接乗る exact-safe cut で、`netkeiba_race_card` の load residual を減らし、reduced smoke の `loading training table` をさらに押し下げる。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- `netkeiba_race_card` read / normalize / select path
- repeated table-level timing
- reduced smoke and summary equivalence

## Non-Goals

- canonical drift を許容する optimization
- feature / policy / benchmark family changes
- broad data-loader rewrite
- NAR ingestion/readiness

## Success Criteria

- `netkeiba_race_card` load に対する exact-safe cut を 1 本以上 landing できる
- reduced smoke summary equality を維持できる
- `loading training table` が current mainline より改善する

## Suggested Validation

- table-level timing for `netkeiba_race_card`
- repeated phase budget snippets before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`

## Expected Outputs

- accepted exact-safe race-card load cut or explicit reject read
- updated table-level timing
- updated phase budget and reduced smoke read

## Starting Context

current baseline path already includes 次の cut を含んでいる。

- narrow `usecols` for join-based supplemental reads
- no-op selection reuse
- `Index/MultiIndex.isin` based key restriction
- exact-safe source-range skip for out-of-range raw and materialized supplemental tables

materialized race-card path は table-level では効く一方、end-to-end default 勝ちにはまだ届いていない。したがって `#29` の本線は materialized default 化ではなく、baseline path の `netkeiba_race_card` load residual revisit である。

## Current Read

`netkeiba_race_card` を current mainline で分解すると、支配的なのは `read_csv` 本体で、normalize はほぼ no-op だった。

- `raw_read`: 約 `0.63s`
- `normalize`: 約 `0.00s`
- `select/reorder`: 約 `0.03s`

このため first candidate として、join-based supplemental read で `usecols` callable を毎回通す代わりに、header から exact header list を引ける場合は list `usecols` を使う path を追加した。

`#29` の first accepted read は次のとおり。

- `netkeiba_race_card` table-level read: `0.5575s -> 0.5299s` 近辺
- phase budget:
  - `supplemental 1.0970s -> 1.0934s`
  - total `15.08s -> 14.93s`
- reduced smoke `perf_smoke_racecard_usecols_v1`:
  - `loading training table 0m14s`
  - total `0m24s`
  - `summary_equivalence_perf_smoke_supplemental_skip_v1_vs_racecard_usecols_v1.json` で `exact_equal=true`

したがってこの cut は small but safe として mainline に残せる。次の観点は、同じ `race_card` path の残差がまだ `read_csv` 本体なのか、post-read reorder / copy なのかをもう一段詰めることにある。
