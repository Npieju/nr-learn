# Next Issue: Supplemental Dominant Phase Revisit

## Summary

`#27` の revisit で、`_read_csv_tail` の known exact-safe candidate は current mainline にほぼ吸収済みだと確認できた。`compact_reset_late`, `compact_iterable`, `single_chunk_fastpath` はすべて tail equivalence exact gate を通ったが、source diff と repeated A/B の読みでは meaningful remaining gap は見えない。

一方で repeated phase budget は引き続き `_merge_supplemental_tables` を next dominant phase と示している。

- `_read_csv_tail`: `12.8691s - 13.5398s`
- `_merge_supplemental_tables`: `3.2791s - 3.4708s`
- `_ensure_minimum_columns`: `1.1898s - 1.3054s`
- `_append_external_tables`: `0.6867s - 0.6897s`

したがって runtime 本線は read-tail revisit を close し、supplemental dominant phase に戻す。

## Objective

current mainline における `_merge_supplemental_tables` の dominant residual を再度 profiling し、baseline path に直接乗る exact-safe cut を 1 本以上見つける。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- supplemental table load / restrict / merge path
- repeated phase-budget snippets
- reduced smoke and summary equivalence
- existing opt-in materialized supplemental configs の再評価

## Non-Goals

- canonical-only drift を許容する optimization
- feature / policy / benchmark family changes
- NAR ingestion/readiness
- broad data-loader rewrite

## Success Criteria

- `_merge_supplemental_tables` に対する exact-safe cut を 1 本以上 landing できる
- reduced smoke summary equality を維持できる
- current mainline の phase budget で supplemental residual が目に見えて下がる

## Suggested Validation

- repeated phase budget snippets before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`
- if tail path changes indirectly, `scripts/run_tail_loader_equivalence.py --fail-gate exact`

## Expected Outputs

- accepted supplemental exact-safe cut or explicit reject read
- updated supplemental phase budget
- next hotspot decision if supplemental still does not move mainline

## Starting Context

`#23` ではすでに次の baseline-path cut を landing している。

- narrow `usecols` for join-based supplemental reads
- no-op selection reuse in `_select_table_columns(...)`
- `Index/MultiIndex.isin` based restrict in `_restrict_table_to_join_keys(...)`

また `race_card` materialized path は table-level では効く一方、end-to-end default win にはまだ届いていない。したがって `#28` の first priority は、materialize default 化ではなく current baseline path の dominant residual revisit である。

## Current Read

current mainline の `tail_training_table(10k)` で、append 後 base frame を作ってから `_merge_supplemental_tables` を table 単位に timing し直した。

- `laptime`: `load 0.1168s`, `dedupe 0.0015s`, `restrict 0.0026s`, final `restricted_empty`
- `corner_passing_order`: `load 0.2325s`, `dedupe 0.0318s`, `restrict 0.0424s`, final `restricted_empty`
- `netkeiba_race_result_keys`: `load 0.2406s`, `dedupe 0.0056s`, `restrict 0.0170s`, `merge 0.0121s`
- `netkeiba_race_card`: `load 0.5735s`, `dedupe 0.0167s`, `restrict 0.0180s`, `merge 0.1093s`
- `netkeiba_pedigree`: `load 0.0491s`, `dedupe 0.0057s`, `restrict 0.0033s`, `merge 0.0105s`

ここで重要なのは、`laptime` と `corner_passing_order` が単なる年代レンジ不一致ではなく、append 後 base frame が 2025 rows に押し上がっていることだった。current 10k tail raw 自体は 2021 だが、append 後 base frame の `race_id` は `202506040102 .. 202509050805` だった。一方で source 側は次で打ち止めである。

- `laptime`: max `202110030612`
- `corner_passing_order`: max `202110030612`

したがって current runtime の空振り 2 本は、source range beyond-base mismatch と読める。`#28` の next candidate は、materialize default 化ではなく、parseable source-range を使った exact-safe early skip を検討することにある。

## Accepted Result

`#28` では `_load_matching_table(...)` に exact-safe source-range skip を追加した。

- raw range-named CSV は filename prefix `YYYYMMDD-YYYYMMDD` から source date range を読む
- materialized supplemental は `materialized_manifest_file` から `race_id_date_start/end` を読む

あわせて `materialize_supplemental_table(...)` は manifest summary に `race_id_min/max` と `race_id_date_start/end` を出すようにし、`corner_passing_order` manifest も再生成した。

結果は次のとおり。

- `laptime load 0.1168s -> 0.0045s`
- `corner_passing_order load 0.2325s -> 0.0040s`
- `_merge_supplemental_tables`: `3.2791s - 3.4708s -> 1.0970s`
- phase budget total: `18.34s - 18.94s -> 15.08s`
- reduced smoke: `loading training table 0m15s -> 0m14s`, total `0m25s -> 0m24s`
- summary equivalence: `summary_equivalence_perf_smoke_phase_budget_a_vs_supplemental_skip_v1.json` で `exact_equal=true`

この cut により `#28` は close し、next runtime issue は `netkeiba_race_card` load residual を対象にする。
