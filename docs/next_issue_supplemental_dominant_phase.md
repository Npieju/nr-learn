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
