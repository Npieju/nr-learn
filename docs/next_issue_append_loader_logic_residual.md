# Next Issue: Append Loader Logic Residual

## Summary

`#32` で append source candidate を 2 本見たが、どちらも current mainline に勝てなかった。

- full-history materialized append: reject
- exact-floor narrow materialized append: reject

したがって次の本線は source shaping ではなく、`_append_external_tables(...)` 自体の exact-safe logic residual を削ることである。

## Objective

append source は据え置いたまま、`_append_external_tables(...)` の load/filter/dedupe/sort tail path で exact-safe に削れる固定コストを見つける。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- append-path profiling
- exact-safe logic cuts
- reduced smoke and summary equivalence

## Non-Goals

- model / feature / policy behavior の変更
- canonical-only drift の受容
- broad loader rewrite
- NAR work

## Success Criteria

- append residual を phase 単位で再現よく説明できる
- exact-safe logic cut が loader-only と reduced smoke の両方で勝つ
- summary equivalence が substantive drift なしで通る

## Suggested Validation

- append phase timing before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`

## Starting Context

current append-path read は次のとおり。

- `append_load_sec=0.5518`
- `recent_filter_sec=0.0297`
- `dedupe_concat_sec=0.0917`
- `sort_tail_sec=0.2063`

source-shaping candidate の結果は次のとおり。

- full-history materialized append: current `13.5230s` vs candidate `13.8917s`
- exact-floor narrow append: current `13.7082s` vs candidate `13.8568s`

したがって `#33` は source を変えるのではなく、

- recent-date filter
- concat / dedupe
- sort / tail

の exact-safe logic cut を優先する issue である。
