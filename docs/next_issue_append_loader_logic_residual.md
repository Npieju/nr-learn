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

## Current Read

first accepted cut として、append_frame を concat 前に `max_rows + len(base_frame)` まで prelimit する path を入れた。final tail が `max_rows` 本である以上、base frame 全量が final tail に残ったとしても append 側はそのぶん上乗せした budget を持てば exact-safe に比較できる、という前提である。

same-input micro compare は次のとおり。

- old path: `0.3561s`
- new path: `0.1438s`
- `equal=True`
- `limited_rows=20000`

reduced smoke も通っている。

- candidate smoke: `perf_smoke_append_logic_v1`
- `loading training table 0m15s`, total `0m25s`
- summary compare:
  - `summary_equivalence_perf_smoke_racecard_default_mainline_v1_vs_append_logic_v1.json`
  - `exact_equal=true`

したがって current read は、「append prelimit logic cut は exact-safe かつ accepted」である。`#33` の next move は、この cut を base に append residual の次の logic surface を探すことである。

second accepted cut として、recent-date filter の `.copy()` を除去した。append path では filter 後の frame をそのまま read-only で prelimit / concat / drop_duplicates に流すだけなので、この copy は不要である。

ad hoc read は次のとおり。

- with copy: `0.0274s`
- without copy: `0.0142s`
- `equal=True`

reduced smoke でも差分はきれいに保たれた。

- candidate smoke: `perf_smoke_append_logic_v2`
- `loading training table 0m13s`, total `0m23s`
- summary compare:
  - `summary_equivalence_perf_smoke_append_logic_v1_vs_v2.json`
  - `exact_equal=true`

したがって current best read は、「append logic residual は `prelimit` と `recent-filter no-copy` の 2 本で段階的に改善しており、current mainline は `loading training table 0m13s` まで前進した」である。

third accepted cut として、append table に explicit `keep_columns` を与え、base frame に存在しない不要列を source read 時点で落とした。

- dropped columns:
  - `jockey_key`
  - `trainer_key`
  - `owner_key`
  - `passing_order_raw`

この cut は current mainline config に直接入れて reduced smoke で確認した。

- candidate smoke: `perf_smoke_append_keepcols_mainline_v1`
- `loading training table 0m14s`, total `0m24s`
- summary compare:
  - `summary_equivalence_perf_smoke_append_logic_v2_vs_keepcols_mainline_v1.json`
  - `exact_equal=true`

したがって current best read は、「append logic residual は `prelimit`、`recent-filter no-copy`、`append keep_columns` の 3 本が accepted であり、current mainline は same-summary のまま `0m24s` 帯を維持している」である。
