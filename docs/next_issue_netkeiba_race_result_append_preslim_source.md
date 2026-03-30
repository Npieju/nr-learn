# Next Issue: Pre-Slim Netkeiba Race-Result Append Source Candidate

## Summary

`#31` で `netkeiba_race_result_keys` の materialized source は default 昇格に到達した。これにより current load residual の主対象は append path の `netkeiba_race_result` 側へ移る。

current accepted state:

- `corner_passing_order` source-range skip は live
- `netkeiba_race_card` materialized source は live
- `netkeiba_race_result_keys` materialized source は live
- `exact header list usecols` は live

したがって next path は supplemental join table ではなく、append table `netkeiba_race_result` の pre-slim / pre-shaped source candidate を独立に評価することである。

## Objective

pre-slim netkeiba race-result append candidate が、current accepted append path を table-level と reduced smoke の両方で上回れるかを評価する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- append source shaping / materialization helper
- repeated table-level timing
- reduced smoke and summary equivalence
- new source artifact / manifest handling

## Non-Goals

- model / feature / policy behavior の変更
- canonical-only drift の受容
- broad loader rewrite
- NAR work

## Success Criteria

- pre-slim append candidate が current accepted path を table-level timing で明確に上回る
- reduced smoke が exact-equal を維持する
- end-to-end wall time が current mainline より改善する

## Suggested Validation

- table-level append timing before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`
- if new source artifacts are generated, explicit manifest and provenance

## Starting Context

current mainline の 10k tail first read では、`netkeiba_race_result load=0.52s` がまだ最大の個別 source である。

`#32` はまず current append residual を current mainline で再計測し、

- `netkeiba_race_result` load
- recent-date filter / dedupe / append
- reduced smoke end-to-end

を 1 本の read に揃えてから、pre-slim append candidate を比較する。

## Current Read

current mainline の append-path first read は次のとおり。

- `append_load_sec=0.5518`
- `recent_filter_sec=0.0297`
- `dedupe_concat_sec=0.0917`
- `sort_tail_sec=0.2063`
- `append_rows_final=101810`
- `recent_date_floor=2021-01-16`

つまり current residual の本体は、recent-date filter や dedupe ではなく `netkeiba_race_result` の load 自体と、その後の sort/tail である。

このため `#32` の next move は、small post-read tweak ではなく、

- pre-slim append source
- append-specific materialization
- exact-equal summary compare

の順で candidate を比較することである。

append / supplemental の両方を同じ手順で materialize できるように、`run_materialize_supplemental_table.py` は `--table-kind append|supplemental|auto` を受ける generic runner に拡張した。

そのうえで first append candidate として `netkeiba_race_result` の full materialized source を作成した。

- output: `data/processed/append/netkeiba_race_result.csv`
- manifest: `artifacts/reports/materialize_netkeiba_race_result_append.json`

loader-only repeated compare は次のとおりだった。

- current: `[13.2767, 13.6774, 13.6150]`, average `13.5230s`
- full materialized append candidate: `[13.7303, 13.9262, 14.0185]`, average `13.8917s`

したがって current read では、「full materialized append source は standardization value はあるが、performance candidate としては reject」である。`#32` の next move は、full materialize ではなく recent-window を意識した narrower append source candidate に移ることになる。
