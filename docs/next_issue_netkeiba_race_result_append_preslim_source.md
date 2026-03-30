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
