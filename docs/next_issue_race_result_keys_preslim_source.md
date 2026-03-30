# Next Issue: Pre-Slim Race-Result-Keys Source Candidate

## Summary

`#30` で `netkeiba_race_card` の pre-slim / materialized source は default 昇格に到達した。これにより supplemental residual の主対象は次に `netkeiba_race_result_keys` 側へ移る。

current accepted state:

- `corner_passing_order` source-range skip は live
- `netkeiba_race_card` materialized source は live
- `exact header list usecols` は live

したがって next path は、parser micro tuning の延長ではなく、`netkeiba_race_result_keys` の pre-slim / pre-shaped source candidate を独立に評価することである。

## Objective

pre-slim race-result-keys source candidate が、current accepted supplemental path を table-level と reduced smoke の両方で上回れるかを評価する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- race-result-keys source shaping / materialization helper
- repeated table-level timing
- reduced smoke and summary equivalence
- new source artifact / manifest handling

## Non-Goals

- model / feature / policy behavior の変更
- canonical-only drift の受容
- broad loader rewrite
- NAR work

## Success Criteria

- pre-slim race-result-keys candidate が current accepted path を table-level timing で明確に上回る
- reduced smoke が exact-equal を維持する
- end-to-end wall time が current mainline より改善する

## Suggested Validation

- table-level race-result-keys timing before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`
- if new source artifacts are generated, explicit manifest and provenance

## Starting Context

`#28` と `#29` の profiling では、supplemental residual の dominant table は `netkeiba_race_card` だったが、これが default 昇格すると次点は `netkeiba_race_result_keys` に寄る。

`#31` はまず current phase budget を current mainline で再計測し、

- `netkeiba_race_result_keys` load
- `netkeiba_race_result_keys` restrict / dedupe / merge
- reduced smoke end-to-end

を 1 本の read に揃えてから、pre-slim candidate を比較する。
