# Next Issue: Pre-Slim Race-Result-Keys Source Candidate

Historical note:

- この draft は `#31` として pre-slim race-result-keys source evaluation まで完了している。
- pre-slim race-result-keys source candidate は completed runtime reference として扱い、この文書は historical issue source / artifact reference として使う。

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

## Current Read

current mainline の 10k tail first read は次のとおり。

- `tail_training_table_sec=13.853`
- `netkeiba_race_result load=0.52s`
- `netkeiba_race_card load=0.3207s`
- `netkeiba_race_result_keys load=0.23s`
- `netkeiba_pedigree load=0.0483s`

このため、race-card 昇格後の next supplemental residual は `netkeiba_race_result_keys` と見てよい。

そのうえで `netkeiba_race_result_keys` を materialize し、

- `data/processed/supplemental/netkeiba_race_result_keys.csv`
- `artifacts/reports/supplemental_materialize_netkeiba_race_result_keys.json`

を生成した。

loader-only repeated compare は次のとおり。

- current: `[14.0172, 13.9594, 15.1751]`, average `14.3839s`
- materialized candidate: `[14.1395, 13.8796, 14.1632]`, average `14.0607s`

reduced smoke も通っている。

- candidate: `perf_smoke_result_keys_preslim_v1`
- `loading training table 0m15s`, total `0m25s`
- summary compare:
  - `summary_equivalence_perf_smoke_racecard_default_mainline_v1_vs_result_keys_preslim_v1.json`
  - difference は `run_context.data_config` の 1 件だけ

したがって final read は「materialized race-result-keys candidate は exact-equivalent behavior を保ったまま loader-only で持続的に勝ち、reduced smoke でも near-par を維持した」である。これにより `netkeiba_race_result_keys` の materialized source は default 昇格でよい。
