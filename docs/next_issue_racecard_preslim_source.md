# Next Issue: Pre-Slim Race-Card Source Candidate

Historical note:

- この draft は `#30` として pre-slim race-card source evaluation まで完了している。
- pre-slim race-card source candidate は completed runtime reference として扱い、この文書は historical issue source / artifact reference として使う。

## Summary

`#29` で `netkeiba_race_card` residual を parser / pre-read に切り分けた結果、残差の本体は parser 側だと分かった。一方で、small exact-safe parser tweak は durable win にならなかった。

current accepted state:

- `exact header list usecols` は live baseline path に対する small accepted win

rejected follow-ups:

- `_select_table_columns` reorder/copy skip: same-summary, no wall-time win
- `memory_map=True`: same-summary, worse phase budget / reduced smoke
- explicit dtype hints: table-level average でも current accepted path に負ける

したがって next path は、baseline parser tuning の延長ではなく、pre-slim / pre-shaped race-card source artifact を別トラックとして評価することである。

## Objective

pre-slim race-card source candidate が、current accepted race-card path を table-level と reduced smoke の両方で上回れるかを評価する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- race-card source shaping / materialization helper
- repeated table-level timing
- reduced smoke and summary equivalence
- new source artifact / manifest handling

## Non-Goals

- model / feature / policy behavior の変更
- canonical-only drift の受容
- broad loader rewrite
- NAR work

## Success Criteria

- pre-slim race-card candidate が current accepted race-card path を table-level timing で明確に上回る
- reduced smoke が exact-equal を維持する
- end-to-end wall time が current mainline より改善する

## Suggested Validation

- table-level race-card timing before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`
- if new source artifacts are generated, explicit manifest and provenance

## Starting Context

`#29` の終点は次のとおり。

- header read は `~0.001s` 未満で無視できる
- residual は parser-dominant
- parser option / dtype hint の micro tweak は exact-safe でも durable win にならない

よって `#30` は parser micro tuning を繰り返すのではなく、source shape そのものを変えた candidate を評価する issue である。

## Current Read

既存の pre-slim candidate として、`configs/data_2025_latest_materialized_racecard.yaml` を current runtime rules に合わせて再読した。

- `corner_passing_order` と `netkeiba_race_card` の `materialized_manifest_file` を config に追加
- `artifacts/reports/supplemental_materialize_netkeiba_race_card.json` を再生成し、`race_id_date_start/end` を持たせた

first read は次のとおり。

- loader-only single run:
  - current `14.3436s`
  - materialized `13.8866s`
- loader-only repeated A/B:
  - average current `14.5743s`
  - average materialized `14.3479s`
- reduced smoke:
  - `perf_smoke_racecard_preslim_v1` は `loading training table 0m14s`, total `0m24s`
  - current accepted `perf_smoke_racecard_usecols_v1` と同着
- summary compare:
  - `summary_equivalence_perf_smoke_racecard_usecols_v1_vs_racecard_preslim_v1.json`
  - difference は `run_context.data_config` の 1 件だけ

その後の repeated reduced smoke compare でも、

- current accepted `perf_smoke_racecard_current_repeat_v2`
- materialized `perf_smoke_racecard_preslim_repeat_v2`

の差分は `run_context.data_config` の 1 件だけだった。`summary_equivalence_perf_smoke_racecard_current_repeat_v2_vs_preslim_repeat_v2.json` でも substantive drift は出ていない。

したがって final read は「materialized race-card candidate は exact-equivalent behavior を保ったまま loader-only で持続的に勝ち、reduced smoke でも durable near-par を維持した」である。これにより `netkeiba_race_card` の materialized source は default 昇格でよい。
