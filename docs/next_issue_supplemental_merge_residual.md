# Next Issue: Supplemental Merge Residual Reduction

## Summary

`#22` の profiling で、`tail_training_table` loading の residual は次の順に大きいと分かった。

- `_read_csv_tail`: 約 `12.2s`
- `_merge_supplemental_tables`: 約 `2.1s`
- `_append_external_tables`: 約 `0.9s`

tail core 自体は `#21` で small exact-safe cut を複数積んでいるため、次の実装 issue は generic tail optimization ではなく、`_merge_supplemental_tables` を exact-safe に減らすことが自然である。

## Objective

`tail_training_table` path における `_merge_supplemental_tables` の residual cost を減らし、reduced smoke の `loading training table` をさらに短縮する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- supplemental table restrict / dedupe / merge path
- materialized supplemental path との整合確認
- `scripts/run_evaluate.py`
- `scripts/run_summary_equivalence.py`

## Non-Goals

- `canonical` only の drift を許容して speedup を取ること
- feature / policy / benchmark family changes
- NAR ingestion/readiness

## Success Criteria

- `_merge_supplemental_tables` に対して 1 つ以上の exact-safe cut を landing できる
- reduced smoke summary equality を維持できる
- `loading training table` の end-to-end wall time が改善する

## Suggested Validation

- loader-phase timing breakdown before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`
- if `_read_csv_tail` is touched, `scripts/run_tail_loader_equivalence.py --fail-gate exact`

## Expected Outputs

- supplemental merge optimization or explicit reject
- updated residual timing read
- summary equivalence manifest

## Current Status

`netkeiba_race_card` の materialized path を `configs/data_2025_latest_materialized_racecard.yaml` として追加し、既存の `scripts/run_materialize_supplemental_table.py` で `data/processed/supplemental/netkeiba_race_card.csv` を生成できる状態にした。

2026-03-29 の current read は次のとおりである。

- loader-only `tail_training_table(10k)`:
  - baseline `14.628s`
  - materialized race card `14.160s`
- full residual timing:
  - `_merge_supplemental_tables`: `3.044s -> 2.883s`
- table-level read:
  - `netkeiba_race_card` load: `0.480s -> 0.282s`
- reduced smoke summary:
  - metric drift なし
  - `summary_equivalence_perf_smoke_baseline_vs_materialized_racecard.json` の差分は `run_context.data_config` 1 件だけ

一方で reduced smoke wall time は baseline rerun `0m24s` に対して materialized race card `0m26s` で、end-to-end ではまだ勝ち切れていない。したがって現時点の位置づけは `default candidate` ではなく `analysis-only opt-in config` である。
