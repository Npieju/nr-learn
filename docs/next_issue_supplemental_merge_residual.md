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
