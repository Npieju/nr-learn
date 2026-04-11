# Next Issue: Read-Tail Dominant Phase Revisit

## Summary

`#26` で current mainline の `tail_training_table` phase budget を repeated read した結果、append path は `~0.69s` で安定しており、次の dominant phase は明確に `_read_csv_tail` だった。

3-run read は次の通り。

- `read_tail`: `12.8691s - 13.5398s`
- `append`: `0.6867s - 0.6897s`
- `supplemental`: `3.2791s - 3.4708s`
- `minimum`: `1.1898s - 1.3054s`
- total: `18.3422s - 18.9425s`

したがって current runtime 本線は append ではなく、`_read_csv_tail` の exact-safe revisit に戻すべきである。

## Objective

`tail_training_table` current mainline における dominant phase として、`_read_csv_tail` の exact-safe residual reduction を再開し、reduced smoke で phase-budget level の改善が残る candidate を見つける。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- `src/racing_ml/data/tail_equivalence.py`
- `scripts/run_tail_loader_equivalence.py`
- `scripts/run_evaluate.py`
- repeated phase-budget snippets

## Non-Goals

- canonical drift を許容する tail optimization
- feature / policy / benchmark family changes
- NAR ingestion/readiness
- broad data-loader rewrite

## Success Criteria

- `_read_csv_tail` に対して 1 本以上の exact-safe candidate を再提示できる
- tail equivalence gate は `exact` を通る
- reduced smoke で `loading training table` または total elapsed に改善が残る

## Suggested Validation

- `run_tail_loader_equivalence.py --fail-gate exact`
- repeated phase budget snippets before/after
- reduced smoke A/B with fixed model artifact suffix
- summary equivalence manifest

## Expected Outputs

- accepted exact-safe tail candidate or explicit reject read
- updated phase budget
- next hotspot decision if tail still does not move mainline

## Current Read

`#27` の revisit では、known exact-safe candidate を current mainline に対して再照合した。

- `compact_reset_late`
- `compact_iterable`
- `single_chunk_fastpath`

3 candidate はすべて `scripts/run_tail_loader_equivalence.py --fail-gate exact` を通過した。manifest は次のとおり。

- `artifacts/reports/tail_loader_equivalence_compact_reset_late_r20260330.json`
- `artifacts/reports/tail_loader_equivalence_compact_iterable_r20260330.json`
- `artifacts/reports/tail_loader_equivalence_single_chunk_fastpath_r20260330.json`

direct read timing (`tail_rows=10000`) は次のレンジだった。

- `current`: 約 `12.90s`
- `compact_reset_late`: 約 `12.74s`
- `compact_iterable`: 約 `12.40s`
- `single_chunk_fastpath`: 約 `12.66s`

ただし source diff を詰めると、current `_read_csv_tail` は `single_chunk_fastpath` candidate と実質同型で、差分は関数名だけだった。current vs `single_chunk_fastpath` の repeated A/B も `12.8624s vs 12.8146s` で、remaining gap は noise と見てよい。

したがって `#27` の結論は、current mainline は既知の best exact-safe tail cuts をほぼ吸収済みであり、runtime 本線は `_merge_supplemental_tables` の dominant residual revisit に戻すべき、である。
