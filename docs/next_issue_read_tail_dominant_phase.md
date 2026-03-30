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
