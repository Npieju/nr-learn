# Next Issue: Append-External Residual Reduction

## Summary

`#24` では `_ensure_minimum_columns` に対して 2 本の exact-safe cut を landing できた。

- numeric dtype 列の fast path
- object-heavy `weight` / `odds` 系の hybrid parse

これにより `_ensure_minimum_columns` residual は `1.663s -> 1.134s` まで下がり、reduced smoke も `loading training table 0m14s`, total `0m24s` を維持している。

current tail loading の residual read では、次にまとまって残っているのは `_append_external_tables` で、おおむね `0.75s-0.80s` 帯にある。したがって次の実装 issue は `_read_csv_tail` へ戻るよりも、`append_table` path の exact-safe residual reduction に進むのが自然である。

## Objective

`tail_training_table` path における `_append_external_tables` residual cost を減らし、reduced smoke の `loading training table` をさらに短縮する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- append table load / dedupe / recent-date filtering
- exact-safe no-op skip
- `scripts/run_evaluate.py`

## Non-Goals

- canonical drift を許容して speedup を取ること
- feature / policy / benchmark family changes
- NAR ingestion/readiness
- broad schema rewrite

## Success Criteria

- `_append_external_tables` に対して 1 つ以上の exact-safe cut を landing できる
- reduced smoke summary equality を維持できる
- `loading training table` の wall time を悪化させない

## Suggested Validation

- loader-phase timing breakdown before/after
- reduced smoke with fixed model artifact suffix
- targeted unit regressions for append no-op paths

## Expected Outputs

- append-path optimization or explicit reject
- updated residual timing read
- summary-equivalent reduced smoke
