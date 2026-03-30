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

## Current Read

append path を分解すると、現時点の main external table は `netkeiba_race_result` 1 本だけで、residual 内訳はおおむね次のとおりだった。

- `load`: `0.5455s`
- `date_parse`: `0.0061s`
- `filter`: `0.0243s`
- `concat`: `0.0380s`
- `dedupe + tail/reset`: `0.1682s`

append 側は no-op ではなく、`dedupe_on=['race_id', 'horse_id']` で見ると `101,810` append rows のうち `96,676` rows が base tail frame に対する new key だった。したがって append 全体の skip は不可で、exact-safe residual cut を局所的に積む方針が正しい。

第 1 の landed cut として、append merge 後の dedupe を `drop_duplicates(..., ignore_index=True)` に寄せ、さらに `_sort_and_tail(..., max_rows=None)` がすでに `RangeIndex(0..)` の frame では余分な `reset_index(drop=True)` を行わないようにした。ad hoc benchmark では append dedupe/tail section が `0.1882s -> 0.0506s` まで下がり、reduced smoke `perf_smoke_append_external_v1` も次の通り summary drift なしで通っている。

- `loading training table`: `0m14s`
- total elapsed: `0m24s`
- summary equivalence: `artifacts/reports/summary_equivalence_perf_smoke_surface_vs_append_external_v1.json` で `exact_equal=true`
