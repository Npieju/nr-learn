# Next Issue: Minimum-Columns Residual Reduction

## Summary

`#23` では `_merge_supplemental_tables` に対して複数の exact-safe cut を landing できた。

- `_select_table_columns(...)` の no-op copy 削減
- `_restrict_table_to_join_keys(...)` の `Index/MultiIndex.isin` 化

これにより baseline path の `_merge_supplemental_tables` residual は `3.044s -> 2.894s` まで下がり、reduced smoke も `loading training table 0m14s`, total `0m24s` を維持している。

その一方で、current residual read では `_ensure_minimum_columns` が約 `1.5s-1.7s` あり、tail training-table loading の中では次の明確な hotspot になっている。したがって次の実装 issue は supplemental merge の延長ではなく、minimum-column normalization path を exact-safe に減らすことが自然である。

## Objective

`tail_training_table` path における `_ensure_minimum_columns` residual cost を減らし、reduced smoke の `loading training table` をさらに短縮する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- date / rank / distance / gate / weight / odds / popularity normalization
- exact-safe no-op skip
- `scripts/run_evaluate.py`
- reduced smoke validation

## Non-Goals

- `canonical` drift を許容して speedup を取ること
- feature / policy / benchmark family changes
- NAR ingestion/readiness
- broad data schema rewrite

## Success Criteria

- `_ensure_minimum_columns` に対して 1 つ以上の exact-safe cut を landing できる
- reduced smoke summary equality を維持できる
- `loading training table` の end-to-end wall time を悪化させない

## Suggested Validation

- loader-phase timing breakdown before/after
- reduced smoke with fixed model artifact suffix
- if needed, targeted unit tests for no-op normalization paths

## Expected Outputs

- minimum-column optimization or explicit reject
- updated residual timing read
- summary-equivalent reduced smoke

## Current Status

2026-03-30 の first read では、`_ensure_minimum_columns` の主要コストは次に集中していた。

- `popularity_normalize`: 約 `0.13s`
- `distance_normalize`: 約 `0.11s`
- `frame_no_normalize`: 約 `0.08s`
- `gate_no_normalize`: 約 `0.08s`
- `weight_normalize`: 約 `0.07s`
- `odds_normalize`: 約 `0.06s`

このうち `distance`, `frame_no`, `gate_no`, `popularity` は tail loading 後の frame ですでに numeric dtype だったため、regex extract を通さず `pd.to_numeric(..., errors='coerce')` だけで処理する fast path を追加した。

結果:

- `_ensure_minimum_columns`: `1.663s -> 1.192s`
- reduced smoke `perf_smoke_minimum_columns_v1_numeric_fastpath`:
  - `loading training table 0m14s`
  - total `0m24s`
  - summary drift なし

続く second cut では、object dtype の `weight` / `odds` 系に対しても `pd.to_numeric(..., errors='coerce')` を先に試し、失敗した rows にだけ regex extract を掛ける hybrid path を入れた。current tail frame では次の読みだった。

- `weight`: non-null `106658`, direct numeric `20819`, regex salvage `85712`
- `odds`: direct numeric と regex の successful rows が同数
- `rank`: 既存の `pd.to_numeric` で十分

この hybrid path により `_ensure_minimum_columns` は `1.192s -> 1.134s` まで下がり、reduced smoke `perf_smoke_minimum_columns_v2_hybrid_parse` も `loading training table 0m14s`, total `0m24s`、summary drift なしで通った。
