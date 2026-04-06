# Next Issue: Append Parser Residual After Logic Cuts

Historical note:

- この draft は `#34` として parser/read-path residual の decision まで完了している。
- append parser residual は completed runtime reference として扱い、この文書は historical issue source / artifact reference として使う。

## Summary

`#33` では append loader logic residual を 4 本の exact-safe cut で削った。

- append frame を concat 前に `max_rows + len(base_frame)` まで prelimit
- recent-date filter の no-copy 化
- append table の explicit `keep_columns`
- append table の exact header-list `usecols`

この結果、reduced smoke は same-summary のまま `loading training table 0m13s`, total `0m22s` まで前進した。

## Objective

残る `netkeiba_race_result` append residual のうち、logic cut ではなく parser / read-path 側にある exact-safe 改善余地を切り分ける。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- append source parser profiling
- exact-safe append read candidates
- loader-only compare
- reduced smoke and summary equivalence

## Non-Goals

- canonical-only drift の受容
- broad source reshaping の再開
- feature / model / policy behavior の変更
- NAR work

## Success Criteria

- append residual の支配要因が parser/read-path であると説明できる
- exact-safe append read cut が current mainline に勝つ
- reduced smoke で summary equivalence が通る

## Suggested Validation

- append phase timing before/after
- loader-only compare on `netkeiba_race_result`
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`

## Starting Context

`#33` close 時点の append phase read は次のとおり。

- `append_load_sec≈0.50s`
- `recent_filter_sec≈0.02s`
- `prelimit_sec≈0.04s`
- `dedupe_concat_sec≈0.03s`
- `sort_tail_sec≈0.03s`

つまり current residual の支配要因は、ほぼ `append_load_sec` 単独である。

reduced smoke の current best read は次のとおり。

- candidate smoke: `perf_smoke_append_logic_v3_exact_usecols`
- `loading training table 0m13s`, total `0m22s`
- summary compare:
  - `summary_equivalence_perf_smoke_append_keepcols_mainline_v1_vs_v3_exact_usecols.json`
  - `exact_equal=true`

ad hoc profiling では、`low_memory=True` 系の parser option は keep-columns / exact-usecols 後でも依然として exact-safe ではなかった。したがって `#34` は、parser/read-path の narrow exact-safe candidate を切るか、「append parser residual はほぼ exhausted」と formal に言い切るかを判断する issue である。
