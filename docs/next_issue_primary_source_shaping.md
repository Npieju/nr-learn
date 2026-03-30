# Next Issue: Primary Source Shaping After Tail Micro-Cut Exhaustion

## Summary

`#35` では `_read_csv_tail(...)` の exact-safe candidate を既存 harness で切り分けたが、current mainline に対して landing できる新規 cut は見つからなかった。

current exact-gate read は次のとおり。

- `compact_reset_late`: exact pass, but slower (`13.8977s` avg)
- `compact_iterable`: exact pass, but slower (`14.0331s` avg)
- `single_chunk_fastpath`: exact pass and faster-looking (`12.8287s` avg), but this shape is already structurally absorbed in current mainline
- `deque_trim`: still canonical-only, exact fail
- chunked `memory_map=True`: exact pass, but slower (`13.2036s` avg vs current `12.8289s`)

したがって、残る tail residual は narrow in-function micro cut の対象としてはかなり硬い。

## Objective

runtime work を `_read_csv_tail(...)` micro-optimization から、primary JRA source 自体の source-shaping へ移す。pre-shaped tail source、partitioned source、cached recent slice のような source-level candidate が、挙動を変えずに wall time を下げられるかを切り分ける。

## In Scope

- primary JRA source loading strategy
- pre-shaped / cached recent source candidates
- exact-safe source-level comparisons
- reduced smoke and summary equivalence

## Non-Goals

- canonical-only drift
- append / supplemental parser tweak の再開
- feature / model / policy behavior の変更
- NAR work

## Success Criteria

- current primary source load cost を source level で説明できる
- source-shaped candidate が loader-only か end-to-end で勝つ
- summary equivalence が通る

## Suggested Validation

- source-level loader-only compare
- `scripts/run_summary_equivalence.py --fail-on-diff`
- reduced smoke with fixed model artifact suffix
- 必要なら tail equivalence exact gate

## Starting Context

current mainline の global phase budget (`tail_training_table(10k)`) は次のとおり。

- `read_csv_tail_sec=14.7710`
- `normalize_primary_sec=0.0545`
- `append_sec=0.7279`
- `supplemental_sec=0.6346`
- `ensure_minimum_columns_sec=0.1194`

accepted append / supplemental cuts のあとで、primary source loading が wall time をほぼ支配している。次の runtime track は parser option の小手先ではなく、source-level shaping である。

## Current Read

first source-level probes として、plain CSV の pre-shaped primary source を 2 本 ad hoc で見た。

1. broad recent slice `race_result_since_20210116.csv`
2. exact tail cache `race_result_tail10000_exact.csv`

loader-only の読みはかなり強く、次のとおりだった。

- current `_read_csv_tail(10k)`: `~12.5s-13.2s`
- broad recent slice: `~0.17s`
- exact tail cache: `~0.05s`

ただし、どちらも current mainline の behavior をそのままは再現しなかった。

broad recent slice は row set 自体が current tail と一致しない。

- `rows_ge_2021-01-16 = 26,859`
- `_read_csv_tail(...)` compare: `exact_equal=false`

exact tail cache は raw file としては速いが、reduced smoke で substantive drift が出た。

- candidate smoke: `perf_smoke_primary_tail_exact_abs_v1`
- `loading training table 0m01s`, total `0m10s`
- summary compare:
  - `summary_equivalence_perf_smoke_append_logic_v3_vs_primary_tail_exact_abs_v1.json`
  - `exact_equal=false`
  - `difference_count=19`

差分は `run_context.data_config` や `primary_source_rows_total` だけではなく、`top1_roi`, `ev_top1_roi`, `auc`, `ev_threshold_1_0_bets` にも及んだ。したがって current best read は、「plain CSV への source shaping は wall time には効くが、schema / parse behavior を保てず、そのままは mainline に昇格できない」である。

このため `#36` の next move は、plain CSV cache を続けるのではなく、schema-preserving な primary cache path を検討することになる。
