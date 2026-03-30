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
