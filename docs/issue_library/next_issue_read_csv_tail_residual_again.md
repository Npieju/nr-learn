# Next Issue: `read_csv_tail` Residual After Append Wrap-Up

Historical note:

- この draft は `#35` として residual decision まで完了している。
- `read_csv_tail` residual after append wrap-up は completed runtime reference として扱い、この文書は historical issue source / profiling reference として使う。

## Summary

`#34` では append parser residual を parser option レベルで切り分けたが、current mainline に勝つ narrow exact-safe candidate は見つからなかった。

repeated loader-only compare on `netkeiba_race_result` は次のとおり。

- `current_like`: avg `0.5841s`
- `memory_map=True`: avg `0.6450s`
- `engine="c"`: avg `0.6726s`

`low_memory=True` 系も keep-columns / exact-usecols 後で依然 exact-safe ではなかった。したがって append parser residual は practical に exhausted とみなせる。

## Objective

current mainline の `tail_training_table(10k)` で支配的な `_read_csv_tail(...)` residual を再び本線として扱い、exact-safe に削れる領域を見つける。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- `_read_csv_tail(...)`
- tail reader phase profiling
- exact-safe tail reader candidates
- tail equivalence gate / reduced smoke

## Non-Goals

- canonical-only drift の受容
- append / supplemental parser tweak の継続
- feature / model / policy behavior の変更
- NAR work

## Success Criteria

- current mainline の global phase budget を踏まえて `_read_csv_tail(...)` が next dominant phase だと説明できる
- exact-safe tail reader cut が current mainline に勝つ
- reduced smoke / equivalence harness が通る

## Suggested Validation

- `tail_training_table(10k)` phase timing before/after
- `scripts/run_tail_loader_equivalence.py --fail-gate exact`
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`

## Starting Context

current mainline の global phase budget は次のとおり。

- `read_csv_tail_sec=14.7710`
- `normalize_primary_sec=0.0545`
- `append_sec=0.7279`
- `supplemental_sec=0.6346`
- `ensure_minimum_columns_sec=0.1194`

つまり、accepted append / supplemental cuts のあとで、next dominant phase は再び `_read_csv_tail(...)` に戻っている。

過去には `deque_trim` などの aggressive candidate が speed では有望だった一方、exact gate を通せなかった。現在は tail equivalence harness と gate standard が揃っているので、`#35` は「exact gate に通る tail cut」だけを対象に進める issue である。
