# Primary Tail Cache Operational Policy

## Summary

primary tail cache は、current mainline と same-summary equivalent な runtime shortcut として扱う。ただし default source そのものではなく、freshness guard 付きの opt-in cache alias として運用する。

## Role

- 目的は `load_training_table_tail(...)` の wall time を大きく下げること
- 対象は JRA primary source tail read のみ
- feature / model / policy behavior は変えない

## Current Standard

- candidate config は [configs/data_2025_latest_primary_tail_cache.yaml](/workspaces/nr-learn/configs/data_2025_latest_primary_tail_cache.yaml) を使う
- cache materialize は `scripts/run_materialize_primary_tail_cache.py` を使う
- cache file は `data/processed/primary/race_result_tail10000_exact.pkl`
- manifest は `artifacts/reports/primary_tail_cache_tail10000.json`

## Freshness Rule

cache を使ってよいのは、manifest が次を満たす場合だけである。

- requested `tail_rows` が一致する
- `source_dataset` が current raw primary source と一致する
- `source_dataset_size_bytes` が一致する
- `source_dataset_mtime_ns` が一致する

どれか 1 つでも崩れた場合、loader は cache を使わず raw tail read に fallback する。

## Refresh Rule

次の場合は cache refresh を実行する。

- raw primary source file が更新された
- `tail_rows` を変えた
- primary source selection rule が変わった
- cache file / manifest を削除した

まず status command で現在地を確認する。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_primary_tail_cache_status.py \
  --data-config configs/data_2025_latest_primary_tail_cache.yaml \
  --tail-rows 10000
```

`fresh` ならそのまま使う。`stale` / `missing` / `tail_mismatch` / `cache_invalid` / `cache_short` なら refresh を実行する。

通常運用では、次の wrapper をそのまま使ってよい。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_primary_tail_cache_refresh_if_needed.py \
  --data-config configs/data_2025_latest_primary_tail_cache.yaml \
  --tail-rows 10000
```

refresh command はこれを使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_materialize_primary_tail_cache.py \
  --data-config configs/data_2025_latest_primary_tail_cache.yaml \
  --tail-rows 10000
```

## Promotion Rule

現時点では、primary tail cache は opt-in alias として採用済みである。default mainline へ昇格するには、次を追加で満たす必要がある。

- refresh 実行の責務が運用上あいまいでない
- status command と refresh command の runbook が標準化されている
- stale cache 時の fallback だけでなく refresh 導線も標準化されている
- current mainline との reduced smoke equivalence が維持される

## Decision

2026-03-30 時点の正本判断は次である。

- `primary tail cache` は freshness-guarded opt-in alias として keep
- default 昇格は refresh automation / operational ownership が整ってから再判定
