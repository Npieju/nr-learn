# Next Issue: Primary Tail Cache Refresh Automation

## Summary

`#36` で primary tail cache path は same-summary equivalent かつ freshness-guarded opt-in candidate として成立した。残る論点は cache correctness ではなく、refresh を誰がいつ回すかである。

## Objective

primary tail cache を default promotion 候補にする前提として、refresh command / ownership / execution timing を標準化し、必要なら自動化する。

## In Scope

- cache refresh command の標準化
- refresh の trigger 条件
- stale cache fallback と refresh runbook の接続
- default promotion に必要な operational checklist

## Non-Goals

- new parser optimization
- feature / model / policy changes
- NAR work

## Success Criteria

- refresh の責務が docs と issue だけで迷わず追える
- stale cache 時の fallback 後に何をすべきかが明文化される
- default promotion を判断する checklist が揃う

## Starting Context

current accepted state:

- candidate config: `configs/data_2025_latest_primary_tail_cache.yaml`
- cache file: `data/processed/primary/race_result_tail10000_exact.pkl`
- manifest: `artifacts/reports/primary_tail_cache_tail10000.json`
- reduced smoke:
  - mainline `loading training table 0m20s`, total `0m32s`
  - candidate `loading training table 0m02s`, total `0m15s`
- summary compare:
  - `summary_equivalence_perf_smoke_primary_tail_cache_mainline_v2_vs_candidate_v2.json`
  - only diff: `run_context.data_config`

## Suggested Validation

- refresh command dry-run / real run
- reduced smoke re-run after refresh
- doc sync for command reference and runtime queue
