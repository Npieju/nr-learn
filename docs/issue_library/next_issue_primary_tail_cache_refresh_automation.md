# Next Issue: Primary Tail Cache Refresh Automation

Historical note:

- この draft は `#37` として refresh automation / ownership decision まで完了している。
- primary tail cache refresh automation は completed runtime reference として扱い、この文書は historical issue source / operational reference として使う。

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

## Current Read

first cut として、operator が refresh 要否を 1 本で読める status CLI を追加した。

- command: `scripts/run_primary_tail_cache_status.py`
- current candidate config での real run:
  - `status=fresh`
  - `recommended_action=use_cache`
  - exit code `0`

status command は `fresh=0`, `stale/missing/tail_mismatch/cache_invalid=2`, `not_configured=1` を返す。したがって refresh runbook は次の一本道になった。

1. `run_primary_tail_cache_status.py`
2. `fresh` ならそのまま継続
3. `stale/missing/...` なら `run_materialize_primary_tail_cache.py`
4. 必要なら reduced smoke で再確認

これで `#37` の current best read は、「refresh ownership は operator に残るが、判断入口と実行導線は CLI と docs で標準化できた」である。次の判断は、これで `#37` を閉じてよいか、あるいは refresh をさらに scheduler/hook 化する follow-up を切るかである。

second cut として、status 判定をそのまま refresh 実行へつなぐ wrapper を追加した。

- command: `scripts/run_primary_tail_cache_refresh_if_needed.py`
- behavior:
  - `fresh` なら `action=skipped_refresh`
  - `stale/missing/...` なら materialize 実行後に再判定
- current candidate config での real run:
  - `status=fresh`
  - `action=skipped_refresh`
  - exit code `0`

regression では stale cache から `action=refreshed`, final `status=fresh` まで確認している。

したがって `#37` の current best read は、「primary tail cache refresh automation は scheduler なしでも operator-facing CLI と runbook の形で十分に標準化できた」である。残る follow-up は default promotion 判断や、必要なら scheduler/hook integration であって、refresh automation そのものではない。
