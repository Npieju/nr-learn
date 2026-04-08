# Next Issue: Primary Tail Cache Default Promotion Decision

## Summary

`#37` で primary tail cache の operator-facing refresh automation は標準化できた。残る論点は、opt-in alias を default mainline に昇格するかどうかである。

## Objective

primary tail cache を default data config に昇格してよいかを、runtime gain、freshness safety、runbook completeness の 3 点から判断する。

## In Scope

- default promotion checklist
- reduced smoke / summary equivalence の再確認
- refresh runbook と ownership の十分性確認
- keep opt-in vs promote default の decision

## Non-Goals

- new parser optimization
- feature / model / policy changes
- scheduler integration の実装
- NAR work

## Success Criteria

- default promotion の判断基準が docs と issue で固定される
- promote するなら rollback 条件も明文化される
- promote しないなら opt-in keep の理由が明文化される

## Starting Context

current accepted state:

- candidate config: `configs/data_2025_latest_primary_tail_cache.yaml`
- status command: `scripts/run_primary_tail_cache_status.py`
- refresh wrapper: `scripts/run_primary_tail_cache_refresh_if_needed.py`
- freshness guard:
  - `source_dataset`
  - `source_dataset_size_bytes`
  - `source_dataset_mtime_ns`
- reduced smoke A/B:
  - mainline `loading training table 0m20s`, total `0m32s`
  - candidate `loading training table 0m02s`, total `0m15s`
  - only summary diff: `run_context.data_config`

## Suggested Validation

- `run_primary_tail_cache_status.py`
- `run_primary_tail_cache_refresh_if_needed.py`
- reduced smoke re-run
- summary equivalence manifest review

## Current Read

promotion read は affirmative である。

- default config に primary tail cache keys を追加した
- default config の real status run:
  - `status=fresh`
  - `recommended_action=use_cache`
- default config の real refresh-if-needed run:
  - `status=fresh`
  - `action=skipped_refresh`
- reduced smoke:
  - `perf_smoke_primary_tail_cache_default_promotion_v1`
  - `loading training table 0m02s`
  - total `0m15s`
- candidate compare:
  - `summary_equivalence_perf_smoke_primary_tail_cache_candidate_v2_vs_default_promotion_v1.json`
  - `difference_count=1`
  - only diff: `run_context.data_config`

したがって current best read は、「freshness guard と refresh runbook が整ったため、primary tail cache は default mainline に昇格してよい」である。

## Decision

- promote default: yes
- keep explicit alias: yes
- rollback condition: summary drift / refresh runbook failure / stale fallback failure
