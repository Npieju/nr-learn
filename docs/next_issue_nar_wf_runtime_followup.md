# Next Issue: NAR WF Runtime Follow-up

## Summary

`#54` の local Nankan baseline は evaluation summary までは strong に進んでいるが、matching `wf_feasibility` が極端に長い。

したがって result read とは別に、NAR support diagnostics の runtime を separate issue として扱う必要がある。

## Objective

local Nankan baseline の `run_wf_feasibility_diag.py` が長時間化している原因を分解し、NAR line で next exact-safe runtime hypothesis を 1 本に narrow する。

## Current Read

- evaluation summary は already strong
- runtime bottleneck は `wf_feasibility` に集中している
- 現時点では stall ではなく CPU-bound search である
- result decision (`#57`) と runtime work は混ぜない

## In Scope

- `scripts/run_wf_feasibility_diag.py`
- local Nankan baseline config / policy surface
- `#54` runtime observation

## Non-Goals

- pending `#54` decision の変更
- NAR benchmark criteria の変更
- broad JRA runtime work

## Success Criteria

- runtime bottleneck が phase / search-space 単位で localized される
- next exact-safe runtime issue が 1 measurable hypothesis に narrow される
