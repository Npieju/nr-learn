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
- long silent interval 自体が operator risk なので、progress 粒度だけでなく interrupt guard も必要
- default `policy_search` は `outer=72`、`strategy_evals_per_outer=10`、`total=720 evals/fold` で重い
- first runtime hypothesis は code optimization ではなく NAR 用 narrow `policy_search` candidate の比較である

## First Runtime Hypothesis

`configs/model_local_baseline_wf_runtime_narrow.yaml` を使い、NAR baseline の `wf_feasibility` search grid を次に狭める。

- `blend_weights`: `[0.2, 0.4]`
- `min_edges`: `[0.01, 0.03, 0.05]`
- `min_probabilities`: `[0.03, 0.05]`
- `odds_maxs`: `[25.0, 40.0]`
- `fractional_kelly_values`: `[0.25]`
- `max_fraction_values`: `[0.02]`
- `top_ks`: `[1]`
- `min_expected_values`: `[1.0, 1.05]`

これで per-fold search space は `outer=24`, `strategy_evals_per_outer=3`, `total=72 evals/fold` になる。default `720` に対して 10x smaller で、`kelly` と `portfolio` の両方は維持する。

## Interim Execution Read

actual run では `artifacts/logs/wf_feasibility_r20260330_local_nankan_baseline_wf_runtime_narrow_v1.log` を正本に使う。現時点で fold 1 は `72` candidates を `18m04s` で完了し、`feasible=26` だった。fold 2 も同じ `72` step grid で `12/72` まで clean に進んでいる。したがって first runtime hypothesis は机上比較ではなく、actual execution でも runtime-shortening として成立している。

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
- bounded no-output interrupt rule を実装候補として評価できる
