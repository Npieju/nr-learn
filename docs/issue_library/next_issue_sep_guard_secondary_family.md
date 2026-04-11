# Next Issue: September Guard Secondary Family Ordering

## 1. Purpose

この文書は、`current_long_horizon_serving_2025_latest` を first seasonal alias と確定した後、`current_sep_guard_candidate` family を second-layer seasonal family としてどう扱うかを固定するための次 issue 下書きである。

## 2. Why Next

`#14` で long-horizon の位置づけは固まった。

- baseline default は `current_recommended_serving_2025_latest`
- first seasonal de-risk alias は `current_long_horizon_serving_2025_latest`

次に曖昧さが残るのは、`current_sep_guard_candidate` と selected-rows September guard family の扱いである。

既存 evidence では次が分かっている。

- `current_sep_guard_candidate` は September sparse-selection regime に対して強い
- ただし broad replacement ではない
- `current_long_horizon_serving` は operational alias としてより保守的で説明しやすい

したがって次は、「Sep guard family を second operational fallback に置くのか」「analysis-only に留めるのか」を固定するのが自然である。

## 3. Proposed GitHub Issue

### Title

`[experiment] September guard secondary family ordering`

### Recommended labels

- `experiment`
- `serving`
- `policy`
- `jra`

### Template body

```md
Universe
JRA

Category
Serving Policy

Objective
`current_sep_guard_candidate` と selected-rows September guard family を整理し、long-horizon alias の下で second-layer seasonal family としてどう扱うかを固定する。

Hypothesis
if we explicitly rank the September guard family behind the long-horizon alias using the same September/control read order, then seasonal operations become easier to explain and future September-only probes can be compared against a stable secondary reference instead of a moving target.

In-Scope Surface
- `current_sep_guard_candidate`
- `current_long_horizon_serving_2025_latest`
- selected-rows September guard configs
- `docs/serving_validation_guide.md`
- `docs/benchmarks.md`
- compare / dashboard artifacts for September guard family

Non-Goals
- baseline default の変更
- long-horizon alias の再判定
- new retrain family の導入
- NAR seasonal policy への展開

Success Metrics
- `current_sep_guard_candidate` の位置づけが `second seasonal fallback` か `analysis-only` のどちらかで固定される
- selected-rows family の current frontier が 1 本に要約できる
- long-horizon / sep-guard / tighter-policy の三者順が docs 上で一貫する

Eval Plan
- smoke: existing September guard artifact の再読
- compare: September guard vs long-horizon / baseline の direct compare read
- decision: secondary fallback ranking を docs に固定する

Validation Commands
- `python scripts/run_serving_profile_compare.py ...`
- `python scripts/run_serving_compare_dashboard.py ...`
- `python scripts/run_serving_stage_path_compare.py ...`

Expected Artifacts
- updated seasonal fallback ordering doc
- issue comment with direct compare read
- if needed, compact dashboard reference table

Stop Condition
- sep-guard family が long-horizon より説明しにくいまま優位も示せない
- family 内 frontier が分散しすぎて single secondary line に要約できない
```

## 4. Default Expectation

現時点の default expectation は次である。

- first seasonal alias は引き続き `current_long_horizon_serving_2025_latest`
- `current_sep_guard_candidate` は second seasonal fallback として保持する
- selected-rows family のその他 variant は analysis-only に留める

## 5. Result

existing artifact reread だけで ordering を固定できる。

- broad September 10 日 compare:
  - baseline `58 bets / total net -21.6`
  - `current_sep_guard_candidate` `25 bets / total net +12.7067`
  - pure path bankroll:
    - baseline `0.2590`
    - candidate `1.0008`
- May-Sep full recent 35 shared dates compare:
  - baseline `136 bets / total net -42.2080`
  - `current_sep_guard_candidate` `103 bets / total net -7.9014`
  - pure path bankroll:
    - baseline `0.1886`
    - candidate `0.7290`
  - differing policy dates は September 10 日だけ
- current long-horizon alias compare:
  - September 2025 latest:
    - baseline `32 bets / total net -27.3`
    - long-horizon `9 bets / total net -4.3`
  - December 2025 control:
    - baseline `3 bets / total net +14.9`
    - long-horizon `3 bets / total net +14.9`

したがって current ordering は次で固定する。

- baseline default:
  - `current_recommended_serving_2025_latest`
- first seasonal alias:
  - `current_long_horizon_serving_2025_latest`
- second seasonal fallback:
  - `current_sep_guard_candidate`
- selected-rows September guard variants:
  - analysis-only

next action は Sep guard family の再探索ではなく、別 family hypothesis か shared portfolio bottleneck の改善である。
