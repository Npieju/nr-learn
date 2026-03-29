# Next Issue: Seasonal De-Risk Long-Horizon Formalization

## 1. Purpose

この文書は、Kelly runtime family 完了後の次の本線 issue をそのまま起票できる粒度で固定するための下書きである。

次の対象は broad replacement ではなく、`current_long_horizon_serving_2025_latest` を中心にした `seasonal de-risk` family である。

## 2. Why Next

`tighter policy` と `kelly runtime` はどちらも formal benchmark まで読み切り、現行 anchor `r20260329_tighter_policy_ratio003_abs90` を置き換えるには至らなかった。

一方で serving 側では、September difficult window に対して次の読みがすでに強い。

- `current_long_horizon_serving_2025_latest` は non-September baseline をほぼ保ったまま、September exposure だけを削る
- December tail では baseline と broad に同化し、controlled override として読める
- broad replacement ではなく `regime-specific de-risk alias` として operational meaning が明確

したがって次の本線は、global KPI 更新ではなく `seasonal / controlled override` family の formal execution standard を固めるのが自然である。

## 3. Proposed GitHub Issue

### Title

`[experiment] Seasonal de-risk long-horizon formalization`

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
`current_long_horizon_serving_2025_latest` を中心とした `seasonal de-risk` family を正式に整理し、September difficult window での controlled override としての採用基準、比較順、artifact read order を固定する。

Hypothesis
if we formalize the long-horizon seasonal family as a regime-specific de-risk track rather than a broad replacement candidate, then we can operate September difficult windows with lower damage and clearer decision rules, while keeping non-September baseline behavior stable.

In-Scope Surface
- `current_long_horizon_serving_2025_latest`
- `current_recommended_serving_2025_latest`
- `current_tighter_policy_search_candidate_2025_latest`
- latest 2025 serving compare dashboard artifacts
- `docs/serving_validation_guide.md`
- `docs/public_benchmark_snapshot.md`
- `docs/benchmarks.md`

Non-Goals
- global operational anchor の即時置換
- 新しい model retrain family の導入
- NAR serving policy への展開
- September 以外へ broad override を拡張すること

Success Metrics
- September difficult window で `long_horizon` を最初の de-risk alias として説明できる
- December control で broad replacement ではないことを確認できる
- compare read order が `long_horizon -> tighter policy -> recent-2018` で固定される
- operational note と benchmark note が同じ結論を指す

Eval Plan
- smoke: existing dashboard summary / compare artifact の再読
- formal: representative September / December compare の spot rerun が必要なら最小限で実施
- decision: operational alias と analysis-only candidates の境界を文書化する

Validation Commands
- `python scripts/run_serving_profile_compare.py ...`
- `python scripts/run_serving_compare_dashboard.py ...`
- `python scripts/run_serving_compare_aggregate.py ...`

Expected Artifacts
- updated seasonal decision doc
- compare dashboard summary references
- issue comment with September / December decision read

Stop Condition
- long-horizon reading が broad replacement にしか見えない
- September de-risk と non-September baseline 維持の両立が崩れる
- tighter policy candidate のほうが seasonal role でも優勢だと判明する
```

## 4. Reading Order

issue 着手時の参照順は次で固定する。

1. `docs/serving_validation_guide.md`
2. `docs/public_benchmark_snapshot.md`
3. `docs/benchmarks.md`
4. September dashboard summary
5. December control dashboard summary

## 5. Expected Decision Shape

今回の issue で欲しい結論は 3 択で十分である。

- `keep long_horizon as first seasonal de-risk alias`
- `demote long_horizon behind tighter policy candidate`
- `keep as analysis-only and avoid operational aliasing`

現時点の default expectation は、最初の選択肢である。
