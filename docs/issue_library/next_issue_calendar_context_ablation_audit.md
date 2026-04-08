# Next Issue: Calendar Context Ablation Audit

## Summary

current JRA high-coverage baseline には `race_year`, `race_month`, `race_dayofweek` が force include として残っている。

これらは Tier B `base race context` family の一部で、単独 alpha よりも conditioning 用の土台として扱ってきた。一方で、owner ablation まで進んだ現時点でも calendar context 単体の寄与はまだ formal に監査していない。

artifact / manifest read では、calendar context 3 列は current baseline と recent promoted candidates の used feature set に継続して現れている。したがって next JRA hypothesis は add-on ではなく ablation に切るのが自然である。

## Objective

`race_year`, `race_month`, `race_dayofweek` を current JRA high-coverage baseline から外した selective ablation を formal compare し、calendar context が current serving family の必須土台か、それとも pruning 候補かを判定する。

## Hypothesis

if `race_year`, `race_month`, `race_dayofweek` を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then calendar context は current serving family の必須要素ではなく pruning 候補になる。

## In Scope

- `race_year`, `race_month`, `race_dayofweek` を明示除外した high-coverage config 1 本
- JRA true component retrain flow
- formal compare
- 必要なら September difficult window と December control window の actual-date role split

## Non-Goals

- weather / track / ground_condition を含む base race context family 全体の broad ablation
- policy rewrite
- new feature family add-on
- NAR work

## Candidate Definition

keep current high-coverage baseline core and exclude only:

- `race_year`
- `race_month`
- `race_dayofweek`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_calendar_context_ablation.yaml`

## Why This Is Next

- `feature_family_ranking.md` では `base race context` は Tier B で、conditioning 用の必須土台として扱ってきた
- current high-coverage baseline config では calendar context 3 列が force include に固定されている
- recent promoted / compare-reference manifests にもこの 3 列が継続して現れている
- それでも current queue には calendar context 単体の keep / prune judgment が無い

したがって、この 3 列を narrow に外した first ablation を 1 issue で読む価値がある。

## Success Metrics

- win / roi component の actual used feature set から `race_year`, `race_month`, `race_dayofweek` が消える
- formal compare を 1 本作れる
- actual-date read が必要かどうかを formal / evaluation shape だけで判断できる

## Validation Plan

1. calendar context ablation config を追加する
2. true component retrain
3. stack rebuild
4. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`
5. if formal candidate remains viable, actual-date compare へ進む

## Stop Condition

- actual selected set から calendar context 3 列が外れない
- component retrain が no-op に近い
- formal support が baseline 比で明確に崩れ、analysis value も薄い