# Next Issue: NAR Selection Fix For Buildable Replay

## Summary

`#62` の audit で、`#61` が no-op だった理由は 2 つに分離できた。

1. `horse_days_since_last_race`, `horse_weight_change`, `horse_distance_change` は build されていたが、explicit selection が `features.base/history/include` しか読まないため selected set に入らなかった。
2. `horse_surface_switch`, `race_class_score`, `horse_class_change` は local Nankan source/builder input が足りず build されていなかった。

したがって次の measurable hypothesis は、source schema 拡張ではなく、まず buildable な 3 features を actual replay candidate にする selection fix である。

## Objective

explicit selection が declared feature groups を拾えるようにして、local Nankan replay candidate を no-op ではない measurable replay に変える。

## Hypothesis

if explicit feature selection が `features.base/history/include` 以外の declared feature groups も選択対象に含める, then local Nankan replay config で build 済みの class/rest features 3 本を selected set に載せられる。

## Current Read

- build 済みで coverage も十分:
  - `horse_days_since_last_race` (`non_null_ratio=0.82359`)
  - `horse_weight_change` (`non_null_ratio=0.8112`)
  - `horse_distance_change` (`non_null_ratio=0.82359`)
- build 不可:
  - `horse_surface_switch`
  - `race_class_score`
  - `horse_class_change`
- no-op の first cause は selection schema 側である

## In Scope

- `src/racing_ml/features/selection.py`
- `configs/features_local_baseline_class_rest_surface_replay.yaml`
- replay feature gap artifacts
- matching-tuple local rerun after selection fix

## Non-Goals

- local Nankan source schema 拡張
- class/surface raw columns の materialization
- broad feature family widening

## Success Metrics

- buildable 3 features が selected feature set に入る
- no-op replay が解消する
- matching-tuple rerun で denominator-first compare ができる

## Stop Condition

- fix が既存 JRA configs の semantics を広く変えてしまう
- buildable 3 features を入れても support / exposure が baseline より明確に悪化する
