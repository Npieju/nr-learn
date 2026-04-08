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

## Final Read

`#63` では selection fix 自体は成功した。train manifest の used feature set は baseline 13 本から 16 本へ増え、次の 3 features が actual candidate に入った。

- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`

一方で formal compare は baseline を下回った。

- baseline narrow:
  - `auc=0.8775353752835744`
  - `ev_top1_roi=1.940849373663306`
  - `formal_benchmark_weighted_roi=3.6903437891931246`
  - `formal_benchmark_bets_total=3525`
  - `wf_feasible_fold_count=3`
- replay v2 selection fix:
  - `auc=0.8737311965910113`
  - `ev_top1_roi=0.37983501374885426`
  - `formal_benchmark_weighted_roi=1.011422845691383`
  - `formal_benchmark_bets_total=499`
  - `wf_feasible_fold_count=1`

race 分母 `9819` に対する formal bet 率も `35.90% -> 5.08%` まで落ちていて、support / exposure とも baseline 劣後である。したがって `#63` は selection-fix issue として close し、feature promotion は行わない。
