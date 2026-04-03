# Next Issue: Pace Closing-Fit Selective Candidate

## Summary

feature ranking では `pace / corner / closing-fit` family は Tier D で、low-coverage なまま primary line にするのは非推奨である。

ただし builder には pace side の派生がすでに存在し、全てが同じ coverage リスクではない。

- `horse_last_3_avg_closing_time_3f`
  - recent-form側の raw で coverage は `~0.76`
- `course_baseline_race_pace_balance_3f`
  - gate/frame/course extension probe では coverage `0.57273`
- `horse_closing_vs_course`
  - builder 実装済み
- 一方で `horse_closing_pace_fit`, `horse_front_pace_fit` は corner 系依存が強く、先に試す candidate としては弱い

したがって次の pace family issue は、low-coverage corner interactions を避けて、`closing_time + course pace baseline` の selective candidate に絞る。

## Objective

`pace / closing-fit` family を low-coverage のまま broad に再開せず、coverage を読める narrow candidate だけで formal compare 候補を 1 本作れるかを判定する。

## Hypothesis

if `horse_last_3_avg_closing_time_3f` と `horse_closing_vs_course` を中心にした narrow candidate が actual selected set に入り、support を壊さず current baseline と formal compare できる, then pace family は low-priority のままでも selective replay 候補として再開できる。

## In Scope

- `horse_last_3_avg_closing_time_3f`
- `course_baseline_race_pace_balance_3f`
- `horse_closing_vs_course`
- 必要なら `horse_closing_pace_fit` は gap read だけ確認
- JRA true component retrain flow

## Non-Goals

- low-coverage corner family の broad 再開
- pedigree / owner family への拡張
- serving policy rewrite
- NAR work

## Candidate Definition

keep current high-coverage baseline core and add only:

- `horse_last_3_avg_closing_time_3f`
- `course_baseline_race_pace_balance_3f`
- `horse_closing_vs_course`

defer:

- `horse_closing_pace_fit`
- `horse_front_pace_fit`
- corner-gain heavy interactions

## Success Metrics

- feature-gap で candidate の coverage / presence が確認できる
- win / roi component の actual used feature set に focal features が入る
- baseline と formal compare できる revision を 1 本作れる

## Validation Plan

1. feature-gap / coverage read
2. narrow config を追加
3. true component retrain
4. stack rebuild
5. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`

## Stop Condition

- focal features が gap read で low coverage / missing 扱いになる
- actual selected set に入らない
- support が baseline 比で明確に壊れる
