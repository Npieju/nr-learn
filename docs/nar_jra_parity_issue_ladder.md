# NAR JRA Parity Issue Ladder

## Purpose

NAR を JRA と同水準の model-development discipline で扱えるようにするための issue ladder を固定する。

ここでいう `同水準` は、まず次を意味する。

- provenance-defensible benchmark
- JRA と同じ stage / gate / artifact discipline
- JRA と同じ layer separation
  - dataset
  - feature
  - model / ensemble
  - policy
- JRA と同じ compare / serving / promotion decision の読み方

これは `JRA と同じ ROI を即座に出す` という意味ではない。  
先に揃えるのは評価の質と開発導線であり、その後に architecture / feature / policy parity を詰める。

## Current State

2026-04-04 時点の NAR は次の位置にある。

- provenance corrective:
  - `#100` close
- strict pre-race benchmark rebuild:
  - `#101` open
- pre-race capture expansion:
  - `#102` close

したがって current blocker は model family ではなく dataset readiness である。`#102` は negative read で閉じ、current primary path は `#101` result-arrival handoff に戻っている。

## Parity Definition

NAR parity は次の 4 段で定義する。

1. benchmark parity
   - strict `pre_race_only` universe
   - held-out formal benchmark
   - actual-date compare
2. architecture parity
   - CatBoost win
   - LightGBM ROI
   - `value_blend` stack
3. feature-family parity
   - JRA Tier A / Tier B family を NAR 用に selective replay
4. policy / serving parity
   - runtime / threshold / seasonal compare
   - promoted vs operational role split

## Ordered Issue Ladder

### Stage 0: Dataset / Benchmark Readiness

1. `#101` strict `pre_race_only` benchmark rebuild
2. `#102` pre-race capture window expansion

exit condition:

- result-ready strict `pre_race` racesで primary / benchmark rerun が 1 本完走する

### Stage 1: Architecture Parity Bootstrap

3. `#103` NAR value-blend architecture bootstrap

objective:

- strict `pre_race_only` benchmark 上で JRA と同じ `CatBoost win + LightGBM ROI + value_blend stack` を NAR に持ち込む

success metric:

- component retrain
- stack bundle
- formal evaluation / promotion gate まで end-to-end で通る

### Stage 2: Feature-Family Parity

4. combo family selective child
5. class/rest/surface selective child
6. pace/closing-fit family selective child

rule:

- JRA で効いた family を broad に再現せず、NAR 用 selective child に narrow する
- 1 issue = 1 measurable hypothesis を維持する

### Stage 3: Policy / Serving Parity

7. runtime / threshold family replay
8. actual-date role split
9. September / control / seasonality compare

rule:

- formal `pass / promote` と operational default は分ける
- bet-rate / feasible folds / actual-date bankroll を必須にする

## Non-Goals

- JRA と NAR の早期 mixed-universe train
- provenance 不足のまま architecture parity を先に進めること
- broad multi-layer change を 1 issue に混ぜること

## Decision Rule

次に着手する NAR issue は、常にこの ladder 上の直近未完了段から切る。

現時点の次順は次で固定する。

1. `#101` result-ready rerun
2. `#103`

## Queue Sync Template

`#101` / `#103` 実行後に ladder と current queue を同期するときは、次の read を使う。

```text
Queue sync read:
- Stage 0 benchmark parity: <open|completed>
- Stage 1 architecture parity: <blocked|open|completed>
- current blocker: <external result arrival|architecture bootstrap execution|none>
- next candidate: <issue number or draft path>
```

promotion rule:

- Stage 0 が未完了なら Stage 1 は blocked のまま維持する
- Stage 1 が completed したときだけ Stage 2 selective replay family を current next candidate に上げる
- Stage 2 へ上げる first candidate は現時点では [next_issue_nar_class_rest_surface_replay.md](/workspaces/nr-learn/docs/next_issue_nar_class_rest_surface_replay.md) を起点にする
