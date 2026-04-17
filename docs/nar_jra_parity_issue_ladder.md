# NAR JRA Parity Issue Ladder

## Purpose

NAR を JRA と同水準の model-development discipline で扱えるようにするための issue ladder を固定する。

この文書での top-level 完了条件は明示的に次とする。

- `NAR が解決した` と言えるのは、`JRA相当の信頼度で運用判断できるモデル line` が構築されたときだけである
- ここでいう `JRA相当の信頼度` とは、単に一時的な ROI が良いことではなく、dataset / benchmark / architecture / feature / policy / serving の各 layer が provenance-defensible な artifact と gate で支えられていることを指す

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

2026-04-16 時点の NAR は次の位置にある。

- universe scope:
  - current corpus は `local Nankan`、すなわち South Kanto local track only
- interpretation guardrail:
  - current local Nankan ROI は `full NAR` や `JRA 相当の信頼性` をまだ主張しない
- market provenance guardrail:
  - `#120` が strict provenance trust gate として current top-priority corrective である
- source timing guardrail:
  - `#121` は historical result-ready `pre_race` recoverability が `future_only_pre_race_capture_available` に narrowed された current corrective である
- operator-default readiness path:
  - `#122` は historical downgrade 後の future-only pre-race readiness track である

- provenance corrective:
  - `#100` close
- strict pre-race benchmark rebuild:
  - `#101` formal rerun completed on trust-ready historical corpus
- pre-race capture expansion:
  - `#102` close

したがって Stage 0 benchmark parity は historical trust-ready corpus 上では一度通過している。`#102` は negative read で閉じ、current next candidate は `#103` architecture bootstrap である。一方で `#122` の future-only readiness track は live operator path として分離維持する。

ただし parallel に unresolved な reading が 3 つある。

- local Nankan only という universe coverage gap
- historical high ROI に対する reliability gap
- provable pre-race market capture が未証明な provenance gap

この 3 点が解消されるまでは、current local Nankan line を readiness track として扱う。

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

## Completion Definition

NAR の完了条件は次をすべて満たすことである。

1. benchmark trust
  - strict `pre_race_only` universe で result-ready benchmark が再現可能
  - held-out / actual-date の formal read が artifact で追える
2. architecture trust
  - JRA と同じ layer separation で baseline から architecture parity line まで end-to-end 実行できる
3. decision trust
  - promoted / hold / reject の判断が JRA と同じ gate discipline で説明可能
4. operator trust
  - readiness / rerun / compare / serving の運用導線が artifact と status board で追跡可能

したがって、current local Nankan line のような `future-only readiness track` は完了そのものではない。これは completion へ至るための blocker resolution / readiness path にすぎない。

## Ordered Issue Ladder

### Stage 0: Trust / Dataset Readiness

1. `#120` strict provenance trust gate
2. `#121` historical source timing corrective
3. `#122` future-only pre-race readiness track
4. `#101` strict `pre_race_only` benchmark rebuild
5. `#102` pre-race capture window expansion

exit condition:

- strict provenance と historical source timing の current read が揃い、trust-bearing result-ready strict `pre_race` races で primary / benchmark rerun が 1 本完走する
- 2026-04-15 run `r20260415_local_nankan_pre_race_ready_formal_v1` により、この exit condition は historical trust-ready corpus 上で充足した

### Stage 1: Architecture Parity Bootstrap

6. `#103` NAR value-blend architecture bootstrap

objective:

- strict `pre_race_only` benchmark 上で JRA と同じ `CatBoost win + LightGBM ROI + value_blend stack` を NAR に持ち込む

success metric:

- component retrain
- stack bundle
- formal evaluation / promotion gate まで end-to-end で通る

### Stage 2: Feature-Family Parity

7. combo family selective child
8. class/rest/surface selective child
9. pace/closing-fit family selective child

rule:

- JRA で効いた family を broad に再現せず、NAR 用 selective child に narrow する
- 1 issue = 1 measurable hypothesis を維持する

### Stage 3: Policy / Serving Parity

10. runtime / threshold family replay
11. actual-date role split
12. September / control / seasonality compare

rule:

- formal `pass / promote` と operational default は分ける
- bet-rate / feasible folds / actual-date bankroll を必須にする

## Non-Goals

- JRA と NAR の早期 mixed-universe train
- provenance 不足のまま architecture parity を先に進めること
- broad multi-layer change を 1 issue に混ぜること
- local Nankan only result を full-NAR representative と見なすこと

## Decision Rule

次に着手する NAR issue は、常にこの ladder 上の直近未完了段から切る。

現時点の次順は次で固定する。

1. `#120` current trust artifact maintenance
2. `#121` historical downgrade / source timing corrective
3. `#103` architecture bootstrap
4. `#122` future-only readiness handoff

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
- Stage 2 へ上げる first candidate は現時点では [next_issue_nar_class_rest_surface_replay.md](/workspaces/nr-learn/docs/issue_library/next_issue_nar_class_rest_surface_replay.md) を起点にする
