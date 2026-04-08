# Next Issue: NAR Promotion Alignment Short-Circuit

## Summary

`#70` と `#71` で、local Nankan baseline narrow line の phase gap は exact に確認できた。  
baseline pathfix は evaluation 段で `3/3 no_bet`, `wf_nested_test_bets_total=0` だった一方、formal 段では `4323 bets`, `weighted_roi=3.2930` まで持ち上がった。

## Objective

NAR revision gate で、evaluation 段の nested WF が `no_bet` に寄っている candidate をそのまま formal promote しないようにし、`wf_feasibility / promotion` 側の optimistic uplift を conservative に抑える。

## Hypothesis

if revision gate が evaluation nested WF と formal benchmark の phase gap を short-circuit 条件として扱う, then suspiciously optimistic NAR promote decisions can be blocked without redesigning the whole policy search.

## Current Read

- baseline pathfix evaluation:
  - `auc=0.8775363015459904`
  - `ev_top1_roi=1.940849373663306`
  - nested WF: `3/3 no_bet`
  - `wf_nested_test_bets_total=0`
- baseline pathfix formal:
  - `formal_benchmark_weighted_roi=3.293016875212951`
  - `formal_benchmark_bets_total=4323`
  - `bets / races = 4323 / 28997 = 14.91%`
- no-market evaluation:
  - nested WF: `3/3 portfolio`
  - `wf_nested_test_bets_total=2302`
- no-market formal:
  - `formal_benchmark_weighted_roi=0.8103764478764478`
  - `formal_benchmark_bets_total=8288`
  - `bets / races = 8288 / 28997 = 28.58%`

つまり current suspicious line は、evaluation で `no_bet` でも formal で promote される。

## In Scope

- `scripts/run_revision_gate.py`
- 必要なら `scripts/run_evaluate.py`
- promotion gate decision rules
- NAR local baseline pathfix / no-market artifacts

## Non-Goals

- 新しい feature family 実験
- no-market baseline の改善
- policy search 空間の全面 redesign

## Success Metrics

- evaluation nested WF `3/3 no_bet` の candidate は formal promote されない
- phase gap が manifest / promotion gate に明示される
- conservative short-circuit が 1 measurable rule として実装できる

## Validation

1. pathfix baseline artifact を fixture にして rule を dry-run する
2. no-market artifact を対照にして false positive にならないことを確認する
3. promotion gate output に block reason が残ることを確認する

## Stop Condition

- conservative rule が JRA / NAR の両方で過剰 block を起こす
- issue が short-circuit ではなく policy redesign に拡散する
