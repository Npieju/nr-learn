# Next Issue: NAR Promotion Threshold Realignment

## Summary

`#72` で evaluation `3/3 no_bet + bets=0` の line は formal 前に `hold` へ short-circuit するようにした。  
`#73` で formal benchmark source は held-out test metrics に揃えた。  
それでも local Nankan no-market rerun は held-out formal `weighted_roi=0.7234` なのに `status=pass`, `decision=promote` になった。

## Objective

NAR promotion gate の `pass / promote` 判定を、held-out formal benchmark の経済性と整合するように揃える。

## Hypothesis

if current NAR promotion decision is too permissive after held-out alignment, then adding a minimal economic guard on held-out formal benchmark will convert sub-unit-ROI lines from `promote` to `hold` without affecting clearly stronger lines.

## Current Read

- no-market held-out rerun:
  - revision: `r20260403_local_nankan_baseline_no_market_holdout_audit_v1`
  - evaluation:
    - `auc=0.7671690217292251`
    - `ev_top1_roi=0.47997759445972094`
    - `wf_nested_test_roi_weighted=0.6824500434404865`
    - `wf_nested_test_bets_total=2302`
  - formal:
    - `weighted_roi=0.7234044858597899`
    - `bets_total=9229`
    - `bet_rate=31.83%`
    - `metric_source_counts={test: 3}`
  - current gate result:
    - `status=pass`
    - `decision=promote`

- baseline pathfix evaluation short-circuit case:
  - evaluation nested WF: `3/3 no_bet`
  - `wf_nested_test_bets_total=0`
  - current gate result after `#72`:
    - `status=block`
    - `decision=hold`

つまり現在は、

- `no_bet` line の formal promote は止められる
- しかし held-out formal `ROI < 1` line はまだ `promote` になる

という状態で、promotion threshold 自体の整合性が残課題である。

## In Scope

- `scripts/run_promotion_gate.py`
- 必要なら `scripts/run_revision_gate.py`
- promotion gate tests
- local Nankan held-out artifacts を使った regression check

## Non-Goals

- 新 feature family 実験
- JRA threshold の同時変更
- policy search 空間の再設計

## Success Metrics

- held-out formal `weighted_roi < 1.0` の NAR line が `promote` されない
- baseline pathfix short-circuit と矛盾しない
- clearly stronger line の gate を不必要に壊さない

## Validation

1. current held-out artifacts で gate の現状判定を fixture 化する
2. `weighted_roi < 1.0` line が `hold` になる unit test を追加する
3. historical stronger line が不要に `hold` へ落ちないことを確認する

## Stop Condition

- 単純な ROI floor では current stronger line まで壊す
- threshold decision が JRA/NAR 共通 redesign を要求し、1 issue で閉じない
