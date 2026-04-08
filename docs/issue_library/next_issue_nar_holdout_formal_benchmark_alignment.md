# Next Issue: NAR Hold-Out Formal Benchmark Alignment

## Summary

`#70` `#71` `#72` で、local Nankan baseline narrow line の suspicious uplift は `wf_feasibility / promotion` 段にあることが固まった。  
さらに code read で、current `promotion_gate` は `wf_summary.folds[].best_feasible` の valid metrics をそのまま formal benchmark に集計していることが確認できた。

## Objective

NAR formal benchmark を held-out test metrics ベースに揃え、valid-only optimistic uplift を止める。

## Hypothesis

if `run_wf_feasibility_diag.py` が valid で選んだ feasible policy を fold test に再適用し、`run_promotion_gate.py` がその held-out test metrics を formal benchmark に使う, then suspiciously optimistic formal ROI will drop toward a defensible hold-out reading.

## Current Read

- baseline pathfix evaluation:
  - `auc=0.8775363015459904`
  - nested WF: `3/3 no_bet`
  - `wf_nested_test_bets_total=0`
- baseline pathfix current formal:
  - `weighted_roi=3.293016875212951`
  - `bets_total=4323`
  - `bet_rate=14.91%`
- no-market evaluation:
  - nested WF: `3/3 portfolio`
  - `wf_nested_test_bets_total=2302`
- no-market current formal:
  - `weighted_roi=0.8103764478764478`
  - `bets_total=8288`
  - `bet_rate=28.58%`

code read:

- `scripts/run_wf_feasibility_diag.py`
  - fold summary は `best_feasible` を valid metrics で保持
- `scripts/run_promotion_gate.py`
  - `_summarize_formal_benchmark(...)` は `best_feasible.bets/roi` をそのまま集計

つまり current formal benchmark は held-out test benchmark ではなく、valid-selected metrics の aggregate になっている。

## Final Read

implementation:

- `scripts/run_wf_feasibility_diag.py`
  - valid で選んだ feasible params を fold test に再適用し、`folds[].best_feasible_test` を出力
- `scripts/run_promotion_gate.py`
  - formal benchmark は `best_feasible_test` を優先集計
  - historical artifact だけ `valid_fallback` を許容

local Nankan no-market held-out rerun:

- revision:
  - `r20260403_local_nankan_baseline_no_market_holdout_audit_v1`
- evaluation:
  - `auc=0.7671690217292251`
  - `top1_roi=0.7877482432019554`
  - `ev_top1_roi=0.47997759445972094`
  - `wf_nested_test_roi_weighted=0.6824500434404865`
  - `wf_nested_test_bets_total=2302`
- old formal:
  - `weighted_roi=0.8103764478764478`
  - `bets_total=8288`
  - `bet_rate=28.58%`
- held-out formal:
  - `weighted_roi=0.7234044858597899`
  - `bets_total=9229`
  - `bet_rate=31.83%`
  - `metric_source_counts={test: 3}`

interpretation:

- held-out alignment で no-market formal ROI は `0.8104 -> 0.7234` に低下
- したがって、旧 formal benchmark は valid-based な optimistic uplift を含んでいた
- `metric_source_counts={test: 3}` により、new formal benchmark は fold test のみを source にしている
- `#72` の short-circuit が入ったため、baseline narrow 自体は held-out formal まで進まない
- それでも `#73` の corrective 自体は no-market rerun で十分に validation できた

## In Scope

- `scripts/run_wf_feasibility_diag.py`
- `scripts/run_promotion_gate.py`
- 必要な unit tests
- NAR baseline narrow / no-market compare artifact

## Non-Goals

- 新 feature family 実験
- policy search 空間の全面 redesign
- JRA promotion policy の同時変更

## Success Metrics

- `wf_summary` に held-out test-side feasible metrics が fold 単位で残る
- `promotion_gate` formal benchmark が held-out test metrics を source として集計される
- valid-only aggregate による optimistic uplift を説明可能な artifact 差分として示せる

## Validation

1. fold fixture で `best_feasible_test` serialization を unit test する
2. `promotion_gate` が `best_feasible_test` を優先して集計する unit test を追加する
3. local Nankan baseline narrow path-fixed rerun を 1 本回し、held-out test formal ROI を旧 formal ROI と比較する

## Stop Condition

- held-out alignment だけでは uplift がほぼ消えず、policy search redesign が先に必要と分かる
- 変更が historical artifact 読みを不必要に壊す

## Artifacts

- `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260403_local_nankan_baseline_no_market_holdout_audit_v1.json`
- `artifacts/reports/wf_feasibility_diag_r20260403_local_nankan_baseline_no_market_holdout_audit_v1.json`
- `artifacts/reports/promotion_gate_r20260403_local_nankan_baseline_no_market_holdout_audit_v1.json`
- `artifacts/reports/nar_holdout_formal_compare_no_market_v1.json`
