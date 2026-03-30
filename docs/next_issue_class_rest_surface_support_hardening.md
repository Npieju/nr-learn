# Next Issue: Class Rest Surface Support Hardening

## Summary

`#41` の `r20260330_class_rest_surface_interactions_v1` は、evaluation summary では A anchor と明確に異なり、`auc=0.8417`、`ev_top1_roi=0.6837`、`wf_nested_test_roi_weighted=1.1003` と強い値を出した。一方で matching WF feasibility は `1/5` feasible folds に留まり、promotion gate は support block になった。

したがって次の仕事は family を捨てることではなく、`class / rest / surface` interaction family を support-aware に詰め直すことである。

## Objective

summary-level の改善を壊さず、formal support、特に `min_bets` で落ちた early folds の feasible-fold count を引き上げる。

## Current Read

- `#41` evaluation summary は A anchor に対して `different`
- `auc=0.8417001497939913`
- `ev_top1_roi=0.6837278786485865`
- `wf_nested_test_roi_weighted=1.100299401197605`
- `wf_nested_test_bets_total=668`
- promotion gate は `block`
- blocking reason は `Walk-forward feasible fold count is below threshold: 1 < 3`
- folds 1-4 は `min_bets` で不通
- fold 5 だけ `portfolio` で `2` feasible candidates が出た
- fold 5 best feasible は `blend_weight=0.8`, `min_prob=0.03`, `min_expected_value=0.95`, `bets=331`, `roi=1.1356`
- component importance の current read では、ROI side で gain を持った新 interaction は `horse_surface_switch_short_turnaround` と `horse_surface_switch_long_layoff` の 2 本だけだった
- class up/down interaction 4 本は ROI side gain が `0.0` で、next narrower candidate では first remove 候補になる

## Current Narrower Candidate

最初の support-hardening candidate は `surface interaction only` とする。

- keep:
  - `horse_surface_switch_short_turnaround`
  - `horse_surface_switch_long_layoff`
- drop first:
  - `horse_class_up_short_turnaround`
  - `horse_class_down_short_turnaround`
  - `horse_class_up_long_layoff`
  - `horse_class_down_long_layoff`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_surface_interactions.yaml`

## In Scope

- `r20260330_class_rest_surface_interactions_v1` の WF feasibility diagnostic 読み直し
- interaction family のうち support を削っている要素の切り分け
- support-aware な narrower variant 設計
- next formal candidate の issue / config / validation plan 準備

## Non-Goals

- broad policy family widening
- unrelated runtime work
- NAR work

## Success Criteria

- summary と support の divergence を説明できる
- next candidate を 1 本に絞れる
- next candidate が formal-gate-ready な粒度まで落ちる
