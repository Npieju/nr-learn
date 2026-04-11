# Next Issue: Post-Promotion Serving Validation For Surface Plus Class-Layoff Candidate

## Summary

`#44` の `r20260330_surface_plus_class_layoff_interactions_v1` は formal に `pass / promote` まで到達した。

- promotion gate: `pass`
- formal benchmark weighted ROI: `1.1379979394080304`
- feasible folds: `3/5`

これは prior A anchor の formal benchmark `1.1042287961989103` を上回る。一方で nested summary weighted ROI は anchor を下回っているため、次は serving / compare の実運用解釈を詰める必要がある。

## Objective

promoted feature-interaction candidate を operational / analysis のどちらに置くかを、serving compare と benchmark docs refresh を通して固定する。

## Current Read

- `#44` は support-hardening family の first promoted candidate
- nested summary:
  - `auc=0.8415731002797395`
  - `ev_top1_roi=0.7099517352332797`
  - `wf_nested_test_roi_weighted=0.8118827160493829`
- matching WF feasibility:
  - fold 1 feasible
  - fold 2 feasible
  - fold 3 infeasible
  - fold 4 infeasible
  - fold 5 feasible
  - net feasible folds `3/5`
- formal benchmark:
  - `weighted_roi=1.1379979394080304`
  - `bets_total=519`
  - `feasible_fold_count=3`

## In Scope

- prior A anchor との serving compare
- latest 2025 role interpretation
- project / benchmark docs refresh
- candidate の baseline / promoted / analysis-first の位置づけ確定

## Non-Goals

- 同 family のさらなる widening
- unrelated runtime work
- NAR work

## Success Criteria

- promoted candidate の operational role が文章で固定できる
- docs が prior anchor ではなく current promoted line を正本として参照する
- next experiment issue が updated promoted baseline を起点に切れる

## Result

`#45` の serving compare は完了した。

- September difficult window:
  - prior A anchor: `32 bets / 216 races / bet_rate=14.81% / total_net=-27.3`
  - promoted candidate: `8 bets / 216 races / bet_rate=3.70% / total_net=-8.0`
- December tail control window:
  - prior A anchor: `45 bets / 264 races / bet_rate=17.05% / total_net=+21.8`
  - promoted candidate: `13 bets / 264 races / bet_rate=4.92% / total_net=+20.0`

この読みでは、promoted candidate は actual-date でも strong de-risk を示した一方、September / December の両方で baseline よりかなり low-frequency に打っている。したがって現時点の役割は broad serving default replacement ではなく、formal には promoted、serving 上は analysis-first conservative promoted candidate とするのが妥当である。

次の issue は、この low bet-rate / low exposure shape が genuine edge なのか fragile overfit suspicion なのかを denominator 付きで監査する。
