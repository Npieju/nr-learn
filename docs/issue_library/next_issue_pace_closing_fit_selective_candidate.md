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

## Actual Execution Read

- feature-gap:
  - `artifacts/reports/feature_gap_summary_pace_closing_fit_selective_v1.json`
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
- focal coverage:
  - `horse_last_3_avg_closing_time_3f = 0.747725`
  - `course_baseline_race_pace_balance_3f = 0.706467`
  - `horse_closing_vs_course = 0.507808`

true component retrain:

- win:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260403_pace_closing_fit_selective_v1.json`
  - `auc=0.8396600832597482`
  - `best_iteration=375`
- roi:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260403_pace_closing_fit_selective_v1.json`
  - `top1_roi=0.9250361794500714`
  - `best_iteration=78`

actual used feature set:

- win / roi の両方で次が selected に入った
  - `horse_last_3_avg_closing_time_3f`
  - `course_baseline_race_pace_balance_3f`
  - `horse_closing_vs_course`

execution note:

- `value_blend` family の compare は suffixed build config ではなく base stack config を使い、`--evaluate-model-artifact-suffix r20260403_pace_closing_fit_selective_v1` を渡して formal compare した
- suffixed stack config を直接 `revision_gate` に渡すと model path が二重 suffix になる

## Final Read

- revision:
  - `r20260403_pace_closing_fit_selective_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260403_pace_closing_fit_selective_v1.json`
  - `auc=0.8411224406691354`
  - `top1_roi=0.7982417834980464`
  - `ev_top1_roi=0.6212479889680533`
  - `wf_nested_test_roi_weighted=0.8857142857142858`
  - `wf_nested_test_bets_total=357`
  - nested WF: `no_bet / portfolio / portfolio`
- promotion / formal:
  - `artifacts/reports/revision_gate_r20260403_pace_closing_fit_selective_v1.json`
  - `status=pass`
  - `decision=promote`
  - `formal_benchmark_weighted_roi=1.0307760253096685`
  - `formal_benchmark_bets_total=938`
  - `formal_benchmark_feasible_fold_count=3`
  - `bets / races / bet_rate = 938 / 6244 = 15.02%`

baseline refresh compare:

- baseline refresh:
  - `auc=0.8400959298075428`
  - `top1_roi=0.8070328660078143`
  - `ev_top1_roi=0.5568030337853367`
  - `wf_nested_test_roi_weighted=0.7628366750468021`
  - `wf_nested_test_bets_total=544`
- candidate read:
  - `auc` は小幅に上
  - `ev_top1_roi` と nested WF weighted ROI は上
  - `top1_roi` と nested WF bet count は下
  - held-out formal `weighted_roi` は `> 1.0`

decision:

- selective pace family は formal candidate として成立した
- ただし serving role は未確定
- 次は actual-date role split を切り、serving default contender か `analysis-first promoted candidate` かを分ける
