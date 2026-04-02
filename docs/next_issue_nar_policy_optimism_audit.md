# Next Issue: NAR Policy Optimism Audit

## Summary

`#69` の evaluation integrity audit で、local Nankan line の高 AUC / 高 EV ROI の大部分は `odds / popularity` 依存であることが確認できた。一方で no-market ablation でも policy 自体は成立したため、残る主論点は `promotion / wf_feasibility` 側の optimism である。

## Objective

NAR formal line の高すぎる `formal_benchmark_weighted_roi` が、どの policy / selection phase で過大化しているかを分解し、1 measurable hypothesis で corrective action に落とす。

## Hypothesis

if NAR formal benchmark の過大な ROI は policy / promotion 側の optimistic selection に起因する, then `evaluation summary`, `wf_feasibility`, `promotion gate` の比較で、過大化が発生する phase と binding constraints を特定できる。

## Current Read

- baseline narrow formal:
  - `formal_benchmark_weighted_roi=3.6903437891931246`
  - `formal_benchmark_bets_total=3725`
  - `bets / races = 3725 / 28997 = 12.85%`
- no-market formal:
  - `formal_benchmark_weighted_roi=0.8103764478764478`
  - `formal_benchmark_bets_total=8288`
  - `bets / races = 8288 / 28997 = 28.58%`
- no-market evaluation:
  - `auc=0.7671689422296566`
  - `ev_top1_roi=0.47997759445972094`
  - `wf_nested_test_roi_weighted=0.6824500434404865`
  - `wf_nested_test_bets_total=2302`

この差分から、current formal line の異常な強さは feature set より policy / promotion 側の selection によって大きく増幅されている可能性が高い。

## In Scope

- `scripts/run_evaluate.py`
- `scripts/run_wf_feasibility_diag.py`
- `scripts/run_revision_gate.py`
- local Nankan baseline narrow artifacts
- no-market audit artifacts

## Non-Goals

- 新しい NAR feature family 実験
- JRA baseline の変更
- CatBoost / stack への移行

## Success Metrics

- `evaluation summary -> wf_feasibility -> promotion gate` のどこで ROI が持ち上がるか説明できる
- `formal_benchmark_weighted_roi` 上振れの主因を 1 つか 2 つに narrowed できる
- next corrective issue を 1 本に絞れる

## Validation

1. artifact compare
   - baseline narrow
   - no-market audit
   の `evaluation_summary`, `wf_feasibility`, `promotion_gate` を phase ごとに比較する

2. policy-path breakdown
   - selected params
   - feasible / infeasible split
   - `min_final_bankroll`, `max_drawdown`, `min_bets` などの gate failure を集計する

3. short-circuit candidate
   - optimistic phase が特定できたら、そこで conservative short-circuit を入れる最小修正案を切り出す

## Stop Condition

- optimism source が feature timing leak に戻る
- corrective action が broad redesign になり、1 measurable hypothesis に落ちない
