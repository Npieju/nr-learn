# Next Issue: NAR Policy Optimism Audit

Historical note:

- この draft は `#70` として optimism diagnosis まで完了している。
- NAR policy optimism audit は completed diagnostic reference として扱い、この文書は historical issue source / artifact reference として使う。

## Summary

`#69` の evaluation integrity audit で、local Nankan line の高 AUC / 高 EV ROI の大部分は `odds / popularity` 依存であることが確認できた。一方で no-market ablation でも policy 自体は成立したため、残る主論点は `promotion / wf_feasibility` 側の optimism である。

## Objective

NAR formal line の高すぎる `formal_benchmark_weighted_roi` が、どの policy / selection phase で過大化しているかを分解し、1 measurable hypothesis で corrective action に落とす。

## Hypothesis

if NAR formal benchmark の過大な ROI は policy / promotion 側の optimistic selection に起因する, then `evaluation summary`, `wf_feasibility`, `promotion gate` の比較で、過大化が発生する phase と binding constraints を特定できる。

## Current Read

exact compare は path-fixed baseline rerun まで含めて確定した。

- compare artifacts:
  - `artifacts/reports/nar_policy_optimism_phase_compare_baseline_vs_no_market_v1.json`
  - `artifacts/reports/nar_policy_optimism_phase_compare_baseline_pathfix_vs_no_market_v2.json`

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
  - `auc=0.7671689422296566`
  - `ev_top1_roi=0.47997759445972094`
  - nested WF: `3/3 portfolio`
  - `wf_nested_test_bets_total=2302`
- no-market formal:
  - `formal_benchmark_weighted_roi=0.8103764478764478`
  - `formal_benchmark_bets_total=8288`
  - `bets / races = 8288 / 28997 = 28.58%`

phase compare の exact read は次である。

- evaluation AUC:
  - `0.8775 -> 0.7672`
- evaluation EV top1 ROI:
  - `1.9408 -> 0.4800`
- evaluation nested bets:
  - `0 -> 2302`
- formal weighted ROI:
  - `3.2930 -> 0.8104`
- formal bets total:
  - `4323 -> 8288`
- formal bet rate:
  - `14.91% -> 28.58%`

この差分でいちばん重要なのは、baseline pathfix line が evaluation 段では `3/3 no_bet` なのに、formal 段では `4323 bets`, `weighted_roi=3.2930` まで持ち上がることである。  
つまり current NAR line の異常な強さは feature set より `wf_feasibility / promotion` 側の selection で発生している、と読むのが妥当である。

historical baseline narrow の旧 read より path-fixed rerun のほうが lower ROI (`3.6903 -> 3.2930`) ではあるが、phase gap 自体は再現した。  
したがって `#70` は diagnosis issue として close してよく、次の実行単位は conservative short-circuit / promotion alignment の corrective issue に切り替える。

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
