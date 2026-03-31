# Next Issue: NAR Baseline Post-Formal Read

## Summary

`#54` の local Nankan baseline は first formal execution まで進んでいる。

したがって次の line は、ROI を単独で読むことではなく、final support artifact を使って `bets / races / bet-rate` を分母つきで確定し、NAR separate-universe line の initial role と next family を 1 本に絞ることである。

## Objective

local Nankan baseline の final support / promotion artifact を denominator-first で読み切り、NAR line の next execution issue を 1 本に narrow する。

## Current Read

- evaluation summary は already strong
- `wf_nested_test_bets_total=381` は interim read として出ている
- ただし正式な `bets / races / bet-rate` は support / promotion artifact 未出力のため未確定
- NAR では JRA よりも低 exposure high-ROI の読み違いを厳しく避ける必要がある

## In Scope

- `artifacts/reports/wf_feasibility_diag_r20260330_local_nankan_baseline_v1.json`
- `artifacts/reports/promotion_gate_r20260330_local_nankan_baseline_v1.json`
- `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_baseline_v1.json`
- `docs/nar_formal_read_template.md`
- NAR next-family decision

## Non-Goals

- JRA / NAR integration
- NAR operational default の即時 promotion
- final denominator read 前の broad family implementation

## Success Criteria

- `bets / races / bet-rate` を明示した formal read が残る
- support / exposure / ROI をまとめて読んだ decision summary が残る
- next NAR child issue が 1 measurable hypothesis に narrow される
