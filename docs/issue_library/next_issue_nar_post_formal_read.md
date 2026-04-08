# Next Issue: NAR Baseline Post-Formal Read

## Summary

`#54` の local Nankan baseline は first formal execution まで進んでいる。

したがって次の line は、ROI を単独で読むことではなく、final support artifact を使って `bets / races / bet-rate` を分母つきで確定し、NAR separate-universe line の initial role と next family を 1 本に絞ることである。

## Objective

local Nankan baseline の final support / promotion artifact を denominator-first で読み切り、NAR line の next execution issue を 1 本に narrow する。

## Current Read

- evaluation summary は already strong
- matching-tuple rerun `r20260330_local_nankan_baseline_wf_runtime_narrow_v1` が完了した
- formal read は `bets_total=3525`, `test_races_total=28997`, `bet_rate=12.16%`
- promotion gate は `status=pass`, `decision=promote`, `wf_feasible_fold_count=3`, `formal_benchmark_weighted_roi=3.6903`
- NAR では JRA よりも低 exposure high-ROI の読み違いを厳しく避ける必要がある
- `#58` の narrow runtime line は完走しており、support read 自体は strong
- runtime / denominator の両面で、`configs/model_local_baseline_wf_runtime_narrow.yaml` が current best line になった

## In Scope

- `artifacts/reports/wf_feasibility_diag_r20260330_local_nankan_baseline_v1.json`
- `artifacts/reports/promotion_gate_r20260330_local_nankan_baseline_v1.json`
- `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_baseline_v1.json`
- `artifacts/reports/wf_feasibility_diag_local_baseline_wf_runtime_narrow_wf_full_nested.json`
- `artifacts/reports/promotion_gate_r20260330_local_nankan_baseline_wf_runtime_narrow_v1.json`
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

## Decision Summary

NAR baseline の first denominator-first formal read は `pass / promote` で確定した。current best line は `configs/model_local_baseline_wf_runtime_narrow.yaml` で、formal read は `3525 / 28997 races = 12.16%`, `weighted_roi=3.6903`, `wf_feasible_fold_count=3` だった。

次の NAR family は broad widening ではなく `class/rest/surface` の narrow replay にする。理由は、baseline に jockey/trainer history はすでに入っている一方で、JRA Tier A family の core な class/rest/surface はまだ NAR baseline に入っていないためである。
