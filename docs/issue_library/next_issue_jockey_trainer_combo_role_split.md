# Next Issue: Jockey Trainer Combo Role Split

## Summary

`jockey / trainer / combo` family の first child `r20260330_jockey_trainer_combo_style_distance_v1` は formal gate まで `pass / promote` に到達した。

ただし、formal benchmark は `weighted_roi=0.95`、`feasible_folds=3/3`、`bets_total=480` であり、既存の promoted line

- `r20260330_surface_plus_class_layoff_interactions_v1`
- `r20260329_tighter_policy_ratio003_abs90`

より top-line は弱い。

したがって次の判断は family widening ではなく、この promoted line を

- family anchor
- analysis-first promoted candidate
- rejected-but-informative family read

のどれで扱うかを固定することである。

## Objective

`r20260330_jockey_trainer_combo_style_distance_v1` を、current operational default と既存 promoted alternatives に対して compare し、`jockey / trainer / combo` family の current role を固定する。

## Current Read

- formal gate は `pass / promote`
- summary は `auc=0.8431`, `wf_nested_test_roi_weighted=0.8907`, `wf_nested_test_bets_total=516`
- matching support は `3/3 feasible folds`
- formal benchmark top-line は `0.95` で、surface+layoff promoted line の `1.1380` と tighter-policy anchor の `1.1042` を下回る

## Mandatory Read Order

1. `promotion_gate_r20260330_jockey_trainer_combo_style_distance_v1.json`
2. `promotion_gate_r20260330_surface_plus_class_layoff_interactions_v1.json`
3. `promotion_gate_r20260329_tighter_policy_ratio003_abs90.json`
4. serving compare / bankroll compare on representative actual-date windows
5. explicit `bets / races / bet-rate` read

## In Scope

- serving compare against operational default
- compare against current formal promoted alternatives
- bankroll / date concentration read
- role decision wording for docs and issue close

## Non-Goals

- new jockey/trainer feature widening
- NAR baseline work
- runtime optimization work
- broad policy redesign

## Success Criteria

- `jockey / trainer / combo` family の current role が 1 行で言える
- `bets / races / bet-rate` が compare read に必ず残る
- next child issue を widen / harden / close のどれにするか決まる

## Decision Boundary

- operational anchor candidate に上げるには、formal top-lineだけでなく actual-date compare でも exposure と bankroll の説明が必要
- top-line が既存 promoted lines に劣後し、actual-date role も narrow なら analysis-first promoted candidate に留める
- compare で優位な regime が明確なら、その regime を明示した conditional role を与える
