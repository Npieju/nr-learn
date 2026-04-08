# Next Issue: Shared Portfolio Bottleneck Regime Split

## Summary

baseline と `current_sep_guard_candidate` は serving outcome では差があるが、formal 側では同じ shared portfolio bottleneck を抱えている。

- shared blocked family:
  - `portfolio blend=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95`
- direct compare では threshold frontier も blocked signature も一致
- blocked occurrence は `34`
  - `min_bets=24`
  - `min_final_bankroll=10`
- shortlist 上位は同じ
  - `portfolio_lower_blend`
  - `portfolio_ev_only`

ただし existing probe は broad operational fix にならなかった。

- August weekends:
  - baseline `34 bets / +20.1 net`
  - `portfolio_lower_blend` `2 bets / -2.0`
  - `portfolio_ev_only` `9 bets / -9.0`
  - `staged_mitigation_ev_guard_probe` `6 bets / -6.0`
- late-September:
  - baseline `31 bets / -25.6`
  - `portfolio_lower_blend` `1 bet / -1.0`
  - `portfolio_ev_only` `12 bets / -12.0`

したがって次の仮説は broad rewrite ではなく、shared portfolio bottleneck を narrower regime split で救えるかである。

## Objective

`portfolio_lower_blend` / `portfolio_ev_only` を broad に当てず、profitable regime を壊さない narrower regime split として formal support を改善できるかを検証する。

## Hypothesis

if shared portfolio bottleneck を broad policy rewrite ではなく narrower regime split として切り出す, then August profitable regime を壊さずに fold4 bankroll block か early-fold `min_bets` block の一部を緩和できる可能性がある。

## In Scope

- `portfolio_lower_blend` / `portfolio_ev_only` / staged mitigation artifact の reread
- `staged_aug_baseline_stage1_probe`
- `sep_selected_rows_kelly_only_candidate`
- shared blocked family の fold-level read
- next narrow regime-split candidate の issue comment / config framing

## Non-Goals

- broad serving default の置換
- NAR work
- unrelated runtime work
- already rejected broad mitigation probe の再実行

## Success Metrics

- next candidate が broad rewrite ではなく narrow split として 1 本に定義される
- current evidence から expected gain と expected breakage が文章で説明できる
- formal rerun に載せる価値がある候補だけを 1 本に絞れる

## Validation Plan

- existing actual-date compare で broad probe failure を再確認する
- shared blocked family の failure reason を `min_bets` / `min_final_bankroll` に分けて読む
- `staged_aug_baseline_stage1` と September Kelly-only split を組み合わせた narrow split 仮説が最も自然かを判断する

## Validation Commands

- `python scripts/run_serving_profile_compare.py ...`
- `python scripts/run_serving_stage_path_compare.py ...`
- `python scripts/run_serving_compare_dashboard.py ...`

## Expected Artifacts

- shared bottleneck reread summary
- chosen narrow regime-split hypothesis
- if justified, next formal candidate config / issue read

## Stop Condition

- narrow split にしても expected gain より expected breakage のほうが大きい
- candidate definition が 1 本に収束しない
- existing evidence の rereadだけで rerun 不要と確定する

## Result

existing artifact reread だけで close してよい。

negative side はすでに十分固い。

- broad mitigation probe は August profitable regime を壊す
  - August weekends baseline `34 bets / +20.1 net`
  - `portfolio_lower_blend` `2 bets / -2.0`
  - `portfolio_ev_only` `9 bets / -9.0`
  - `staged_mitigation_ev_guard_probe` `6 bets / -6.0`
- `staged_aug_baseline_stage1` は August weekends を baseline と完全一致で守る
  - `34 bets / +20.1 net`
  - ただし late-September は `38 bets / -32.6` で baseline `31 bets / -25.6` より悪化
- `sep_selected_rows_kelly_only_candidate` は late-September を改善する
  - baseline `31 bets / -25.6`
  - candidate `10 bets / -10.0`
  - August weekends は baseline と完全一致 `34 bets / +20.1 net`

しかし formal 側が弱い。

- `wf_feasibility_diag_*sep_selected_rows_kelly_only_candidate_20240601_20240929_wf_full_nested.json`
  - folds `5/5`
  - all folds `feasible_candidates=0`
  - dominant failure は全 fold `min_bets`
  - fold4 だけ `min_final_bankroll` も一部に残る
  - best fallback bets は `28, 19, 17, 9, 11`

したがって current evidence の結論は次で固定する。

- broad portfolio rewrite rerun は不要
- `staged_aug_baseline_stage1 + sep_selected_rows_kelly_only` の mixed regime split も、現時点では formal support 不足で rerun を正当化しない
- 次に進めるなら shared portfolio bottleneck の runtime/policy repair ではなく、feature family か policy family の別 hypothesis を切る
