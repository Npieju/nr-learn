# Next Issue: Jockey Trainer Combo Closing-Time Role Split

## Summary

`#90` の second child `r20260404_jockey_trainer_combo_closing_time_selective_v1` は formal `pass / promote` に到達した。

ただし、JRA の serving role は formal `pass / promote` だけでは決めない。次に必要なのは、current operational default と difficult / control windows の actual-date compare を読み、この line を

- serving contender
- analysis-first promoted candidate
- compare reference

のどこに置くかを固定することである。

## Objective

`r20260404_jockey_trainer_combo_closing_time_selective_v1` を current operational default と actual-date windows で比較し、operational role を固定する。

## Hypothesis

if `r20260404_jockey_trainer_combo_closing_time_selective_v1` が September difficult window で baseline を de-risk しつつ December control を壊さない, then this line can be considered a serving contender rather than an analysis-first promoted candidate.

## Current Read

- formal `pass / promote`
- evaluation:
  - `auc=0.8426169492248933`
  - `top1_roi=0.8082087836284203`
  - `ev_top1_roi=0.8213612324672338`
  - nested WF `portfolio / portfolio / portfolio`
  - `wf_nested_test_bets_total=457`
- formal:
  - `weighted_roi=1.2311149102465346`
  - `bets_total=9806`
  - `bet_rate=17.02%`

## In Scope

- broad September actual-date compare
- December control actual-date compare
- `bets / total net / pure bankroll` compare
- operational wording fix

## Non-Goals

- feature widening
- new component retrain
- policy redesign
- NAR work

## Success Criteria

- role が 1 行で固定される
- compare は `bets / total net / pure bankroll` で並ぶ
- next serving queue に置くか、analysis-first に留めるかが決まる

## Validation Plan

1. broad September actual-date compare
2. December control actual-date compare
3. 必要なら late-September narrow window compare
4. docs / queue / issue thread に decision summary を固定

## Stop Condition

- actual-date compare で current operational default に一貫して劣後
- control window を壊し、serving contender を主張できない

## Actual-Date Read

Actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260404_jockey_trainer_combo_closing_time_selective_v1` の true retrain suffix を fresh 推論で読んだ。

September difficult window:

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_combo_ct_base_vs_sep25_combo_ct_cand.json`
- dashboard summary:
  - `artifacts/reports/dashboard/serving_compare_dashboard_sep25_combo_ct_base_vs_sep25_combo_ct_cand.json`
- baseline:
  - `32 bets`
  - `total net = -27.3`
  - `pure bankroll = 0.2958869306148325`
- candidate:
  - `5 bets`
  - `total net = -3.6`
  - `pure bankroll = 0.8808888460219477`

December control window:

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_combo_ct_base_vs_dec25_combo_ct_cand.json`
- dashboard summary:
  - `artifacts/reports/dashboard/serving_compare_dashboard_dec25_combo_ct_base_vs_dec25_combo_ct_cand.json`
- baseline:
  - `45 bets`
  - `total net = +21.8`
  - `pure bankroll = 1.6711564921099862`
- candidate:
  - `30 bets`
  - `total net = -9.3`
  - `pure bankroll = 0.7513642270122226`

補足:

- late-September narrow window は追加で切らなかった
- broad September と December control の split だけで operational role は十分に固定できる

## Decision Summary

`r20260404_jockey_trainer_combo_closing_time_selective_v1` は formal `pass / promote` だが、actual-date role は serving contender ではない。

- September difficult window では baseline より strong downside control
- December control window では baseline の upside を明確に壊す

したがって operational role は `analysis-first promoted candidate` に据え置く。

- serving default は引き続き `current_recommended_serving_2025_latest`
- this line は `jockey / trainer / combo` family の compare reference として保持する
- broad replacement / default promotion は行わない
