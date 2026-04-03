# Next Issue: Gate Frame Course Actual-Date Role Split

## Summary

`#79` の first child `r20260403_gate_frame_course_regime_extension_v1` は formal `pass / promote` まで通った。

- `auc=0.8426169492248933`
- `ev_top1_roi=0.8213612324672338`
- held-out formal `weighted_roi=1.2311149102465346`
- `formal_benchmark_bets_total=9806`

ただし、serving default に寄せるには actual-date role が未読である。

## Objective

`gate / frame / course` extension line の actual-date role を baseline と比較し、serving default 候補か analysis-first promoted candidate かを切り分ける。

## Hypothesis

if `r20260403_gate_frame_course_regime_extension_v1` が formal だけでなく actual-date windows でも baseline 比で安定した downside control か upside を示す, then this family can advance from promoted candidate to serving-role contender.

## In Scope

- `r20260403_gate_frame_course_regime_extension_v1`
- current recommended serving baseline compare
- September / December / latest windows の actual-date compare
- role split conclusion

## Non-Goals

- new feature family
- policy rewrite
- NAR work

## Success Criteria

- actual-date compare で role が 1 つに絞れる
- serving default 候補か analysis-first promoted candidate かを明文化できる

## Validation Plan

- compare dashboard / serving smoke artifact を baseline と並べる
- September, December, latest windows の net, bets, bankroll を読む
- operational role を fixed wording でまとめる

## Stop Condition

- actual-date read が mixed で role を 1 本に決められない
- baseline より明確に悪化して serving role を主張できない

## Actual-Date Read

Actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260403_gate_frame_course_regime_extension_v1` の true retrain suffix を fresh 推論で読んだ。

September difficult window:

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_gfc_base_vs_sep25_gfc_cand.json`
- dashboard summary:
  - `artifacts/reports/dashboard/serving_compare_dashboard_sep25_gfc_base_vs_sep25_gfc_cand.json`
- shared 8 dates:
  - `2025-09-06`
  - `2025-09-07`
  - `2025-09-13`
  - `2025-09-14`
  - `2025-09-20`
  - `2025-09-21`
  - `2025-09-27`
  - `2025-09-28`
- baseline:
  - `32 bets`
  - `total net = -27.3`
  - `pure bankroll = 0.2959`
- candidate:
  - `5 bets`
  - `total net = -3.6`
  - `pure bankroll = 0.8809`

December control window:

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_gfc_base_vs_dec25_gfc_cand.json`
- dashboard summary:
  - `artifacts/reports/dashboard/serving_compare_dashboard_dec25_gfc_base_vs_dec25_gfc_cand.json`
- shared 8 dates:
  - `2025-12-06`
  - `2025-12-07`
  - `2025-12-13`
  - `2025-12-14`
  - `2025-12-20`
  - `2025-12-21`
  - `2025-12-27`
  - `2025-12-28`
- baseline:
  - `45 bets`
  - `total net = +21.8`
  - `pure bankroll = 1.6712`
- candidate:
  - `30 bets`
  - `total net = -9.3`
  - `pure bankroll = 0.7514`

補足:

- `latest` separate window は追加で切らなかった。理由は、2025 latest line の最新 operational dates が December tail control window に含まれており、role split の判定に十分だったため。

## Decision

`r20260403_gate_frame_course_regime_extension_v1` は formal `pass / promote` だが、actual-date role は serving default 候補ではない。

- September difficult window では strong downside control
- December control window では baseline 優位を崩している

したがって operational role は `analysis-first promoted candidate` に固定する。

- serving default は引き続き `current_recommended_serving_2025_latest`
- この family は September-like difficult regime の compare reference として保持する
- broad replacement / default promotion は行わない
