# Next Issue: Role Split For Surface Plus Class-Layoff Promoted Line

Historical note:

- この draft は `#48` として role split decision まで完了している。
- promoted surface-plus-class-layoff line の role split は確定済みであり、この文書は historical decision reference として使う。

## Summary

`#47` の first two policy-side probes では、promoted feature line の exposure widening はできなかった。

- current promoted serving path:
  - September `8 / 216 races = 3.70%`
- `sep_date_selected_rows_kelly_candidate` on promoted line:
  - September `1 / 216 = 0.46%`
- `portfolio_lower_blend` on promoted line:
  - September `0 / 216 = 0.00%`

したがって、promoted feature line に対して単純な policy tweak を重ねても broadening にはならず、むしろさらに suppressive になる。

## Objective

promoted feature line を broad serving replacement として広げるのではなく、explicit role split によって「formal promoted line」と「operational default line」の併存をどう標準化するかを決める。

## Current Read

- formal benchmark では promoted line が勝っている
- actual-date serving では low-frequency / concentrated
- simple policy-only probes は widening に失敗した

## In Scope

- promoted line の role split standardization
- September / December の explicit operator reading
- docs / benchmark phrasing の更新

## Non-Goals

- feature family の再設計
- unrelated runtime work
- NAR work

## Success Criteria

- `formal promoted` と `operational default` の関係が曖昧でなくなる
- low bet-rate candidate を無理に broad replacement 扱いしない標準文面ができる
- next experiment queue が role ambiguity を引きずらない
