# NAR Formal Read Template

## 目的

NAR baseline や challenger を読むときに、ROI だけで誤読しないための最小テンプレートを固定する。

この文書は `#52` 系の NAR separate-universe line で、issue comment / docs summary / decision summary にそのまま使う。

## 必須項目

次をこの順で書く。

1. `policy_bets`
2. `race_count`
3. `row_count`
4. `feasible_race_count`
5. `bet_rate_races = policy_bets / race_count`
6. `bet_rate_rows = policy_bets / row_count`
7. `bet_rate_feasible_races = policy_bets / feasible_race_count`
8. `weighted ROI`
9. `total net`
10. `feasible_folds / total_folds`
11. `zero_bet_dates / total_dates`
12. concentration / low-support warning
13. `promote / hold / reject`

## Standard Template

次の形で書く。

```text
Formal read:
- policy_bets: <bets>
- race_count: <races>
- row_count: <rows>
- feasible_race_count: <feasible_races>
- bet_rate_races: <bets/races>
- bet_rate_rows: <bets/rows>
- bet_rate_feasible_races: <bets/feasible_races>
- weighted_roi: <weighted_roi>
- total_net: <total_net>
- feasible_folds: <feasible_folds>/<total_folds>
- zero_bet_dates: <zero_bet_dates>/<total_dates>

Interpretation:
- <support/readability note>
- <concentration / low-frequency warning or explicit no-warning>

Decision:
- <promote|hold|reject>
- <if needed: formal promoted but operational hold>
```

## Warning 例

次のどれかに当たる場合は、必ず warning を書く。

- `bet_rate_races` が低い
- `feasible_folds` がぎりぎり
- `zero_bet_dates` が多い
- gain の大半が少数日・少数 race に集中している
- prior baseline より ROI が良くても、単に exposure が縮んだだけに見える

## 比較用 Template

baseline / challenger compare では次を使う。

```text
Compare:
- baseline: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- challenger: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- delta_bets: <challenger - baseline>
- delta_net: <challenger - baseline>
- delta_weighted_roi: <challenger - baseline>

Read:
- <is this real improvement, suppression, or concentration shift?>
```

## Current Use

`#54` の local Nankan baseline formalization が最初の適用対象である。

ここでは interim read と final read を分ける。

- interim read: evaluation summary まで
- final read: `wf_feasibility` / `promotion_gate` を含む denominator-fixed read
