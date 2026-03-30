# Next Issue: Bet-Rate Robustness Audit For Surface Plus Class-Layoff Candidate

## Summary

`r20260330_surface_plus_class_layoff_interactions_v1` は formal benchmark では `pass / promote` に到達したが、actual-date の serving compare では exposure を大きく削る shape が目立つ。

- September difficult window:
  - prior A anchor: `32 bets / 216 races / bet_rate=14.81% / total_net=-27.3`
  - promoted candidate: `8 bets / 216 races / bet_rate=3.70% / total_net=-8.0`
- December tail control window:
  - prior A anchor: `45 bets / 264 races / bet_rate=17.05% / total_net=+21.8`
  - promoted candidate: `13 bets / 264 races / bet_rate=4.92% / total_net=+20.0`

このため、actual-date 上の改善が genuine edge なのか、あるいは low bet-rate 由来の fragile win なのかを切り分ける必要がある。

## Objective

promoted candidate の low bet-rate / low exposure shape を監査し、serving default へ昇格させる根拠があるかではなく、analysis-first conservative role を維持すべきかどうかを evidence で固定する。

## Current Read

- formal benchmark:
  - `weighted_roi=1.1379979394080304`
  - `feasible_folds=3/5`
  - `bets_total=519`
- actual-date September:
  - `8 / 216 races = 3.70%`
- actual-date December:
  - `13 / 264 races = 4.92%`
- prior A anchor:
  - September `32 / 216 races = 14.81%`
  - December `45 / 264 races = 17.05%`
- concentration:
  - September promoted candidate は `8 bets` が 4/8 dates にしか出ず、4 dates は `0 bet`
  - December promoted candidate の total net `+20.0` のうち `+21.9` が top 2 dates (`2025-12-06`, `2025-12-07`) 由来
- bankroll sweep:
  - September pure path final bankroll は prior A `0.2959` に対して promoted `0.7333`
  - December pure path final bankroll は prior A `1.6712` に対して promoted `1.6551`
  - December best hybrid は `1.7395` で、promoted を使うのは `2025-12-06` だけ

## In Scope

- September / December compare artifact の denominator 監査
- date-level bet concentration の確認
- optional stateful bankroll / bankroll sweep
- docs への low bet-rate caution 反映

## Non-Goals

- 同 family のさらなる widening
- unrelated runtime work
- NAR work
- serving default への即時昇格

## Success Criteria

- bet count だけでなく `bets / races` と `bets / rows` が明示される
- low bet-rate が role decision にどう効くかが文章で固定される
- next experiment line が overfit suspicion を無視せずに切れる
