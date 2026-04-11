# Next Issue: Class-Rest-Surface Conditional Selective Candidate

Historical note:

- この draft は `#94` として selective child の formal candidate read まで完了している。
- conditional interaction child は no-op ではないことが確認済みであり、この文書は historical issue source / feature reference として使う。

## Summary

`class / rest / surface` family 自体は、JRA current baseline の中核に残っている。ただし current high-coverage baseline の actual used set を見ると、conditional interaction 群は builder に存在し `force_include` にも入っているのに、selected set には残っていない。

current baseline used set には次が入っている。

- `horse_days_since_last_race`
- `horse_days_since_last_race_log1p`
- `horse_is_short_turnaround`
- `horse_is_long_layoff`
- `horse_weight_change`
- `horse_distance_change`
- `horse_surface_switch`
- `horse_last_class_score`
- `horse_class_change`
- `horse_is_class_up`
- `horse_is_class_down`

一方で、次の conditional interaction 群は current used set に入っていない。

- `horse_surface_switch_short_turnaround`
- `horse_surface_switch_long_layoff`
- `horse_class_up_short_turnaround`
- `horse_class_down_short_turnaround`
- `horse_class_up_long_layoff`
- `horse_class_down_long_layoff`

したがって、次の `class / rest / surface` 仮説は broad family の再実行ではなく、この conditional interaction 群を selective child として切り出すべきである。

## Objective

current JRA high-coverage baseline に対して、未採用の `class / rest / surface` conditional interaction 群を selective child として formal compare する価値があるかを first read で判定する。

## Hypothesis

if current baseline で未採用の conditional interaction 群を narrow selective child として切り出す, then broad family を再実行せずに、September difficult regime 向けの defensive alpha を追加で読める可能性がある。

## Current Read

- `feature_family_ranking.md` では `class / rest / surface change` は Tier A / 最上位
- `r20260330_surface_plus_class_layoff_interactions_v1` は formal promoted まで到達済み
- ただし current baseline actual used set では、conditional interaction 群は未採用
- direct rerun だけでは no-op risk があるため、先に selective child として issue を切る必要がある

## Result

- focal 6 列は feature-gap で全て buildable
  - `selected=True`
  - `present=True`
  - `status=ok`
  - `non_null_ratio=0.68878`
- win / roi component の actual used set に focal 6 列は全て入った
- formal compare は `pass / promote`

主要値:

- evaluation
  - `auc=0.8426169492248933`
  - `top1_roi=0.8082087836284203`
  - `ev_top1_roi=0.8213612324672338`
  - nested WF: `3/3 portfolio`
  - `wf_nested_test_roi_weighted=0.7632385120350111`
  - `wf_nested_test_bets_total=457`
- formal
  - `weighted_roi=1.2311149102465346`
  - `bets_total=9806`
  - `feasible_fold_count=3`
  - metric source: `test=3`

formal benchmark fold winner:

- fold 1: `kelly`, `roi=0.9775776609688788`, `bets=4745`
- fold 2: `kelly`, `roi=1.5410960940613454`, `bets=3408`
- fold 3: `kelly`, `roi=1.3198132607496211`, `bets=1653`

Decision:

- selective child は no-op ではなかった
- baseline refresh 比で evaluation は小幅改善
- held-out formal `weighted_roi > 1.20` を満たした
- 次段は actual-date role split を切る

## In Scope

- current baseline used feature inventory の確認
- conditional interaction 群だけを対象にした selective child 仮説
- first candidate config の定義
- feature-gap / selection read

## Non-Goals

- broad class/rest/surface family の再実行
- unrelated policy work
- NAR work
- serving role split の再議論

## Success Criteria

- selective child の focal feature 群が 1 本に固定される
- no-op risk が事前に説明できる
- actual retrain に進むか、selection-audit だけで止めるかを決められる

## Validation Plan

1. current baseline used set と `force_include` の差分を固定する
2. focal interaction 群の feature-gap / coverage を読む
3. narrow selective config を 1 本に絞る
4. no-op risk が低ければ true component retrain へ進む

## Stop Condition

- focal interaction 群が low coverage / unstable で current high-coverage line に載せる根拠が弱い
- selection 上 no-op が濃厚で、actual retrain の前に止めるほうが合理的
