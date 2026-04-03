# Next Issue: Breeder Signal Selective Candidate

## Summary

Tier A/Tier B/Tier C の主要 family は JRA で一通り読み終わった。lineage 側は long-standing に low priority だが、broad pedigree ではなく breeder 単体なら narrow に buildability を再確認する価値がある。

current high-coverage baseline は `breeder_last_50_win_rate` を意図的に外している。一方で builder には `breeder_name -> breeder_last_50_win_rate` が実装済みで、full lineage を混ぜずに breeder 単体だけを切り出せる。

## Objective

`breeder_last_50_win_rate` 単体を current JRA high-coverage baseline に narrow add-on し、coverage / selection / support を壊さず formal compare 候補にできるかを判定する。

## Hypothesis

if breeder signal を single-feature selective candidate として narrow に切れば, then low-priority lineage family でも broad pedigree 再開なしに buildability と marginal contribution を読める。

## In Scope

- `breeder_last_50_win_rate` 単体 add-on
- feature-gap / coverage read
- buildable なら true component retrain flow へ進む

## Non-Goals

- sire / damsire / sire_track_distance の broad 再開
- pedigree-heavy family の大量追加
- policy rewrite
- NAR work

## Candidate Definition

keep current high-coverage baseline core and add only:

- `breeder_last_50_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_breeder_signal_selective.yaml`

## Success Metrics

- feature-gap で `breeder_last_50_win_rate` の presence / coverage が読める
- actual selected set に入る可能性を判定できる
- buildable なら formal compare 候補へ進める

## Validation Plan

1. feature-gap / coverage read
2. if buildable, true component retrain
3. stack rebuild
4. formal compare

## Stop Condition

- coverage が低すぎる
- selected set に入る見込みが薄い
- broad pedigree に広げないと意味がないと判明する

## First Gap Read

- artifact:
  - `artifacts/reports/feature_gap_summary_breeder_signal_selective_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_breeder_signal_selective_v1.csv`
- rows evaluated: `100000`
- selected feature count: `110`
- categorical feature count: `37`
- `priority_missing_raw_columns=[]`
- `missing_force_include_features=[]`
- `low_coverage_force_include_features=['breeder_last_50_win_rate']`
- focal candidate:
  - `breeder_last_50_win_rate`
  - `selected=True`
  - `present=True`
  - `non_null_ratio=0.18955`
  - `status=low_coverage`

interpretation:

- breeder 単体は missing ではない
- ただし coverage `0.18955` で current high-coverage line に対しては薄すぎる
- single-feature selective candidate としても first gate を通さない

## Decision

- breeder signal selective candidate は reject
- lineage family は引き続き primary な next bet に戻さない
- broad pedigree 再開は行わない
