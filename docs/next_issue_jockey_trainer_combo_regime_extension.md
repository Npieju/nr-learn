# Next Issue: Jockey Trainer Combo Regime Extension

## Summary

`class / rest / surface` family は `r20260330_surface_plus_class_layoff_interactions_v1` まで読み切り、formal promoted と operational default の role split も固定した。

次の Tier A feature family は `jockey / trainer / combo` である。

## Objective

`jockey_last_30_*`, `trainer_last_30_*`, `jockey_trainer_combo_last_50_*` を中心に、regime-aware interaction を追加して、current strong family の次の feature line を formal に試す。

## Current Read

- `feature_family_ranking.md` では Tier A 第二候補
- current strong family に一貫して残る
- class/rest/surface family の次に最も自然な extension

## In Scope

- jockey / trainer / combo interaction hypothesis
- feature config candidate
- true component retrain ready な issue body

## Non-Goals

- unrelated runtime work
- NAR work
- current promoted line の role split の再議論

## Success Criteria

- next feature experiment family が 1 本に絞れる
- role split 後の queue が自然につながる

## Current First Candidate

first candidate は `style + track-distance` extension とする。

理由:

- builder にすでに存在し、新規実装を増やさずに試せる
- `fundamental_enriched_no_lineage` で採用済みの richer side である
- current core family を壊さず narrow add-on として読める

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_regime_extension.yaml`

detail:

- `docs/next_issue_jockey_trainer_combo_style_distance_candidate.md`

## Final Read

- first child:
  - `r20260330_jockey_trainer_combo_style_distance_v1`
  - formal `pass / promote`
- role split:
  - family anchor にはならない
  - current role は `analysis-first promoted candidate`
- actual-date read:
  - September: `5 / 216 races = 2.31%`, `total net -3.6`
  - December control: `30 / 264 races = 11.36%`, `total net -9.3`

interpretation:

- family の first child は model family としては成立した
- ただし top-line と actual-date role は既存 strong line を上回らず、family anchor にはならない
- よって `#49` は hypothesis issue として完了
- 次回この family を再開する場合は、新しい child hypothesis issue を別途切る
