# Next Issue: Pre-Slim Race-Card Source Candidate

## Summary

`#29` で `netkeiba_race_card` residual を parser / pre-read に切り分けた結果、残差の本体は parser 側だと分かった。一方で、small exact-safe parser tweak は durable win にならなかった。

current accepted state:

- `exact header list usecols` は live baseline path に対する small accepted win

rejected follow-ups:

- `_select_table_columns` reorder/copy skip: same-summary, no wall-time win
- `memory_map=True`: same-summary, worse phase budget / reduced smoke
- explicit dtype hints: table-level average でも current accepted path に負ける

したがって next path は、baseline parser tuning の延長ではなく、pre-slim / pre-shaped race-card source artifact を別トラックとして評価することである。

## Objective

pre-slim race-card source candidate が、current accepted race-card path を table-level と reduced smoke の両方で上回れるかを評価する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- race-card source shaping / materialization helper
- repeated table-level timing
- reduced smoke and summary equivalence
- new source artifact / manifest handling

## Non-Goals

- model / feature / policy behavior の変更
- canonical-only drift の受容
- broad loader rewrite
- NAR work

## Success Criteria

- pre-slim race-card candidate が current accepted race-card path を table-level timing で明確に上回る
- reduced smoke が exact-equal を維持する
- end-to-end wall time が current mainline より改善する

## Suggested Validation

- table-level race-card timing before/after
- reduced smoke with fixed model artifact suffix
- `scripts/run_summary_equivalence.py --fail-on-diff`
- if new source artifacts are generated, explicit manifest and provenance

## Starting Context

`#29` の終点は次のとおり。

- header read は `~0.001s` 未満で無視できる
- residual は parser-dominant
- parser option / dtype hint の micro tweak は exact-safe でも durable win にならない

よって `#30` は parser micro tuning を繰り返すのではなく、source shape そのものを変えた candidate を評価する issue である。
