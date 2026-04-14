# Next Issue: local Nankan Historical Pre-Race Source Timing Corrective

## Summary

`#120` の strict provenance fail-closed gate により、local Nankan current high ROI の root blocker は次へ narrowed された。

- stale primary: resolved
- missing provenance columns: resolved
- remaining blocker:
  - historical racecard cache の大半が `post_race` capture
  - repaired strict audit: `pre_race=0`, `unknown=257`, `post_race=728850`

したがって次の corrective は、`historical result-ready rows に provable pre-race market capture を作る source timing corrective` である。

## Objective

local Nankan historical dataset について、result-ready benchmark に使う race rows が `pre_race market capture` を持つように source timing を補正し、strict provenance audit の `pre_race_rows>0` を達成する。

## Hypothesis

if historical local Nankan cache の大半が post-race snapshot であり、それが abnormal ROI trust を壊している, then source timing corrective を入れて pre-race capture corpus を再構築すれば、strict provenance audit を通る benchmark subset を materialize できる。

## In Scope

- historical race_id universe の再取得方法整理
- source timing corrective:
  - true pre-race capture source
  - or historical benchmark universe を pre-race provable rows のみに限定する fallback
- repaired racecard / primary / provenance audit の再実行
- `#101` strict pre-race benchmark handoff を trust-ready subset 上で再開できるか確認

## Non-Goals

- value-blend bootstrap `#103`
- full NAR multi-region expansion
- JRA/NAR mixed compare
- ROI 改善を claim する policy retune

## Success Metrics

- strict provenance audit で `pre_race_rows>0`
- strict provenance audit で `post_race_ratio=0`
- trust-carrying benchmark subset を artifact 化できる
- `#101` を trust-ready evidence として再開できる

## Validation Plan

1. current cache / crawler timing の source path を棚卸しする
2. historical pre-race capture を復元できる source があるか確認する
3. 復元できるなら racecard / primary を rebuild する
4. strict provenance audit を再実行する
5. trust-ready subset ができたら `#101` を再開する

## Stop Condition

- historical pre-race capture source が存在せず、past local Nankan benchmark は trust-carrying corpus を作れないと判明する
- その場合は historical benchmark を freeze し、future-only pre-race capture readiness track に縮退する

## Current Evidence

- `artifacts/reports/local_nankan_racecard_provenance_repair_issue120.json`
- `artifacts/reports/local_nankan_primary_materialize_issue120_repaired.json`
- current canonical read:
  - `artifacts/reports/local_nankan_provenance_audit.json`
  - `artifacts/reports/local_nankan_source_timing_audit.json`
- historical issue snapshot retained for audit trail:
  - `artifacts/reports/local_nankan_provenance_audit_issue120_repaired.json`
  - `artifacts/reports/local_nankan_source_timing_audit_issue121.json`
- `artifacts/reports/local_nankan_source_timing_audit_issue121_by_year.csv`

source timing audit により、current cache からの復元可否はさらに narrowed された。

- overall racecard provenance:
  - `pre_race=562`
  - `unknown=1959`
  - `post_race=729982`
- result-ready historical races:
  - `race_count=62503`
  - `pre_race=0`
  - `unknown=0`
  - `post_race=728859`
- current pre-race rows は `2026-04-06 .. 2026-04-07` の `24 races / 562 rows` に限られる
- `fetch_mode_summary` では historical result-ready corpus の大半が `cache_legacy|cache_legacy` かつ `post_race`

したがって `historical pre-race capture を current cache から復元する` という仮説は、現時点では stop condition に達したとみなす。

- historical pre-race recoverability: `future_only_pre_race_capture_available`
- recommended action: `downgrade_historical_benchmark_to_diagnostic_only`

## Decision Rule

- `#121` が closed になるまで historical local Nankan ROI は trust-carrying evidence と見なさない
- historical pre-race capture を作れない場合、past local Nankan line は readiness / diagnostic only に降格する
- `#101` handoff は source timing audit で `result_ready_pre_race_rows=0` が確認された場合、historical benchmark 再開ではなく future-only readiness track へ分岐する
- future-only readiness の execution handoff は `#122 [nar] local Nankan future-only pre-race readiness track` に分離する
