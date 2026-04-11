# JRA Pruning Staged Decision Summary 2026-04-11

## Purpose

この文書は、JRA staged pruning review の human decision 用に、結論だけを短く固定した summary である。

## Decision

- `approve for staged simplification review, not for one-shot baseline rewrite`
- current defendable stopping point は `stage-7`
- `stage-8` は narrow hold boundary artifact として retain し、baseline rewrite 候補には上げない

## Why

- one-shot pruning bundle は actual-date では harmless でも formal gate では `0/5 feasible folds` で `hold` だった
- staged simplification は branch-sensitive に support が分岐し、supported line は `stage-7` まで延長できた
- `stage-7 dispatch metadata` add-on は formal `pass / promote`, `feasible_fold_count=3/5`, actual-date Sep/Dec equivalence を維持した
- `stage-8 condition quartet` add-on は top-line を維持しながらも `feasible_fold_count=0/5`, `status=block`, `decision=hold` に戻った
- したがって current risk は operational breakage ではなく、formal support の `min_bets` 崩れである

## Final Read

- supported branch:
  - `stage-1 calendar + recent-history`
  - `stage-2 gate/frame/course`
  - `stage-3 track/weather/surface`
  - `stage-4 class/rest/surface`
  - `stage-5 jockey/trainer ID`
  - `stage-6 jockey/trainer/combo`
  - `stage-7 dispatch metadata`
- hold boundary:
  - `stage-2 race-condition/dispatch early block`
  - `stage-8 condition quartet`

最も重要な読みは次である。

- `actual-date equivalence` だけでは staged simplification promote の根拠にならない
- current benchmark gate では `stage-7` までが defendable で、`stage-8` 以降は hold 側へ戻る

## Recommended Action

1. baseline feature config の one-shot rewrite は行わない
2. staged simplification を検討する場合は `stage-7` を natural stopping point として扱う
3. implementation へ進む場合でも group 単位で rollback point を明示する
4. current next execution は pruning 継続ではなく、別 family の `1 measurable hypothesis` へ戻す

review-ready support docs:

- rollout guardrail issue:
  - `docs/issue_library/next_issue_pruning_stage7_rollout_guardrails.md`
- implementation review checklist:
  - `docs/jra_pruning_stage7_implementation_review_checklist.md`
- rollback checklist:
  - `docs/jra_pruning_stage7_rollback_checklist.md`

## Key Evidence

### Stage-7 Supported

- revision:
  - `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1`
- formal:
  - `auc=0.842043836749433`
  - `top1_roi=0.8029459148446491`
  - `ev_top1_roi=0.7295512082853856`
  - `wf_nested_test_roi_weighted=1.1925373134328359`
  - `feasible_fold_count=3/5`
  - `status=pass`
  - `decision=promote`
- actual-date:
  - broad September 2025 baseline 完全同値
  - December control 2025 baseline 完全同値

### Stage-8 Hold

- revision:
  - `r20260411_pruning_stage8_condq_v1`
- formal:
  - `auc=0.8422288519737056`
  - `top1_roi=0.8064556962025317`
  - `ev_top1_roi=0.7503567318757192`
  - `wf_nested_test_roi_weighted=0.9622002820874471`
  - `feasible_fold_count=0/5`
  - `status=block`
  - `decision=hold`
- actual-date:
  - broad September 2025 baseline 完全同値
  - December control 2025 baseline 完全同値

## Artifacts To Open First

- `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`
- `artifacts/reports/promotion_gate_r20260411_pruning_stage8_condq_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_stage8_condq_v1.json`
- `artifacts/reports/serving_smoke_compare_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`
- `artifacts/reports/serving_smoke_compare_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
- `artifacts/reports/serving_smoke_compare_sep25_ps8_condq_base_vs_sep25_ps8_condq_cand.json`
- `artifacts/reports/serving_smoke_compare_dec25_ps8_condq_base_vs_dec25_ps8_condq_cand.json`

## Relationship To Public Docs

- この staged pruning judgment は internal review material である
- `docs/public_benchmark_operational_reading_guide.md` の public message には混ぜない

## Issue Thread Draft

```text
Decision summary:
- one-shot pruning bundle は actual-date Sep/Dec では harmless だったが、formal gate は `0/5 feasible folds` で `hold`
- staged simplification は branch-sensitive で、supported line は `stage-7 dispatch metadata` まで延長できた
- `stage-8 condition quartet` は top-line を維持しても `0/5 feasible folds` に戻り、`status=block`, `decision=hold`
- したがって current defendable boundary は `stage-7` で止まり、remaining quartet は narrow hold boundary artifact として retain する

Recommended read:
- approve for staged simplification review, not for one-shot baseline rewrite
- natural stopping point は `stage-7`
- if implementation is considered later, use group-wise rollback points and do not treat this as a public benchmark update reason

Open first:
- `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`
- `artifacts/reports/promotion_gate_r20260411_pruning_stage8_condq_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_stage8_condq_v1.json`
```