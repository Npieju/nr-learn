# Next Issue: NAR Class-Rest-Surface Availability Audit

## Summary

`#61` の class/rest/surface replay は matching-tuple formal run まで完了したが、trained manifest の used features は baseline と完全一致だった。

したがって next line は family widening ではなく、NAR local Nankan で class/rest/surface の core 6 features が実際に build されているか、selection で落ちているか、source columns 自体が足りないかを切り分けることになる。

## Objective

local Nankan baseline に追加した class/rest/surface core 6 features が、build / coverage / selection のどこで消えているかを特定し、next replay candidate を no-op ではなく measurable にする。

## Hypothesis

if local Nankan の class/rest/surface replay が no-op だった理由を feature-level に切り分ける, then next NAR feature issue は source / builder / selection のどこを触るべきかを 1 hypothesis に narrow できる。

## Current Read

- `#61` は `pass / promote` まで進んだが、used features は baseline と同じ 13 本だった
- replay で追加した `horse_days_since_last_race`, `horse_weight_change`, `horse_distance_change`, `horse_surface_switch`, `race_class_score`, `horse_class_change` は final manifest に出ていない
- したがって current result は gain ではなく no-op read である

## In Scope

- `configs/features_local_baseline_class_rest_surface_replay.yaml`
- `artifacts/models/local_nankan_baseline_model.manifest_r20260330_local_nankan_class_rest_surface_replay_v1.json`
- `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_class_rest_surface_replay_v1.json`
- feature build / selection diagnostics
- local Nankan source availability

## Non-Goals

- broad new feature family の追加
- JRA / NAR integration
- class/rest/surface replay の再学習を闇雲に回すこと

## Success Metrics

- core 6 features の each status が `built / missing / low-coverage / filtered` のどれかで説明できる
- next NAR child issue が 1 measurable hypothesis に narrow される
- no-op replay を current best line と誤読しない decision summary が残る

## Stop Condition

- feature-level の explainability が取れず、実質的に source schema 問題としか読めない
- current local Nankan source では replay family を正しく評価できないと判断される
