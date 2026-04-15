# GitHub Issue Queue: Current

## Purpose

この文書は current open issue と次の実行順だけを保持する。completed history や長い判断経緯はここへ積まない。

詳細は GitHub issue thread、artifact、tagged snapshot/reference を見る。この file は「今どれを正本として追うか」を短く固定するための入口である。

## Current Priority

1. JRA benchmark 正本は維持する。
2. NAR は readiness track として分離して扱う。
3. current queue では open issue だけを追い、completed issue の説明は繰り返さない。

## Open Issues

### `#120` local Nankan strict provenance trust gate

- role: current NAR root blocker
- status: open, fail-closed
- source-of-truth: GitHub issue `#120`
- first read:
  - [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
  - [../artifacts/reports/local_nankan_provenance_audit.json](../artifacts/reports/local_nankan_provenance_audit.json)
  - `read_order`
  - [../artifacts/reports/benchmark_gate_local_nankan.json](../artifacts/reports/benchmark_gate_local_nankan.json)
- current meaning:
  - historical local Nankan result-ready rows は `provable pre-race market capture` を持たず、strict provenance read は `pre_race=0`, `post_race=728850`, `unknown=257`
  - この状態では historical local Nankan ROI は trust-carrying benchmark evidence と見なさない
  - runtime guard と wrapper は current alias `../artifacts/reports/local_nankan_provenance_audit.json` を優先し、未生成時だけ repaired snapshot を fallback として読む
  - `#101` と `#103` は provenance trust gate の後段に blocked のまま維持する

### `#121` local Nankan historical source timing corrective

- role: `#120` の narrow corrective
- status: open, stop-condition reached on historical recoverability
- source-of-truth: GitHub issue `#121`
- first read:
  - [../artifacts/reports/local_nankan_source_timing_audit.json](../artifacts/reports/local_nankan_source_timing_audit.json)
  - `read_order`
  - [issue_library/next_issue_local_nankan_source_timing_corrective.md](issue_library/next_issue_local_nankan_source_timing_corrective.md)
- current meaning:
  - current cache から historical result-ready `pre_race` capture を復元する仮説は negative read で narrowed された
  - past local Nankan benchmark は diagnostic-only に降格し、future-only readiness track `#122` を正本にする
  - current alias read は `result_ready_pre_race_rows=0`, `future_only_pre_race_rows=426`, `recommended_action=downgrade_historical_benchmark_to_diagnostic_only`
  - runtime guard は current alias `../artifacts/reports/local_nankan_source_timing_audit.json` を優先し、未生成時だけ `issue121` snapshot を fallback として読む

### `#124` JRA pruning stage-7 rollout guardrails

- role: JRA 側の current human-review surface
- status: open, review pending
- source-of-truth: GitHub issue `#124`
- local entrypoints:
  - [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
  - [jra_pruning_package_review_20260410.md](jra_pruning_package_review_20260410.md)
  - [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)
- next action:
  - reviewer が `approve implementation-candidate review package` か `keep docs-only` を判断する

### `#122` local Nankan future-only pre-race readiness track

- role: current NAR operator-default path after historical trust downgrade
- status: open, blocked on external result arrival
- source-of-truth: GitHub issue `#122`
- first read:
  - [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)
  - `read_order`
  - `readiness_surfaces.readiness_supervisor`
  - `operator_runtime`
  - `readiness_surfaces.readiness_supervisor.current_refs.capture_upcoming_only|capture_as_of|capture_pre_filter_row_count|capture_filtered_out_count`
  - `highlights.supervisor_capture_upcoming_only|supervisor_capture_as_of|supervisor_capture_pre_filter_rows|supervisor_capture_filtered_out`
- command/source docs:
  - [command_reference.md](command_reference.md)
  - [scripts_guide.md](scripts_guide.md)
- current meaning:
  - future-only pre-race pool は維持されている
  - current top-level operator surfaces から strict upcoming filter の cutoff と母数を child capture manifest 深掘りなしで読める
  - historical local Nankan ROI は diagnostic-only に降格しており、`local_nankan_recommended` も operator convenience alias であって trust-carrying benchmark default ではない
  - `result_ready_races>0` が来るまでは readiness blocker 継続

### `#101` pre-race-only benchmark rebuild

- role: `#120` strict trust ready と `#122` result arrival の両方が揃った後に再開する downstream gate
- status: open, waiting on `#120` and `#122`
- source-of-truth: GitHub issue `#101`

### `#103` value-blend architecture bootstrap

- role: `#120` and `#101` 完了後にだけ再開する NAR model bootstrap
- status: open, waiting on `#120` and `#101`
- source-of-truth: GitHub issue `#103`

### `#123` JRA-equivalent trust completion gate

- role: NAR top-level completion gate
- status: open, not the day-to-day execution entrypoint
- source-of-truth: GitHub issue `#123`

## Execution Order

1. `#120` の provenance audit / benchmark gate を current NAR truth として保守する。
2. `#121` の negative read を踏まえ、historical local Nankan ROI を trust-carrying benchmark として扱わない。
3. `#122` の future-only readiness track を operator-default path として保守する。
4. `#124` の human review decision を待つ。
5. JRA 本線を再開する場合だけ、review 結果の後に次の 1 measurable hypothesis を選ぶ。
6. NAR は `#120 -> #121 -> #122 -> #101 -> #103` の順でしか進めない。

## Reading Order

### JRA current read

1. GitHub issue `#124`
2. [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
3. [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)

### NAR current read

1. [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
2. [../artifacts/reports/local_nankan_provenance_audit.json](../artifacts/reports/local_nankan_provenance_audit.json)
3. [../artifacts/reports/local_nankan_source_timing_audit.json](../artifacts/reports/local_nankan_source_timing_audit.json)
4. [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)
5. GitHub issue `#122`

## Boundaries

- completed issue history はここへ戻さない。
- `next_issue_*.md` を current queue として直接読まない。
- current queue に必要な情報だけ残し、長い historical explanation は issue thread または snapshot へ置く。