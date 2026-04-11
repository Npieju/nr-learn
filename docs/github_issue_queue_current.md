# GitHub Issue Queue: Current

## Purpose

この文書は current open issue と次の実行順だけを保持する。completed history や長い判断経緯はここへ積まない。

詳細は GitHub issue thread、artifact、tagged snapshot/reference を見る。この file は「今どれを正本として追うか」を短く固定するための入口である。

## Current Priority

1. JRA benchmark 正本は維持する。
2. NAR は readiness track として分離して扱う。
3. current queue では open issue だけを追い、completed issue の説明は繰り返さない。

## Open Issues

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

- role: current NAR execution surface
- status: open, blocked on external result arrival
- source-of-truth: GitHub issue `#122`
- first read:
  - [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)
  - `readiness_surfaces.readiness_supervisor`
  - `operator_runtime`
- command/source docs:
  - [command_reference.md](command_reference.md)
  - [scripts_guide.md](scripts_guide.md)
- current meaning:
  - future-only pre-race pool は維持されている
  - `result_ready_races>0` が来るまでは readiness blocker 継続

### `#101` pre-race-only benchmark rebuild

- role: `#122` の result-ready 到着後に再開する downstream gate
- status: open, waiting on `#122`
- source-of-truth: GitHub issue `#101`

### `#103` value-blend architecture bootstrap

- role: `#101` 完了後に再開する NAR model bootstrap
- status: open, waiting on `#101`
- source-of-truth: GitHub issue `#103`

### `#123` JRA-equivalent trust completion gate

- role: NAR top-level completion gate
- status: open, not the day-to-day execution entrypoint
- source-of-truth: GitHub issue `#123`

## Execution Order

1. `#122` の board / manifest / issue thread を current truth として保守する。
2. `#124` の human review decision を待つ。
3. JRA 本線を再開する場合だけ、review 結果の後に次の 1 measurable hypothesis を選ぶ。
4. NAR は `#122 -> #101 -> #103` の順でしか進めない。

## Reading Order

### JRA current read

1. GitHub issue `#124`
2. [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
3. [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)

### NAR current read

1. [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)
2. GitHub issue `#122`
3. [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
4. [command_reference.md](command_reference.md)

## Boundaries

- completed issue history はここへ戻さない。
- `next_issue_*.md` を current queue として直接読まない。
- current queue に必要な情報だけ残し、長い historical explanation は issue thread または snapshot へ置く。