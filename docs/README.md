# Docs Index

## Purpose

この `docs/` の役割は 1 つだけである。必要な current source-of-truth へ最短で到達させる。

細かな経緯や時点依存の判断を毎回広く読む前提にはしない。current 情報は少数の正本に寄せ、差分や過去時点の記録は issue thread、artifact、tagged snapshot/reference で追う。

## First Read

最初に開くのは次の 5 本だけでよい。

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [development_flow.md](development_flow.md)
4. [command_reference.md](command_reference.md)
5. [github_issue_queue_current.md](github_issue_queue_current.md)

戦略や標準手順まで必要になったときだけ、次を追加で読む。

1. [strategy_v2.md](strategy_v2.md)
2. [autonomous_dev_standard.md](autonomous_dev_standard.md)
3. [ml_model_development_standard.md](ml_model_development_standard.md)
4. [development_operational_cautions.md](development_operational_cautions.md)

## Task Routes

### モデル改善

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [ml_model_development_standard.md](ml_model_development_standard.md)
4. [benchmarks.md](benchmarks.md)
5. [ml_stage_checklist.md](ml_stage_checklist.md)

### 評価・昇格判断

1. [benchmarks.md](benchmarks.md)
2. [evaluation_guide.md](evaluation_guide.md)
3. [public_benchmark_snapshot.md](public_benchmark_snapshot.md)
4. NAR denominator 読みが必要なら [nar_bet_denominator_standard.md](nar_bet_denominator_standard.md)

### JRA human review

1. [github_issue_queue_current.md](github_issue_queue_current.md)
2. [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
3. [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)
4. rollback runbook は [jra_pruning_stage7_rollback_checklist.md](jra_pruning_stage7_rollback_checklist.md)

### 日常実行・障害対応

1. [command_reference.md](command_reference.md)
2. [scripts_guide.md](scripts_guide.md)
3. [artifact_guide.md](artifact_guide.md)
4. cache alias 運用は [primary_tail_cache_operational_policy.md](primary_tail_cache_operational_policy.md)
5. GPU / Docker / Notebook の実務メモは [environment_notes.md](environment_notes.md)
6. progress 棚卸し結果は [progress_coverage_audit.md](progress_coverage_audit.md)

### NAR readiness

1. [github_issue_queue_current.md](github_issue_queue_current.md)
2. [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
3. [command_reference.md](command_reference.md)
4. [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)

### 対外説明

1. [public_benchmark_snapshot.md](public_benchmark_snapshot.md)
2. [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md)
3. 時点付きの説明資料が必要なら [public_model_report_20260418.md](public_model_report_20260418.md)

## Source-Of-Truth Rules

- current priority と open issue の読みは [github_issue_queue_current.md](github_issue_queue_current.md) を正本にする。
- 実行手順は [command_reference.md](command_reference.md) を正本にする。
- benchmark と昇格基準は [benchmarks.md](benchmarks.md) と [evaluation_guide.md](evaluation_guide.md) を正本にする。
- 対外向け benchmark の読み方は [public_benchmark_snapshot.md](public_benchmark_snapshot.md) と [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) を正本にする。
- draft や historical issue source は [issue_library/README.md](issue_library/README.md) から辿る。個別 `next_issue_*.md` を入口にしない。

## Keep Or Delete

- current info は current source-of-truth にだけ書く。
- review package や時点付き判断は version/tag で識別できる snapshot として残す。
- 細かい仕様や補助説明のために file を増やす場合は、原則として read-only な version/tag 付き snapshot に倒す。
- current doc の下に mutable な child docs を増やし、親子で平行更新が必要な構造は作らない。
- GitHub issue thread に source-of-truth が移った local draft は削除候補とする。
- 新しい docs を増やす前に、既存の正本へ統合できない理由を確認する。

- snapshot 側に「古い情報」と追記して回るのではなく、どの更新単位の記録かを version/tag で識別できるようにする。

## Update Rules

- current queue が変わったら [github_issue_queue_current.md](github_issue_queue_current.md) を更新する。
- current baseline / current active reading が変わったら [project_overview.md](project_overview.md) と [roadmap.md](roadmap.md) を更新する。
- 実行導線が変わったら [command_reference.md](command_reference.md) を更新する。
- public message が変わったら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) と [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) を更新する。

## Supplemental References

- AI 自動開発の実務標準は [ai_coding_best_practices.md](ai_coding_best_practices.md)
- module 構成の俯瞰は [system_architecture.md](system_architecture.md)
- policy challenger 判定の補助チェックは [policy_challenger_decision_checklist.md](policy_challenger_decision_checklist.md)
- kelly runtime family の候補整理は [kelly_runtime_candidate_matrix.md](kelly_runtime_candidate_matrix.md)
- tail loader 最適化の受け入れ基準は [tail_loader_equivalence_gate_standard.md](tail_loader_equivalence_gate_standard.md)

## Snapshot And Tag Rules

- current source-of-truth は原則として version や日付を file 名に入れない。
- 凍結して残す review package / decision memo / handoff note だけ、`<topic>_<version>.md` または `<topic>_<tag>.md` の形で snapshot 化する。
- source version の正本は `src/racing_ml/version.py` に置く。docs の commit batch が review package、運用 boundary、または current source-of-truth の大きい再編を確定させ、その source 更新と対応づけたい場合だけ semver 形式の git tag を切る。
- docs tag の基本形は `docs-v<major>.<minor>.<patch>` とする。これは source version を置き換えるものではなく、対応する source 更新 batch を指す補助ラベルである。
- 既存の大きい docs 再編は `docs-v0.1.0`、tag workflow 導入は `docs-v0.2.0` のように、意味のある batch 単位で minor を上げる。
- minor の後に小さい docs 修正を積むときだけ `docs-v0.2.1` のように patch を上げる。
- 軽微な wording fix や typo 修正には毎回 tag を切らない。
- snapshot file 名と git tag は 1 対 1 に揃っている必要はないが、どの更新単位を凍結したかが相互に追える状態を保つ。

## Notes

- 長い historical 説明を current docs に抱え込まない。
- docs を読む順番に迷ったら、個別ファイル検索ではなくこの index に戻る。