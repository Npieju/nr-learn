# Docs Index

この `docs/` は、初めて引き継ぐ人が「何を読めば全体像と現在地が分かるか」を最短で判断できるように整理している。

2026-03-29 以降の新規標準化は、既存 docs を参照しつつも、まず [strategy_v2.md](strategy_v2.md) と [autonomous_dev_standard.md](autonomous_dev_standard.md) を入口に進める。

## 方針

- 残すのは、構造理解と運用判断に必要な正本ドキュメントに限定する。
- `next_issue_*.md` の draft library は [issue_library/](issue_library/) に隔離し、指示用の正本 docs と混在させない。
- 日付依存の実験メモ、セッションメモ、途中経過の説明資料は `docs/` に置かない。
- commit しない一時メモや scratch doc は `docs/` に置かず、gitignore 管理の local path に隔離する。
- 古い判断経緯や細かな数値の追跡は `git log` と `artifacts/reports/` を正本にする。
- `issue_library/` は GitHub issue と自動同期されない local source/reference である。GitHub issue thread に objective / hypothesis / decision summary を反映した後は、local に同じ役割の draft を重複保持しない。

## まず読む順番

引き継ぎ時は、まず次の 5 本を順に読む。

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [development_flow.md](development_flow.md)
4. [command_reference.md](command_reference.md)
5. [benchmarks.md](benchmarks.md)

対外説明が必要なら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) と [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) を追加で読む。実装の中身まで入る必要がある場合だけ [system_architecture.md](system_architecture.md) を読む。

latest 2025 の actual-date compare を再開したいだけなら、次の 3 段で足りる。

1. [serving_validation_guide.md](serving_validation_guide.md) の dashboard summary JSON 一覧から、September は `long_horizon -> tighter policy -> recent-2018` の順で見て、次に December control を確認する。
2. compare を回し直す必要があるときだけ [command_reference.md](command_reference.md) の latest 2025 compare 例を同じ順に使う。
3. formal support まで根拠を掘る必要があるときだけ [benchmarks.md](benchmarks.md) と `artifacts/reports/` の promotion gate / evaluation summary に降りる。

新しい issue 駆動運用を始めるときは、次の 3 本を最初に読む。

1. [strategy_v2.md](strategy_v2.md)
2. [autonomous_dev_standard.md](autonomous_dev_standard.md)
3. [initial_issue_backlog.md](initial_issue_backlog.md)

AI coding の標準化を進めるときは、追加で [ai_coding_best_practices.md](ai_coding_best_practices.md) を読む。

運用上の hard caution は [development_operational_cautions.md](development_operational_cautions.md) を使う。

progress 不足の棚卸しは [progress_coverage_audit.md](progress_coverage_audit.md) を使う。

現在 GitHub に起こすべき案件の直近キューは [github_issue_queue_current.md](github_issue_queue_current.md) を使う。

2026-04-05 時点では、active queue は [github_issue_queue_current.md](github_issue_queue_current.md) の `3.0` と `Execution Order` を正本にする。draft/reference library は [issue_library/](issue_library/) に分離してあり、そのまま current priority と読まず、issue 化する 1 hypothesis だけを queue から選ぶ。

2026-04-11 時点の JRA pruning human review package は次の 4 本を正本にする。

1. [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
2. [jra_pruning_package_review_20260410.md](jra_pruning_package_review_20260410.md)
3. [issue_library/next_issue_pruning_stage7_rollout_guardrails.md](issue_library/next_issue_pruning_stage7_rollout_guardrails.md)
4. [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)

rollback まで見る場合だけ [jra_pruning_stage7_rollback_checklist.md](jra_pruning_stage7_rollback_checklist.md) を追加で開く。

issue 下書きや historical reference を追うときは、個別ファイルをこの README に列挙せず、次の正本から辿る。

- JRA / NAR の current priority と execution order: [github_issue_queue_current.md](github_issue_queue_current.md)
- NAR の stage 定義と completion gate: [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
- draft / historical issue source 全体の索引: [issue_library/README.md](issue_library/README.md)

運用上の入口だけ残す。

- ML モデル開発プロセス: [ml_model_development_standard.md](ml_model_development_standard.md)
- `ROI > 1.20` の success 条件: [roi120_kpi_definition.md](roi120_kpi_definition.md)
- JRA baseline artifact inventory: [jra_baseline_artifact_inventory.md](jra_baseline_artifact_inventory.md)
- feature family の優先順位: [feature_family_ranking.md](feature_family_ranking.md)
- policy family の優先順位: [policy_family_shortlist.md](policy_family_shortlist.md)
- revision gate 前の確認: [revision_gate_candidate_checklist.md](revision_gate_candidate_checklist.md)
- pruning stage-7 review 前の確認: [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)
- promoted と operational role の切り分け: [promoted_vs_operational_role_split_standard.md](promoted_vs_operational_role_split_standard.md)
- NAR separate-universe 方針: [nar_model_transfer_strategy.md](nar_model_transfer_strategy.md)
- NAR formal read の分母ルール: [nar_bet_denominator_standard.md](nar_bet_denominator_standard.md)
- NAR formal read template: [nar_formal_read_template.md](nar_formal_read_template.md)
- tail loader optimization の gate 運用: [tail_loader_equivalence_gate_standard.md](tail_loader_equivalence_gate_standard.md)

local Nankan future-only readiness は、個別 draft ではなく current queue と status board を一次参照にする。

- issue / blocker の正本: [github_issue_queue_current.md](github_issue_queue_current.md)
- parity / completion gate の位置づけ: [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
- operator の current board: [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)

`issue_library/` の個別 `next_issue_*.md` は、queue や ladder から必要になった 1 hypothesis だけを開く。まず [issue_library/README.md](issue_library/README.md) を開き、その後に個別 file へ降りる。入口 docs に個別 draft を増やさない。

GitHub issue に起票済みで、thread 側が最新の objective / hypothesis / validation / decision を保持している場合は、その local draft は削除候補として扱う。local に残すのは次だけに限定する。

- current queue が直接参照している未起票 source draft
- package review / decision summary から参照される historical decision source
- GitHub thread では保持しづらい長文の internal review memo

## 正本

このプロジェクトの理解と運用に必要な中核文書は次の 8 本である。

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [development_flow.md](development_flow.md)
4. [benchmarks.md](benchmarks.md)
5. [evaluation_guide.md](evaluation_guide.md)
6. [command_reference.md](command_reference.md)
7. [system_architecture.md](system_architecture.md)
8. [public_benchmark_snapshot.md](public_benchmark_snapshot.md)

## 補助資料

必要なときだけ読む補助資料は次のとおりである。

1. [artifact_guide.md](artifact_guide.md)
2. [environment_notes.md](environment_notes.md)
3. [serving_validation_guide.md](serving_validation_guide.md)
4. [data_extension.md](data_extension.md)
5. [scripts_guide.md](scripts_guide.md)

## 文書の役割

### 正本

- [project_overview.md](project_overview.md)
  - プロジェクトの目的、現在の operational baseline、直近の正式通過候補、未決定事項をまとめた入口資料。
- [roadmap.md](roadmap.md)
  - いま何が完了していて、次に何を決めるかを管理する実行計画の正本。
- [development_flow.md](development_flow.md)
  - smoke / probe と full revision gate をどう分けるか、`revision` と commit をどの単位で切るかを定義する運用ルール。
- [benchmarks.md](benchmarks.md)
  - 何を良い結果とみなすか、現在の baseline/candidate をどう扱うかを定義する内部採用基準。
- [evaluation_guide.md](evaluation_guide.md)
  - `run_evaluate.py`、`stability_assessment`、`run_promotion_gate.py`、`revision` 単位の正式判断をまとめた評価ガイド。
- [command_reference.md](command_reference.md)
  - 日常で使う主要 CLI の入口を用途別にまとめた実行リファレンス。
- [system_architecture.md](system_architecture.md)
  - 実コードに対応したシステム構成、主要モジュール、学習から serving までの流れを整理した技術資料。
- [public_benchmark_snapshot.md](public_benchmark_snapshot.md)
  - 対外説明で使う current benchmark の要約と、現在の採用位置づけをまとめた公開向け snapshot。
- [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md)
  - 対外説明で benchmark 数字と operational role の関係を読むための公開向けガイド。

### 補助資料

- [artifact_guide.md](artifact_guide.md)
  - 各 CLI がどこに何を出力するか、正式判断でどの artifact を見るかを整理した資料。
- [environment_notes.md](environment_notes.md)
  - GPU / Docker / Notebook 周りの実務メモとトラブルシュートをまとめた補助資料。
- [serving_validation_guide.md](serving_validation_guide.md)
  - serving smoke / replay / compare / bankroll sweep / dashboard の導線をまとめた検証ガイド。
  - latest 2025 の actual-date compare を再開するときは、この文書の quickstart を最初に見る。
- [data_extension.md](data_extension.md)
  - Kaggle/JRA 主表に対して、netkeiba などの外部データをどう追加し、どう検証するかを整理した資料。
- [scripts_guide.md](scripts_guide.md)
  - `scripts/` 配下の CLI と補助 shell を用途別に引けるようにした索引資料。

## 更新するときの基準

- 現在地や優先順位が変わったら [roadmap.md](roadmap.md) を更新する。
- baseline / candidate の位置づけが変わったら [project_overview.md](project_overview.md) を更新する。
- 採用基準や formal result の読み方が変わったら [benchmarks.md](benchmarks.md) と [evaluation_guide.md](evaluation_guide.md) を更新する。
- 対外説明に出す数値が変わったら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) を更新する。
- benchmark 数字と運用上の位置づけの説明順が変わったら [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) も更新する。
- public / internal の説明境界が変わったら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) と [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) を同時に見直す。
- 実行手順や Git 運用が変わったら [development_flow.md](development_flow.md) と [command_reference.md](command_reference.md) を更新する。

## コードの入口

- 学習: [scripts/run_train.py](../scripts/run_train.py)
- 評価: [scripts/run_evaluate.py](../scripts/run_evaluate.py)
- 予測: [scripts/run_predict.py](../scripts/run_predict.py)
- バックテスト: [scripts/run_backtest.py](../scripts/run_backtest.py)
- revision gate: [scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
- latest revision gate wrapper: [scripts/run_netkeiba_latest_revision_gate.py](../scripts/run_netkeiba_latest_revision_gate.py)
- serving smoke: [scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- serving compare dashboard: [scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- serving compare aggregate: [scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)
- script 全体の索引: [scripts_guide.md](scripts_guide.md)

## 補足

- 実験の細かい経緯や旧判断を確認したい場合は、削除した文書名を探すのではなく、コミット履歴と artifact を起点にたどるほうが正確である。
- 今後 `docs/` に新しい文書を追加する前に、その内容が既存の 8 本の正本か 5 本の補助資料のどこへ統合できるかを先に確認する。
