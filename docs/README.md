# Docs Index

この `docs/` は、初めて引き継ぐ人が「何を読めば全体像と現在地が分かるか」を最短で判断できるように整理している。

## 方針

- 残すのは、構造理解と運用判断に必要な正本ドキュメントに限定する。
- 日付依存の実験メモ、セッションメモ、途中経過の説明資料は `docs/` に置かない。
- 古い判断経緯や細かな数値の追跡は `git log` と `artifacts/reports/` を正本にする。

## まず読む順番

引き継ぎ時は、まず次の 5 本を順に読む。

1. [project_overview.md](project_overview.md)
2. [roadmap.md](roadmap.md)
3. [development_flow.md](development_flow.md)
4. [command_reference.md](command_reference.md)
5. [benchmarks.md](benchmarks.md)

対外説明が必要なら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) を追加で読む。実装の中身まで入る必要がある場合だけ [system_architecture.md](system_architecture.md) を読む。

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

### 補助資料

- [artifact_guide.md](artifact_guide.md)
  - 各 CLI がどこに何を出力するか、正式判断でどの artifact を見るかを整理した資料。
- [environment_notes.md](environment_notes.md)
  - GPU / Docker / Notebook 周りの実務メモとトラブルシュートをまとめた補助資料。
- [serving_validation_guide.md](serving_validation_guide.md)
  - serving smoke / replay / compare / bankroll sweep / dashboard の導線をまとめた検証ガイド。
- [data_extension.md](data_extension.md)
  - Kaggle/JRA 主表に対して、netkeiba などの外部データをどう追加し、どう検証するかを整理した資料。
- [scripts_guide.md](scripts_guide.md)
  - `scripts/` 配下の CLI と補助 shell を用途別に引けるようにした索引資料。

## 更新するときの基準

- 現在地や優先順位が変わったら [roadmap.md](roadmap.md) を更新する。
- baseline / candidate の位置づけが変わったら [project_overview.md](project_overview.md) を更新する。
- 採用基準や formal result の読み方が変わったら [benchmarks.md](benchmarks.md) と [evaluation_guide.md](evaluation_guide.md) を更新する。
- 対外説明に出す数値が変わったら [public_benchmark_snapshot.md](public_benchmark_snapshot.md) を更新する。
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