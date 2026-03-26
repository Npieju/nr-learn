# Docs Index

この `docs/` は、初めてこのリポジトリを見る人が、短時間でプロジェクトの全体像を理解できることを目的に整理している。

## 方針

- 残すのは、構造を理解するための正本ドキュメントだけに限定する。
- 日付依存の実験メモ、セッションメモ、途中経過の説明資料は `docs/` から外す。
- 履歴を追いたい場合は `git log` と `artifacts/reports/` を参照する。

## 正本

このプロジェクトの理解に必要な中核文書は次の 7 本である。

1. [project_overview.md](project_overview.md)
2. [system_architecture.md](system_architecture.md)
3. [benchmarks.md](benchmarks.md)
4. [development_flow.md](development_flow.md)
5. [evaluation_guide.md](evaluation_guide.md)
6. [command_reference.md](command_reference.md)
7. [public_benchmark_snapshot.md](public_benchmark_snapshot.md)

進行中の優先順位と最新の計画は [roadmap.md](roadmap.md) を参照する。

## 補助資料

必要に応じて参照する補助資料は次のとおりである。

1. [artifact_guide.md](artifact_guide.md)
2. [environment_notes.md](environment_notes.md)
3. [serving_validation_guide.md](serving_validation_guide.md)
4. [data_extension.md](data_extension.md)
5. [scripts_guide.md](scripts_guide.md)

## 文書の役割

### 正本

- [project_overview.md](project_overview.md)
  - プロジェクトの目的、現在の到達点、主要スコア、今後の改善方針をまとめた入口資料。
- [system_architecture.md](system_architecture.md)
  - 実コードに対応したシステム構成、主要モジュール、学習から serving までの流れを整理した技術資料。
- [benchmarks.md](benchmarks.md)
  - 何を良い結果とみなすか、その判断基準を定義した資料。外部比較の目安と内部の採用基準を含む。
- [development_flow.md](development_flow.md)
  - smoke / probe と full revision gate をどう分けるか、`revision` をどう切るか、progress をどう扱うかを定義した開発フロー資料。
- [evaluation_guide.md](evaluation_guide.md)
  - `run_evaluate.py`、`stability_assessment`、`run_promotion_gate.py`、`revision` 単位の正式判断をまとめた評価ガイド。
- [command_reference.md](command_reference.md)
  - 日常で使う主要 CLI の入口を用途別にまとめた実行リファレンス。
- [public_benchmark_snapshot.md](public_benchmark_snapshot.md)
  - 対外説明で使う current benchmark の要約と、現在の採用位置づけをまとめた公開向け snapshot。
- [roadmap.md](roadmap.md)
  - 進行中の優先順位、判断済み事項、次に進める順番を管理する開発ロードマップ。

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

## コードの入口

- 学習: [scripts/run_train.py](../scripts/run_train.py)
- 評価: [scripts/run_evaluate.py](../scripts/run_evaluate.py)
- 予測: [scripts/run_predict.py](../scripts/run_predict.py)
- バックテスト: [scripts/run_backtest.py](../scripts/run_backtest.py)
- revision gate: [scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
- serving smoke: [scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- serving compare dashboard: [scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- serving compare aggregate: [scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)
- script 全体の索引: [scripts_guide.md](scripts_guide.md)

## 補足

- 実験の細かい経緯や旧判断を確認したい場合は、削除した文書名を探すのではなく、コミット履歴と artifact を起点にたどるほうが正確である。
- 今後 `docs/` に追加する文書も、この 6 本の正本のどこに属するかを先に決めてから追加する。