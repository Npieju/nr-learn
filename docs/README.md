# Docs Index

この `docs/` は、初めてこのリポジトリを見る人が、短時間でプロジェクトの全体像を理解できることを目的に整理している。

## 方針

- 残すのは、構造を理解するための正本ドキュメントだけに限定する。
- 日付依存の実験メモ、セッションメモ、途中経過の説明資料は `docs/` から外す。
- 履歴を追いたい場合は `git log` と `artifacts/reports/` を参照する。

## まず読む順番

1. [project_overview.md](project_overview.md)
2. [system_architecture.md](system_architecture.md)
3. [benchmarks.md](benchmarks.md)
4. [data_extension.md](data_extension.md)

## 各文書の役割

- [project_overview.md](project_overview.md)
  - プロジェクトの目的、現在の到達点、主要スコア、今後の改善方針をまとめた入口資料。
- [system_architecture.md](system_architecture.md)
  - 実コードに対応したシステム構成、主要モジュール、学習から serving までの流れを整理した技術資料。
- [benchmarks.md](benchmarks.md)
  - 何を良い結果とみなすか、その判断基準を定義した資料。外部比較の目安と内部の採用基準を含む。
- [data_extension.md](data_extension.md)
  - Kaggle/JRA 主表に対して、netkeiba などの外部データをどう追加し、どう検証するかを整理した資料。

## コードの入口

- 学習: [scripts/run_train.py](../scripts/run_train.py)
- 評価: [scripts/run_evaluate.py](../scripts/run_evaluate.py)
- 予測: [scripts/run_predict.py](../scripts/run_predict.py)
- バックテスト: [scripts/run_backtest.py](../scripts/run_backtest.py)
- serving smoke: [scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- serving compare dashboard: [scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- serving compare aggregate: [scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)

## 補足

- 実験の細かい経緯や旧判断を確認したい場合は、削除した文書名を探すのではなく、コミット履歴と artifact を起点にたどるほうが正確である。
- 今後 `docs/` に追加する文書も、この 4 本の正本のどこに属するかを先に決めてから追加する。