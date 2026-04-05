# nr-learn

JRA競馬予想を、再現可能な機械学習パイプラインとして扱うためのリポジトリです。

この README は詳細手順ではなく、プロジェクト全体への入口として使います。

## New Operating System

既存の `docs/` は参照用として維持しつつ、これからの標準化は次の 3 本を入口に進めます。

1. [docs/strategy_v2.md](docs/strategy_v2.md)
2. [docs/autonomous_dev_standard.md](docs/autonomous_dev_standard.md)
3. [docs/initial_issue_backlog.md](docs/initial_issue_backlog.md)
4. [docs/ai_coding_best_practices.md](docs/ai_coding_best_practices.md)
5. [docs/ml_model_development_standard.md](docs/ml_model_development_standard.md)
6. [docs/roi120_kpi_definition.md](docs/roi120_kpi_definition.md)
7. [docs/development_operational_cautions.md](docs/development_operational_cautions.md)

方針の要点:

- 第 1 目標は、JRA 正本データで `ROI > 1.20` を狙う長期運用モデル開発
- NAR は並行して ingestion / readiness を進める別トラック
- 開発は issue 駆動、短命 branch、artifact ベースの formal gate を標準にする
- 実装、反復、定型検証は基本的に AI に寄せる
- 数秒超の処理には progress を必須にする

## 何があるか

- 学習: CatBoost / LightGBM を使った勝率・ROI 系モデル学習
- 評価: nested walk-forward と promotion gate による採用判定
- serving 検証: actual calendar の smoke / compare / dashboard
- データ拡張: netkeiba など外部 CSV の追加と backfill

## 引き継ぎの最短導線

引き継ぎ時は次の順で見れば、現状把握と再開判断に必要な情報が揃います。

1. [docs/project_overview.md](docs/project_overview.md)
2. [docs/roadmap.md](docs/roadmap.md)
3. [docs/development_flow.md](docs/development_flow.md)
4. [docs/command_reference.md](docs/command_reference.md)
5. [docs/benchmarks.md](docs/benchmarks.md)

対外向け説明だけが必要なら [docs/public_benchmark_snapshot.md](docs/public_benchmark_snapshot.md) を見ます。docs 全体の役割分担は [docs/README.md](docs/README.md) を参照してください。

## まず読む場所

1. [docs/project_overview.md](docs/project_overview.md)
2. [docs/system_architecture.md](docs/system_architecture.md)
3. [docs/benchmarks.md](docs/benchmarks.md)
4. [docs/development_flow.md](docs/development_flow.md)
5. [docs/evaluation_guide.md](docs/evaluation_guide.md)
6. [docs/command_reference.md](docs/command_reference.md)
7. [docs/roadmap.md](docs/roadmap.md)
8. [docs/public_benchmark_snapshot.md](docs/public_benchmark_snapshot.md)

補助資料:

1. [docs/artifact_guide.md](docs/artifact_guide.md)
2. [docs/environment_notes.md](docs/environment_notes.md)
3. [docs/serving_validation_guide.md](docs/serving_validation_guide.md)
4. [docs/data_extension.md](docs/data_extension.md)
5. [docs/scripts_guide.md](docs/scripts_guide.md)

docs の索引と更新ルールは [docs/README.md](docs/README.md) にあります。

## 役割ごとの入口

- プロジェクトの目的と現状: [docs/project_overview.md](docs/project_overview.md)
- システム構造: [docs/system_architecture.md](docs/system_architecture.md)
- benchmark と採用基準: [docs/benchmarks.md](docs/benchmarks.md)
- 日々の進め方と revision の切り方: [docs/development_flow.md](docs/development_flow.md)
- 正式な評価と promotion gate: [docs/evaluation_guide.md](docs/evaluation_guide.md)
- 主要コマンド: [docs/command_reference.md](docs/command_reference.md)
- 現在の優先順位と次の実行順: [docs/roadmap.md](docs/roadmap.md)
- 対外向けの benchmark 要約: [docs/public_benchmark_snapshot.md](docs/public_benchmark_snapshot.md)
- artifact の見方: [docs/artifact_guide.md](docs/artifact_guide.md)
- GPU / Docker / Notebook メモ: [docs/environment_notes.md](docs/environment_notes.md)
- serving の検証導線: [docs/serving_validation_guide.md](docs/serving_validation_guide.md)
- 外部データ追加: [docs/data_extension.md](docs/data_extension.md)
- script 全体の索引: [docs/scripts_guide.md](docs/scripts_guide.md)

## よく使う script

- 学習: [scripts/run_train.py](scripts/run_train.py)
- 評価: [scripts/run_evaluate.py](scripts/run_evaluate.py)
- 予測: [scripts/run_predict.py](scripts/run_predict.py)
- 当日 JRA live 予測: [scripts/run_jra_live_predict.py](scripts/run_jra_live_predict.py)
- 2026 YTD netkeiba backfill: [scripts/run_netkeiba_2026_ytd_backfill.py](scripts/run_netkeiba_2026_ytd_backfill.py)
- 2026 YTD netkeiba coverage snapshot: [scripts/run_netkeiba_2026_ytd_snapshot.py](scripts/run_netkeiba_2026_ytd_snapshot.py)
- 2026 YTD live handoff: [scripts/run_netkeiba_2026_live_handoff.py](scripts/run_netkeiba_2026_live_handoff.py)
- 2026 YTD status board: [scripts/run_netkeiba_2026_status_board.py](scripts/run_netkeiba_2026_status_board.py)
- バックテスト: [scripts/run_backtest.py](scripts/run_backtest.py)
- revision gate: [scripts/run_revision_gate.py](scripts/run_revision_gate.py)
- serving smoke: [scripts/run_serving_smoke.py](scripts/run_serving_smoke.py)
- serving compare: [scripts/run_serving_profile_compare.py](scripts/run_serving_profile_compare.py)
- script 一覧: [docs/scripts_guide.md](docs/scripts_guide.md)

## 最短の開始手順

1. データを用意する
2. 学習する
3. 評価する
4. 必要なら正式な revision gate を回す

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_ingest.py --config configs/data.yaml
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_best_eval
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --profile current_best_eval --max-rows 120000
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py --profile current_best_eval --revision r20260321a --evaluate-max-rows 200000 --evaluate-wf-mode full
```

2025 backfill 済みの最新データをそのまま使いたいときは、profile 名の末尾に `_2025_latest` を付ける。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_train.py --profile current_best_eval_2025_latest
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py --profile current_best_eval_2025_latest --max-rows 120000
/workspaces/nr-learn/.venv/bin/python scripts/run_predict.py --profile current_recommended_serving_2025_latest --race-date 2025-12-28
/workspaces/nr-learn/.venv/bin/python scripts/run_jra_live_predict.py --profile current_recommended_serving_2025_latest --race-date 2026-04-05 --headline-contains 大阪杯 --refresh
/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_2026_ytd_backfill.py --max-cycles 1
/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_2026_ytd_snapshot.py
/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_2026_live_handoff.py --race-date 2026-04-05 --headline-contains 大阪杯
/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_2026_status_board.py
```

`run_jra_live_predict.py` は、当日の race_list から JRA race_id を発見し、live racecard と pedigree を一時ディレクトリへ crawl した上で、現時点 odds を使って prediction CSV / summary / markdown report を出力する。標準出力先は `artifacts/predictions/predictions_<date>_jra_live.*`。

`run_netkeiba_2026_ytd_backfill.py` は、2026-01-01 から当日までを `race_list` 起点で backfill するための wrapper で、same-day serving 用の 2026 履歴補完を繰り返し回しやすくする。標準では各 cycle 後に `run_netkeiba_2026_ytd_snapshot.py` を自動実行し、cycle 完了は post-cycle hook 実行前に `artifacts/reports/netkeiba_backfill_manifest_2026_ytd.json` へ反映される。

`run_netkeiba_2026_ytd_snapshot.py` は、2026 YTD backfill の target manifest と lock file を見ながら coverage snapshot を別名出力する wrapper で、running 中でも current stage と readiness を読みやすくする。標準出力先は `artifacts/reports/netkeiba_coverage_snapshot_2026_ytd.json` で、成功時は status board も更新する。

`run_netkeiba_2026_live_handoff.py` は、2026 YTD snapshot と外部結果 CSV の max date を見ながら、履歴が `race_date - history_lag_days` まで追いついた時点で `run_jra_live_predict.py` へ handoff する wrapper である。未達時も waiting manifest を残し、poll ごとに status board も更新する。

`run_netkeiba_2026_status_board.py` は、backfill / snapshot / live handoff の manifest を 1 つに集約し、same-day serving readiness の current phase と next action を 1 ファイルで読めるようにする。進行中 cycle では crawl target manifest を直接使って `active_cycle` と各 target の `processed_ids` / `rows_written` を出し、history frontier は外部 CSV から直接 max date を読むので、snapshot や handoff polling を待たずに board 側で途中経過を追える。さらに `race_result_gap_days` / `race_card_gap_days` / `limiting_history_target` も出すので、あと何日足りないかがすぐ分かる。

`run_netkeiba_2026_backfill_rollover.py` は、進行中 target が 0 本の cycle 境界を待って旧 2026 backfill process に `SIGINT` を送り、その後に最新コードの `run_netkeiba_2026_ytd_backfill.py` を再起動する one-shot watcher である。long-running backfill を安全に新コードへ切り替えたいときに使う。

詳しいコマンドは [docs/command_reference.md](docs/command_reference.md) を参照してください。

## ディレクトリの見取り図

```text
nr-learn/
├── configs/
├── data/
├── docs/
├── notebooks/
├── scripts/
├── src/
└── artifacts/
```

## 主な出力先

- 学習モデル: `artifacts/models/`
- prediction: `artifacts/predictions/`
- report: `artifacts/reports/`
- dashboard: `artifacts/reports/dashboard/`

## 注意

- これは投資助言ではなく、機械学習の学習プロジェクトです。
- 短い smoke / probe は方向確認用であり、昇格判断には full revision gate を使います。
