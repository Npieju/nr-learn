# nr-learn

JRA競馬予想を、再現可能な機械学習パイプラインとして扱うためのリポジトリです。

この README は詳細手順ではなく、プロジェクト全体への入口として使います。

## New Operating System

まず [docs/README.md](docs/README.md) を入口にして、current source-of-truth だけを辿ります。個別 docs を広く読む前提にはしません。

標準手順そのものを確認したいときは次の 4 本を開きます。

1. [docs/strategy_v2.md](docs/strategy_v2.md)
2. [docs/autonomous_dev_standard.md](docs/autonomous_dev_standard.md)
3. [docs/ml_model_development_standard.md](docs/ml_model_development_standard.md)
4. [docs/development_operational_cautions.md](docs/development_operational_cautions.md)

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
5. [docs/github_issue_queue_current.md](docs/github_issue_queue_current.md)

対外向け説明だけが必要なら [docs/public_benchmark_snapshot.md](docs/public_benchmark_snapshot.md) と [docs/public_benchmark_operational_reading_guide.md](docs/public_benchmark_operational_reading_guide.md) を見ます。docs 全体の役割分担は [docs/README.md](docs/README.md) を参照してください。

必要な資料を役割別に辿りたいときは [docs/README.md](docs/README.md) を使います。README では重複した索引を持たず、最短導線だけを残します。

## よく使う script

- 学習: [scripts/run_train.py](scripts/run_train.py)
- 評価: [scripts/run_evaluate.py](scripts/run_evaluate.py)
- 予測: [scripts/run_predict.py](scripts/run_predict.py)
- 当日 JRA live 予測: [scripts/run_jra_live_predict.py](scripts/run_jra_live_predict.py)
- 2026 YTD netkeiba backfill: [scripts/run_netkeiba_2026_ytd_backfill.py](scripts/run_netkeiba_2026_ytd_backfill.py)
- 2026 YTD netkeiba coverage snapshot: [scripts/run_netkeiba_2026_ytd_snapshot.py](scripts/run_netkeiba_2026_ytd_snapshot.py)
- 2026 YTD live handoff: [scripts/run_netkeiba_2026_live_handoff.py](scripts/run_netkeiba_2026_live_handoff.py)
- 2026 YTD status board: [scripts/run_netkeiba_2026_status_board.py](scripts/run_netkeiba_2026_status_board.py)
- 2026 post-race benchmark gate: [scripts/run_netkeiba_2026_benchmark_gate.py](scripts/run_netkeiba_2026_benchmark_gate.py)
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
/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_2026_benchmark_gate.py --skip-train --skip-evaluate
```

`run_jra_live_predict.py` は、当日の race_list から JRA race_id を発見し、live racecard と pedigree を一時ディレクトリへ crawl した上で、現時点 odds を使って prediction CSV / summary / markdown report を出力する。標準出力先は `artifacts/predictions/predictions_<date>_jra_live.*`。

`run_netkeiba_2026_ytd_backfill.py` は、2026-01-01 から当日までを `race_list` 起点で backfill するための wrapper で、same-day serving 用の 2026 履歴補完を繰り返し回しやすくする。標準では各 cycle 後に `run_netkeiba_2026_ytd_snapshot.py` を自動実行し、cycle 完了は post-cycle hook 実行前に `artifacts/reports/netkeiba_backfill_manifest_2026_ytd.json` へ反映される。

`run_netkeiba_2026_ytd_snapshot.py` は、2026 YTD backfill の target manifest と lock file を見ながら coverage snapshot を別名出力する wrapper で、running 中でも current stage と readiness を読みやすくする。標準出力先は `artifacts/reports/netkeiba_coverage_snapshot_2026_ytd.json` で、成功時は status board も更新する。

`run_netkeiba_2026_live_handoff.py` は、2026 YTD snapshot と外部結果 CSV の max date を見ながら、履歴が `race_date - history_lag_days` まで追いついた時点で `run_jra_live_predict.py` へ handoff する wrapper である。未達時も waiting manifest を残し、poll ごとに status board も更新する。既に同じ `race_date` の completed manifest と live 出力が揃っていれば、既定では rerun せず即終了する。強制 rerun したいときだけ `--force` を使う。

`run_netkeiba_2026_status_board.py` は、backfill / snapshot / live handoff の manifest を 1 つに集約し、same-day serving readiness の current phase と next action を 1 ファイルで読めるようにする。進行中 cycle では crawl target manifest を直接使って `active_cycle` と各 target の `processed_ids` / `rows_written` を出し、history frontier は外部 CSV から直接 max date を読むので、snapshot や handoff polling を待たずに board 側で途中経過を追える。さらに `race_result_gap_days` / `race_card_gap_days` / `limiting_history_target` と、completed 後の `policy_selected_rows` / `num_races` も出し、crawl lock も snapshot 由来の stale 値ではなく実 lock file 状態を直接読む。

`run_netkeiba_2026_backfill_rollover.py` は、進行中 target が 0 本の cycle 境界を待って旧 2026 backfill process に `SIGINT` を送り、その後に最新コードの `run_netkeiba_2026_ytd_backfill.py` を再起動する one-shot watcher である。long-running backfill を安全に新コードへ切り替えたいときに使う。

`run_netkeiba_2026_same_day_ops.py` は、2026 same-day serving の orchestration 入口で、status board を更新したうえで backfill / handoff / rollover を必要なときだけ起動し、既に completed なら何もせず summary manifest を返す。日次運用の first command として使う。

`run_netkeiba_2026_benchmark_gate.py` は、same-day serving 完了後に 2026 YTD netkeiba データで enriched benchmark rerun を再現するための wrapper である。既定では `configs/data_2025_latest.yaml`、`configs/model_catboost_fundamental_enriched.yaml`、`configs/features_catboost_fundamental_enriched.yaml` を束ねて `run_netkeiba_benchmark_gate.py` を呼び、完了後に status board も更新する。

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
