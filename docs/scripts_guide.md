# Scripts Guide

## 1. この文書の役割

この文書は、`scripts/` 配下の CLI と補助 shell を用途別に引けるようにした索引である。

日常運用の代表コマンドは [command_reference.md](command_reference.md) に寄せ、この文書では「どの場面でどの script を見に行くか」を整理する。

## 2. 基本の見方

- まずは [command_reference.md](command_reference.md) の代表コマンドを使う。
- そこに載っていない補助 script を探すときに、この文書でカテゴリを絞る。
- 正式な採用判断に関わるものは、用途が近くても smoke / probe 系と混同しない。
- 長時間かかる入口は progress 付きで整備している前提で読み、無言の長時間停止は異常の可能性として扱う。

## 3. データ取り込みと品質確認

### 3.1 主表の取り込み

- [../scripts/run_ingest.py](../scripts/run_ingest.py)
  - `configs/data.yaml` に従って raw から学習用テーブルを作る入口。

### 3.2 取り込み後の品質確認

- [../scripts/run_validate_data_sources.py](../scripts/run_validate_data_sources.py)
  - データソースの存在、join key、重複、正規化状態を検証する。
- [../scripts/run_feature_gap_report.py](../scripts/run_feature_gap_report.py)
  - 特徴量の low coverage や raw 列不足を点検する。
- [../scripts/run_netkeiba_coverage_snapshot.py](../scripts/run_netkeiba_coverage_snapshot.py)
  - 外部データの coverage を時点付きで確認する。

## 4. 学習と正式評価

### 4.1 学習

- [../scripts/run_train.py](../scripts/run_train.py)
  - 単体モデルの標準学習入口。
  - 2025 backfill 済みデータを使うときは、既存 profile 名に `_2025_latest` を付ければ同じ family を最新 split で呼べる。
- [../scripts/run_build_value_stack.py](../scripts/run_build_value_stack.py)
  - 学習済み component artifact から value blend bundle を構築する。
- [../scripts/run_bundle_models.py](../scripts/run_bundle_models.py)
  - 複数 component を bundle 化して serving / evaluation 用にまとめる。

### 4.2 正式評価と gate

- [../scripts/run_evaluate.py](../scripts/run_evaluate.py)
  - nested walk-forward を含む正式評価の入口。
- [../scripts/run_promotion_gate.py](../scripts/run_promotion_gate.py)
  - 昇格条件を gate としてまとめて判定する。
- [../scripts/run_revision_gate.py](../scripts/run_revision_gate.py)
  - train → evaluate → promotion gate を revision 単位で直列実行する。
- [../scripts/run_validate_evaluation_manifest.py](../scripts/run_validate_evaluation_manifest.py)
  - `evaluation_manifest.json` と versioned manifest の整合を検証する。

これらの正式評価系 script は、概ね設定読込、特徴量処理、fold 処理、artifact 書き出しの段階を progress で出す。

`run_revision_gate.py` は `--dry-run` も持ち、重い train / evaluate を回さずに planned command と revision manifest だけ確認できる。

正式な評価の読み筋は [evaluation_guide.md](evaluation_guide.md) を参照する。

## 5. 予測、バックテスト、比較

- [../scripts/run_predict.py](../scripts/run_predict.py)
  - 指定日付に対する batch prediction の入口。
- [../scripts/run_backtest.py](../scripts/run_backtest.py)
  - prediction を使ったバックテストの入口。
- [../scripts/run_ab_compare.py](../scripts/run_ab_compare.py)
  - base/challenger の比較をまとめて実行する。
- [../scripts/run_dashboard.py](../scripts/run_dashboard.py)
  - 既存 report から dashboard 向け出力を作る。

これらの profile 対応 CLI では、`current_best_eval` や `current_recommended_serving` のような既存 family に `_2025_latest` を付けるだけで、`configs/data_2025_latest.yaml` を使う variant を選べる。

## 6. serving 検証

### 6.1 単発 smoke と比較

- [../scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
  - 1 候補を単日で smoke する。
- [../scripts/run_serving_smoke_compare.py](../scripts/run_serving_smoke_compare.py)
  - 単日比較を軽く回す。
- [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py)
  - 複数日で 2 候補を比較し、必要なら sweep / dashboard までつなぐ。

### 6.2 replay、集計、bankroll

- [../scripts/run_serving_replay_from_predictions.py](../scripts/run_serving_replay_from_predictions.py)
  - 既存 prediction から policy replay を再現する。
- [../scripts/run_serving_stateful_bankroll_sweep.py](../scripts/run_serving_stateful_bankroll_sweep.py)
  - bankroll 条件を変えながら stateful に sweep する。
- [../scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
  - compare 結果を dashboard 用に整形する。
- [../scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)
  - compare 結果を window 横断で集計する。

serving 系の判断範囲は [serving_validation_guide.md](serving_validation_guide.md) を参照する。

serving 系の重い script は、smoke 本体、replay、bankroll sweep、dashboard 生成のどこにいるかが分かるように進捗を出す。

## 7. walk-forward と policy 調整

### 7.1 feasibility / liquidity / threshold

- [../scripts/run_wf_feasibility_diag.py](../scripts/run_wf_feasibility_diag.py)
  - fold ごとの成立性を診断する。
- [../scripts/run_wf_liquidity_probe.py](../scripts/run_wf_liquidity_probe.py)
  - 日付窓や fold を絞って liquidity / policy 成立性を調べる。
- [../scripts/run_wf_threshold_sweep.py](../scripts/run_wf_threshold_sweep.py)
  - threshold を広く sweep する。
- [../scripts/run_wf_threshold_compare.py](../scripts/run_wf_threshold_compare.py)
  - threshold 候補同士を比較する。

### 7.2 mitigation / signature 分析

- [../scripts/run_wf_threshold_mitigation_focus.py](../scripts/run_wf_threshold_mitigation_focus.py)
  - mitigation 効き目の強い領域へ絞り込む。
- [../scripts/run_wf_threshold_mitigation_policy_probe.py](../scripts/run_wf_threshold_mitigation_policy_probe.py)
  - mitigation と policy 選択の関係を掘る。
- [../scripts/run_wf_threshold_mitigation_shortlist.py](../scripts/run_wf_threshold_mitigation_shortlist.py)
  - 有望候補を shortlist 化する。
- [../scripts/run_wf_threshold_signature_report.py](../scripts/run_wf_threshold_signature_report.py)
  - signature 単位で集計レポートを出す。
- [../scripts/run_wf_threshold_signature_family_compare.py](../scripts/run_wf_threshold_signature_family_compare.py)
  - signature family を比較する。
- [../scripts/run_wf_threshold_signature_drilldown.py](../scripts/run_wf_threshold_signature_drilldown.py)
  - 特定 signature の詳細を掘る。

### 7.3 value stack 調整

- [../scripts/run_tune_value_stack.py](../scripts/run_tune_value_stack.py)
  - value stack の blend 系パラメータを調整する。
- [../scripts/run_tune_top3.py](../scripts/run_tune_top3.py)
  - top-3 系の補助調整を行う。

## 8. serving 候補の生成と書き出し

- [../scripts/run_generate_serving_candidates_from_mitigation_probe.py](../scripts/run_generate_serving_candidates_from_mitigation_probe.py)
  - mitigation probe の集計から runtime 候補を組み立てる。
- [../scripts/run_generate_serving_config_variants_from_candidates.py](../scripts/run_generate_serving_config_variants_from_candidates.py)
  - 候補を serving config variant 群に展開する。
- [../scripts/run_export_serving_from_summary.py](../scripts/run_export_serving_from_summary.py)
  - summary / report から serving block を YAML として書き出す。

## 9. netkeiba と外部データ運用

- [../scripts/run_prepare_netkeiba_ids.py](../scripts/run_prepare_netkeiba_ids.py)
  - crawl 対象 ID を事前に準備する。
- [../scripts/run_collect_netkeiba.py](../scripts/run_collect_netkeiba.py)
  - 指定 target を収集する。
- [../scripts/run_backfill_netkeiba.py](../scripts/run_backfill_netkeiba.py)
  - 期間を切って backfill を進める。
- [../scripts/run_netkeiba_benchmark_gate.py](../scripts/run_netkeiba_benchmark_gate.py)
  - coverage と readiness を見て benchmark 再実行可否を判定する。
- [../scripts/run_netkeiba_wait_then_cycle.py](../scripts/run_netkeiba_wait_then_cycle.py)
  - 待機と再試行を含む連続運転に使う。

netkeiba 系は lock 待機、収集、backfill、gate 実行の各段で heartbeat または status を出す。

外部データの設計意図は [data_extension.md](data_extension.md) を参照する。

## 10. 長時間バッチと運転監視

- [../scripts/run_meeting_full_train_high_coverage.sh](../scripts/run_meeting_full_train_high_coverage.sh)
  - 高 coverage 条件の長時間学習・評価をまとめて回す meeting 用バッチ。
- [../scripts/check_meeting_run_status.sh](../scripts/check_meeting_run_status.sh)
  - 上記バッチの status file、log tail、稼働中 process を確認する。

補足:

- `check_meeting_run_status.sh` は長時間 worker ではなく短命な status 確認用なので、`ProgressBar` を持たないこと自体は異常とみなさない。
- progress 退行を軽く確認したいときは、[command_reference.md](command_reference.md) の「進捗表示の軽量 smoke」を使う。

## 11. 迷ったときの基準

- 正式な改善判断をしたいなら `run_revision_gate.py` を起点にする。
- 実運用寄りの候補比較をしたいなら serving 系へ進む。
- 外部データを触るなら、収集前後で data quality 系 script を通す。
- `wf_` 系 script は探索用であり、単体では昇格判断を確定しない。