# Scripts Guide

## 1. この文書の役割

この文書は、`scripts/` 配下の CLI と補助 shell を用途別に引けるようにした索引である。

日常運用の代表コマンドは [command_reference.md](command_reference.md) に寄せ、この文書では「どの場面でどの script を見に行くか」を整理する。

latest 2025 の actual-date compare を再開するときは、この索引から script を探し始めるのではなく、まず `serving_validation_guide.md` の quickstart と `command_reference.md` の latest compare 入口を使う。

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
  - recent-heavy split を試すときは `_2025_recent_2018` / `_2025_recent_2020` を使う。
  - `value_blend` profile は、同じ `artifact_suffix` で先に学習済みの component artifact があれば、それを優先して bundle する。
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

これらの profile 対応 CLI では、`current_best_eval` や `current_recommended_serving` のような既存 family に `_2025_latest`、`_2025_recent_2018`、`_2025_recent_2020` を付けるだけで、対応する data split variant を選べる。

ただし、`_2025_latest` が付く profile は自動派生で広く生成される一方、2026-03-27 時点で docs 上の stable family として明示的に参照するのは次の 4 本である。

| profile | 種別 | 現在の扱い | 主な使い道 |
| --- | --- | --- | --- |
| `current_best_eval_2025_latest` | evaluation mainline | latest holdout 上の評価基準線 | train / evaluate / revision gate の比較起点 |
| `current_recommended_serving_2025_latest` | operational baseline | 現在の既定運用 profile | predict / smoke / compare の baseline |
| `current_long_horizon_serving_2025_latest` | seasonal de-risk alias | September difficult regime 向けの実運用寄り候補 | baseline との replay/fresh compare |
| `current_tighter_policy_search_candidate_2025_latest` | analysis-first defensive candidate | formal support 改善を持つ比較候補 | threshold sweep と fresh compare |

一方、次の `_2025_latest` variant も profile 解決自体はできるが、現時点の docs では stable family として前面には出さない。

| profile | 位置づけ |
| --- | --- |
| `current_bankroll_candidate_2025_latest` | 旧 mitigation 系の generated variant。現行 latest 運用の主導線には置かない。 |
| `current_ev_candidate_2025_latest` | 旧 mitigation 系の generated variant。現行 latest 運用の主導線には置かない。 |
| `current_sep_guard_candidate_2025_latest` | `current_long_horizon_serving_2025_latest` と同系統だが、latest docs では long-horizon alias を正面名として使う。 |

したがって、`_2025_latest` suffix が付いているだけで current stable family とは見なさない。まず上の 4 本を基準に読み、追加 variant は必要が生じたときだけ config / artifact 単位で掘る。

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

latest 2025 の current reading は、September を `long_horizon -> tighter policy -> recent-2018` の順で dashboard summary から見て、必要なときだけこの節の script に降りる。

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
- [../scripts/run_local_coverage_snapshot.py](../scripts/run_local_coverage_snapshot.py)
  - local-only universe 用に snapshot 名、manifest 名、source path を切った wrapper 雛形。
- [../scripts/run_local_benchmark_gate.py](../scripts/run_local_benchmark_gate.py)
  - local-only universe 用に gate manifest、baseline reference、model / feature config を切った wrapper 雛形。
- [../scripts/run_local_data_source_validation.py](../scripts/run_local_data_source_validation.py)
  - local-only universe 用に data integrity report を別名出力する wrapper。
- [../scripts/run_local_feature_gap_report.py](../scripts/run_local_feature_gap_report.py)
  - local-only universe 用に feature gap summary / coverage CSV を別名出力する wrapper。
- [../scripts/run_local_evaluate.py](../scripts/run_local_evaluate.py)
  - local-only evaluation 実行後に、versioned output へのポインタ manifest を別名で残す wrapper。
- [../scripts/run_local_feasibility_manifest.py](../scripts/run_local_feasibility_manifest.py)
  - snapshot / validation / feature gap / evaluation pointer を 1 本の orchestration manifest に束ねる入口。
- [../scripts/run_local_revision_gate.py](../scripts/run_local_revision_gate.py)
  - benchmark gate、revision slug、evaluation pointer、promotion / revision artifact を 1 lineage manifest に束ねる入口。
- [../scripts/run_local_public_snapshot.py](../scripts/run_local_public_snapshot.py)
  - local revision lineage を public 向けの要約 manifest に潰し、local-only の読み順を固定する入口。
- [../scripts/run_mixed_universe_compare.py](../scripts/run_mixed_universe_compare.py)
  - local public snapshot または lineage と JRA public reference を束ね、mixed-universe compare の pointer manifest を作る入口。
- [../scripts/run_mixed_universe_readiness.py](../scripts/run_mixed_universe_readiness.py)
  - mixed compare の前提条件を点検し、left readiness / representative evaluation / right public reference の揃い具合を manifest 化する入口。
- [../scripts/run_mixed_universe_schema.py](../scripts/run_mixed_universe_schema.py)
  - mixed compare で左右に並べる指標名と source artifact を schema manifest に落とす入口。
- [../scripts/run_public_benchmark_reference.py](../scripts/run_public_benchmark_reference.py)
  - JRA latest baseline の public benchmark を machine-readable な reference manifest に切り出す入口。
- [../scripts/run_mixed_universe_numeric_compare.py](../scripts/run_mixed_universe_numeric_compare.py)
  - mixed readiness / compare / schema と right-side reference manifest を受け、row 単位の left/right 値と delta を numeric compare manifest / CSV に落とす入口。
- [../scripts/run_mixed_universe_numeric_summary.py](../scripts/run_mixed_universe_numeric_summary.py)
  - numeric compare manifest から promote-safe な要点だけを summary manifest に落とす入口。
- [../scripts/run_netkeiba_wait_then_cycle.py](../scripts/run_netkeiba_wait_then_cycle.py)
  - 待機と再試行を含む連続運転に使う。

netkeiba 系は lock 待機、収集、backfill、gate 実行の各段で heartbeat または status を出す。

`run_netkeiba_coverage_snapshot.py` と `run_netkeiba_benchmark_gate.py` は、すでに `universe`, `source-scope`, `baseline-reference`, `schema-version` を受け取れる。既定値のままなら従来どおり JRA / netkeiba 向けとして動き、payload 側には universe-aware な metadata と `completed_step`, `error_code`, `recommended_action` が残る。

地方 universe を将来追加する場合も、CLI 契約はこの系統に寄せる。つまり `data-config`, `tail-rows`, `snapshot-output`, `manifest-output`, `skip-train`, `skip-evaluate` を基底にし、追加は同じ 4 引数だけに留める。

さらに local-only の最小雛形として、`configs/data_local_nankan.yaml`、`configs/model_local_baseline.yaml`、`configs/features_local_baseline.yaml` と、`run_local_coverage_snapshot.py` / `run_local_benchmark_gate.py` を置いた。ここでは real local data の存在を前提にせず、artifact 名、source path、baseline reference を JRA 系から分離した状態で smoke できるところまでを入口にしている。

その次段として、`run_local_data_source_validation.py`、`run_local_feature_gap_report.py`、`run_local_evaluate.py` も追加した。validation / feature gap は local-only artifact 名へ直接出し、evaluation は既存 `run_evaluate.py` の versioned output を再利用しつつ `evaluation_local_nankan_pointer.json` で local-only 側の入口を固定する。

さらに `run_local_feasibility_manifest.py` を追加し、readiness snapshot、data integrity、feature gap、evaluation pointer を fail-fast で直列実行しつつ、`local_feasibility_manifest_local_nankan.json` から停止点と artifact lineage をまとめて読めるようにした。

その次段として `run_local_revision_gate.py` も追加し、local benchmark gate の readiness 確認、`run_revision_gate.py` による suffixed train / evaluate / promotion、evaluation pointer の書き出しを、`local_revision_gate_<revision>.json` で 1 lineage として追えるようにした。

さらに `run_local_public_snapshot.py` を追加し、`local_revision_gate_<revision>.json` を起点に `local_public_snapshot_<revision>.json` を出せるようにした。public 向けに local-only を読むときは、この snapshot を先頭にして、必要な場合だけ lineage / promotion / revision / evaluation pointer の順へ降りる。

mixed-universe 比較も `run_mixed_universe_compare.py` を追加して、`mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` を最小 pointer manifest として出せるようにした。ここでは数値比較そのものではなく、left 側の local public snapshot または lineage と、right 側の JRA public reference を 1 本へ束ねる。

さらに `run_mixed_universe_readiness.py` を追加し、mixed compare の前に `mixed_universe_readiness_<left_universe>_vs_<right_universe>_<revision>.json` で前提条件を確認できるようにした。ここでは left 側の `benchmark_rerun_ready`、evaluation pointer の有無、`stability_assessment=representative`、right 側の public reference を check として並べる。

その次段として `run_mixed_universe_schema.py` も追加し、readiness manifest と pointer-only compare manifest を受けて、`mixed_universe_schema_<left_universe>_vs_<right_universe>_<revision>.json` へ comparison axes と metric rows を固定できるようにした。現段階では numeric compare ではなく、`decision`, `stability_assessment`, `auc`, `top1_roi`, `ev_top1_roi`, `nested_wf_weighted_test_roi`, `formal_benchmark_weighted_roi`, `formal_benchmark_feasible_folds` を左右でどこから読むかを揃える役割に留める。

right 側の JRA baseline については `run_public_benchmark_reference.py` も追加し、promotion / revision / evaluation artifact から `public_benchmark_reference_<reference>.json` を生成できるようにした。mixed compare/readiness/schema はこの JSON を right-side の machine-readable reference として参照してよい。

その次段として `run_mixed_universe_numeric_compare.py` を追加し、schema manifest の row 名を基準に left/right の値を解決して `mixed_universe_numeric_compare_<left_universe>_vs_<right_universe>_<revision>.json` と CSV を出せるようにした。left 値が未整備の行は `missing_left_value` として残し、取得できた numeric 行だけ `delta_left_minus_right` と `delta_direction` を計算する。

さらに `run_mixed_universe_numeric_summary.py` を追加し、numeric compare から `verdict`, `missing_left_rows`, `missing_right_rows`, `positive_rows`, `negative_rows` を promote-safe にまとめた `mixed_universe_numeric_summary_<left_universe>_vs_<right_universe>_<revision>.json` を出せるようにした。

step 名も既存 gate の読み方に寄せてあり、snapshot 側は `load_config -> load_source_tables -> compute_alignment -> compute_coverage -> write_snapshot`、gate 側は `init_manifest -> run_snapshot -> validate_readiness -> run_train -> run_evaluate -> write_manifest` を基本系列として読める。

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