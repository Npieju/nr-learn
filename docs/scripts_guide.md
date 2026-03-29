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

`run_promotion_gate.py` の report も `read_order`, `current_phase`, `recommended_action`, `highlights` を返す。したがって formal decision 入口でも、operator は WF support 不足による block なのか、matching WF summary 不足なのか、入力不備の hard failure なのかを top-level だけで追える。

`run_revision_gate.py` は `--dry-run` も持ち、重い train / evaluate を回さずに planned command と revision manifest だけ確認できる。planned manifest には `read_order`, `current_phase`, `recommended_action`, `highlights` と step ごとの output preview も入るので、wrapper なしでも operator が次の artifact を追いやすい。

`run_netkeiba_latest_revision_gate.py --dry-run` も同様に planned wrapper manifest を返し、`read_order`, `current_phase`, `recommended_action`, `highlights`, readiness preview, revision gate preview を残す。したがって latest wrapper の入口でも、実 readiness snapshot を回す前に停止点と downstream artifact path を同じ読み口で確認できる。

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

`run_serving_profile_compare.py` の compare manifest も `read_order`, `current_phase`, `recommended_action`, `highlights` を返す。したがって left/right smoke の途中失敗や compare 完了後の dashboard 待ちでも、operator は compare manifest 単体で停止点と次アクションを追える。

`run_serving_compare_dashboard.py` の summary JSON も `status`, `read_order`, `current_phase`, `recommended_action`, `highlights` を返す。したがって serving validation の最初の読む入口でも、operator は completed と hard failure のどちらでも summary 単体で現在地を追える。

`run_serving_compare_aggregate.py` の aggregate summary も `status`, `read_order`, `current_phase`, `recommended_action`, `highlights` を返す。したがって window 横断の入口でも、operator は completed と hard failure のどちらでも aggregate summary 単体で現在地を追える。

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
- [../scripts/run_archive_local_nankan_window.py](../scripts/run_archive_local_nankan_window.py)
  - local_nankan の 6 か月 window backfill を実行し、その window の CSV slice と manifest を GitHub 向け archive/tarball/index に固める入口。
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
- [../scripts/run_mixed_universe_left_gap_audit.py](../scripts/run_mixed_universe_left_gap_audit.py)
  - missing_left_rows について、必要 source artifact と local revision lineage に残っている command preview を gap audit manifest に落とす入口。
- [../scripts/run_mixed_universe_left_recovery_plan.py](../scripts/run_mixed_universe_left_recovery_plan.py)
  - left gap audit を受け、欠損解消に必要な command preview を重複除去した recovery plan manifest に落とす入口。
- [../scripts/run_mixed_universe_recovery.py](../scripts/run_mixed_universe_recovery.py)
  - status board を起点に local revision lineage の再実行と mixed-universe manifest 群の再生成をまとめて回す入口。
- [../scripts/run_mixed_universe_status_board.py](../scripts/run_mixed_universe_status_board.py)
  - public snapshot から recovery plan までの mixed-universe manifest 群を 1 本の status board manifest に束ねる入口。
- [../scripts/run_netkeiba_wait_then_cycle.py](../scripts/run_netkeiba_wait_then_cycle.py)
  - 待機と再試行を含む連続運転に使う。

netkeiba 系は lock 待機、収集、backfill、gate 実行の各段で heartbeat または status を出す。

`run_netkeiba_coverage_snapshot.py` と `run_netkeiba_benchmark_gate.py` は、すでに `universe`, `source-scope`, `baseline-reference`, `schema-version` を受け取れる。既定値のままなら従来どおり JRA / netkeiba 向けとして動き、payload 側には universe-aware な metadata と `completed_step`, `error_code`, `recommended_action` が残る。

`run_netkeiba_benchmark_gate.py` はさらに `read_order`, `current_phase`, `highlights` も返すようにした。したがって gate 入口でも、operator は preflight / snapshot / readiness / train / evaluate のどこで止まったかを top-level だけで追える。

`run_netkeiba_wait_then_cycle.py` も `read_order`, `current_phase`, `recommended_action`, `highlights` を返す。したがって wait/handoff 入口でも、operator は lock 待機中なのか、handoff backfill で失敗したのか、post-cycle gate が not_ready/failed だったのかを top-level だけで追える。

地方 universe を将来追加する場合も、CLI 契約はこの系統に寄せる。つまり `data-config`, `tail-rows`, `snapshot-output`, `manifest-output`, `skip-train`, `skip-evaluate` を基底にし、追加は同じ 4 引数だけに留める。

さらに local-only の最小雛形として、`configs/data_local_nankan.yaml`、`configs/model_local_baseline.yaml`、`configs/features_local_baseline.yaml` と、`run_local_coverage_snapshot.py` / `run_local_benchmark_gate.py` を置いた。ここでは real local data の存在を前提にせず、artifact 名、source path、baseline reference を JRA 系から分離した状態で smoke できるところまでを入口にしている。

`run_local_benchmark_gate.py` は現在、source preflight も先に実行する。ここで `data_preflight_<revision>.json` または `data_preflight_local_nankan.json` を通して raw dir / primary CSV / required source table の不足を早期に `not_ready` として返し、snapshot 深部の `No CSV files found ...` まで潜らなくても停止理由を読める。

さらに `--materialize-primary-before-gate` を付けると、wrapper は preflight の前に `run_materialize_local_nankan_primary.py` を呼び、external の `race_result/racecard/pedigree` から `data/local_nankan/raw` 相当の primary CSV を組み立てようとする。materialize が `status=not_ready` でも wrapper 自体は benchmark gate を続行し、正式な blocker は従来どおり preflight / benchmark manifest 側へ残す。

`run_local_backfill_then_benchmark.py` を使うと、backfill は常に `--materialize-after-collect` 付きで回り、`current_phase=materialized_primary_raw` に達した場合のみ local benchmark gate へ handoff する。したがって Phase 0 の入口でも、operator は backfill blocker と benchmark blocker を wrapper manifest 1 本で追える。

`run_local_revision_gate.py --backfill-before-benchmark` を使うと、この handoff wrapper が revision lineage の benchmark 前段へ入る。したがって local revision lineage でも、backfill 側で止まったのか benchmark 側で止まったのかを lineage manifest 1 本で追えるようになる。

`configs/data_local_nankan.yaml` の `local_nankan_race_result_keys` は、まず results を見て schema が足りなければ racecard を fallback として見る。これにより `horse_key` が result CSV 側にまだ無い段階でも、racecard に join key が揃っていれば preflight の supplemental blocker を 1 段先へ進められる。

その次段として、`run_local_data_source_validation.py`、`run_local_feature_gap_report.py`、`run_local_evaluate.py` も追加した。validation / feature gap は local-only artifact 名へ直接出し、evaluation は既存 `run_evaluate.py` の versioned output を再利用しつつ `evaluation_local_nankan_pointer.json` で local-only 側の入口を固定する。

さらに `run_local_feasibility_manifest.py` を追加し、readiness snapshot、data integrity、feature gap、evaluation pointer を fail-fast で直列実行しつつ、`local_feasibility_manifest_local_nankan.json` から停止点と artifact lineage をまとめて読めるようにした。`--dry-run` でも top-level を `status=planned` で閉じつつ、`read_order`, `current_phase`, `highlights`, `snapshot_payload`, `validation_payload`, `feature_gap_payload`, `evaluation_payload` の planned preview を返すので、local feasibility 入口も completed payload と同じ読み口で確認できる。real run の completed / failed manifest でも `current_phase` と `highlights` を保つので、snapshot / validation / feature gap / evaluation のどこで止まったかを top-level だけで追える。

その次段として `run_local_revision_gate.py` も追加し、local benchmark gate の readiness 確認、`run_revision_gate.py` による suffixed train / evaluate / promotion、evaluation pointer の書き出しを、`local_revision_gate_<revision>.json` で 1 lineage として追えるようにした。さらに `--dry-run` でも top-level を `status=planned` で閉じつつ、`read_order`, `current_phase`, `recommended_action`, `highlights` と `benchmark_gate_payload` / `data_preflight_payload` / `revision_manifest_payload` / `evaluation_pointer_payload` の planned preview を返す。real run の benchmark gate blocked / revision gate failed / interrupted でも `current_phase` と `highlights` を保つので、local revision lineage の入口も completed payload と同じ読み口で確認できる。

さらに `run_local_public_snapshot.py` を追加し、`local_revision_gate_<revision>.json` を起点に `local_public_snapshot_<revision>.json` を出せるようにした。public 向けに local-only を読むときは、この snapshot を先頭にして、必要な場合だけ lineage / promotion / revision / evaluation pointer の順へ降りる。mixed revision 名から alias で辿って生成した場合でも、snapshot payload 内の `compare_contract.local_only_public_snapshot` と `mixed_compare_manifest` は入力 alias ではなく resolved lineage revision と実 output path を基準に書く。`--dry-run` でも `read_order`, `current_phase`, `recommended_action`, `readiness`, `promotion_summary`, `benchmark_gate_summary`, `evaluation_summary`, `highlights` を返し、lineage 不足などの hard failure でも summary artifact を残すので、public bridge 入口も completed payload と同じ読み口で確認できる。

この lineage が `--backfill-before-benchmark` を使っている場合、`run_local_public_snapshot.py` は `backfill_handoff_summary` も併記し、`current_phase` と `recommended_action` も upstream lineage の planned / blocked / failed に追従する。したがって public bridge 入口でも、backfill/materialize handoff で止まったのか、その後段で止まったのかを 1 本で読める。

mixed-universe 比較も `run_mixed_universe_compare.py` を追加して、`mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` を最小 pointer manifest として出せるようにした。ここでは数値比較そのものではなく、left 側の local public snapshot または lineage と、right 側の JRA public reference を 1 本へ束ねる。planned / dry-run でも `left_summary`, `right_summary`, `comparison_contract` に加えて `current_phase`, `highlights` を返すので、pointer bridge 入口も completed payload と同じ shape で読める。

`run_mixed_universe_compare.py` は、mixed revision 名と local public snapshot revision 名がズレる場合でも、`local_public_snapshot_*` / `local_revision_gate_*` と snapshot 内の `lineage_manifest` を辿って left input を自動解決する。

さらに `run_mixed_universe_readiness.py` を追加し、mixed compare の前に `mixed_universe_readiness_<left_universe>_vs_<right_universe>_<revision>.json` で前提条件を確認できるようにした。ここでは left 側の `benchmark_rerun_ready`、evaluation pointer の有無、`stability_assessment=representative`、right 側の public reference を check として並べる。planned / dry-run でも `checks`, `left_summary`, `compare_command_preview` に加えて `current_phase`, `highlights` も返すので、left input 未生成や left readiness blocker の段階から不足条件と次に回す compare CLI を確認できる。

`run_mixed_universe_readiness.py` も同様に left-side artifact を自動探索するので、`reference_bridge` 系の revision でも `--left-public-snapshot` や `--left-lineage-manifest` を省略したまま再生成できる。

left 側 public snapshot に `backfill_handoff_summary` がある場合、`run_mixed_universe_readiness.py` はそれも `left_summary` と highlights に写す。したがって `left_readiness_blocked` のときも、単に readiness 不足と出すだけでなく upstream handoff の `status`, `current_phase`, `recommended_action` をその場で確認できる。

その次段の `run_mixed_universe_left_gap_audit.py` も、local revision lineage に `backfill_handoff_payload` が残っている場合は benchmark/preflight より先にその blocker を採用する。したがって `missing_left_rows` を見ている段階でも、真の停止点が local handoff なら `current_phase=local_backfill_then_benchmark`, `recommended_action=run_local_backfill_then_benchmark` を top-level から読める。

mixed-universe 系 manifest の `revision` は引き続き要求した mixed revision alias を表す。一方で alias 解決後に実際に参照した local revision は `resolved_left_revision` として readiness / compare / status board / recovery に併記し、入口側の readiness / compare では `resolved_left_source_kind` と `resolved_left_artifact` も合わせて出して、operator が left-side の実体と参照元 path をすぐ追えるようにしている。

この bridge 情報は downstream の schema / numeric compare / numeric summary / left gap audit / left recovery plan にも引き継ぐ。さらに status board の `phase_summaries` でも `public_snapshot` 行を含めて同じ値を補完し、planned / dry-run でも同じキー集合を出すので、mixed alias を保ったまま生成を続けても、どの段で見ても `requested_revision`, `resolved_left_revision`, `resolved_left_source_kind`, `resolved_left_artifact` を読める。

local public snapshot / local revision lineage の fallback は revision prefix だけでなく `left_universe` も条件に入れて解決する。したがって未知 universe の dry-run や別 local universe の将来追加時にも、既存 `local_nankan` artifact を誤って拾わない。

その次段として `run_mixed_universe_schema.py` も追加し、readiness manifest と pointer-only compare manifest を受けて、`mixed_universe_schema_<left_universe>_vs_<right_universe>_<revision>.json` へ comparison axes と metric rows を固定できるようにした。現段階では numeric compare ではなく、`decision`, `stability_assessment`, `auc`, `top1_roi`, `ev_top1_roi`, `nested_wf_weighted_test_roi`, `formal_benchmark_weighted_roi`, `formal_benchmark_feasible_folds` を左右でどこから読むかを揃える役割に留める。planned / dry-run でも `read_order`, `comparison_axes`, `metric_rows`, `blocking_context` に加えて `current_phase`, `highlights` を返すので、schema 入口も completed payload と同じ shape で読める。

right 側の JRA baseline については `run_public_benchmark_reference.py` も追加し、promotion / revision / evaluation artifact から `public_benchmark_reference_<reference>.json` を生成できるようにした。manifest には `read_order`, `current_phase`, `recommended_action`, `blocker_summary`, `highlights` も入り、mixed compare/readiness/schema はこの JSON を right-side の machine-readable reference として参照しつつ、operator も right-side baseline の readiness を top-level で読める。

その次段として `run_mixed_universe_numeric_compare.py` を追加し、schema manifest の row 名を基準に left/right の値を解決して `mixed_universe_numeric_compare_<left_universe>_vs_<right_universe>_<revision>.json` と CSV を出せるようにした。left 値が未整備の行は `missing_left_value` として残し、取得できた numeric 行だけ `delta_left_minus_right` と `delta_direction` を計算する。planned / dry-run でも `read_order`, `blocking_context`, `summary`, `current_phase`, `highlights`, `row_results=[]` を返すので、numeric compare 入口も completed payload と同じ shape で読める。

`run_mixed_universe_schema.py` と `run_mixed_universe_numeric_compare.py` は、exact revision 名の upstream manifest が無いときでも、同じ universe の既存 mixed manifest を fallback で拾う。

さらに `run_mixed_universe_numeric_summary.py` を追加し、numeric compare から `verdict`, `severity`, `missing_left_rows`, `missing_right_rows`, `positive_rows`, `negative_rows`, `notes` を promote-safe にまとめた `mixed_universe_numeric_summary_<left_universe>_vs_<right_universe>_<revision>.json` を出せるようにした。planned / dry-run でも `read_order`, `current_phase`, `highlights`, `promote_safe_summary` を返すので、numeric compare 未生成でも summary 入口を completed payload と同じ shape で読める。

その次段として `run_mixed_universe_left_gap_audit.py` を追加し、`missing_left_rows` ごとにどの left-side artifact が必要か、現時点で存在するか、local revision lineage にどの command preview が残っているかを `mixed_universe_left_gap_audit_<left_universe>_vs_<right_universe>_<revision>.json` で追えるようにした。local benchmark が preflight で止まった場合は、`populate_primary_raw_dir` のような upstream blocker も gap audit に残す。`--left-lineage-manifest` を省略した場合でも、numeric compare から readiness/public snapshot を辿って local lineage を自動解決する。planned / dry-run でも `read_order`, `summary`, `lineage_blocker`, `current_phase`, `highlights`, `gap_rows=[]` を返すので、gap audit 入口も completed payload と同じ shape で読める。

さらに `run_mixed_universe_left_recovery_plan.py` を追加し、gap audit に残った command preview を重複除去して `mixed_universe_left_recovery_plan_<left_universe>_vs_<right_universe>_<revision>.json` へ並べられるようにした。command が無い場合でも upstream blocker の `recommended_action` を手動 step として残すので、plan が空のまま終わりにくい。local handoff blocker がある場合は、その manual step を downstream `run_revision_gate` preview より前へ並べ、first step 自体も `run_local_backfill_then_benchmark` を返す。planned / dry-run でも `read_order`, `summary`, `current_phase`, `highlights`, `plan_steps=[]` を返すので、recovery plan 入口も completed payload と同じ shape で読める。
planned / dry-run でも `read_order`, `summary`, `plan_steps=[]` を返すので、gap audit 未生成の段階でも completed payload と同じ shape で recovery plan の入口を読める。

その次段として `run_mixed_universe_recovery.py` も追加し、既存の `mixed_universe_status_board_<left_universe>_vs_<right_universe>_<revision>.json` を起点に、local revision lineage の再実行、public snapshot の更新、readiness / compare / schema / numeric compare / summary / gap audit / recovery plan / status board の再生成までを直列に回せるようにした。実データ未配置の段階でも、dry-run で planned steps を確認し、source board 由来の `recommended_action` / `current_phase` / `highlights` を写すだけでなく、recovery 自身の `read_order`, `status_board_preview`, `highlights` も返したうえで、実行時は partial を許容しつつ最新の停止点まで一気に更新できる。source board と recovery plan の両方が local handoff blocker を指している場合は、recovery wrapper 自身の first step も `local_backfill_then_benchmark` に切り替わる。

この alias 解決ロジックは script ごとの重複実装ではなく共通 helper に寄せてあり、`run_mixed_universe_recovery.py` と `run_mixed_universe_status_board.py` も readiness / compare / schema / numeric compare / gap audit と同じ規則で `reference_bridge` 系の local artifact を拾う。

さらに `run_mixed_universe_status_board.py` を追加し、public snapshot / readiness / compare / schema / numeric compare / numeric summary / left gap audit / left recovery plan を 1 本の `mixed_universe_status_board_<left_universe>_vs_<right_universe>_<revision>.json` に束ねられるようにした。ここでは `current_phase`, `next_action_source`, `recommended_action`, `phase_summaries` を先頭にして全体の現在地を読む。phase_summaries 自体にも各 manifest の `current_phase`, `error_code`, `highlights` を写すので、board を開くだけで各段の blocker 要約まで追える。

`run_mixed_universe_status_board.py` は explicit path 指定も受けられるので、tmp smoke や recovery 途中の artifact を読むときは `--public-snapshot` や `--readiness-manifest` を渡して入力を固定してよい。public snapshot が `status=completed` でも `current_phase=lineage_planned` のような upstream stop を持つ場合、board はそこを incomplete と見なして `status=partial`, `current_phase=public_snapshot` を返す。

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