# nr-learn

JRA競馬予想の**学習用**プロジェクトです。  
データソースは KaggleHub の `takamotoki/jra-horse-racing-dataset` を主表にしつつ、後から外部CSVを追加マージできる構成です。

## 目的
- 競馬予想を「勘」ではなく、再現可能なMLパイプラインとして学ぶ
- 時系列リークを避けた検証方法を身につける
- ベースラインから改善を積み上げる

## アーキテクチャ概要
- **Data Layer**: 生データ取得・正規化・特徴量生成
- **Feature Selection Layer**: `explicit` / `all_safe` による共通特徴量解決とカテゴリ列判定
- **Model Layer**: LightGBM比較系 + CatBoost本命系で classification / ranking / Top3 / ROI / alpha を学習
- **Scoring Layer**: モデル出力と Top3 正規化を共通 API 化
- **Policy Layer**: ROI評価、market signal、gating、Kelly / portfolio を統一
- **Walk-Forward Layer**: calibration / blend / nested WF 最適化を統一
- **Artifact Layer**: train manifest と stack bundle manifest を管理
- **Serving Layer**: 次レース予測バッチ出力（CSV/PNG）

詳細は [docs/README.md](docs/README.md) を参照。

## ディレクトリ
```text
nr-learn/
├── configs/
│   ├── data.yaml
│   ├── features.yaml
│   └── model.yaml
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
├── docs/
│   ├── README.md
│   ├── project_overview.md
│   ├── system_architecture.md
│   ├── benchmarks.md
│   └── data_extension.md
├── notebooks/
├── src/
│   └── racing_ml/
│       ├── common/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       ├── serving/
│       └── pipeline/
└── scripts/
    ├── run_bundle_models.py
    ├── run_train.py
    ├── run_backtest.py
    └── run_predict.py
```

## まずの進め方（推奨）
1. `data/raw/` にデータを配置（または取得スクリプトで同期）
2. 必要に応じて `data/external/...` に外部CSVを配置し、`configs/data.yaml` の `append_tables` / `supplemental_tables` で取り込む
3. 外部収集を自前で行う場合は `scripts/run_collect_netkeiba.py` で `race_result` / `race_card` / `pedigree` を canonical CSV 化する
4. CatBoost分類モデルで広めの安全特徴量を使って `win` を予測
5. 時系列スライスで評価し、AUCだけでなくROIも確認
6. Top3 / ROI / alpha / ranking に横展開
7. LightGBM baseline とA/B比較して改善幅を固定化

## 実行手順（MVP）
1. データ取得
    - `python scripts/run_ingest.py --config configs/data.yaml`
    - Kaggle認証が未設定 / データ取得失敗時は、学習確認用の `data/raw/sample_races.csv` を自動生成
    - `configs/data.yaml` の `dataset.external_raw_dirs` に定義した外部 raw dir も同時に作成されます
    - 外部サイト由来CSVを後から足す場合は、`append_tables` で行追加、`supplemental_tables` で列追加を定義します
    - stable な競走馬 ID は `horse_id` ではなく `horse_key` として別保持します。feature builder は `horse_key` がある行で常にこれを履歴 key に使い、不足分だけ `horse_name` / `horse_id` へ fallback します。netkeiba crawler はまず `race_result` または `race_card` で `horse_key` を補完し、その後 `pedigree` を `horse_key` で結合する前提です
    - `race_card` は shutuba page から `horse_key`, 枠番, 馬番, 性齢, 斤量, 騎手, 調教師などの pre-race 列を収集します。owner / breeder は現レイアウトでは直接出ないため、`pedigree` 側で補います
    - 取り込み前の整合確認: `python scripts/run_validate_data_sources.py --config configs/data.yaml`
    - 外部 pre-race 情報の欠損確認: `python scripts/run_feature_gap_report.py --config configs/data.yaml --model-config configs/model_catboost_fundamental_enriched.yaml --feature-config configs/features_catboost_fundamental_enriched.yaml`
    - netkeiba の tail coverage / race 整合確認: `python scripts/run_netkeiba_coverage_snapshot.py --config configs/data.yaml --tail-rows 5000`
    - snapshot JSON には race_result / race_card / pedigree の manifest 状態と `readiness.benchmark_rerun_ready` も出るので、mid-cycle の一時的不整合と再学習可能状態を分けて判断できます。pid が死んで lock も消えた古い manifest は `status=stale` として検出されます
    - `configs/data.yaml` の netkeiba table 群は default では optional 扱いです。CSV がまだ無い段階でも validation は `optional_missing` 扱いで通り、収集が始まると自動で merge 対象に入ります
    - crawler 用 ID を自動生成する場合:
        - `python scripts/run_prepare_netkeiba_ids.py --data-config configs/data.yaml --crawl-config configs/crawl_netkeiba_template.yaml --target race_result --start-date 2020-01-01`
        - `python scripts/run_prepare_netkeiba_ids.py --data-config configs/data.yaml --crawl-config configs/crawl_netkeiba_template.yaml --target pedigree`
        - benchmark を先に押し上げたいときは `--date-order desc` で最新日付を優先します
        - base dataset より新しい年の race ID を取りたいときは mobile race list を source にします。例: `python scripts/run_prepare_netkeiba_ids.py --data-config configs/data.yaml --crawl-config configs/crawl_netkeiba_template.yaml --target race_result --race-id-source race_list --start-date 2024-01-01 --end-date 2024-12-31 --date-order desc`
        - 既存出力に含まれる ID を再投入したいときだけ `--include-completed` を付けます
    - 年単位の backfill は `python scripts/run_backfill_netkeiba.py --data-config configs/data.yaml --crawl-config configs/crawl_netkeiba_template.yaml --start-date 2020-01-01 --end-date 2021-12-31 --date-order desc --race-batch-size 100 --pedigree-batch-size 500` のように回します
    - base dataset 以後の年を掘るときは `--race-id-source race_list` を付けると、training table ではなく netkeiba mobile race list から race IDs を起こしてそのまま backfill できます
    - backfill の cycle 要約は `artifacts/reports/netkeiba_backfill_manifest.json` に出ます
        - cycle 完了直後の安定スナップショットで benchmark を回したいときは `--post-cycle-command` に gate script を渡せます。例: `python scripts/run_backfill_netkeiba.py --data-config configs/data.yaml --crawl-config configs/crawl_netkeiba_template.yaml --start-date 2021-01-01 --end-date 2021-07-31 --date-order desc --race-batch-size 100 --pedigree-batch-size 500 --max-cycles 1 --post-cycle-command "/workspaces/nr-learn/.venv/bin/python scripts/run_netkeiba_benchmark_gate.py --data-config configs/data.yaml --model-config configs/model_catboost_fundamental_enriched.yaml --feature-config configs/features_catboost_fundamental_enriched.yaml --max-rows 200000 --wf-mode off"`
        - `scripts/run_netkeiba_benchmark_gate.py` は snapshot を 1 回更新し、`benchmark_rerun_ready=true` のときだけ train/evaluate を実行します。manifest は `artifacts/reports/netkeiba_benchmark_gate_manifest.json` に出ます
        - すでに old-style backfill が lock を握っている最中に follow-up の 1 cycle を予約したいときは `python scripts/run_netkeiba_wait_then_cycle.py --data-config configs/data.yaml --crawl-config configs/crawl_netkeiba_template.yaml --model-config configs/model_catboost_fundamental_enriched.yaml --feature-config configs/features_catboost_fundamental_enriched.yaml --start-date 2021-01-01 --end-date 2021-07-31 --date-order desc --race-batch-size 100 --pedigree-batch-size 500 --max-rows 200000 --wf-mode off` を使うと、lock 解放待ちのあとに `--max-cycles 1` backfill と gate を自動でつなげられます。待機 manifest は `artifacts/reports/netkeiba_wait_then_cycle_manifest.json` に出ます
    - `artifacts/reports/netkeiba_crawl_manifest.json.lock` に lock を置くので、netkeiba の collect/backfill は同時起動しないでください。同時実行はエラーで弾かれます
    - 長い batch の途中経過は `artifacts/reports/netkeiba_crawl_manifest_<target>.json` の `status=running` と `processed_ids` で追えます。プロセスが落ちて lock も消えた場合は snapshot 側で `status=stale` として拾います
    - `race_card` はレイアウト差で同一馬が二重に出るケースがあるため、crawler 側で `race_id + horse_id` 重複を除外します
    - netkeiba crawler の初期実行例:
        - `python scripts/run_collect_netkeiba.py --config configs/crawl_netkeiba_template.yaml --target race_result --limit 50`
        - `python scripts/run_collect_netkeiba.py --config configs/crawl_netkeiba_template.yaml --target race_card --limit 50`
        - `python scripts/run_collect_netkeiba.py --config configs/crawl_netkeiba_template.yaml --target pedigree --limit 50`
        - `race_result` / `race_card` は `data/external/netkeiba/ids/race_ids.csv`、`pedigree` は `data/external/netkeiba/ids/horse_keys.csv` を読む想定です
        - crawler の output CSV は batch ごとに累積更新され、重複キーは最新 batch 側で上書きされます
        - 推奨フローは `run_prepare_netkeiba_ids.py --target race_result` → `run_collect_netkeiba.py --target race_result` or `race_card` → `run_prepare_netkeiba_ids.py --target pedigree` → `run_collect_netkeiba.py --target pedigree` です
2. 学習
    - stable alias で主力評価系を呼ぶときは `python scripts/run_train.py --profile current_best_eval`、簡易運用候補を呼ぶときは `python scripts/run_train.py --profile current_recommended_serving` を使えます
    - alias の説明と config 対応を見たいときは `python scripts/run_train.py --list-profiles` を使えます。同じ `--list-profiles` は `run_predict.py` / `run_backtest.py` / `run_evaluate.py` でも使えます
    - validation や probe で canonical artifact を上書きしたくないときは `--artifact-suffix train_probe` のように suffix を付けて、model/report/manifest を別名で出せます
    - probe を軽く回したいときは `--max-train-rows 50000 --max-valid-rows 10000` のように学習/検証行数を CLI から一時的に上書きできます
    - `value_blend` 系 config を `run_train.py` で呼んだ場合は、component artifact を読み込んで stack bundle を再構築します。base component の再学習までは行いません
    - `python scripts/run_train.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （推奨 / CatBoost win）`python scripts/run_train.py --config configs/model_catboost.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （benchmark 用 / CatBoost fundamental win）`python scripts/run_train.py --config configs/model_catboost_fundamental.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental.yaml`
    - （benchmark 拡張 / pace-corner enriched）`python scripts/run_train.py --config configs/model_catboost_fundamental_enriched.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental_enriched.yaml`
    - （benchmark 診断 / no-lineage）`python scripts/run_train.py --config configs/model_catboost_fundamental_enriched_no_lineage.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental_enriched_no_lineage.yaml`
    - enriched benchmark の train / evaluate artifact には `feature_coverage` が入り、missing force-include 特徴が自動記録されます
    - （推奨 / CatBoost Top3）`python scripts/run_train.py --config configs/model_catboost_top3.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （推奨 / CatBoost Ranker）`python scripts/run_train.py --config configs/model_catboost_ranker.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （推奨 / CatBoost ROI）`python scripts/run_train.py --config configs/model_catboost_roi.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （推奨 / CatBoost alpha）`python scripts/run_train.py --config configs/model_catboost_alpha.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （長期ROI向け stack build）`python scripts/run_build_value_stack.py --config configs/model_catboost_value_stack.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （GPU / win component）`python scripts/run_train.py --config configs/model_catboost_gpu.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （GPU / time-deviation component）`python scripts/run_train.py --config configs/model_catboost_time_deviation_gpu.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （GPU / ROI mainline stack build）`python scripts/run_build_value_stack.py --config configs/model_catboost_value_stack_time_gpu.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （Ranker）`python scripts/run_train.py --config configs/model_ranker.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （Top3確率）`python scripts/run_train.py --config configs/model_top3.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （ROI回帰）`python scripts/run_train.py --config configs/model_roi.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （市場乖離/Layer2）`python scripts/run_train.py --config configs/model_alpha.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
3. 生成物確認
    - モデル: `artifacts/models/baseline_model.joblib`
    - レポート: `artifacts/reports/train_metrics.json`
    - manifest: `artifacts/models/baseline_model.manifest.json`
    - 各 train レポートには `run_context` / `leakage_audit` / `policy_constraints` が保存されます
4. stack bundle 作成
    - `python scripts/run_bundle_models.py --bundle-name policy_stack_v1 --primary-component win --component win=configs/model.yaml --component top3=configs/model_top3.yaml --component alpha=configs/model_alpha.yaml --component roi=configs/model_roi.yaml`
    - bundle manifest: `artifacts/models/policy_stack_v1.bundle.json`
    - 現時点の bundle は registry / orchestration 用の束ね方であり、学習済み meta-model ではありません
5. 予測と可視化
    - stable alias での実行は `python scripts/run_predict.py --profile current_best_eval --race-date 2021-07-31` または `python scripts/run_predict.py --profile current_recommended_serving --race-date 2021-07-31` を使えます
    - `python scripts/run_predict.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - （Top3確率）`python scripts/run_predict.py --config configs/model_top3.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - 予測CSV: `artifacts/predictions/predictions_YYYYMMDD.csv`
    - 可視化PNG: `artifacts/predictions/predictions_YYYYMMDD.png`
    - `predict` は sidecar として `artifacts/predictions/predictions_YYYYMMDD.summary.json` も出力し、`profile / score_source / policy_name` を保存します
    - 対応する manifest が存在する場合は利用 artifact も出力されます
    - Top3確率モデルでは `p_rank1 / p_rank2 / p_rank3` 列が出力されます（各レース内で正規化済み）
    - `p_top3`（= `p_rank1 + p_rank2 + p_rank3`）も出力され、複勝系の期待値計算に利用できます
6. バックテスト
    - stable alias での実行は `python scripts/run_backtest.py --profile current_best_eval` または `python scripts/run_backtest.py --profile current_recommended_serving` を使えます
    - `python scripts/run_backtest.py --config configs/model.yaml`
    - （任意）`python scripts/run_backtest.py --config configs/model.yaml --predictions-file artifacts/predictions/predictions_20210731.csv`
    - レポートJSON: `artifacts/reports/backtest_YYYYMMDD.json`
    - 可視化PNG: `artifacts/reports/backtest_YYYYMMDD.png`
    - alias で backtest した場合、JSON には `profile` も保存されます
7. Serving smoke validation
    - 今後の短い別名として、主力比較は `current_best_eval` 系、簡易運用候補は `current_recommended_serving` 系を使えます
    - representative date の score source / fixed policy routing を一括確認するときは `python scripts/run_serving_smoke.py --profile best_policy_may`、`python scripts/run_serving_smoke.py --profile fallback_hybrid`、または June-only simplification probe の `python scripts/run_serving_smoke.py --profile fallback_hybrid_june_strict`
    - 同じ内容を stable alias で呼ぶ場合は `python scripts/run_serving_smoke.py --profile current_best_eval` と `python scripts/run_serving_smoke.py --profile current_recommended_serving` を使えます
    - `2024-05-25..2024-06-23` の運用 calendar window をまとめて確認するときは `python scripts/run_serving_smoke.py --profile best_policy_may_window`、`python scripts/run_serving_smoke.py --profile fallback_hybrid_window`、または June-only simplification probe の `python scripts/run_serving_smoke.py --profile fallback_hybrid_june_strict_window`
    - stable alias では `python scripts/run_serving_smoke.py --profile current_best_eval_window` と `python scripts/run_serving_smoke.py --profile current_recommended_serving_window` が使えます
    - 2024-05 の実週末 8 日 (`05-04/05/11/12/18/19/25/26`) をまとめて確認するときは `python scripts/run_serving_smoke.py --profile best_policy_may_may_weekends` または `python scripts/run_serving_smoke.py --profile fallback_hybrid_june_strict_may_weekends`
    - stable alias では `python scripts/run_serving_smoke.py --profile current_best_eval_may_weekends` と `python scripts/run_serving_smoke.py --profile current_recommended_serving_may_weekends` が使えます
    - 特定日だけ確認したいときは `--date 2024-09-14` のように絞れます。preset に入っていない日付でも、`serving.score_regime_overrides` / `serving.policy_regime_overrides` を config から自動解決して smoke できます
    - summary は `artifacts/reports/serving_smoke_<profile>.json` に保存され、prediction/backtest artifact は `_policy_may` や `_fallback_hybrid` suffix 付きでも退避されます
    - 2 つの smoke summary を横比較するときは `python scripts/run_serving_smoke_compare.py --left-summary artifacts/reports/serving_smoke_best_policy_may.json --right-summary artifacts/reports/serving_smoke_fallback_hybrid.json`
    - compare 結果は `artifacts/reports/serving_smoke_compare_<left>_vs_<right>.json` と `.csv` に保存されます
    - compare summary には shared representative dates 上の `policy_bets` / `policy_selected_rows` 合計差、`policy_roi` の日次平均差、さらに `policy_return` / `policy_net` の合計差も含まれます
    - mitigation probe 由来の runtime-ready candidate を実際の config variant に落としたいときは `python scripts/run_generate_serving_config_variants_from_candidates.py --base-config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml --candidate-report artifacts/reports/generated_serving_candidates_from_mitigation_probe_current_profiles_h1_vs_h2_2024.json` を使えます。single-policy probe と、May-July の Kelly override を残した `hybrid_keep_kelly` proxy の両方を `configs/` に生成し、対応する report JSON も残します
    - 代表 5 日 (`2024-05-11`, `2024-06-29`, `2024-07-27`, `2024-08-17`, `2024-09-07`) の actual-calendar smoke では、single-policy portfolio 候補は直接 load できる一方で baseline `current_recommended_serving` を上回りませんでした。`hybrid_keep_kelly` でも差分は Aug/Sep の portfolio 月に集約されるだけで、shared-date total net は baseline `39.6` に対し `portfolio_ev_only=8.0`, `portfolio_lower_blend=-5.0` でした。現時点では analysis/proxy 用であり、serving alias の置き換え候補ではありません
    - shared prediction frame は 16GB 環境でも 4GB 以上の RSS を使うことがあるため、`run_serving_smoke.py` は直列で回してください。スクリプト側でも lock で多重起動を弾きます
8. モデル評価（全体＋日別）
    - stable alias で主力評価系を再計算するときは `python scripts/run_evaluate.py --profile current_best_eval --max-rows 80000`、簡易運用候補を確認するときは `python scripts/run_evaluate.py --profile current_recommended_serving --max-rows 80000` を使えます
    - `python scripts/run_evaluate.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - （CatBoost win）`python scripts/run_evaluate.py --config configs/model_catboost.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 200000`
    - （benchmark 用 / CatBoost fundamental win）`python scripts/run_evaluate.py --config configs/model_catboost_fundamental.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental.yaml --max-rows 200000 --wf-mode off`
    - （benchmark 診断 / no-lineage）`python scripts/run_evaluate.py --config configs/model_catboost_fundamental_enriched_no_lineage.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental_enriched_no_lineage.yaml --max-rows 200000 --wf-mode off`
    - （CatBoost Top3）`python scripts/run_evaluate.py --config configs/model_catboost_top3.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 200000`
    - （CatBoost ROI）`python scripts/run_evaluate.py --config configs/model_catboost_roi.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 200000 --wf-mode off`
    - （CatBoost alpha）`python scripts/run_evaluate.py --config configs/model_catboost_alpha.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 150000 --wf-mode off`
    - （長期ROI向け stack）`python scripts/run_evaluate.py --config configs/model_catboost_value_stack.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 150000 --wf-mode fast`
    - （GPU / ROI mainline stack）`python scripts/run_evaluate.py --config configs/model_catboost_value_stack_time_gpu.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 100000 --wf-mode off`
    - （ROI回帰）`python scripts/run_evaluate.py --config configs/model_roi.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - （市場乖離/Layer2）`python scripts/run_evaluate.py --config configs/model_alpha.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - 最新年だけを切って見たいときは `--start-date` / `--end-date` を併用できます。例: `python scripts/run_evaluate.py --config configs/model_catboost_fundamental_enriched_no_lineage.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental_enriched_no_lineage.yaml --start-date 2024-01-01 --end-date 2024-01-08 --wf-mode off`
    - 低メモリ環境で feature build 前に入力を絞りたいときは `--pre-feature-max-rows 5000` を使えます。`run_evaluate.py` はこの指定時に recent tail loader を使い、artifact には `data_load_strategy=tail_training_table` と `primary_source_rows_total` も残します。既存の `--max-rows` は feature build 後の evaluation slice のままです
    - 全体指標: `artifacts/reports/evaluation_summary.json`（常に最新実行）
    - 最新 evaluation manifest: `artifacts/reports/evaluation_manifest.json`
    - summary / manifest には `stability_assessment` と `stability_guardrail` も入り、レース数・日数・calendar span・EV>=1.0 bet 数が小さい run は `probe_only` / `caution` として明示的に警告されます。短い smoke / probe の ROI を昇格判断に使わないための guardrail です
    - モデル別保存: `artifacts/reports/evaluation_summary_<model>.json`
            - `wf_mode` / `wf_scheme` がデフォルト (`fast` / `nested`) 以外のときは versioned artifact に `_wf_<mode>_<scheme>` suffix が付き、raw / walk-forward 実験が同じ date window でも上書きされません
            - 対応する `evaluation_manifest_<model>.json` も保存され、summary / by-date の path、checksum、row-count 整合性を確認できます
            - `selection_mode=gate_then_roi` で feasible 候補が 1 つも無い walk-forward fold は `strategy_kind=no_bet` / `selection_reason=no_feasible_candidate` として保存され、infeasible な fallback policy は採用されません
            - `run_context`: 実行条件（config, max_rows, wf設定 など）
            - `artifact_manifest`: 利用モデルの manifest パス
            - `leakage_audit`: 特徴量リーク疑義の自動点検結果
            - `public_pseudo_r2` / `model_pseudo_r2` / `benter_combined_pseudo_r2` / `benter_delta_pseudo_r2`: public を超える追加情報があるかを測る benchmark 指標
    - 日別指標: `artifacts/reports/evaluation_by_date.csv`（常に最新実行）
    - モデル別日別保存: `artifacts/reports/evaluation_by_date_<model>.csv`
        - by-date が空の run では summary / manifest には `latest_by_date` / `versioned_by_date = null` が入り、存在しない CSV path を指さないようにします
        - 回収率指標（主目的）:
            - `top1_roi`: スコア1位を毎レース購入
            - `ev_top1_roi`: `score × odds` が最大の馬を毎レース購入
            - `ev_threshold_1_0_roi`: 期待値1.0以上のみ購入
            - `ev_threshold_1_2_roi`: 期待値1.2以上のみ購入
    - manifest の整合性を点検するときは `python scripts/run_validate_evaluation_manifest.py --manifest artifacts/reports/evaluation_manifest.json` を使えます
    - latest + versioned manifest をまとめて点検するときは `python scripts/run_validate_evaluation_manifest.py --all-manifests` を使えます
    - versioned manifest 単体も独立に検証できます。latest が別 run に進んだ後でも、`evaluation_manifest_<model>_...json` は自分の versioned summary / by-date / manifest と整合していれば `ok` になります
    - 必要に応じて `--output artifacts/reports/evaluation_manifest_validation.json` で validation report を保存できます
    - 昇格可否を機械判定したいときは `python scripts/run_promotion_gate.py --evaluation-manifest artifacts/reports/evaluation_manifest.json` を使えます。matching な `wf_feasibility_diag_*.json` があれば config/data/feature tuple で自動解決し、見つからない場合は `--wf-summary <path>` を明示します
    - promotion gate は evaluation manifest 整合性、evaluation の `stability_assessment=representative`、matching な walk-forward feasibility summary の `stability_assessment=representative`、そして feasible fold 数をまとめて見ます。fold の valid/test は短窓になりやすいため `probe_only` でも即 fail にはせず warning として扱います
    - gate report には `wf_diagnostics` も入り、dominant な failure reason、fold ごとの `min_bets_required`、`best_fallback` / `closest_infeasible` の要約を見られます。`ratio_bets_required_by_fold` と `binding_min_bets_source_by_fold` / `binding_min_bets_source_counts` も出るので、`min_bets` block が `min_bet_ratio` 起因か `min_bets_abs` 起因かを report 単体で追えます
    - `min_bets_abs` の感度を軽く見たいときは `python scripts/run_wf_threshold_sweep.py --wf-summary artifacts/reports/wf_feasibility_diag_<config>_...json --min-bets-abs-values 100,80,60,40 --target-feasible-fold-counts 1,3,5` を使えます。既存の WF detail CSV を再判定するだけなので、再学習や再推論なしで feasible fold 数、fold ごとの初回通過閾値、`best_feasible` / `best_fallback` の変化に加えて、「何 fold 通したいか」ごとの strictest threshold、`best_feasible` の bet support 要約、閾値を下げたときの `best_feasible_changed_folds` も JSON/CSV に保存できます
    - ただし threshold sweep は policy frontier を見るための analysis です。元の WF fold が `probe_only` なら、report の `stability_context.warnings` にも出る通り、長期運用の昇格根拠としては使わず representative な evaluation / WF gate と分けて扱ってください
    - 複数期間・複数 profile の threshold report を並べて比較するときは `python scripts/run_wf_threshold_compare.py --reports <report1> <report2> ...` を使えます。strictest threshold (`1/3/5 folds`)、warning 数、閾値ごとの feasible fold 数 / dominant failure / support summary に加えて、fold 別 `feasible/min_bets/min_final_bankroll/max_drawdown` 内訳、feasible 候補の final bankroll 要約、bankroll block の gap 要約、blocked candidate の top signature も JSON/CSV にまとめます。fold 明細は既定で `<summary>_folds.csv` にも出るので、繰り返し出る blocked strategy parameter を period 間で直接比較できます
    - blocked signature family を period 横断で集約したいときは `python scripts/run_wf_threshold_signature_report.py --compare-report artifacts/reports/wf_threshold_compare_....json` を使えます。`fold_snapshots` から blocked signature の出現回数、status 別件数、window 別件数、bankroll gap / min_bets gap 要約を JSON/CSV にまとめます
    - dominant blocked signature family の代替候補を fold 単位で掘りたいときは `python scripts/run_wf_threshold_signature_drilldown.py --compare-report artifacts/reports/wf_threshold_compare_....json` を使えます。blocked occurrence ごとに、target signature の実績、同じ fold/threshold での最良 feasible 候補、同じ strategy 内の最良 feasible 代替候補、bet/final_bankroll 差分を JSON/CSV に出します
    - dominant blocked signature の近傍 variant を parameter 軸で横並び比較したいときは `python scripts/run_wf_threshold_signature_family_compare.py --compare-report artifacts/reports/wf_threshold_compare_....json` を使えます。既定では target と同じ strategy / odds band / `top_k` を shared family とみなし、同一 blocked occurrence 群の中で `blend_weight`・`min_prob`・`min_expected_value` などの variant が何回現れ、何回 feasible になり、target 比で bets / final_bankroll がどう動くかを JSON/CSV にまとめます
    - 次に試す mitigation を ranking したいときは `python scripts/run_wf_threshold_mitigation_shortlist.py --family-report ... --drilldown-report ...` を使えます。portfolio family 内の variant 候補と drilldown の lower-threshold recovery signature をまとめ、occurrence 数・median bet 差分・median bankroll 差分・best-in-family / recovery 回数から候補順を JSON/CSV に落とします
    - shortlist 上位候補を fold 単位で直接比べたいときは `python scripts/run_wf_threshold_mitigation_focus.py --compare-report ... --shortlist-report ... --candidate-ranks 1,3` を使えます。各 blocked occurrence ごとに target と 2 候補の bets / final_bankroll / base_score を並べ、どちらが bankroll 優位か、どちらで十分か、どの window/fold が lower-blend を必要とするかを JSON/CSV にまとめます
    - 段階的 mitigation policy を probe したいときは `python scripts/run_wf_threshold_mitigation_policy_probe.py --focus-report ... --drilldown-report ...` を使えます。既定では `portfolio blend=0.8,min_ev=1.0` を first step、`portfolio blend=0.6,min_ev=1.0` を second step、必要なら drilldown の Kelly recovery signature を third step として選び、各 blocked occurrence がどの段階でどこまで改善するかを JSON/CSV にまとめます
    - mitigation probe から serving 候補の YAML/JSON を起こしたいときは `python scripts/run_generate_serving_candidates_from_mitigation_probe.py --policy-probe ...` を使えます。runtime にそのまま入れられる portfolio 候補と、現行 runtime では date regime だけでは表現しきれない staged hybrid spec を分けて出すので、実装可能な部分と未実装ギャップを切り分けられます
    - さらに runtime-ready serving 候補を現行 serving config 派生の実ファイルへ落とすときは `python scripts/run_generate_serving_config_variants_from_candidates.py --base-config ... --candidate-report ...` を使えます。pure portfolio の `single_policy` probe と、既存 Kelly month override を残した `hybrid_keep_kelly` proxy を同時に生成できるので、runtime で今すぐ試せる差し替えと、baseline を大きく崩さない近似 proxy を分けて検証できます
    - gate report はデフォルトで `artifacts/reports/promotion_gate_report.json` に保存されます。長期運用の昇格判断は短い probe ROI ではなく、この report を起点に確認する想定です
9. ベースライン vs Ranker 比較（同一データでA/B）
    - stable alias で比較する場合は `python scripts/run_ab_compare.py --base-profile current_best_eval --challenger-profile current_recommended_serving --max-rows 30000` を使えます
    - 同一 artifact 比較を禁止したいときは `--require-distinct-artifacts` を付けると、重い feature build 前に fail-fast します
    - 低メモリ環境で feature build 前に入力を絞りたいときは `--pre-feature-max-rows 5000` を使えます。`run_ab_compare.py` はこの指定時に recent tail loader を使い、summary / `run_context` に `data_load_strategy` と `primary_source_rows_total` も残します。既存の `--max-rows` は feature build 後の evaluation slice のままです
    - `python scripts/run_ab_compare.py --base-config configs/model.yaml --challenger-config configs/model_ranker.yaml --max-rows 30000`
    - （CatBoost win vs LightGBM baseline）`python scripts/run_ab_compare.py --base-config configs/model.yaml --challenger-config configs/model_catboost.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 30000`
    - 比較サマリ: `artifacts/reports/ab_compare_summary.json`
    - versioned 保存: `artifacts/reports/ab_compare_summary_<base>_vs_<challenger>_*.json`
    - compare summary にも `stability_assessment` / `stability_guardrail` が入り、小標本 compare を `probe_only` / `caution` として区別できます
    - summary には `run_context` / `artifact_manifest` / `date_window` / `comparison_warnings` / `expected_artifacts` も保存されます
    - 両 side が同じ `model_file` / `manifest_file` に解決された場合は warning が残ります。serving policy だけが違う profile 同士では、このケースが起こりえます
    - Top3確率モデルを比較する場合は `--challenger-config configs/model_top3.yaml` を指定
    - Top3チューニング結果（`artifacts/reports/tune_top3_summary.json`）には `run_context` / `leakage_audit` / `policy_constraints` が保存されます
    - Top3 tuning でも 低メモリ環境で feature build 前に入力を絞りたいときは `--pre-feature-max-rows 5000` を使えます。`run_tune_top3.py` はこの指定時に recent tail loader を使い、candidate report を含む `run_context` に `data_load_strategy` と `primary_source_rows_total` も残します
    - 互換のため `strategy_constraints` も同時に残します
    - value stack tuning は `python scripts/run_tune_value_stack.py --summary-path artifacts/reports/tune_value_stack_summary.json` で実行でき、summary / csv に加えて `tune_value_stack_summary.manifest.json` も出力します
    - 低メモリ環境で feature build 前に入力を絞りたいときは `--pre-feature-max-rows 5000` を使えます。`run_tune_value_stack.py` はこの指定時に recent tail loader を使い、summary / manifest / `run_context` に `data_load_strategy` と `primary_source_rows_total` も残します。既存の `--max-rows` は feature build 後の evaluation slice のままです
    - value stack tuning summary には `run_context` / `search_space` / `component_artifacts` / `output_files` / `loaded_rows` / `data_load_strategy` / `primary_source_rows_total` / `pre_feature_max_rows` が、manifest には summary/csv の checksum と row-count 整合性が保存されます
    - value stack tuning summary / manifest、Top3 tuning summary、walk-forward 診断系 (`run_wf_feasibility_diag.py` / `run_wf_liquidity_probe.py`) にも `stability_assessment` / `stability_guardrail` が入り、短い tuning/probe window を昇格判断に使わないための support 情報を残します。Top3 tuning は candidate ごとの `roi_detail.stability_guardrail` も保存します
    - walk-forward の診断系 (`run_wf_feasibility_diag.py` / `run_wf_liquidity_probe.py`) も `--pre-feature-max-rows` 指定時は recent tail loader を使い、summary `run_context` に `data_load_strategy` と `primary_source_rows_total` を残します
    - walk-forward 診断 artifact の file 名 slug は model file 名ではなく config 名を優先して作ります。runtime score override で同じ model bundle を共有していても、report 名が実行 config と一致するようにするためです
10. ダッシュボード（Notebookが止まるときのCLI代替）
    - `python scripts/run_dashboard.py`
    - 必要に応じて `--predictions-file` / `--backtest-file` / `--train-metrics-file` で参照する artifact を固定できます
    - provenance を厳密に揃えたいときは `--strict-input-match` を付けると、prediction / backtest / train の参照先や `profile` が食い違う場合に失敗します
    - 概要JSON: `artifacts/reports/dashboard/dashboard_summary_YYYYMMDD.json`
    - 概要JSON には `profile / score_source / policy_name / prediction_file / backtest_file / train_metrics_file` に加えて `input_consistency_ok / input_consistency_issues` も保存されます
    - 可視化PNG: `artifacts/reports/dashboard/dashboard_YYYYMMDD.png`
    - Top20 CSV: `artifacts/reports/dashboard/dashboard_top20_YYYYMMDD.csv`
11. 実データで重い場合
    - `configs/model.yaml` の `training.max_train_rows` / `training.max_valid_rows` で学習件数を調整

## Artifact運用
- 学習ごとに `model_file` / `report_file` / `manifest_file` の3点が揃います
- manifest には task、config パス、used_features、categorical_columns、metrics、policy_constraints、run_context が保存されます
- stack bundle は複数 manifest / model を運用単位として束ねるための JSON です
- 推奨運用順: `run_train` 群 → `run_bundle_models.py` → `run_predict` / `run_evaluate` / `run_ab_compare`

## CatBoost長期運用メモ
- 現在の本命系は `configs/model_catboost*.yaml` と `configs/features_catboost_rich.yaml` の組み合わせです
- GPU 用の CatBoost config は `configs/model_catboost_*_gpu.yaml` を使います。CatBoost GPU は pairwise 以外で `rsm` 非対応なので、CPU config をそのまま流用せず GPU 専用 config を使ってください
- `features_catboost_rich.yaml` は `selection.mode: all_safe` を使い、`horse_id`、`horse_name`、`レース名`、`馬主` のような超高カーディナリティ列や払戻系列を除外します
- public benchmark を測るときは `configs/model_catboost_fundamental.yaml` と `configs/features_catboost_fundamental.yaml` を使い、`odds` / `popularity` を切った fundamental model を別管理します
- 学習済みCatBoost bundle には `feature_columns` と `categorical_columns` が埋め込まれるため、推論・評価側は model metadata を優先して同じ入力列を再現します
- データ拡張は `configs/data.yaml` の multi-source loader で扱います。外部CSVを足すときは、主表を壊さず `append_tables` と `supplemental_tables` に追加してください
- CPUのCatBoost ranking は pairwise 制約のため `one_hot_max_size=1` に自動補正されます
- 長期ROI改善用には `configs/model_catboost_value_stack.yaml` を使い、win確率を土台に alpha / ROI シグナルで logit を補正した `value_blend_model` を構築できます

## Notebookトラブルシュート
- `dashboard.ipynb` が止まる場合は、まずカーネルを `.venv` に再選択して先頭セルから順に実行
- それでも止まる場合は Notebook を使わず `python scripts/run_dashboard.py` で同等の集計・可視化を生成
- CLI実行はすべてエラーハンドリング済みで、失敗時は原因を標準出力に表示

## LightGBM / GPUメモ
- 現在は `configs/model.yaml` の `training.allow_fallback_model: false` により、LightGBMが使えない場合は明示的に失敗します（精度劣化フォールバック防止）。
- Docker Desktop + WSL2 を使っている場合は、WSL内に `nvidia-container-toolkit` を別途入れなくてもGPU利用できます（Windows側ドライバ + Docker DesktopのWSL連携前提）。
- LinuxネイティブのDocker Engineを使う場合のみ、host側で `nvidia-container-toolkit` が必要です。
- コンテナ内で `nvidia-smi` が見えない場合、コード側ではGPU利用できません。
- このプロジェクトでは LightGBM の `device_type: "cuda"` を使用します。
- Docker Desktop + WSL2 で `cuInit rc=500` が出る場合は、`/usr/lib/wsl` をコンテナにマウントして `LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:/usr/local/cuda/lib64` を設定します（`docker-compose.yml` に反映済み）。

### LightGBMをCUDA有効で入れ直す（重要）
- `pip install lightgbm` の標準wheelは、環境によってはCUDA無効ビルドです。
- `CUDA Tree Learner was not enabled in this build` が出る場合は、`.venv` でソースビルドしてください。
    - `CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDACXX=/usr/bin/nvcc /workspaces/nr-learn/.venv/bin/python -m pip install --force-reinstall --no-binary lightgbm lightgbm --config-settings=cmake.define.USE_CUDA=ON --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/gcc-13 --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/g++-13 --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13`
- 上記は CUDA 12.4 + GCC 13 の組み合わせを前提にしています（GCC 14 だと `nvcc` 側で失敗する場合があります）。

### `nvidia-smi` は見えるのに学習でGPUが使えない場合
- 症状例:
    - `clinfo` が `Number of platforms 0`
    - LightGBM(OpenCL) が `No OpenCL device found`
    - CUDA初期化テストが `cuInit rc=500`
- これはコンテナ設定だけではなく、WSL/Windows側のGPUコンピュート提供が不足している状態です。
- 対応:
    - Windows側NVIDIAドライバをWSL対応の最新版へ更新
    - `wsl --update` 実行後に Windows 再起動
    - Docker Desktop の WSL Integration / GPU利用設定を有効化
    - `docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi` を再確認
- 補足: WSL側に `libnvidia-opencl.so` が存在しない環境では、LightGBM の OpenCL (`device_type: "gpu"`) は使用できません。

### Docker(WSL)での権限トラブル回避
- `docker-compose.yml` の `nr-learn-gpu` は `UID/GID` を引き継いで起動する設定です。
- `UID` はbashのreadonly変数のため、起動時は `env UID=... GID=... docker compose ...` 形式を使います。
- 以前に `root` 所有で生成されたファイルがあると `unable to write` になるため、最初に一度だけ所有者を戻します。
    - `sudo chown -R $(id -u):$(id -g) artifacts data`
- GPUコンテナ起動例:
    - `DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 env UID=$(id -u) GID=$(id -g) docker compose up -d --build nr-learn-gpu`
    - `docker compose exec nr-learn-gpu nvidia-smi`

### ビルド高速化（BuildKitキャッシュ）
- `Dockerfile.gpu` は BuildKit cache mount（apt/pip）を使用しています。
- 初回ビルド後は同一依存で再ビルドが高速化されます。
- 例:
    - `DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose build nr-learn-gpu`

## 注意
- これは投資助言ではなく、機械学習の学習プロジェクトです。
- 実運用前に必ず長期バックテストと破綻ケース分析を行ってください。
