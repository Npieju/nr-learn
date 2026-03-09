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

詳細は [docs/architecture.md](docs/architecture.md)、[docs/architecture_long_term.md](docs/architecture_long_term.md)、[docs/model_artifacts.md](docs/model_artifacts.md)、[docs/external_benchmark_targets.md](docs/external_benchmark_targets.md) を参照。

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
│   ├── architecture.md
│   ├── architecture_long_term.md
│   └── model_artifacts.md
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
3. CatBoost分類モデルで広めの安全特徴量を使って `win` を予測
4. 時系列スライスで評価し、AUCだけでなくROIも確認
5. Top3 / ROI / alpha / ranking に横展開
6. LightGBM baseline とA/B比較して改善幅を固定化

## 実行手順（MVP）
1. データ取得
    - `python scripts/run_ingest.py --config configs/data.yaml`
    - Kaggle認証が未設定 / データ取得失敗時は、学習確認用の `data/raw/sample_races.csv` を自動生成
    - `configs/data.yaml` の `dataset.external_raw_dirs` に定義した外部 raw dir も同時に作成されます
    - 外部サイト由来CSVを後から足す場合は、`append_tables` で行追加、`supplemental_tables` で列追加を定義します
    - 取り込み前の整合確認: `python scripts/run_validate_data_sources.py --config configs/data.yaml`
    - 外部 pre-race 情報の欠損確認: `python scripts/run_feature_gap_report.py --config configs/data.yaml --model-config configs/model_catboost_fundamental_enriched.yaml --feature-config configs/features_catboost_fundamental_enriched.yaml`
2. 学習
    - `python scripts/run_train.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （推奨 / CatBoost win）`python scripts/run_train.py --config configs/model_catboost.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （benchmark 用 / CatBoost fundamental win）`python scripts/run_train.py --config configs/model_catboost_fundamental.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental.yaml`
    - （benchmark 拡張 / pace-corner enriched）`python scripts/run_train.py --config configs/model_catboost_fundamental_enriched.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental_enriched.yaml`
    - enriched benchmark の train / evaluate artifact には `feature_coverage` が入り、missing force-include 特徴が自動記録されます
    - （推奨 / CatBoost Top3）`python scripts/run_train.py --config configs/model_catboost_top3.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （推奨 / CatBoost Ranker）`python scripts/run_train.py --config configs/model_catboost_ranker.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （推奨 / CatBoost ROI）`python scripts/run_train.py --config configs/model_catboost_roi.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （推奨 / CatBoost alpha）`python scripts/run_train.py --config configs/model_catboost_alpha.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
    - （長期ROI向け stack build）`python scripts/run_build_value_stack.py --config configs/model_catboost_value_stack.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml`
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
    - `python scripts/run_predict.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - （Top3確率）`python scripts/run_predict.py --config configs/model_top3.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - 予測CSV: `artifacts/predictions/predictions_YYYYMMDD.csv`
    - 可視化PNG: `artifacts/predictions/predictions_YYYYMMDD.png`
    - 対応する manifest が存在する場合は利用 artifact も出力されます
    - Top3確率モデルでは `p_rank1 / p_rank2 / p_rank3` 列が出力されます（各レース内で正規化済み）
    - `p_top3`（= `p_rank1 + p_rank2 + p_rank3`）も出力され、複勝系の期待値計算に利用できます
6. バックテスト
    - `python scripts/run_backtest.py --config configs/model.yaml`
    - （任意）`python scripts/run_backtest.py --config configs/model.yaml --predictions-file artifacts/predictions/predictions_20210731.csv`
    - レポートJSON: `artifacts/reports/backtest_YYYYMMDD.json`
    - 可視化PNG: `artifacts/reports/backtest_YYYYMMDD.png`
7. モデル評価（全体＋日別）
    - `python scripts/run_evaluate.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - （CatBoost win）`python scripts/run_evaluate.py --config configs/model_catboost.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 200000`
    - （benchmark 用 / CatBoost fundamental win）`python scripts/run_evaluate.py --config configs/model_catboost_fundamental.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_fundamental.yaml --max-rows 200000 --wf-mode off`
    - （CatBoost Top3）`python scripts/run_evaluate.py --config configs/model_catboost_top3.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 200000`
    - （CatBoost ROI）`python scripts/run_evaluate.py --config configs/model_catboost_roi.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 200000 --wf-mode off`
    - （CatBoost alpha）`python scripts/run_evaluate.py --config configs/model_catboost_alpha.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 150000 --wf-mode off`
    - （長期ROI向け stack）`python scripts/run_evaluate.py --config configs/model_catboost_value_stack.yaml --data-config configs/data.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 150000 --wf-mode fast`
    - （ROI回帰）`python scripts/run_evaluate.py --config configs/model_roi.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - （市場乖離/Layer2）`python scripts/run_evaluate.py --config configs/model_alpha.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - 全体指標: `artifacts/reports/evaluation_summary.json`（常に最新実行）
    - モデル別保存: `artifacts/reports/evaluation_summary_<model>.json`
            - `run_context`: 実行条件（config, max_rows, wf設定 など）
            - `artifact_manifest`: 利用モデルの manifest パス
            - `leakage_audit`: 特徴量リーク疑義の自動点検結果
            - `public_pseudo_r2` / `model_pseudo_r2` / `benter_combined_pseudo_r2` / `benter_delta_pseudo_r2`: public を超える追加情報があるかを測る benchmark 指標
    - 日別指標: `artifacts/reports/evaluation_by_date.csv`（常に最新実行）
    - モデル別日別保存: `artifacts/reports/evaluation_by_date_<model>.csv`
        - 回収率指標（主目的）:
            - `top1_roi`: スコア1位を毎レース購入
            - `ev_top1_roi`: `score × odds` が最大の馬を毎レース購入
            - `ev_threshold_1_0_roi`: 期待値1.0以上のみ購入
            - `ev_threshold_1_2_roi`: 期待値1.2以上のみ購入
8. ベースライン vs Ranker 比較（同一データでA/B）
    - `python scripts/run_ab_compare.py --base-config configs/model.yaml --challenger-config configs/model_ranker.yaml --max-rows 30000`
    - （CatBoost win vs LightGBM baseline）`python scripts/run_ab_compare.py --base-config configs/model.yaml --challenger-config configs/model_catboost.yaml --feature-config configs/features_catboost_rich.yaml --max-rows 30000`
    - 比較サマリ: `artifacts/reports/ab_compare_summary.json`
    - Top3確率モデルを比較する場合は `--challenger-config configs/model_top3.yaml` を指定
    - Top3チューニング結果（`artifacts/reports/tune_top3_summary.json`）には `run_context` / `leakage_audit` / `policy_constraints` が保存されます
    - 互換のため `strategy_constraints` も同時に残します
9. ダッシュボード（Notebookが止まるときのCLI代替）
    - `python scripts/run_dashboard.py`
    - 概要JSON: `artifacts/reports/dashboard/dashboard_summary_YYYYMMDD.json`
    - 可視化PNG: `artifacts/reports/dashboard/dashboard_YYYYMMDD.png`
    - Top20 CSV: `artifacts/reports/dashboard/dashboard_top20_YYYYMMDD.csv`
10. 実データで重い場合
    - `configs/model.yaml` の `training.max_train_rows` / `training.max_valid_rows` で学習件数を調整

## Artifact運用
- 学習ごとに `model_file` / `report_file` / `manifest_file` の3点が揃います
- manifest には task、config パス、used_features、categorical_columns、metrics、policy_constraints、run_context が保存されます
- stack bundle は複数 manifest / model を運用単位として束ねるための JSON です
- 推奨運用順: `run_train` 群 → `run_bundle_models.py` → `run_predict` / `run_evaluate` / `run_ab_compare`

## CatBoost長期運用メモ
- 現在の本命系は `configs/model_catboost*.yaml` と `configs/features_catboost_rich.yaml` の組み合わせです
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
