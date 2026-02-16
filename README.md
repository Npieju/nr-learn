# nr-learn

JRA競馬予想の**学習用**プロジェクトです。  
データソースは KaggleHub の `takamotoki/jra-horse-racing-dataset` を前提にしています。

## 目的
- 競馬予想を「勘」ではなく、再現可能なMLパイプラインとして学ぶ
- 時系列リークを避けた検証方法を身につける
- ベースラインから改善を積み上げる

## アーキテクチャ概要
- **Data Layer**: 生データ取得・正規化・特徴量生成
- **Model Layer**: Baseline（LightGBM分類）→ Ranker（LightGBM LambdaRank）
- **Evaluation Layer**: AUC / NDCG / 回収率 / 的中率
- **Serving Layer**: 次レースの予測バッチ出力（CSV/JSON）
- **Ops Layer**: 設定管理、実験ログ、モデルレジストリ（ローカル）

詳細は [docs/architecture.md](docs/architecture.md) を参照。

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
│   └── architecture.md
├── notebooks/
├── src/
│   └── racing_ml/
│       ├── data/
│       ├── features/
│       ├── models/
│       ├── evaluation/
│       ├── serving/
│       └── pipeline/
└── scripts/
    ├── run_train.py
    ├── run_backtest.py
    └── run_predict.py
```

## まずの進め方（推奨）
1. `data/raw/` にデータを配置（または取得スクリプトで同期）
2. Baseline分類モデルで `win/place` を予測
3. 時系列CVで評価
4. 回収率を含む指標で改善
5. Rankモデルへ拡張

## 実行手順（MVP）
1. データ取得
    - `python scripts/run_ingest.py --config configs/data.yaml`
    - Kaggle認証が未設定 / データ取得失敗時は、学習確認用の `data/raw/sample_races.csv` を自動生成
2. 学習
    - `python scripts/run_train.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （Ranker）`python scripts/run_train.py --config configs/model_ranker.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
    - （Top3確率）`python scripts/run_train.py --config configs/model_top3.yaml --data-config configs/data.yaml --feature-config configs/features.yaml`
3. 生成物確認
    - モデル: `artifacts/models/baseline_model.joblib`
    - レポート: `artifacts/reports/train_metrics.json`
4. 予測と可視化
    - `python scripts/run_predict.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - （Top3確率）`python scripts/run_predict.py --config configs/model_top3.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - 予測CSV: `artifacts/predictions/predictions_YYYYMMDD.csv`
    - 可視化PNG: `artifacts/predictions/predictions_YYYYMMDD.png`
    - Top3確率モデルでは `p_rank1 / p_rank2 / p_rank3` 列が出力されます（各レース内で正規化済み）
    - `p_top3`（= `p_rank1 + p_rank2 + p_rank3`）も出力され、複勝系の期待値計算に利用できます
5. バックテスト
    - `python scripts/run_backtest.py --config configs/model.yaml`
    - （任意）`python scripts/run_backtest.py --config configs/model.yaml --predictions-file artifacts/predictions/predictions_20210731.csv`
    - レポートJSON: `artifacts/reports/backtest_YYYYMMDD.json`
    - 可視化PNG: `artifacts/reports/backtest_YYYYMMDD.png`
6. モデル評価（全体＋日別）
    - `python scripts/run_evaluate.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --max-rows 80000`
    - 全体指標: `artifacts/reports/evaluation_summary.json`
    - 日別指標: `artifacts/reports/evaluation_by_date.csv`
        - 回収率指標（主目的）:
            - `top1_roi`: スコア1位を毎レース購入
            - `ev_top1_roi`: `score × odds` が最大の馬を毎レース購入
            - `ev_threshold_1_0_roi`: 期待値1.0以上のみ購入
            - `ev_threshold_1_2_roi`: 期待値1.2以上のみ購入
7. ベースライン vs Ranker 比較（同一データでA/B）
    - `python scripts/run_ab_compare.py --base-config configs/model.yaml --challenger-config configs/model_ranker.yaml --max-rows 30000`
    - 比較サマリ: `artifacts/reports/ab_compare_summary.json`
    - Top3確率モデルを比較する場合は `--challenger-config configs/model_top3.yaml` を指定
8. ダッシュボード（Notebookが止まるときのCLI代替）
    - `python scripts/run_dashboard.py`
    - 概要JSON: `artifacts/reports/dashboard/dashboard_summary_YYYYMMDD.json`
    - 可視化PNG: `artifacts/reports/dashboard/dashboard_YYYYMMDD.png`
    - Top20 CSV: `artifacts/reports/dashboard/dashboard_top20_YYYYMMDD.csv`
9. 実データで重い場合
    - `configs/model.yaml` の `training.max_train_rows` / `training.max_valid_rows` で学習件数を調整

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
