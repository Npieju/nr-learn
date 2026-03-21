# 環境メモとトラブルシュート

## 1. この文書の役割

この文書は、GPU・Docker・Notebook 周りの実務メモをまとめた補助資料である。

## 2. Notebook が止まるとき

- `dashboard.ipynb` が止まる場合は、まずカーネルを `.venv` に再選択して先頭セルから順に実行する。
- それでも止まる場合は Notebook を使わず、次を使う。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_dashboard.py
```

- CLI 実行はエラーハンドリング済みで、失敗時は原因を標準出力に出す。

## 3. LightGBM / GPU の前提

- 現在は `configs/model.yaml` の `training.allow_fallback_model: false` により、LightGBM が使えない場合は明示的に失敗する。
- このプロジェクトでは LightGBM の `device_type: "cuda"` を使う。
- コンテナ内で `nvidia-smi` が見えない場合、コード側でも GPU は利用できない。

## 4. Docker Desktop + WSL2 の前提

- Docker Desktop + WSL2 を使っている場合は、WSL 内に `nvidia-container-toolkit` を別途入れなくても GPU 利用できる。
- Linux ネイティブの Docker Engine を使う場合のみ、host 側で `nvidia-container-toolkit` が必要になる。
- Docker Desktop + WSL2 で `cuInit rc=500` が出る場合は、`/usr/lib/wsl` をコンテナにマウントし、`LD_LIBRARY_PATH=/usr/lib/wsl/lib:/usr/lib/wsl/drivers:/usr/local/cuda/lib64` を設定する。

## 5. LightGBM を CUDA 有効で入れ直す

- `pip install lightgbm` の標準 wheel は、環境によっては CUDA 無効ビルドである。
- `CUDA Tree Learner was not enabled in this build` が出る場合は、`.venv` でソースビルドする。

例:

```bash
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 CUDACXX=/usr/bin/nvcc \
  /workspaces/nr-learn/.venv/bin/python -m pip install --force-reinstall --no-binary lightgbm lightgbm \
  --config-settings=cmake.define.USE_CUDA=ON \
  --config-settings=cmake.define.CMAKE_C_COMPILER=/usr/bin/gcc-13 \
  --config-settings=cmake.define.CMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  --config-settings=cmake.define.CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-13
```

上記は CUDA 12.4 + GCC 13 を前提にしている。

## 6. `nvidia-smi` は見えるのに GPU 学習できないとき

症状例:

- `clinfo` が `Number of platforms 0`
- LightGBM(OpenCL) が `No OpenCL device found`
- CUDA 初期化テストが `cuInit rc=500`

この場合、コンテナ設定だけではなく WSL / Windows 側の GPU コンピュート提供が不足していることがある。

確認項目:

1. Windows 側 NVIDIA ドライバを WSL 対応の最新版へ更新する。
2. `wsl --update` 実行後に Windows を再起動する。
3. Docker Desktop の WSL Integration / GPU 利用設定を有効化する。
4. 次で再確認する。

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-runtime-ubuntu22.04 nvidia-smi
```

## 7. Docker(WSL) の権限トラブル回避

- `docker-compose.yml` の `nr-learn-gpu` は `UID/GID` を引き継いで起動する設定である。
- `UID` は bash の readonly 変数なので、起動時は `env UID=... GID=... docker compose ...` 形式を使う。
- 以前に `root` 所有で生成されたファイルがあると `unable to write` になるため、必要なら所有者を戻す。

```bash
sudo chown -R $(id -u):$(id -g) artifacts data
```

GPU コンテナ起動例:

```bash
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 env UID=$(id -u) GID=$(id -g) docker compose up -d --build nr-learn-gpu
docker compose exec nr-learn-gpu nvidia-smi
```

## 8. BuildKit キャッシュ

- `Dockerfile.gpu` は BuildKit cache mount を使っている。
- 初回ビルド後は、同一依存の再ビルドが高速化される。

例:

```bash
DOCKER_BUILDKIT=1 COMPOSE_DOCKER_CLI_BUILD=1 docker compose build nr-learn-gpu
```

## 9. CatBoost 長期運用メモ

- 本命系は `configs/model_catboost*.yaml` と `configs/features_catboost_rich.yaml` の組み合わせを使う。
- GPU 用の CatBoost config は `configs/model_catboost_*_gpu.yaml` を使う。
- `features_catboost_rich.yaml` は `selection.mode: all_safe` を使い、超高カーディナリティ列や払戻系列を除外する。
- public benchmark を測るときは `configs/model_catboost_fundamental.yaml` と `configs/features_catboost_fundamental.yaml` を使い、`odds` / `popularity` を切った fundamental model を別管理する。
- 学習済み CatBoost bundle には `feature_columns` と `categorical_columns` が埋め込まれるため、推論・評価側は model metadata を優先して同じ入力列を再現する。
- CPU の CatBoost ranking は pairwise 制約のため `one_hot_max_size=1` に自動補正される。
- 長期 ROI 改善用には `configs/model_catboost_value_stack.yaml` を使い、`value_blend_model` を構築できる。