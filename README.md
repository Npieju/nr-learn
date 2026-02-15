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
3. 生成物確認
    - モデル: `artifacts/models/baseline_model.joblib`
    - レポート: `artifacts/reports/train_metrics.json`
4. 予測と可視化
    - `python scripts/run_predict.py --config configs/model.yaml --data-config configs/data.yaml --feature-config configs/features.yaml --race-date 2021-07-31`
    - 予測CSV: `artifacts/predictions/predictions_YYYYMMDD.csv`
    - 可視化PNG: `artifacts/predictions/predictions_YYYYMMDD.png`
5. バックテスト
    - `python scripts/run_backtest.py --config configs/model.yaml`
    - （任意）`python scripts/run_backtest.py --config configs/model.yaml --predictions-file artifacts/predictions/predictions_20210731.csv`
    - レポートJSON: `artifacts/reports/backtest_YYYYMMDD.json`
    - 可視化PNG: `artifacts/reports/backtest_YYYYMMDD.png`
6. 実データで重い場合
    - `configs/model.yaml` の `training.max_train_rows` / `training.max_valid_rows` で学習件数を調整

## 注意
- これは投資助言ではなく、機械学習の学習プロジェクトです。
- 実運用前に必ず長期バックテストと破綻ケース分析を行ってください。
