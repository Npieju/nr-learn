# 競馬予想MLアーキテクチャ（学習用）

## 補足（ROI再整備方針）
- ROI最大化の基礎設計と実装順序は [roi_foundation_design.md](roi_foundation_design.md) を参照。
- 以後の改善はこの方針を優先し、チューニングは最終段階で実施する。
- 長期運用向けの責務分割は [architecture_long_term.md](architecture_long_term.md) を参照。
- 学習成果物と bundle の運用は [model_artifacts.md](model_artifacts.md) を参照。

## 1. なぜ `takamotoki/jra-horse-racing-dataset` が良いか
- JRAのレース/馬/騎手などの構造化データがまとまっている
- 学習用として「前処理・特徴量設計・時系列評価」を一通り学べる
- まずはこのデータで再現可能なベースラインを作り、後でオッズや外部要因を追加しやすい

## 2. 全体構成

```mermaid
flowchart LR
  A[Raw Data] --> B[Data Validation / Cleaning]
  B --> C[Feature Store-like Table]
  C --> D[Train / Validation Split by Time]
  D --> E[Model Training]
  E --> F[Evaluation]
  F --> G[Backtest / ROI Analysis]
  E --> H[Model Registry Local]
  H --> I[Batch Inference]
```

---

## 3. レイヤー設計

### 3.1 Data Layer
**責務**
- 生データの取り込み
- スキーマ統一
- 欠損/外れ値処理
- リーク防止（未来情報を遮断）

**主要コンポーネント**
- `data/ingest.py`: データ取得/配置
- `data/validate.py`: 型・キー・重複チェック
- `data/build_dataset.py`: 学習テーブル生成

**ポイント**
- 主キーの候補: `race_id`, `horse_id`, `date`
- 後処理でオッズを使う場合、締切時点で取得可能だった値のみ利用

### 3.2 Feature Layer
**責務**
- 集約特徴量（直近N走成績、騎手成績、コース適性など）
- カテゴリ処理（競馬場・距離・馬場状態）
- 過去時点で再現できる特徴量のみを採用
- モデルごとに同じ特徴量集合を再現できるよう、選択結果を明示管理する

**設計指針**
- `features/base_features.py`
- `features/history_features.py`
- `features/encoding.py`
- `features/selection.py`

**現在の運用方針**
- LightGBM向けの明示列指定は残すが、本命系は `selection.mode: all_safe` で広めの安全特徴量を採用する
- `horse_id`、`horse_name`、`レース名`、`馬主` のような超高カーディナリティ列はメモリ保護のため除外する
- `jockey_id`、`trainer_id`、馬場・天候・条件系はCatBoostのネイティブカテゴリ処理に渡す

**リーク回避ルール**
- 各サンプル時点 `t` で、`t` 以降の情報を参照しない
- rolling集計は `shift(1)` を原則

### 3.3 Model Layer
段階的に2系統を持つのが学習しやすいです。

1) **Baseline分類**
- 目的変数: `is_win`（1着）または `is_place`（3着以内）
- モデル: LightGBM / CatBoost
- 現在の推奨: CatBoost + 広めの安全特徴量
- メリット: カテゴリ列を大規模に活用しやすく、競馬データの構造と相性が良い

2) **Rankモデル**
- 1レース内での相対順位を学習
- モデル: LightGBM LambdaRank / CatBoost YetiRankPairwise
- メリット: 競馬の本質（同レース内比較）に自然

3) **派生タスク**
- Top3: 1着 / 2着 / 3着確率を独立学習して共通Scoring Layerで扱う
- ROI回帰: 払戻を直接近似して高期待値候補を探索する
- 市場乖離(alpha): 市場確率との差分を学習してオッズ由来の歪みを拾う

### 3.4 Evaluation Layer
**最低限の指標**
- 分類: AUC, LogLoss
- ランキング: NDCG@k, MAP
- 馬券指標: 回収率, 的中率, 破産確率に近い簡易指標

**検証方法（必須）**
- 時系列CV（例: expanding window）
- ランダム分割は使わない

### 3.5 Serving Layer
**目的**
- 当日レースに対して予測スコアをバッチ生成
- 出力: `race_id, horse_id, score, rank, confidence`

**実装**
- `serving/predict_batch.py`
- `scripts/run_predict.py`

### 3.6 Ops Layer
- 設定ファイル: `configs/*.yaml`
- 実験管理: まずはCSVログ/MLflowローカル（任意）
- モデル保存: `artifacts/models/` に日付バージョンで保存
- manifest には `used_features` に加えて `categorical_columns` も残し、推論時は model metadata を優先して入力列を再現する

---

## 4. 学習用におすすめの開発ステップ
1. **P0**: CatBoost分類を `all_safe` 特徴量で時系列評価まで動かす
2. **P1**: 主要特徴量を追加（近走・騎手・距離適性・日付派生）
3. **P2**: Top3 / ROI / alpha に同じ特徴量戦略を横展開する
4. **P3**: Rankモデルを比較導入し、LightGBM baseline とA/Bで差分確認する
5. **P4**: 推論バッチとartifact manifestを固定化する

---

## 5. 最小MVP仕様
- 入力: 過去レーステーブル（特徴量生成済み）
- 出力: 次レースの推奨順位CSV
- 学習: CatBoost分類 + 時系列評価
- 評価: AUC + 回収率
- 保存: モデルと評価レポートを`artifacts/`へ

---

## 6. よくある失敗
- ランダム分割で高スコア → 本番で崩壊
- ゴール後確定情報（確定人気等）を混入
- 特徴量追加で説明不能になり、改善理由が追えない

---

## 7. 次に追加すると良いもの
- Optunaでハイパーパラメータ探索
- SHAPで特徴量寄与の可視化
- ベッティング戦略最適化（ケリー基準の学習用途実装）
