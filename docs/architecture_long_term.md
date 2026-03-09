# 長期運用向けアーキテクチャ再設計

最終更新: 2026-03-08

## 1. 背景
- 既存実装は `run_evaluate.py`、`run_tune_top3.py`、`run_ab_compare.py`、`predict_batch.py` に予測・戦略・評価ロジックが分散していた。
- この構成では、同じモデルでも CLI ごとに損益計算や Top3 正規化や gating 条件が微妙にズレ、長期運用時の比較可能性が壊れる。
- 根本問題はモデル単体ではなく、**予測レイヤーと意思決定レイヤーの責務境界が曖昧**だったことにある。
- あわせて、学習系は LightGBM + one-hot 前提が強く、実データに多いカテゴリ列を十分に使えていなかった。

## 2. 新しい責務分割

### 2.1 Scoring Layer
- ファイル: `src/racing_ml/evaluation/scoring.py`
- 役割:
  - すべてのモデル出力を共通 API で扱う
  - Top3 bundle の正規化を一箇所に固定
  - `score`, `pred_rank`, `expected_value`, `ev_rank` の生成を共通化

### 2.2 Policy Layer
- ファイル: `src/racing_ml/evaluation/policy.py`
- 役割:
  - 単勝フラットベットのシミュレーション
  - `top1`, `top1_filtered`, `ev`, `edge` の候補カタログ生成
  - `market_prob`, `edge`, `blend_prob` の共通計算
  - `min_bets`, `max_drawdown`, `final_bankroll` の gating
  - Kelly / portfolio のような資金配分戦略も同じレイヤーに収容

### 2.3 Walk-Forward Layer
- ファイル: `src/racing_ml/evaluation/walk_forward.py`
- 役割:
  - calibration split / 3-way split / nested split を共通化
  - isotonic / platt / blend optimization を集約
  - inner-loop の policy optimization を統一

### 2.4 Orchestration Layer
- ファイル:
  - `scripts/run_evaluate.py`
  - `scripts/run_tune_top3.py`
  - `scripts/run_ab_compare.py`
  - `src/racing_ml/serving/predict_batch.py`
- 役割:
  - CLI 引数の解釈
  - 設定ロード
  - レポート保存
  - 共通エンジンの呼び出しだけを行う

### 2.5 Feature Selection / Model Input Layer
- ファイル:
  - `src/racing_ml/features/selection.py`
- 役割:
  - `explicit` / `all_safe` の特徴量選択を共通化する
  - CatBoost向けカテゴリ列と数値列の判定を一箇所に固定する
  - 学習済みbundleに保存された `feature_columns` / `categorical_columns` を推論・評価で再利用する
  - 高カーディナリティ列やリーク疑義列を運用上の安全側に倒して除外する

### 2.6 Artifact Layer
- ファイル:
  - `src/racing_ml/common/artifacts.py`
  - `src/racing_ml/pipeline/bundle_pipeline.py`
  - `scripts/run_bundle_models.py`
- 役割:
  - train ごとの model / report / manifest の整合を保つ
  - config から artifact path を一貫して解決する
  - win / top3 / alpha / roi を stack bundle manifest に束ねる

## 3. 設計原則
1. 予測ロジックは一箇所にしか置かない。
2. 単勝 ROI のシミュレーションは一箇所にしか置かない。
3. 採用判定は「減点」ではなく gating を基準にする。
4. CLI はビジネスロジックを持たず、薄いオーケストレータにする。
5. Walk-forward の内側最適化と外側評価は同じ policy engine を使う。
6. モデル入力列は config の推定値ではなく、保存済みartifact metadataを最優先に再現する。

## 4. 長期運用での利点
- 同一モデルを evaluate / tune / predict / ab_compare で同じ解釈で扱える。
- 追加戦略を policy layer に足すだけで全 CLI に反映できる。
- 採用基準の変更が config と policy layer で閉じる。
- 大規模再学習前に、設計変更の影響を軽量に検証しやすい。
- artifact manifest が残るので、後から「どの config / features / policy で作ったモデルか」を逆引きできる。

## 5. 今回の実装方針
- 既存の CLI インターフェースは維持する。
- 既存の `strategy_constraints` は互換のため読み続ける。
- 新しい推奨設定は `evaluation.policy` に集約する。
- 既存モデル資産を壊さず、予測・評価・戦略選択の基盤を先に再編する。

## 6. 2026-03-07 時点の実装済み追加事項
- main config 群と tune config 群は `evaluation.policy` に移行済み。
- `run_train.py` と `run_tune_top3.py` は report に加えて manifest も出力する。
- `run_bundle_models.py` により、複数モデルを stack bundle manifest として束ねられる。
- `predict_batch.py`、`run_evaluate.py`、`run_ab_compare.py` は artifact path 解決を共通化した。
- 現在の bundle は artifact registry / orchestration 用であり、学習済み meta-model を内包する fused scorer ではない。

## 7. 2026-03-08 時点のCatBoost拡張
- `trainer.py` は LightGBM専用実装ではなくなり、CatBoostを classification / ranking / Top3 / ROI / alpha で共通的に扱う。
- `features/selection.py` により、`all_safe` モードで広めの安全特徴量を選択しつつ、`horse_id`、`horse_name`、`レース名`、`馬主` のような超高カーディナリティ列を除外できる。
- 学習済み model bundle は `feature_columns` と `categorical_columns` を保持し、推論・評価・A/B比較で同じ入力列を再現する。
- model manifest は `categorical_count` と `categorical_columns` を保持する。
- CatBoost CPU ranking は pairwise 制約のため `one_hot_max_size=1` を中央で自動補正する。
- 現在の広域CatBoost profile は 57特徴量、36カテゴリ列を前提に検証している。

## 8. 次段階
1. bundle を直接評価できる fused stack scorer / meta-policy learner を導入する。
2. 戦略の採用判定を OOS fold 単位の gate 集計へ拡張する。
3. fold 別 manifest と selection rationale まで保存して再現性をさらに上げる。