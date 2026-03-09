# CatBoost Phase 1 Milestone

更新日: 2026-03-08

## 到達点
- LightGBM中心だった学習系を、CatBoost中心の長期運用構成へ切り替えた。
- `all_safe` 特徴量選択により、57特徴量 / 36カテゴリ列で再現可能な入力を固定した。
- win / Top3 / ranking / alpha / ROI の各タスクで CatBoost artifact を作成し、評価レポートまで保存した。
- `run_evaluate.py` は latest の共通ファイルに加えて、モデル別の versioned 評価レポートも自動保存するようにした。

## 残っていた課題
- 勝率モデル、alpha、ROI 回帰が別々に存在し、長期ROI向けの意思決定に一つの確率スコアとして接続されていなかった。
- ROI回帰は閾値を強くかけたときにだけプラスへ寄るが、単体ではベット数が少なく、長期運用の主軸にしづらい。
- alpha は市場との乖離を捉えるが、確率モデルとしては扱えないため calibration / walk-forward の恩恵を受けにくい。

## 次段階の方針
- CatBoost win を確率の土台にする。
- alpha と ROI 回帰は確率そのものを置き換えるのではなく、`logit(win_prob)` を補正する補助シグナルとして使う。
- 出力は 0-1 の確率に戻し、既存の calibration / walk-forward / policy engine をそのまま再利用する。

## 実装対象
- `value_blend_model` を追加し、win / alpha / ROI の artifact を一つの joblib bundle にまとめる。
- `run_build_value_stack.py` で再学習なしに stack artifact を構築する。
- `run_evaluate.py` / `run_predict.py` からは通常のモデルと同じように扱えるようにする。