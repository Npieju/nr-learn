# Model Artifacts

最終更新: 2026-03-07

## 1. 目的
- 学習済みモデルを `joblib` だけで管理すると、「どの config / feature / policy / split で作ったか」が追えなくなる。
- このプロジェクトでは train ごとに model / report / manifest を揃え、複数モデルを bundle manifest で束ねる。

## 2. train で生成されるもの
- `model_file`
  - 学習済みモデル本体
- `report_file`
  - metrics
  - `run_context`
  - `leakage_audit`
  - `policy_constraints`
- `manifest_file`
  - artifact 種別
  - config パス
  - task / label / model_name
  - used_features
  - metrics
  - report path
  - policy_constraints

## 3. manifest の役割
- モデルとレポートの紐付けを固定する
- 生成元 config を逆引きできる
- train pipeline と tune candidate の成果物を同じ形式で扱える
- 将来の registry / promotion / rollback の入口にできる

## 4. bundle manifest
- 生成コマンド:
  - `python scripts/run_bundle_models.py --bundle-name policy_stack_v1 --primary-component win --component win=configs/model.yaml --component top3=configs/model_top3.yaml --component alpha=configs/model_alpha.yaml --component roi=configs/model_roi.yaml`
- 出力:
  - `artifacts/models/policy_stack_v1.bundle.json`
- 役割:
  - `win`, `top3`, `alpha`, `roi` など複数モデルをひとつの運用単位として束ねる
  - component ごとの model / report / manifest の場所を一括管理する

## 5. 現時点の制約
- bundle は registry / orchestration 用の定義であり、まだ fused scorer ではない
- つまり bundle 自体が新しい予測器として `score` を直接返すわけではない
- まずは artifact の再現性と運用単位の整備を優先している

## 6. 推奨運用順
1. 各 component を `run_train.py` で学習する
2. manifest が生成されたことを確認する
3. `run_bundle_models.py` で stack bundle を作る
4. 個別 component で `run_predict.py` / `run_evaluate.py` / `run_ab_compare.py` を回す
5. 次段階で fused stack 評価に進む