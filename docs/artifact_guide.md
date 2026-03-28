# Artifact ガイド

## 1. この文書の役割

この文書は、`nr-learn` の各 CLI がどこに何を出力し、どの artifact を正式判断で見るべきかを整理した資料である。

## 2. 基本の出力先

- 学習モデル: `artifacts/models/`
- prediction: `artifacts/predictions/`
- report: `artifacts/reports/`
- dashboard: `artifacts/reports/dashboard/`

## 3. 学習 artifact

学習ごとに、基本的には次の 3 点が揃う。

1. `model_file`
2. `report_file`
3. `manifest_file`

manifest には少なくとも次が保存される。

- task
- config path
- used features
- categorical columns
- metrics
- policy constraints
- run context

## 4. 評価 artifact

正式判断でまず見るものは次の 2 つである。

- `artifacts/reports/evaluation_summary.json`
- `artifacts/reports/evaluation_manifest.json`

`run_evaluate.py` の summary / manifest には `stability_assessment` と `stability_guardrail` が入り、短窓 run を `probe_only` / `caution` として区別できる。

モデル別の versioned 保存も行われるため、latest が上書きされた後でも個別 run を追える。

## 5. promotion gate artifact

昇格判断で見る代表 artifact は次のとおりである。

- `artifacts/reports/promotion_gate_report.json`
- `artifacts/reports/promotion_gate_<revision>.json`
- `artifacts/reports/revision_gate_<revision>.json`

特に `revision` 単位で判断するときは、`run_revision_gate.py` が出す `promotion_gate_<revision>.json` と `revision_gate_<revision>.json` を対で扱う。

`run_revision_gate.py` は train / evaluate の途中で失敗した場合も、可能な限り `revision_gate_<revision>.json` に実行済み step と失敗位置を残す。

## 6. prediction / backtest artifact

prediction の主な出力:

- `artifacts/predictions/predictions_YYYYMMDD.csv`
- `artifacts/predictions/predictions_YYYYMMDD.png`
- `artifacts/predictions/predictions_YYYYMMDD.summary.json`

backtest の主な出力:

- `artifacts/reports/backtest_YYYYMMDD.json`
- `artifacts/reports/backtest_YYYYMMDD.png`

prediction summary には `profile / score_source / policy_name` が保存される。

## 7. serving compare artifact

serving 検証では、主に次の artifact を見る。

- `serving_smoke_profile_compare_*.json`
- `serving_smoke_*.json`
- `serving_smoke_compare_*.json`
- `serving_stateful_bankroll_sweep_*.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_*.json`
- `artifacts/reports/dashboard/serving_compare_aggregate_*.json`

短窓 compare は方向確認に使い、正式昇格判断は別途 `evaluation` と `promotion gate` で行う。

latest 2025 の actual-date compare を再開するときは、この節だけを起点にせず、まず `serving_validation_guide.md` の quickstart で dashboard summary JSON を順に見てから、必要な artifact 名をこの文書で確認する。

`run_serving_profile_compare.py` は provenance 用の `serving_smoke_profile_compare_*.json` を出し、途中 step が失敗した場合も可能な限り実行済み step と失敗位置を残す。

`run_serving_compare_dashboard.py` と `run_serving_compare_aggregate.py` の出力には、この compare manifest の `status` / `decision` も引き継がれる。

## 8. dashboard artifact

Notebook を使わずに CLI でダッシュボードを出す場合、主な出力は次のとおりである。

- `artifacts/reports/dashboard/dashboard_summary_YYYYMMDD.json`
- `artifacts/reports/dashboard/dashboard_YYYYMMDD.png`
- `artifacts/reports/dashboard/dashboard_top20_YYYYMMDD.csv`

## 9. stack bundle artifact

stack bundle は、複数 manifest / model を運用単位として束ねるための JSON である。

これは学習済み meta-model そのものではなく、registry / orchestration 用の束ね方である。

## 10. 推奨の見方

普段の確認順は次でよい。

1. latest compare の再開は `serving_validation_guide.md` の quickstart から入る
2. 実運用比較では serving compare / dashboard を見る
3. 正式判断では evaluation summary / manifest を見る
4. 昇格可否は promotion gate report を見る
5. 学習結果を見るときは train report と manifest を見る

外部データを別 universe として試す場合は、snapshot / gate / revision artifact 名を JRA 系と分ける。特に coverage snapshot と benchmark gate manifest は、既存の `netkeiba_*` 系 latest artifact を上書きしない別 slug を持たせる。