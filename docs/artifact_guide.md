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

地方-only snapshot / gate を設ける場合、payload 側でも `universe`、`readiness`、`coverage_summary`、`integrity_summary`、`completed_step` を持たせる。artifact 名だけでなく JSON の中身でも JRA-only / local-only / mixed を判別できる状態にしておく。

local-only の formal 結果を public 向けに要約するときは、まず `artifacts/reports/local_public_snapshot_<revision>.json` を見る。ここでは local-only の status、readiness、promotion decision、evaluation pointer への入口だけを薄くまとめ、必要な場合だけ次の順で深掘りする。

local-only benchmark に入る前の source readiness を分けて読みたいときは、`artifacts/reports/data_preflight_<revision>.json` あるいは `artifacts/reports/data_preflight_local_nankan.json` を見てよい。ここでは raw dir の有無、primary dataset の有無、required table の不足、`recommended_action` を snapshot より手前で確認できる。

1. `local_public_snapshot_<revision>.json`
2. `local_revision_gate_<revision>.json`
3. `promotion_gate_<revision>.json`
4. `revision_gate_<revision>.json`
5. `evaluation_<revision>_pointer.json`

mixed-universe 比較 artifact は JRA-only / local-only と衝突しない別 family として命名する。最小ルールは `artifacts/reports/mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` で、`left_universe` と `right_universe` は `jra`、`local_nankan` のような universe slug をそのまま使う。

この命名にしておくと、public snapshot、local revision lineage、mixed compare を grep だけで並べ替えられる。JRA current baseline の artifact には mixed compare の結果を直接混ぜず、比較結果は常に別 manifest として切り出す。

最小実装としては `run_mixed_universe_compare.py` が pointer-only manifest を出す。ここでの `decision` は promote 判定ではなく `separate_lineage_required` を返し、left 側の local snapshot / lineage と right 側の JRA public reference の入口を固定する役割に留める。

pointer-only compare の前には、`artifacts/reports/mixed_universe_readiness_<left_universe>_vs_<right_universe>_<revision>.json` を見て前提条件を確認してよい。最小 check は次である。

1. left 側 universe の readiness が `benchmark_rerun_ready=true`
2. left 側 evaluation pointer が存在する
3. left 側 evaluation の `stability_assessment` が `representative`
4. right 側の public reference と public doc が存在する

この readiness manifest が `status=ready` なら pointer-only compare へ進み、`status=not_ready` なら mixed compare ではなく left 側の readiness / evaluation を先に補う。

compare 前提が揃った後は、`artifacts/reports/mixed_universe_schema_<left_universe>_vs_<right_universe>_<revision>.json` を見て comparison axes を固定する。最小の軸は次である。

1. `promotion`: `decision`
2. `evaluation`: `stability_assessment`, `auc`, `top1_roi`, `ev_top1_roi`, `nested_wf_weighted_test_roi`, `nested_wf_bets_total`
3. `support`: `formal_benchmark_weighted_roi`, `formal_benchmark_feasible_folds`

この schema manifest は numeric compare そのものではなく、left 側でどの artifact path から値を拾い、right 側ではどの public reference を正本にするかを固定する contract として扱う。

right 側 JRA baseline の machine-readable 正本は `artifacts/reports/public_benchmark_reference_<reference>.json` とする。ここでは promotion manifest、revision manifest、evaluation manifest、evaluation summary を束ね、mixed compare 系 script は docs だけでなくこの reference manifest も参照する。

numeric compare 本体は `artifacts/reports/mixed_universe_numeric_compare_<left_universe>_vs_<right_universe>_<revision>.json` とする。ここでは schema manifest の row ごとに `left_value`, `right_value`, `delta_left_minus_right`, `comparison_status` を持たせ、left 側未整備でも partial compare を残せるようにする。

一覧確認用には `artifacts/reports/mixed_universe_numeric_compare_<left_universe>_vs_<right_universe>_<revision>.csv` も併せて出してよい。CSV には `name`, `left_value`, `right_value`, `delta_left_minus_right`, `delta_direction`, `comparison_status` を最低限並べる。

判読用には `artifacts/reports/mixed_universe_numeric_summary_<left_universe>_vs_<right_universe>_<revision>.json` を置いてよい。ここでは row 全件を再掲せず、`verdict`, `severity`, `missing_left_rows`, `missing_right_rows`, `positive_rows`, `negative_rows`, `notes`, `recommended_action` のような summary だけを持たせる。

left 側の欠損原因を詰めたいときは `artifacts/reports/mixed_universe_left_gap_audit_<left_universe>_vs_<right_universe>_<revision>.json` を見る。ここでは `missing_left_rows` ごとに必要 source artifact、実在状況、local revision lineage に残っている command preview を並べる。

実行順まで落としたいときは `artifacts/reports/mixed_universe_left_recovery_plan_<left_universe>_vs_<right_universe>_<revision>.json` を見る。ここでは gap audit の command preview を重複除去し、`required_for_rows` と必要 artifact path を付けて並べる。

この recovery plan を実際に回して下流 manifest まで更新したいときは、`artifacts/reports/mixed_universe_recovery_<left_universe>_vs_<right_universe>_<revision>.json` を使ってよい。ここでは local lineage 再実行から status board 再生成までの step 実行結果、exit code、更新後 board の `recommended_action` をまとめる。

全体の現在地を 1 本で見たいときは `artifacts/reports/mixed_universe_status_board_<left_universe>_vs_<right_universe>_<revision>.json` を置いてよい。ここでは `current_phase`, `next_action_source`, `recommended_action`, `phase_summaries` と、summary / audit / recovery の主要 severity をまとめる。