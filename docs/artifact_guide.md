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

`artifacts/reports/promotion_gate_report.json` と `promotion_gate_<revision>.json` も `read_order`, `current_phase`, `recommended_action`, `highlights` を持つ。したがって pass/promote、block/hold、入力不備の hard failure のどれでも、operator は promotion gate report 単体で現在地を読める。

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

`run_serving_profile_compare.py` は provenance 用の `serving_smoke_profile_compare_*.json` を出し、途中 step が失敗した場合も可能な限り実行済み step と失敗位置を残す。manifest には `read_order`, `current_phase`, `recommended_action`, `highlights` も入り、left/right smoke, compare, bankroll sweep, dashboard のどこで止まったかを top-level だけで追える。

`run_serving_compare_dashboard.py` と `run_serving_compare_aggregate.py` の出力には、この compare manifest の `status` / `decision` も引き継がれる。

`artifacts/reports/dashboard/serving_compare_dashboard_*.json` も `status`, `read_order`, `current_phase`, `recommended_action`, `highlights` を持つ。したがって compare JSON や smoke summary の入力不備で失敗した場合でも、operator は dashboard summary 単体で停止点を読める。

`artifacts/reports/dashboard/serving_compare_aggregate_*.json` も `status`, `read_order`, `current_phase`, `recommended_action`, `highlights` を持つ。したがって dashboard summary 群や compare manifest 群の入力不備で失敗した場合でも、operator は aggregate summary 単体で停止点を読める。

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

`artifacts/reports/netkeiba_benchmark_gate_manifest.json` も同様に `read_order`, `current_phase`, `recommended_action`, `highlights` を持つ。したがって preflight not-ready、snapshot blocker、train/evaluate failure、completed のどれでも、operator は gate manifest 単体で停止点を読める。

`artifacts/reports/netkeiba_wait_then_cycle_manifest.json` も `read_order`, `current_phase`, `recommended_action`, `highlights` を持つ。したがって lock 待機中、timeout、handoff backfill failure、post-cycle gate not-ready/failed のどれでも、operator は wait manifest 単体で停止点を読める。

local-only の formal 結果を public 向けに要約するときは、まず `artifacts/reports/local_public_snapshot_<revision>.json` を見る。ここでは local-only の status、readiness、promotion decision、evaluation pointer への入口だけを薄くまとめ、必要な場合だけ次の順で深掘りする。

`artifacts/reports/local_public_snapshot_<revision>.json` は dry-run でも `status=planned` のまま閉じ、`read_order`, `current_phase`, `recommended_action`, `readiness`, `promotion_summary`, `benchmark_gate_summary`, `evaluation_summary`, `compare_contract`, `highlights` を返す。さらに lineage 不足などの hard failure でも `status=failed` の summary artifact を残すので、operator は public bridge 入口だけで readiness と downstream mixed compare anchor、停止点、次アクションを completed payload に近い shape で確認できる。

local revision lineage が `--backfill-before-benchmark` で生成されている場合、この public snapshot は `backfill_handoff_summary` も持ち、`read_order` に `local_backfill_then_benchmark -> backfill -> materialize` を含める。したがって public 入口でも、Phase 0 の handoff で止まっているのか、その先の benchmark/revision 側で止まっているのかを top-level から追える。

`artifacts/reports/local_revision_gate_<revision>.json` は dry-run でも `status=planned` のまま閉じ、`read_order`, `current_phase`, `recommended_action`, `highlights` と `benchmark_gate_payload` / `data_preflight_payload` / `revision_manifest_payload` / `evaluation_pointer_payload` の planned preview を返す。さらに benchmark gate blocked / revision gate failed / interrupted などの real run stop state でも `current_phase` と `highlights` を保つので、operator は local raw data 未配置や downstream formal blocker の段階でも、この lineage manifest 単体で停止点と次アクションを completed payload に近い shape で確認できる。

`run_local_revision_gate.py --backfill-before-benchmark` を使った lineage では、上記に加えて `backfill_wrapper_manifest` と `backfill_manifest` も artifacts に入り、`backfill_handoff_payload` / `backfill_summary_payload` も持てる。これにより revision 入口でも `prepare -> collect -> materialize -> benchmark -> revision -> evaluation_pointer` のどこで止まったかを top-level lineage から辿れる。

`artifacts/reports/local_feasibility_manifest_<universe>.json` も dry-run では `status=planned` のまま閉じ、`read_order`, `current_phase`, `highlights`, `snapshot_payload`, `validation_payload`, `feature_gap_payload`, `evaluation_payload` の planned preview を返す。さらに completed / failed でも `current_phase` と `highlights` を保つので、validation / feature gap / evaluation をまだ回していない段階や途中失敗の段階でも、operator はどこで止まり、どの artifact が次に生まれるかを completed payload に近い shape で確認できる。

local-only benchmark に入る前の source readiness を分けて読みたいときは、`artifacts/reports/data_preflight_<revision>.json` あるいは `artifacts/reports/data_preflight_local_nankan.json` を見てよい。ここでは raw dir の有無、primary dataset の有無、required table の不足、`recommended_action` を snapshot より手前で確認できる。

1. `local_public_snapshot_<revision>.json`
2. `local_revision_gate_<revision>.json`
3. `promotion_gate_<revision>.json`
4. `revision_gate_<revision>.json`
5. `evaluation_<revision>_pointer.json`

mixed-universe 比較 artifact は JRA-only / local-only と衝突しない別 family として命名する。最小ルールは `artifacts/reports/mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` で、`left_universe` と `right_universe` は `jra`、`local_nankan` のような universe slug をそのまま使う。

この命名にしておくと、public snapshot、local revision lineage、mixed compare を grep だけで並べ替えられる。JRA current baseline の artifact には mixed compare の結果を直接混ぜず、比較結果は常に別 manifest として切り出す。

最小実装としては `run_mixed_universe_compare.py` が pointer-only manifest を出す。ここでの `decision` は promote 判定ではなく `separate_lineage_required` を返し、left 側の local snapshot / lineage と right 側の JRA public reference の入口を固定する役割に留める。alias 解決を使った場合でも `resolved_left_revision`, `resolved_left_source_kind`, `resolved_left_artifact` を見れば、実際に束ねた left-side 参照元を manifest 自体から追える。

pointer-only compare の前には、`artifacts/reports/mixed_universe_readiness_<left_universe>_vs_<right_universe>_<revision>.json` を見て前提条件を確認してよい。ここでは check 群に加えて `resolved_left_revision`, `resolved_left_source_kind`, `resolved_left_artifact` も見れば、どの left-side artifact を前提に readiness を判定したかがすぐ分かる。最小 check は次である。

1. left 側 universe の readiness が `benchmark_rerun_ready=true`
2. left 側 evaluation pointer が存在する
3. left 側 evaluation の `stability_assessment` が `representative`
4. right 側の public reference と public doc が存在する

この readiness manifest が `status=ready` なら pointer-only compare へ進み、`status=not_ready` なら mixed compare ではなく left 側の readiness / evaluation を先に補う。`mixed_universe_compare_<left>_vs_<right>_<revision>.json` は planned / dry-run でも `left_summary`, `right_summary`, `comparison_contract` に加えて `current_phase`, `highlights` を返して、left input 未生成の段階から pointer bridge の読み口を completed payload と揃える。

left public snapshot に `backfill_handoff_summary` がある場合、`mixed_universe_readiness_<left>_vs_<right>_<revision>.json` は `left_summary.backfill_handoff_summary` を持ち、`left_readiness_blocked` の highlights にも upstream handoff の `status` / `current_phase` / `recommended_action` を補足する。したがって left readiness blocker を読む入口でも、public snapshot のさらに 1 段前にある local handoff 停止点を薄く追える。

compare 前提が揃った後は、`artifacts/reports/mixed_universe_schema_<left_universe>_vs_<right_universe>_<revision>.json` を見て comparison axes を固定する。最小の軸は次である。

1. `promotion`: `decision`
2. `evaluation`: `stability_assessment`, `auc`, `top1_roi`, `ev_top1_roi`, `nested_wf_weighted_test_roi`, `nested_wf_bets_total`
3. `support`: `formal_benchmark_weighted_roi`, `formal_benchmark_feasible_folds`

この schema manifest は numeric compare そのものではなく、left 側でどの artifact path から値を拾い、right 側ではどの public reference を正本にするかを固定する contract として扱う。alias bridge を使っている場合も `resolved_left_revision`, `resolved_left_source_kind`, `resolved_left_artifact` を引き継ぐので、schema の段階で left 側の参照元を見失わない。planned / dry-run でも `read_order`, `comparison_axes`, `metric_rows`, `blocking_context` に加えて `current_phase`, `highlights` を返して、readiness/compare 未生成や readiness blocker の段階から比較契約そのものを同じ shape で読めるようにする。

right 側 JRA baseline の machine-readable 正本は `artifacts/reports/public_benchmark_reference_<reference>.json` とする。ここでは promotion manifest、revision manifest、evaluation manifest、evaluation summary を束ね、`read_order`, `current_phase`, `recommended_action`, `blocker_summary`, `highlights` も持たせる。mixed compare 系 script は docs だけでなくこの reference manifest も参照し、operator も promotion decision・stability・blocking reasons を manifest 単体で追えるようにする。入口の `mixed_universe_readiness_<left>_vs_<right>_<revision>.json` は planned / dry-run でも `checks`, `left_summary`, `compare_command_preview` に加えて `current_phase` と `highlights` も出して、left input 未生成や left readiness blocker の段階から次に回す compare CLI と不足条件を同じ shape で確認できるようにする。

numeric compare 本体は `artifacts/reports/mixed_universe_numeric_compare_<left_universe>_vs_<right_universe>_<revision>.json` とする。ここでは schema manifest の row ごとに `left_value`, `right_value`, `delta_left_minus_right`, `comparison_status` を持たせ、left 側未整備でも partial compare を残せるようにする。ここでも `resolved_left_revision`, `resolved_left_source_kind`, `resolved_left_artifact` を保持して、どの left artifact から比較したかを追えるようにする。planned / dry-run でも `read_order`, `blocking_context`, `summary`, `current_phase`, `highlights`, `row_results=[]` を返して、upstream 未生成や blocker の段階から同じ shape で読めるようにする。

一覧確認用には `artifacts/reports/mixed_universe_numeric_compare_<left_universe>_vs_<right_universe>_<revision>.csv` も併せて出してよい。CSV には `name`, `left_value`, `right_value`, `delta_left_minus_right`, `delta_direction`, `comparison_status` を最低限並べる。

判読用には `artifacts/reports/mixed_universe_numeric_summary_<left_universe>_vs_<right_universe>_<revision>.json` を置いてよい。ここでは row 全件を再掲せず、`verdict`, `severity`, `missing_left_rows`, `missing_right_rows`, `positive_rows`, `negative_rows`, `notes`, `recommended_action` のような summary だけを持たせる。`promote_safe_summary` 自体にも upstream compare の `resolved_left_*` を保持するので、要約だけを読んでも left-side 参照元を確認できる。planned / dry-run でも `read_order`, `current_phase`, `highlights` と空の `promote_safe_summary` を返して、numeric compare 未生成の段階から同じ shape で読めるようにする。

left 側の欠損原因を詰めたいときは `artifacts/reports/mixed_universe_left_gap_audit_<left_universe>_vs_<right_universe>_<revision>.json` を見る。ここでは `missing_left_rows` ごとに必要 source artifact、実在状況、local revision lineage に残っている command preview を並べる。local benchmark が preflight で止まった場合は、`lineage_blocker.recommended_action` とその根拠 artifact もここに残る。mixed revision と local revision が別名でも、gap audit は readiness/public snapshot artifact を使って local lineage を自動で拾い、`summary` にも `resolved_left_*` を残す。planned / dry-run でも `read_order`, `summary`, `lineage_blocker`, `current_phase`, `highlights`, `gap_rows=[]` を返して、numeric compare / lineage 未生成の段階から同じ shape で読めるようにする。

local revision lineage が `backfill_handoff_payload` を持つ場合、gap audit の `lineage_blocker` は benchmark/preflight blocker より先に upstream handoff blocker を採用する。したがって `current_phase` も `local_backfill_then_benchmark` へ切り替わり、`lineage_blocker.source=backfill_handoff` と `recommended_action=run_local_backfill_then_benchmark` をそのまま top-level から読める。

実行順まで落としたいときは `artifacts/reports/mixed_universe_left_recovery_plan_<left_universe>_vs_<right_universe>_<revision>.json` を見る。ここでは gap audit の command preview を重複除去し、`required_for_rows` と必要 artifact path を付けて並べる。もし command preview が無ければ、`populate_primary_raw_dir` のような手動 blocker step が plan に残る。local handoff blocker がある場合は、この manual step が downstream command preview より先頭へ来る。recovery plan は top-level だけでなく `summary` にも `resolved_left_*` を引き継ぐので、要約から left-side 参照元を確認できる。planned / dry-run でも `read_order`, `summary`, `current_phase`, `highlights`, `plan_steps=[]` を先に出して、gap audit 未生成の段階から同じ shape で読めるようにする。

この recovery plan を実際に回して下流 manifest まで更新したいときは、`artifacts/reports/mixed_universe_recovery_<left_universe>_vs_<right_universe>_<revision>.json` を使ってよい。ここでは local lineage 再実行から status board 再生成までの step 実行結果、exit code、更新後 board の `recommended_action` に加えて、実際に参照している left-side の `resolved_left_revision`, `resolved_left_source_kind`, `resolved_left_artifact` もまとめる。planned / dry-run でも source board から `recommended_action`, `refreshed_board_status`, `refreshed_current_phase`, `refreshed_next_action_source`, `refreshed_highlights` を写しつつ、recovery 自身の `read_order`, `status_board_preview`, `highlights` も返して、実行前の停止点と再生成ルートを recovery manifest 単体で読めるようにする。local handoff blocker が recovery plan の first step にいる場合は、この recovery manifest 自体の first step も `local_backfill_then_benchmark` へ切り替わる。

全体の現在地を 1 本で見たいときは `artifacts/reports/mixed_universe_status_board_<left_universe>_vs_<right_universe>_<revision>.json` を置いてよい。ここでは `current_phase`, `next_action_source`, `recommended_action`, `phase_summaries` と、summary / audit / recovery の主要 severity に加えて、各 phase summary にも `requested_revision`, `resolved_left_*`, `current_phase`, `error_code`, `highlights` を持たせ、どの left artifact を読んでいて何が blocker かを段ごとに追えるようにする。planned / dry-run でも同じキー集合を保ち、まだ未生成の phase は `status=missing` の summary 行として先に見えるようにする。

status board は explicit path 指定も受けられるので、tmp smoke や recovery 途中の artifact を読むときは `--public-snapshot` や `--readiness-manifest` を渡して入力を固定してよい。これにより fallback で別 revision の reports を拾う事故を避けられる。