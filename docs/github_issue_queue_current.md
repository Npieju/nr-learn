# GitHub Issue Queue: Current

## Purpose

この文書は current open issue と次の実行順だけを保持する。completed history や長い判断経緯はここへ積まない。

詳細は GitHub issue thread、artifact、tagged snapshot/reference を見る。この file は「今どれを正本として追うか」を短く固定するための入口である。

## Current Priority

1. JRA benchmark 正本は維持する。
2. NAR は readiness track として分離して扱う。
3. current queue では open issue だけを追い、completed issue の説明は繰り返さない。

## Open Issues

### `#120` local Nankan strict provenance trust gate

- role: current NAR root blocker
- status: open, current alias trust-ready
- source-of-truth: GitHub issue `#120`
- first read:
  - [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
  - [../artifacts/reports/local_nankan_provenance_audit.json](../artifacts/reports/local_nankan_provenance_audit.json)
  - `read_order`
  - [../artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json](../artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json)
  - [../artifacts/reports/benchmark_gate_local_nankan_pre_race_ready.json](../artifacts/reports/benchmark_gate_local_nankan_pre_race_ready.json)
- current meaning:
  - trust 判定は fetch 時刻ではなく `classification_basis=pre_race_feature_availability` を使う。current alias read は `pre_race=729107`, `post_race=0`, `unknown=0` で `strict_trust_ready=true`
  - fetch timing は `#121` の source timing audit へ分離されており、`snapshot_timing_bucket_counts` の post-race 優勢だけでは `#120` の trust veto にしない
  - historical pre-race benchmark handoff は provenance block を越えて `local_nankan_pre_race_benchmark_handoff_manifest.json` と `benchmark_gate_local_nankan_pre_race_ready.json` で `completed` まで進んだ。直近 run は `--skip-train --skip-evaluate` で readiness だけ確認している
  - runtime guard と wrapper は current alias `../artifacts/reports/local_nankan_provenance_audit.json` を優先し、未生成時だけ repaired snapshot を fallback として読む
  - `#101` は provenance gate ではもう止まっておらず、次は explicit な formal rerun を回す段階にある。`#103` はその後段に維持する

### `#121` local Nankan historical source timing corrective

- role: `#120` と分離した fetch-timing / recoverability audit
- status: open, historical recoverability negative read maintained
- source-of-truth: GitHub issue `#121`
- first read:
  - [../artifacts/reports/local_nankan_source_timing_audit.json](../artifacts/reports/local_nankan_source_timing_audit.json)
  - `read_order`
  - [issue_library/next_issue_local_nankan_source_timing_corrective.md](issue_library/next_issue_local_nankan_source_timing_corrective.md)
- current meaning:
  - current cache から historical result-ready `pre_race` fetch capture を復元する仮説は negative read のままで、current alias read は `result_ready_pre_race_rows=0`, `future_only_pre_race_rows=426` を返す
  - ここで見ているのは `classification_basis=snapshot_time_vs_scheduled_post_time` の fetch-timing 軸であり、`#120` の trust-bearing feature validity 判定とは別物である
  - future-only readiness track `#122` は引き続き real-time capture / result arrival の operator-default path として保守する
  - runtime guard は current alias `../artifacts/reports/local_nankan_source_timing_audit.json` を優先し、未生成時だけ `issue121` snapshot を fallback として読む

### `#124` JRA pruning stage-7 rollout guardrails

- role: JRA 側の current human-review surface
- status: open, review pending
- source-of-truth: GitHub issue `#124`
- local entrypoints:
  - [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
  - [jra_pruning_package_review_20260410.md](jra_pruning_package_review_20260410.md)
  - [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)
- next action:
  - reviewer が `approve implementation-candidate review package` か `keep docs-only` を判断する

### `#122` local Nankan future-only pre-race readiness track

- role: current NAR operator-default path for live pre-race capture / result arrival
- status: open, blocked on external result arrival
- source-of-truth: GitHub issue `#122`
- first read:
  - [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)
  - `read_order`
  - `readiness_surfaces.readiness_supervisor`
  - `operator_runtime`
  - `readiness_surfaces.readiness_supervisor.current_refs.capture_upcoming_only|capture_as_of|capture_pre_filter_row_count|capture_filtered_out_count`
  - `highlights.supervisor_capture_upcoming_only|supervisor_capture_as_of|supervisor_capture_pre_filter_rows|supervisor_capture_filtered_out`
- command/source docs:
  - [command_reference.md](command_reference.md)
  - [scripts_guide.md](scripts_guide.md)
- current meaning:
  - future-only pre-race pool は維持されている
  - current top-level operator surfaces から strict upcoming filter の cutoff と母数を child capture manifest 深掘りなしで読める
  - この track は live pre-race capture と result arrival を扱う operator path であり、`#120/#101` で扱う historical trust-ready corpus とは役割を分ける
  - `result_ready_races>0` が来るまでは readiness blocker 継続

### `#101` pre-race-only benchmark rebuild

- role: historical pre-race-ready corpus を formal rerun へ進める downstream gate
- status: open, formal rerun completed and review pending
- source-of-truth: GitHub issue `#101`
- first read:
  - [../artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](../artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json)
  - `read_order`
  - [../artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json)
  - [../artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json](../artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json)
- current meaning:
  - formal rerun `r20260415_local_nankan_pre_race_ready_formal_v1` は `status=completed`, `exit_code=0` で完走し、local evaluation pointer が出力された
  - train は `auc=0.8649528380`, `logloss=0.2159883155`, `best_iteration=158` で、source model manifest は `local_nankan_baseline_model.manifest_r20260415_local_nankan_pre_race_ready_formal_v1.json`
  - evaluate は `auc=0.8625051675`, `logloss=0.2171821084`, `top1_roi=0.8402165859`, `ev_threshold_1_0_roi=2.8603751213`, `ev_threshold_1_2_roi=4.5726544989` を返し、nested WF は `wf_nested_test_roi_weighted=3.9660920371`, `wf_nested_test_bets_total=778`, `wf_nested_completed=true`
  - current action は promotion/issue judgement の review であり、`#103` はこの formal rerun artifact を benchmark reference にして次段へ進める

### `#103` value-blend architecture bootstrap

- role: `#101` formal rerun の後段に置く NAR model bootstrap
- status: open, support-corrective candidate formalized and still hold
- source-of-truth: GitHub issue `#103`
- first read:
  - [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json)
  - [../artifacts/reports/wf_feasibility_diag_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_fast_nested.json](../artifacts/reports/wf_feasibility_diag_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_fast_nested.json)
  - [../artifacts/reports/promotion_gate_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_fast_gate.json](../artifacts/reports/promotion_gate_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_fast_gate.json)
- current meaning:
  - revision `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1` で CatBoost win / LightGBM ROI / value stack の first formal execution は completed
  - evaluate は `auc=0.7386067382`, `top1_roi=0.7955827871`, `wf_nested_test_roi_weighted=0.6579177603` で `#101` baseline を下回った
  - bounded fast feasibility でも `wf_feasible_fold_count=0`, `dominant_failure_reason=min_bets`, `formal_benchmark_weighted_roi=null` となり、promotion gate は `status=block`, `decision=hold`
  - follow-up の policy-only support corrective では fast feasibility が `wf_feasible_fold_count=2/3` まで回復し、`min_bets` block が policy/support surface に依存することは確認できた
  - ただし bounded formal candidate でも `auc=0.7235740748`, `top1_roi=0.8005172090`, `wf_nested_test_roi_weighted=0.8160725261` と `#101` baseline を大きく下回り、5 fold 中 2 fold が `no_bet` のままだった
  - 2026-04-17 の bounded October score-switch sweep では `autumn_baseline_recovery` を October fold にだけ適用し、まず `odds_max` を `20 -> 15 -> 12` と絞るほど nested OOS は改善した。versioned summary は `0.8084775967` (`odds_max=20`), `0.8180577022` (`odds_max=15`), `0.8313897764` (`odds_max=12`) だった
  - その後の bounded follow-up では `odds_max=12` を固定して `min_expected_value=1.05` へ上げる仮説は `wf_nested_test_roi_weighted=0.8253844430` まで悪化し、fold 3 strategy も `portfolio -> kelly` に崩れたため reject した
  - 同じ `odds_max=12` を固定したまま October `top_k=2 -> 1` だけを絞る仮説は再び改善し、fold 3 `test_roi=0.7596685083`, `test_final_bankroll=0.9729477612`, `wf_nested_test_roi_weighted=0.8396698616` を記録した
  - さらに October `odds_max=10, top_k=1` まで絞る follow-up では fold 3 `test_roi=0.8598290598`, `test_bets=117`, `test_final_bankroll=0.9898009950`, `wf_nested_test_roi_weighted=0.8489525910` まで改善した。現時点の local best は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_0bac2275217320db.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_0bac2275217320db.json)
  - その後の boundary probe では `odds_max=9, top_k=1` と `odds_max=8, top_k=1` がどちらも fold 3 `no_bet` に崩れ、weighted ROI も `0.8482027107` まで低下したため reject した。October odds tighten の local boundary は `odds_max=10` 近傍とみなしてよい
  - `odds_max=10, top_k=1` を固定した follow-up では October `min_prob=0.025` は `0.02` と完全に同値で no-op だった一方、October `min_expected_value=1.01` と `1.02` はどちらも fold 3 `portfolio -> kelly`, `test_roi=0.7103001518`, `wf_nested_test_roi_weighted=0.8393082237` まで悪化したため reject した。EV gate の局所境界は `1.00` 近傍とみなしてよい
  - 同じく October `min_prob=0.015` も fold 3 `portfolio`, `test_roi=0.8598290598`, `test_bets=117`, `wf_nested_test_roi_weighted=0.8489525910` で `0.02` best と完全一致し、low side も no-op だった。`min_prob` は少なくとも `0.015-0.025` では感度がない
  - October `max_fraction=0.03` も fold 3 `portfolio`, `test_roi=0.8598290598`, `test_bets=117`, `wf_nested_test_roi_weighted=0.8489525910` で現 best と完全一致し、stake cap も binding していなかった
  - October `odds_min=1.5` は fold 3 が `no_bet` に崩れ、`selection_reason=no_feasible_candidate`, `test_bets=0`, `test_final_bankroll=1.0`, `wf_nested_test_roi_weighted=0.8482027107` まで低下したため reject した。低オッズ帯も切りすぎると coverage cliff に入る
  - 中間点の October `odds_min=1.2` も fold 3 `no_bet`, `selection_reason=no_feasible_candidate`, `test_bets=0`, `test_final_bankroll=1.0`, `wf_nested_test_roi_weighted=0.8482027107` で `1.5` と同じく reject だった。low-odds side の cliff は `1.2` 近傍以下には残っている
  - ただし October `odds_min=1.1` は fold 3 `portfolio` を維持したまま `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228856`, `wf_nested_test_roi_weighted=0.8494208494` へ改善し、`odds_max=10, top_k=1` 系の local best を更新した。low-odds side は `1.1` ではまだ改善余地が残る
  - さらに October `odds_min=1.15` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `wf_nested_test_roi_weighted=0.8494208494` で `1.1` と完全一致した。low-odds side の cliff は `1.15` と `1.2` の間にある可能性が高い
  - October `odds_min=1.18` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `wf_nested_test_roi_weighted=0.8494208494` で `1.1/1.15` と完全一致した。改善 plateau は少なくとも `1.18` まで続き、cliff は `1.18` と `1.2` の間へさらに狭まった
  - October `odds_min=1.19` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228856`, `wf_nested_test_roi_weighted=0.8494208494` で `1.1/1.15/1.18` と完全一致した。改善 plateau は `1.19` まで続き、局所 cliff は実質的に `1.2` で始まるとみなせる
  - low-side の October `min_expected_value=0.99` は fold 3 `portfolio` を維持したが、`test_roi=0.8439716312`, `test_bets=141`, `test_final_bankroll=0.9863184080`, `wf_nested_test_roi_weighted=0.8478781284` まで低下したため reject した。October EV gate の局所最適は high side / low side とも `1.00` 近傍にあるとみなしてよい
  - October `blend_weight=0.9` への follow-up も fold 3 `portfolio -> kelly`, `test_roi=0.7509631235`, `test_final_bankroll=0.7888704261`, `wf_nested_test_roi_weighted=0.8377221032` と悪化したため reject した
  - 中間点の `blend_weight=0.85` は fold 3 `portfolio` を維持したが、`test_roi=0.8150684932`, `wf_nested_test_roi_weighted=0.8455778622` でなお `0.8` best を下回った。現状では high-blend 側の局所境界は `0.8` 近傍とみなしてよい
  - 低側 probe の `blend_weight=0.75` は fold 3 が再び `no_bet` へ崩れ、`selection_reason=no_feasible_candidate`, `test_bets=0`, `test_final_bankroll=1.0`, `wf_nested_test_roi_weighted=0.8482027107` まで低下したため reject した。blend の low side も `0.8` を下回ると coverage を失い、October blend の局所最適は `0.8` 近傍とみなしてよい
  - ただし最良の `odds_max=10, top_k=1` でも `#101` baseline の `wf_nested_test_roi_weighted=3.9660920371` には遠く、今回の sweep は「October 高オッズ露出と pick concentration を詰める方向は有効だが Stage 1 parity gap は未解消」という確認に留まる
  - current next action は `odds_min` plateau (`1.1-1.19`) と `min_expected_value=1.00` 局所最適を閉じた前提で、October best (`odds_min=1.1-1.19`, `odds_max=10`, `top_k=1`, `blend_weight=0.8`, `min_expected_value=1.0`) を固定したまま別の非 no-op policy lever を 1 本だけ試すことである

### `#123` JRA-equivalent trust completion gate

- role: NAR top-level completion gate
- status: open, not the day-to-day execution entrypoint
- source-of-truth: GitHub issue `#123`

## Execution Order

1. `#120` の provenance audit を current trust truth として保守する。
2. `#121` の source timing audit は fetch-timing diagnostic として保守し、`#120` の trust 判定には混ぜない。
3. `#101` の formal rerun artifact は current benchmark reference として保守し、pointer / versioned summary / manifest を一次参照にする。
4. `#103` を次の measurable hypothesis として進める。
5. `#122` の future-only readiness track は live capture / result arrival 用の operator path として独立に保守する。
6. `#124` の human review decision を待つ。
7. JRA 本線を再開する場合だけ、review 結果の後に次の 1 measurable hypothesis を選ぶ。

## Reading Order

### JRA current read

1. GitHub issue `#124`
2. [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md)
3. [jra_pruning_stage7_implementation_review_checklist.md](jra_pruning_stage7_implementation_review_checklist.md)

### NAR current read

1. [nar_jra_parity_issue_ladder.md](nar_jra_parity_issue_ladder.md)
2. [../artifacts/reports/local_nankan_provenance_audit.json](../artifacts/reports/local_nankan_provenance_audit.json)
3. [../artifacts/reports/local_nankan_source_timing_audit.json](../artifacts/reports/local_nankan_source_timing_audit.json)
4. [../artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](../artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json)
5. [../artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json)
6. [../artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json](../artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json)
7. [../artifacts/reports/local_nankan_data_status_board.json](../artifacts/reports/local_nankan_data_status_board.json)
8. GitHub issue `#122`

## Boundaries

- completed issue history はここへ戻さない。
- `next_issue_*.md` を current queue として直接読まない。
- current queue に必要な情報だけ残し、長い historical explanation は issue thread または snapshot へ置く。