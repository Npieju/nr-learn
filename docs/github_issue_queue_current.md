# GitHub Issue Queue: Current

## Purpose

この文書は current open issue と次の実行順だけを保持する。completed history や長い判断経緯はここへ積まない。

詳細は GitHub issue thread、artifact、tagged snapshot/reference を見る。この file は「今どれを正本として追うか」を短く固定するための入口である。

## Current Priority

1. JRA benchmark 正本は維持する。
2. NAR は readiness track として分離して扱う。
3. current queue では open issue だけを追い、completed issue の説明は繰り返さない。
4. JRA 本線を再開するときは、`roadmap.md` の stage roadmap を先に更新し、ad-hoc probe を継ぎ足さない。
5. JRA 本線の current read は、Stage 4 bounded reintegration を `hold` で止めたうえで architecture branch を優先する。
6. market-aware probability-path first candidate は `reject`、prediction foundation support diagnostics も `reject` で閉じ、broader composition の first `support_preserving_residual_path` candidate は `hold` だった。その後の bounded residual calibration follow-up も `reject` で、representative / September actual-date ともに first candidate から有意改善を作れなかった。JRA architecture の次手は、support-preserving residual family をこれ以上 ad-hoc に掘らず、human review 後に別 issue として切る。

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
  - 同じ October best に対する `fractional_kelly=0.20` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `wf_nested_test_roi_weighted=0.8494208494` で `0.25` best と完全一致し、low-side Kelly もこの近傍では binding していなかった
  - October `min_edge=0.01` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228856`, `wf_nested_test_roi_weighted=0.8494208494` で現 best と完全一致し、edge gate もこの surface では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_ceef429c7fb166ef.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_ceef429c7fb166ef.json)
  - high-side の October `min_edge=0.02` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00/0.01` と完全一致し、edge gate は少なくとも `0.00-0.02` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_cda02df88bbbcfab.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_cda02df88bbbcfab.json)
  - high-side の October `min_edge=0.03` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00/0.01/0.02` と完全一致し、edge gate は少なくとも `0.00-0.03` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_71d6c305ea3c87d1.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_71d6c305ea3c87d1.json)
  - high-side の October `min_edge=0.04` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00/0.01/0.02/0.03` と完全一致し、edge gate は少なくとも `0.00-0.04` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_1309a93cd597f6a8.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_1309a93cd597f6a8.json)
  - high-side の October `min_edge=0.05` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00/0.01/0.02/0.03/0.04` と完全一致し、edge gate は少なくとも `0.00-0.05` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_ed9aeec951f5ab89.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_ed9aeec951f5ab89.json)
  - high-side の October `min_edge=0.06` 初回 run は [../artifacts/logs/r20260417_oct_sw_conc_odds10_oddsmin119_topk1_minedge006_v1_wf_fast.log](../artifacts/logs/r20260417_oct_sw_conc_odds10_oddsmin119_topk1_minedge006_v1_wf_fast.log) 上で `post-inference 1/3 -> 2/3 -> leakage audit complete -> calibration and walk-forward elapsed=0m30s` までは進んだが、long silent 区間を hang と誤判定して手動 interrupt したため結果は無効で、binding 判定には使わない
  - rerun した high-side の October `min_edge=0.06` は fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00/0.01/0.02/0.03/0.04/0.05` と完全一致し、edge gate は少なくとも `0.00-0.06` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_8901a490907710d8.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_8901a490907710d8.json)
  - high-side の October `min_edge=0.07` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00/0.01/0.02/0.03/0.04/0.05/0.06` と完全一致し、edge gate は少なくとも `0.00-0.07` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_5f8c9e0a79586942.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_5f8c9e0a79586942.json)
  - high-side の October `min_edge=0.08` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.07` と完全一致し、edge gate は少なくとも `0.00-0.08` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_b1798fd37a7fe8db.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_b1798fd37a7fe8db.json)
  - high-side の October `min_edge=0.09` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.08` と完全一致し、edge gate は少なくとも `0.00-0.09` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_e245ea12e3b1619e.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_e245ea12e3b1619e.json)
  - high-side の October `min_edge=0.10` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.09` と完全一致し、edge gate は少なくとも `0.00-0.10` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_92313602f2b64744.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_92313602f2b64744.json)
  - high-side の October `min_edge=0.11` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.10` と完全一致し、edge gate は少なくとも `0.00-0.11` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_b3d0a3819dacdcb2.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_b3d0a3819dacdcb2.json)
  - high-side の October `min_edge=0.12` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.11` と完全一致し、edge gate は少なくとも `0.00-0.12` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_f2572831d216e3a0.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_f2572831d216e3a0.json)
  - high-side の October `min_edge=0.13` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.12` と完全一致し、edge gate は少なくとも `0.00-0.13` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_d8a0365064fd8c0d.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_d8a0365064fd8c0d.json)
  - high-side の October `min_edge=0.14` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.13` と完全一致し、edge gate は少なくとも `0.00-0.14` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_7e83f61cc05fff89.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_7e83f61cc05fff89.json)
  - high-side の October `min_edge=0.15` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.14` と完全一致し、edge gate は少なくとも `0.00-0.15` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_c209f0056b061f9f.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_c209f0056b061f9f.json)
  - high-side の October `min_edge=0.16` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.15` と完全一致し、edge gate は少なくとも `0.00-0.16` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_0d7a3572c63b78ff.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_0d7a3572c63b78ff.json)
  - high-side の October `min_edge=0.17` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.16` と完全一致し、edge gate は少なくとも `0.00-0.17` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_2710eba733ebcc4b.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_2710eba733ebcc4b.json)
  - high-side の October `min_edge=0.18` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.17` と完全一致し、edge gate は少なくとも `0.00-0.18` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_ce6eaa0c5369a41d.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_ce6eaa0c5369a41d.json)
  - high-side の October `min_edge=0.19` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.18` と完全一致し、edge gate は少なくとも `0.00-0.19` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_bc45e85c660b1907.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_bc45e85c660b1907.json)
  - high-side の October `min_edge=0.20` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.19` と完全一致し、edge gate は少なくとも `0.00-0.20` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_fed9381ce3976aef.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_fed9381ce3976aef.json)
  - high-side の October `min_edge=0.21` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.20` と完全一致し、edge gate は少なくとも `0.00-0.21` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_8c625a3df4080f48.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_8c625a3df4080f48.json)
  - high-side の October `min_edge=0.22` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228855721394`, `wf_nested_test_roi_weighted=0.8494208494208492` で `0.00-0.21` と完全一致し、edge gate は少なくとも `0.00-0.22` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_130141bf68ed885c.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_130141bf68ed885c.json)
  - 構造要因として、WF optimizer は `min_edge` を kelly family の候補ループにしか適用しておらず、今回の October fold best は `0.00-0.22` を通して一貫して `portfolio` だった。したがって family winner が `portfolio` のままなら、`min_edge` を線形に増やしても weighted ROI が変わらない no-op を量産しやすい。以後は blind な high-side 線形 sweep を優先せず、まず family-aware な感度診断へ切り替える
  - `r20260418_oct_sw_familydiag_minedge022_v1` で `wf_family_diagnostics` を実 summary に出し、family 差を確認した。fold 1 は `portfolio score=0.9802` に対して best kelly `0.8102`、fold 2 は `0.7306` 対 `0.6119`、fold 3 は `1.2097` 対 `0.9022` で、すべて `portfolio` が明確に優位だった。特に October fold 3 の best kelly は `min_edge=0.22` まで上げても `bets=7`, `roi=0.8879`, `final_bankroll=0.9889` に留まり、support が細すぎて `portfolio` (`bets=93`, `roi=1.1581`) を崩せなかった。したがって current bottleneck は `min_edge` 境界未発見ではなく、October 局所解が kelly family ではなく portfolio family にあること自体である
  - high-side の October `fractional_kelly=0.30` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228856`, `wf_nested_test_roi_weighted=0.8494208494` で `0.20/0.25` と完全一致し、Kelly fraction は少なくとも `0.20-0.30` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_410d8950bfa30ac1.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_410d8950bfa30ac1.json)
  - low-side の October `max_fraction=0.01` も fold 3 `portfolio`, `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228856`, `wf_nested_test_roi_weighted=0.8494208494` で `0.02/0.03` と完全一致し、stake cap も少なくとも `0.01-0.03` では non-binding だった。versioned summary は [../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_f8dd6831ba827b3d.json](../artifacts/reports/evaluation_summary_value_stack_local_nankan_value_blend_bootstrap_r20260417_local_nankan_value_ble_f8dd6831ba827b3d.json)
  - low-side の October `min_expected_value=0.99` は fold 3 `portfolio` を維持したが、`test_roi=0.8439716312`, `test_bets=141`, `test_final_bankroll=0.9863184080`, `wf_nested_test_roi_weighted=0.8478781284` まで低下したため reject した。October EV gate の局所最適は high side / low side とも `1.00` 近傍にあるとみなしてよい
  - October `blend_weight=0.9` への follow-up も fold 3 `portfolio -> kelly`, `test_roi=0.7509631235`, `test_final_bankroll=0.7888704261`, `wf_nested_test_roi_weighted=0.8377221032` と悪化したため reject した
  - 中間点の `blend_weight=0.85` は fold 3 `portfolio` を維持したが、`test_roi=0.8150684932`, `wf_nested_test_roi_weighted=0.8455778622` でなお `0.8` best を下回った。現状では high-blend 側の局所境界は `0.8` 近傍とみなしてよい
  - 低側 probe の `blend_weight=0.75` は fold 3 が再び `no_bet` へ崩れ、`selection_reason=no_feasible_candidate`, `test_bets=0`, `test_final_bankroll=1.0`, `wf_nested_test_roi_weighted=0.8482027107` まで低下したため reject した。blend の low side も `0.8` を下回ると coverage を失い、October blend の局所最適は `0.8` 近傍とみなしてよい
  - ただし最良の `odds_max=10, top_k=1` でも `#101` baseline の `wf_nested_test_roi_weighted=3.9660920371` には遠く、今回の sweep は「October 高オッズ露出と pick concentration を詰める方向は有効だが Stage 1 parity gap は未解消」という確認に留まる
  - 同じ October local best (`blend_weight=0.8`, `min_edge=0.22`, `odds_min=1.19`, `odds_max=10`, `top_k=1`, `min_expected_value=1.0`) を保ったまま `score_regime_overrides` だけ外した control `r20260418_oct_no_scoreswitch_control_v1` では、fold 1/2 は不変だった一方、fold 3 は `portfolio/autumn_baseline_recovery` から `no_bet/default` に崩れた。valid 側の best portfolio も `roi=0.7376515152`, `bets=1320`, `final_bankroll=0.7784388996`, `max_drawdown=0.2402431222` と gate を満たせず、best kelly はさらに悪化して `final_bankroll=0.0013375540` だった。weighted ROI も `0.8494208494 -> 0.8482027107` へ低下した。versioned summary は [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_cf2ab65bc52b3ab0.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_cf2ab65bc52b3ab0.json)
  - ただし `autumn_baseline_recovery` で使っていた baseline model を score-source branch ではなく単独 main model として使う control `r20260418_oct_global_baseline_control_v1` では、fold 3 の October portfolio rescue はそのまま維持され、さらに fold 1/2 は `portfolio` ではなく `kelly` が勝ち、test ROI はそれぞれ `2.3957140289` / `0.8087050966`、nested weighted ROI は `1.4846934582` まで大きく改善した。`score_source_count=1` のまま fold 3 `portfolio/default` が `test_roi=0.8672413793`, `test_bets=116`, `test_final_bankroll=0.9904228856` を再現しているため、October rescue の本体は seasonal routing そのものではなく `autumn_baseline_recovery` で参照していた baseline model 側の優位である可能性が高い。versioned summary は [../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_b2c5197e765eadeb.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_b2c5197e765eadeb.json)
  - したがって current bottleneck の読みは更新が必要である。次に優先すべきは seasonal score-source branch を増やすことではなく、この baseline model 単独 control がなぜ value_blend mainline より広く強いのかを正面から切り分けることである。具体的には、`#101` baseline (`wf_nested_test_roi_weighted=3.9660920371`) との差分が model family 自体なのか、policy search surface / feature path / calibration / market blend 設定差なのかを順に潰し、October seasonal override は暫定 workaround として扱う
  - fast default-surface control `r20260418_baseline_default_surface_control_fast_v1` では、baseline model はそのままに `min_bet_ratio=0.05`, `min_bets_abs=100` と broad default search surface を戻した結果、nested weighted ROI は `1.4846934582 -> 2.3334769858` まで改善し、`#101` との差分の約 3 割を埋めた。fold 1 は `portfolio` (`blend_weight=0.6`, `min_prob=0.05`, `odds_max=40`, `test_roi=3.7818897638`)、fold 2 は `kelly` (`blend_weight=0.6`, `min_prob=0.05`, `odds_max=40`, `min_edge=0.01`, `test_roi=0.7745920467`) を選び、search surface が current control の主要ボトルネックの一つだったことは確認できた。versioned summary は [../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_a03bdbfbf8cb4c15.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_a03bdbfbf8cb4c15.json)
  - ただし同じ fast default-surface control でも fold 3 (`2025-10-02..2026-03-27`) は `no_bet` のままで、best valid portfolio は `roi=0.6818181818`, `bets=22`, `final_bankroll=0.9955214331`、best valid kelly は `roi=0.4115078567`, `bets=16`, `final_bankroll=0.9787665638` に留まり、October rescue は broad search だけでは復元できなかった。したがって残差は search surface 単独ではなく、October-local override か feature path / sample window 差に残っているとみなし、次仮説はそこへ絞る
  - その feature path 仮説を切るため、同じ baseline model / same broad search / same `max_rows=120000` / same constraints で `features_local_baseline.yaml` にだけ差し替えた control `r20260418_baseline_default_surface_baselinefeat_fast_v1` も実行したが、`auc`, `top1_roi`, `wf_nested_test_roi_weighted=2.3334769858`, fold winners (`portfolio`, `kelly`, `no_bet`)、fold 3 family diagnostics まで rich-feature control と全一致した。versioned summary は [../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_93fbff374d4e69b8.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_93fbff374d4e69b8.json)
  - したがって current evaluate-only path では inference-time の feature-config swap は実効差分になっていない。残差は baseline artifact 自体の train-time feature/model state、October-local override、または `#101` 側の sample window / fold geometry 差に残っていると読むべきで、次仮説はそこへ絞る
  - さらに broad default search を維持したまま October-local override だけ戻した control `r20260418_baseline_default_surface_october_override_fast_control_v1` も実行したが、nested weighted ROI はなお `2.3334769858` で不変、fold 3 winner も `no_bet` のままだった。versioned summary は [../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_b83eb7c2626b9ba8.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_b83eb7c2626b9ba8.json)
  - ただし fold 3 family diagnostics 自体は改善しており、October override 下の best portfolio は `blend_weight=0.8`, `min_prob=0.02`, `odds_min=1.19`, `odds_max=10.0`, `top_k=1`, `min_expected_value=1.0` で `roi=1.1580645161`, `bets=93`, `final_bankroll=1.0094049904`, `max_drawdown=0.0111104097` まで回復した。つまり economics は戻るが `min_bets_abs=100` を 7 bet 下回るため gate を通れず、winner は `no_bet` に据え置かれている
  - このため current residual の読みはさらに狭まる。October-local policy 自体は方向として正しいが、現 120k / 3-fold fast geometry では support gate を超えるだけの母数が足りない。次仮説は `#101` に近い sample window / fold geometry（例えば rows と fold 数）か、support gate に効く train-time artifact 差へ絞るべきで、feature-config swap や broad search の追加だけを続けても主要残差は埋まらない
    - その sample-window 仮説を切るため、同じ broad default search + October override を保ったまま `max_rows=200000` へ広げた control `r20260418_baseline_default_surface_october_override_fast_rows200k_v1` も実行した。結果は nested weighted ROI `3.0151249975`、fold winners は 3 fold すべて `kelly` となり、問題だった fold 3 も `valid_bets=140`, `test_bets=145`, `test_roi=0.3695799788`, `test_final_bankroll=0.7416863812` で abstain ではなく gate 通過側へ移った。versioned summary は [../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_9bb93cfbab898233.json](../artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260415_local_nankan_pre_race_ready_formal_v1_r20_9bb93cfbab898233.json)
    - したがって residual の主要因に support geometry が含まれることはほぼ確定である。一方で `3.0151` はなお `#101=3.9660920371` に届かず、200k 化だけでは parity は復元しない。次仮説は 5-fold/full geometry 自体の差、または baseline artifact の train-time state 差へ絞るべきで、inference-time feature swap や October override のみを続けても残差全体は埋まらない
  - 参考として、versioned by-date CSV 同士の比較では `2025-10-02..2026-03-27` の日次 `top1/ev_top1` 指標は不変だった。これは by-date artifact が global prediction summary であり、nested WF の fold-level policy selection 差分を表現していないためで、この問いの判定は fold summary を正本として扱う必要がある

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
7. JRA 本線を再開する場合だけ、review 結果の後に `roadmap.md` の current stage を確認し、その stage に対応する次の 1 measurable hypothesis を選ぶ。
8. current JRA next action は [issue_library/next_issue_jra_prediction_foundation_probability_support_diagnostics.md](issue_library/next_issue_jra_prediction_foundation_probability_support_diagnostics.md) を起点に、baseline `classification` surface の support cliff を formalizeすることである。policy reintegration の再開はその後に判断する。

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