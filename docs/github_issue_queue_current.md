# GitHub Issue Queue: Current

## 1. Purpose

この文書は、今すぐ GitHub issue に起こすべき案件を、GitHub template にそのまま貼れる粒度で整理した current queue である。

`docs/initial_issue_backlog.md` が中長期 backlog であるのに対して、こちらは直近で着手・追跡する issue の正本として使う。

## 2. Why GitHub

現在の `nr-learn` は docs と artifact が十分育ってきたため、次の段階では GitHub issue を execution source of truth にしたほうがよい。

- 進行中 / blocked / done を board 上で見分けやすい
- experiment と ops を混ぜずに追跡できる
- artifact path と decision summary を issue thread に残せる
- AI coding の着手単位を固定しやすい

## 3. Current Issue Set

### 3.0 Current Queue As Of 2026-04-05

2026-04-05 時点の current queue は、JRA latest の source-of-truth docs と latest formal rerun の再現導線を揃えた状態を前提に、次のように固定する。

- `current_recommended_serving_2025_latest` は引き続き operational baseline である。
- `current_best_eval_2025_latest` は `r20260405_2025latest_refresh_reuse_runtimecfg_wfprofilefix` により formal refresh が `pass / promote` まで再現できる evaluation mainline reference である。
- current active priority は、新しい broad 実行を足すことではなく、docs / queue の drift を抑えつつ future option を JRA baseline 本線と切り分けることである。
- したがって JRA / NAR の新しい重い execution は、まず 1 issue = 1 measurable hypothesis の GitHub issue を明示してから再開する。
- `next_issue_*.md` 群の多くは reference draft library として保持しており、この節または `## 4. Execution Order` に明示したものだけを current queue として扱う。

Current reading:

1. JRA 本線は `#118` を current active experiment として再開した。
2. `recent_history_track_distance_selective` は `#96`, role split は `#97` まで進み、next candidate ではなく historical completed reference である。
3. `owner_signal_ablation_audit` は actual execution と hold decision まで確定しており、secondary draft ではなく historical decision reference である。
4. `jockey-trainer combo style-distance` は `#108` の formal child と `#115` の role split まで完了しており、historical completed reference である。
5. NAR separate-universe line は future option として保持するが、JRA baseline の current queue には混ぜない。
6. 既存の NAR corrective / replay / parity 系 draft は archive/reference として残し、必要時のみ個別 issue で再開する。

Primary completed issues:

- latest 2025 serving baseline formalization は完了
- latest eval mainline formal refresh は完了
- latest wrapper reuse-compare pass-through、runtime-config restore、WF `--profile` compatibility は完了
- public / internal source-of-truth docs の同期は完了

Archive note:

- 以降の `3.1+` は historical draft / template library を含む。active queue として読むのは `3.0` と `## 4. Execution Order` を優先する。

Primary next execution order:

1. source-of-truth docs と queue summary が open / closed issue 状態と矛盾しないよう保守する
2. JRA 本線を再開する場合に限り、draft library から次の 1 measurable hypothesis issue を選ぶ
3. 新しい JRA hypothesis は 1 本ずつ進め、並列で open にしない
4. NAR side は future option として別 issue でのみ再開し、current JRA queue と混在させない

Latest promoted-candidate ordering:

1. `#93` で JRA `analysis-first promoted candidate` 群の横断順位を existing artifact だけで固定した
2. first operator reference:
   - `r20260330_surface_plus_class_layoff_interactions_v1`
3. secondary difficult-regime specialist:
   - `r20260403_pace_closing_fit_selective_v1`
4. compare-reference only:
   - `r20260403_gate_frame_course_regime_extension_v1`
   - `r20260404_jockey_trainer_combo_closing_time_selective_v1`
5. 理由は次のとおり
   - `surface_plus_class_layoff` は low-exposure だが、September downside control と December positive carry を同時に残す唯一の line
   - `pace_closing_fit` は formal top-line は弱いが、September difficult-regime specialist として最も clean で、December に実損を作らない
   - `gate_frame_course` と `combo_closing_time` は formal top-line は高いが、December control window を loss に反転させる
6. serving default と September fallback ordering は変えない

Latest reread completions:

- `surface_plus_class_layoff` widening reread は close
- `sep_guard` ordering reread も existing artifact だけで close 可能

Latest reread completions:

- `shared portfolio bottleneck` narrower regime split reread も existing artifact だけで close 可能

Latest actual-date role split:

1. `#79` gate/frame/course regime extension は formal `pass / promote` まで完了した
2. `#80` actual-date role split で、September difficult window は改善したが December control window は baseline 劣後と確定した
3. したがって `r20260403_gate_frame_course_regime_extension_v1` の operational role は `analysis-first promoted candidate` に固定し、serving default には上げない
4. `current_long_horizon_serving_2025_latest` を first seasonal alias、`current_sep_guard_candidate` を second seasonal fallback として据え置く

Latest fallback compare:

1. `#81` gate/frame/course September fallback compare では、candidate は broad September と late-September の両方で `current_sep_guard_candidate` に pure bankroll で劣後した
2. broad September:
   - sep guard `9 bets / -4.3 / pure bankroll 0.9995842542264094`
   - gate-frame-course candidate `5 bets / -3.6 / pure bankroll 0.8808888460219477`
3. late-September:
   - sep guard `6 bets / -6.0 / pure bankroll 0.9857901270555528`
   - gate-frame-course candidate `3 bets / -1.6 / pure bankroll 0.9319444444444444`
4. したがって `current_sep_guard_candidate` は September seasonal fallback のまま据え置き、`r20260403_gate_frame_course_regime_extension_v1` は compare reference に留める

Current open-work expectation:

1. JRA 本線の current active issue は `#118` gate-frame-course pace decomposition selective child read である
2. latest completed JRA policy issue は `#117` であり、`abs90` anchor 維持・`minprob005` / `odds25` reject まで確定した
3. `#118` の next move は `course_baseline_race_pace_front3f` / `course_baseline_race_pace_back3f` の coverage / buildability を current high-coverage baseline 上で first read することである
4. `gate/frame/course`, `pace/closing-fit`, `kelly runtime`, `class-rest-surface conditional`, `recent-history track-distance`, `deque_trim promotion decision`, `tighter policy seasonal narrowing` はこの節では next queue ではなく completed / historical reference として扱う
5. `current_sep_guard_candidate` は September seasonal fallback のまま据え置く
6. NAR side の open issue は `#101`, `#103` のみであり、current JRA queue とは混在させない
7. `docs/next_issue_nar_class_rest_surface_replay.md` は Stage 2 feature-family parity の future option であり、`#103` の後段に置く

Historical note:

1. `#83` / `#84` / `#85` の pace-closing-fit line は formal / role split / fallback compare まで完了した
2. `#92` の kelly runtime family は anchor challenger としては reject まで完了した
3. `#94` / `#95` の class-rest-surface conditional line は formal / role split まで完了した
4. `#96` / `#97` の recent-history track-distance line は formal / role split まで完了した
5. `#116` の deque trim promotion decision は `analysis-only retention` で close した
6. したがって、これらの draft は再利用可能な historical issue source ではあるが、2026-04-05 時点の current next issue draft ではない

Primary next issue draft:

- current JRA primary next issue draft は `docs/next_issue_gate_frame_course_pace_decomposition_selective_candidate.md` である
- `docs/next_issue_tighter_policy_seasonal_regime_narrowing.md` は `#117` 完了後の historical issue source として保持する
- JRA 本線は `#118 [experiment] Gate-frame-course pace decomposition selective child read` を active issue として進める
- NAR residual unissued draft library は `docs/next_issue_nar_class_rest_surface_replay.md` のみである
- ただし NAR residual draft は current next issue ではなく、`#101` と `#103` の後段候補である

Latest next feature queue:

1. `#117` 完了後の再選定結果として、current JRA next issue は `#118` gate-frame-course pace decomposition selective child read に固定した
2. `tighter policy search` family の season-aware narrowing 自体は close し、strict anchor `abs90` 維持で判断が終わった
3. 次段は `course_baseline_race_pace_front3f` / `course_baseline_race_pace_back3f` pair を narrow selective child として読み、gate/frame/course family の second child になりうるかを判定する

Latest combo track-distance selective read:

1. `#98` `jockey / trainer / combo` family の `track-distance` quartet child は formal `pass / promote` まで完了した
2. revision:
   - `r20260404_jockey_trainer_combo_track_distance_selective_v1`
3. evaluation:
   - `auc=0.8449560456255405`
   - `top1_roi=0.458794459525301`
   - `ev_top1_roi=0.832437430649735`
   - nested WF `kelly / no_bet / no_bet`
   - `wf_nested_test_roi_weighted=0.5974478647406058`
   - `wf_nested_test_bets_total=192`
4. formal:
   - `weighted_roi=2.1315595528260776`
   - `bets_total=1814`
   - `feasible_fold_count=2`
   - `metric_source_counts={test: 2}`
5. current judgment:
   - formal promoted candidate
   - operational role は未確定
6. next execution source:
   - `docs/next_issue_jockey_trainer_combo_track_distance_role_split.md`

Latest combo track-distance role split:

1. `#107` actual-date role split は確定した
2. broad September:
   - baseline `32 bets / -27.3 / pure bankroll 0.2958869306148325`
   - candidate `28 bets / -28.0 / pure bankroll 0.30568644833662867`
3. December control:
   - baseline `45 bets / +21.8 / pure bankroll 1.6711564921099862`
   - candidate `121 bets / +2.1999999999999957 / pure bankroll 0.4147571870121275`
4. したがって `r20260404_jockey_trainer_combo_track_distance_selective_v1` は formal `pass / promote` だが、operational role は `compare reference` に固定する
5. 理由は、broad September では near-flat で明確な difficult-window de-risk を示せず、December control では exposure 増加とともに `total net` / `pure bankroll` の両方で大きく劣後するため

Current active issue:

- JRA 本線の active issue は `#118`
- <https://github.com/Npieju/nr-learn/issues/118>
- latest completed JRA issue は `#117`
- <https://github.com/Npieju/nr-learn/issues/117>
- latest completed role split は `#115`
- <https://github.com/Npieju/nr-learn/issues/115>

Current active read:

1. current active JRA issue は `#118` gate-frame-course pace decomposition selective child read である
2. hypothesis は `course_baseline_race_pace_balance_3f` 単体 child の再実行ではなく、`course_baseline_race_pace_front3f` / `course_baseline_race_pace_back3f` の pair を narrow selective child として独立に読むことである
3. rationale は、gate/frame/course family の first child は formal `pass / promote` でも actual-date role が compare reference に留まり、front/back 分解はまだ independent issue として未評価だからである
4. first execution は heavy compare ではなく feature-gap / coverage read から始める
5. `#117` で strict policy anchor は `r20260329_tighter_policy_ratio003_abs90` に維持されたため、same-family threshold narrowing は reopen しない
6. したがって current next move は gate/frame/course family の second child hypothesis を 1 issue で検証し、buildable / no-op / rerun-worthy のいずれかへ固定することである

Blocked parallel track:

1. `#101` timestamped strict `pre_race_only` benchmark handoff は外部結果待ち
2. `result_ready_races=0`
3. NAR side の next move は result arrival 待ち

Queue sync template after NAR progress:

1. `#101` strict `pre_race_only` benchmark rebuild は <completed|blocked|failed>
2. result-ready support は `<rows> rows / <races> races`
3. `#103` value-blend architecture bootstrap は <completed|benchmark_ready|blocked|failed>
4. if `#103` completed:
   - revision `<revision>`
   - held-out formal `weighted_roi=<weighted_roi>`
   - `feasible_folds=<feasible_folds>/<total_folds>`
   - decision `<promote|hold|reject>`
5. next NAR stage は `<Stage 2 selective replay|keep blocked>`

Background read:

1. `#100` timestamped odds rebuild は close 条件を満たして close 済み
2. third cut の live recrawl で、strict `pre_race` row の実在を確認した
3. race list discovery:
   - `2026-04-04 .. 2026-04-10`
   - `24 races`
   - dates `2026-04-06`, `2026-04-07`
4. live overwrite recrawl:
   - [local_nankan_timestamped_recrawl_apr06_07.log](/workspaces/nr-learn/artifacts/logs/local_nankan_timestamped_recrawl_apr06_07.log)
   - `requested_ids=24`
   - `parsed=24`
   - `failed=0`
5. provenance audit:
   - [local_nankan_race_card_provenance_summary_apr06_07.json](/workspaces/nr-learn/artifacts/reports/local_nankan_race_card_provenance_summary_apr06_07.json)
   - [local_nankan_race_card_pre_race_only_apr06_07.csv](/workspaces/nr-learn/artifacts/reports/local_nankan_race_card_pre_race_only_apr06_07.csv)
6. confirmed read:
   - `pre_race_only_rows=281`
   - `post_race_rows=0`
   - `unknown_rows=731941`
   - strict `pre_race` rows cover `24 races`
7. したがって current backfilled benchmark は provenance strict filter で弾ける
8. `#101` first cut で strict subset materialization も通った
9. output:
   - [local_nankan_pre_race_only_materialize_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_only_materialize_summary.json)
10. materialization read:
   - `pre_race_only_rows=281`
   - `pre_race_only_races=24`
   - `result_ready_races=0`
   - `pending_result_races=24`
   - `ready_for_benchmark_rerun=false`
11. second cut として result-ready subset / primary materialize 導線も追加した
12. output:
    - [local_nankan_pre_race_ready_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_ready_summary.json)
    - [nar_pre_race_primary_materialize_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_primary_materialize_smoke.log)
13. confirmed read:
    - `status=not_ready`
    - `current_phase=await_result_arrival`
    - `result_ready_races=0`
    - `pending_result_races=24`
14. third cut として benchmark handoff wrapper も追加した
15. output:
    - [local_nankan_pre_race_benchmark_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json)
    - [nar_pre_race_benchmark_handoff_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_benchmark_handoff_smoke.log)
16. wrapper read:
    - `status=not_ready`
    - `current_phase=await_result_arrival`
    - `recommended_action=wait_for_result_ready_pre_race_races`
17. fourth cut として bounded wait も追加した
18. wait smoke:
    - [nar_pre_race_benchmark_handoff_wait_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_benchmark_handoff_wait_smoke.log)
19. timeout read:
    - `status=not_ready`
    - `attempts=1`
    - `waited_seconds=9`
    - `timed_out=true`
20. したがって `#101` の残作業は result arrival 後の first benchmark rerun に narrowed された
21. 並行 next issue は `pre_race` capture window expansion である
22. execution source:
    - [next_issue_nar_pre_race_capture_window_expansion.md](/workspaces/nr-learn/docs/next_issue_nar_pre_race_capture_window_expansion.md)
23. NAR を JRA と同水準の model-development surface へ上げる中期 ladder は
    - [nar_jra_parity_issue_ladder.md](/workspaces/nr-learn/docs/nar_jra_parity_issue_ladder.md)
24. post-readiness の first architecture issue は `#103`
    - <https://github.com/Npieju/nr-learn/issues/103>
25. execution source:
    - [next_issue_nar_value_blend_architecture_bootstrap.md](/workspaces/nr-learn/docs/next_issue_nar_value_blend_architecture_bootstrap.md)
26. `#102` first read:
    - race-list discovery を `2026-04-30` まで広げても `24 races` のまま
    - strict `pre_race` pool は `281 rows / 24 races`
    - dates:
      - `2026-04-06: 136`
      - `2026-04-07: 145`
27. したがって capture expansion の本丸は wider date range ではなく repeated recrawl cadence と coverage accumulation artifact である
28. `#102` second cut として capture coverage artifact を追加した
29. script:
    - [run_local_nankan_pre_race_capture_coverage.py](/workspaces/nr-learn/scripts/run_local_nankan_pre_race_capture_coverage.py)
30. output:
    - [local_nankan_pre_race_capture_coverage_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_coverage_summary.json)
    - [local_nankan_pre_race_capture_date_coverage.csv](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_date_coverage.csv)
    - [nar_pre_race_capture_coverage_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_capture_coverage_smoke.log)
31. confirmed read:
    - `status=capturing`
    - `current_phase=capturing_pre_race_pool`
    - `pre_race_only_rows=281`
    - `pre_race_only_races=24`
    - `result_ready_races=0`
    - `pending_result_races=24`
32. date coverage:
    - `2026-04-06: 136 rows / 12 races`
    - `2026-04-07: 145 rows / 12 races`
33. baseline compare:
    - `delta_pre_race_only_rows=0`
    - `delta_pre_race_only_races=0`
    - `added_dates=[]`
34. したがって operator read は `await_result_arrival` だけでなく `capturing_pre_race_pool` まで上がった
35. 次段は repeated recrawl cadence 自体を bounded loop / snapshot artifact として入れること
36. `#102` third cut として bounded repeated recrawl wrapper を追加した
37. script:
    - [run_local_nankan_pre_race_capture_loop.py](/workspaces/nr-learn/scripts/run_local_nankan_pre_race_capture_loop.py)
38. output:
    - [local_nankan_pre_race_capture_loop_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_loop_manifest.json)
    - [pass_001_coverage_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_snapshots/pass_001_coverage_summary.json)
    - [pass_001_date_coverage.csv](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_capture_snapshots/pass_001_date_coverage.csv)
    - [nar_pre_race_capture_loop_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_capture_loop_smoke.log)
39. smoke read:
    - `prepare rows=24`
    - `collect requested_ids=24 parsed=24 failed=0`
    - latest summary `pre_race_only_rows=562`
    - `pre_race_only_races=24`
    - `delta_pre_race_only_rows=281`
    - `delta_pre_race_only_races=0`
    - `added_dates=[]`
40. fourth cut の duplicate read:
    - strict `pre_race` full pool `562 rows / 24 races / 281 horses`
    - `races_with_duplicate_rows=24`
    - `mean_duplicate_factor=2.0`
    - latest-only dedupe `281 rows / 24 races`
41. したがって repeated recrawl cadence は current source horizon で unique race support を増やしていない
42. `#102` は negative read で close し、capture expansion は stop する
43. NAR provenance-defensible benchmark の primary path は `#101` result-arrival handoff に戻す
44. post-readiness architecture parity は引き続き `#103` を block で維持する
45. `#104` result-ready bootstrap handoff automation は close 条件を満たした
46. dedicated config:
   - `configs/data_local_nankan_pre_race_ready.yaml`
47. wrapper:
   - `scripts/run_local_nankan_result_ready_bootstrap_handoff.py`
48. smoke は `status=not_ready` だが、`bootstrap_command_plan=4 steps` と `runtime_configs` を manifest に残せている
49. したがって current internal blocker は再び external result arrival のみである
50. active issue は `#101` に戻し、`#103` は result-ready benchmark 到着まで blocked のまま維持する
51. `#105` pre-race readiness probe は close 条件を満たした
52. read-only monitoring は
   - `scripts/run_local_nankan_pre_race_readiness_probe.py`
   - `artifacts/reports/local_nankan_pre_race_readiness_probe_summary.json`
   を正本に使う
53. active issue は `#101` に戻す
54. `#106` watcher handoff は close 条件を満たした
55. bounded watcher の正本は
   - `scripts/run_local_nankan_readiness_watcher.py`
   - `artifacts/reports/local_nankan_readiness_watcher_manifest.json`
   を使う
56. active issue は `#101` に戻す
57. current NAR blocker を 1 file で読む canonical board は
   - `artifacts/reports/local_nankan_data_status_board.json`
   を使う
58. board の `readiness_surfaces` では
   - probe
   - pre-race handoff
   - bootstrap handoff
   - watcher
   をまとめて追えるため、`#101/#103` の current phase はまず board を読む

Latest actual-date role split:

1. `#97` recent-history track-distance role split は December control compare まで確定した
2. broad September:
   - baseline `32 bets / -27.3 / pure bankroll 0.2958869306148325`
   - candidate `28 bets / -28.0 / pure bankroll 0.30568644833662867`
3. December control:
   - baseline `45 bets / +21.8 / pure bankroll 1.6711564921099862`
   - candidate `121 bets / +2.1999999999999957 / pure bankroll 0.4147571870121275`
4. したがって `r20260404_recent_history_track_distance_selective_v1` は formal `pass / promote` だが、operational role は `compare reference` に近く、serving default / seasonal fallback のどちらにも上げない

Latest September fallback compare:

1. `#85` で `r20260403_pace_closing_fit_selective_v1` と `current_sep_guard_candidate_2025_latest` を broad September / late-September で比較した
2. broad September:
   - sep guard `9 bets / -4.3 / pure bankroll 0.9995842542264094`
   - pace-closing-fit `3 bets / -3.0 / pure bankroll 0.892891589506173`
3. late-September:
   - sep guard `6 bets / -6.0 / pure bankroll 0.9857901270555528`
   - pace-closing-fit `2 bets / -2.0 / pure bankroll 0.9184027777777779`
4. net だけ見れば pace-closing-fit は小さく見えるが、pure bankroll は両 window とも sep guard が上
5. したがって `current_sep_guard_candidate_2025_latest` は September seasonal fallback のまま据え置き、`r20260403_pace_closing_fit_selective_v1` は `analysis-first promoted candidate` に留める

Latest actual-date role split:

1. `#84` pace-closing-fit actual-date role split は September difficult window と December control window の compare で確定した
2. broad September:
   - baseline `32 bets / -27.3 / pure bankroll 0.2958869306148325`
   - pace-closing-fit candidate `3 bets / -3.0 / pure bankroll 0.892891589506173`
3. December control:
   - baseline `45 bets / +21.8 / pure bankroll 1.6711564921099862`
   - pace-closing-fit candidate `0 bets / 0.0 / pure bankroll 1.0`
4. したがって `r20260403_pace_closing_fit_selective_v1` は `analysis-first promoted candidate` に据え置き、serving default には上げない

Primary next issue draft:

- none

Latest tighter policy fallback compare:

1. `#89` で `current_tighter_policy_search_candidate_2025_latest` と `current_sep_guard_candidate_2025_latest` を broad September / late-September で比較した
2. broad September:
   - sep guard `9 bets / -4.3 / pure bankroll 0.9995842542264094`
   - tighter policy `9 bets / -4.3 / pure bankroll 0.8394653592417107`
3. late-September:
   - sep guard `6 bets / -6.0 / pure bankroll 0.9857901270555528`
   - tighter policy `6 bets / -6.0 / pure bankroll 0.7701189959490742`
4. `bets` と `total net` は両 window で同値だった
5. pure bankroll は両 window で `sep_guard` が明確に上だった
6. したがって fallback ordering は変えない
7. `current_sep_guard_candidate` を second seasonal fallback のまま据え置き、`current_tighter_policy_search_candidate_2025_latest` は third defensive option に留める

Latest combo reentry queue:

1. `#90` で JRA `jockey / trainer / combo` family を second child として再開した
2. first child `style + track-distance` は formal `pass / promote` まで到達したが、actual-date role は `analysis-first promoted candidate` に留まった
3. したがって broad child を繰り返さず、`closing_time_3f` pair
   - `jockey_last_30_avg_closing_time_3f`
   - `trainer_last_30_avg_closing_time_3f`
   だけを追加する narrow selective candidate を first execution source とする
4. first read は feature-gap / coverage で、low-coverage や no-op なら true component retrain へは進めない
5. `#90` の first read は clean
   - `priority_missing_raw_columns=[]`
   - `missing_force_include_features=[]`
   - `low_coverage_force_include_features=[]`
   - `jockey_last_30_avg_closing_time_3f non_null_ratio=0.99727`
   - `trainer_last_30_avg_closing_time_3f non_null_ratio=0.99635`
6. true component retrain / stack rebuild / formal compare まで完了した
7. revision `r20260404_jockey_trainer_combo_closing_time_selective_v1` は
   - `auc=0.8426169492248933`
   - `top1_roi=0.8082087836284203`
   - `ev_top1_roi=0.8213612324672338`
   - `wf_nested_test_roi_weighted=0.7632385120350111`
   - `wf_nested_test_bets_total=457`
   - formal `weighted_roi=1.2311149102465346`
   - `bets / races / bet_rate = 9806 / 57620 = 17.02%`
   で `pass / promote` に到達した
8. したがって `#90` は feature child hypothesis issueとして完了し、次は actual-date role split を切る
9. actual-date role split issue は `#91` として起票済みである
10. `#91` actual-date role split では、September difficult window は baseline 比で strong downside control だった一方、December control window は baseline に明確劣後した
11. したがって `r20260404_jockey_trainer_combo_closing_time_selective_v1` の operational role は `analysis-first promoted candidate` に固定し、serving default には上げない

Latest breeder signal read:

1. `#88` で `breeder_last_50_win_rate` 単体 add-on を feature-gap で読んだ
2. result:
   - `selected=True`
   - `present=True`
   - `non_null_ratio=0.18955`
   - `status=low_coverage`
3. `low_coverage_force_include_features=['breeder_last_50_win_rate']`
4. したがって breeder 単体でも current high-coverage line には薄すぎる
5. lineage family は引き続き primary な next bet に戻さない

Latest owner signal audit:

1. `#86` で `owner_last_50_win_rate` を current JRA high-coverage baseline から外した selective ablation を formal compare した
2. win / roi component の selected features は `109 -> 108` となり、owner は actual used set から外れた
3. evaluation summary:
   - `auc=0.8449770387983483`
   - `top1_roi=0.4597201921069117`
   - `ev_top1_roi=0.8105412402909629`
   - nested WF `3/3 no_bet`
   - `wf_nested_test_bets_total=0`
4. `#72` の short-circuit により revision gate は `hold`
5. したがって owner signal は current baseline で prune しない

Latest Kelly runtime queue:

1. `#92` で `kelly-centered runtime family` を次の JRA policy 本線として再開した
2. current promoted anchor は `r20260329_tighter_policy_ratio003_abs90`
3. candidate matrix は 3 本に固定する
   - `kelly_runtime_base25`
   - `kelly_runtime_minprob003`
   - `kelly_runtime_edge005`
4. first step は 3 candidate の `run_revision_gate.py --dry-run`
5. 受け入れ基準は
   - candidate matrix が明文化されている
   - 3 candidate の dry-run が通る
   - first formal candidate を 1 本に絞れる
6. dry-run first read は完了
   - `r20260404_kelly_runtime_base25_dryrun_v1`
   - `r20260404_kelly_runtime_minprob003_dryrun_v1`
   - `r20260404_kelly_runtime_edge005_dryrun_v1`
   はすべて current codepath で通過した
7. historical formal top-line は 3 candidate で同値
   - `weighted_roi=1.1038859989058847`
   - `bets_total=598`
   - `feasible_fold_count=5`
8. したがって first formal candidate は最も解釈しやすい `kelly_runtime_base25` に固定する
9. first formal candidate `r20260404_kelly_runtime_base25_v1` は完了
   - evaluation top-line は anchor と同値
   - held-out formal `weighted_roi=0.8702925541408022`
   - `bets_total=993`
   - `feasible_fold_count=5`
10. anchor `r20260329_tighter_policy_ratio003_abs90` 比では
    - held-out formal `weighted_roi=1.1042287961989103`
    - `bets_total=598`
    - `feasible_fold_count=5`
11. support は維持し bets は増えたが、primary KPI の formal ROI が明確に劣後した
12. したがって `#92` の判断は `reject as anchor challenger`
13. current active issue は `none`

Latest evaluate progress fix:

1. `#87` で JRA `run_evaluate.py` の `inference complete -> leakage audit started` 無音区間を corrective した
2. `scripts/run_evaluate.py` に `[evaluate post]` progress を追加し、post-inference phase を
   - `score sources ready`
   - `summary payload ready`
   - `post-inference phases finished`
   に分解した
3. smoke log `artifacts/logs/evaluate_post_inference_progress_gap_smoke.log` で `1/3 -> 2/3 -> 3/3` を確認した
4. calibration 自体は small tail single-class で失敗したが、progress fix の確認目的は達成した

### 3.0 Current Queue After Tail Micro-Cut Exhaustion

2026-03-29 時点で、Kelly runtime family (`#10`, `#11`, `#12`, `#13`)、seasonal ordering (`#14`, `#15`)、runtime broad reduction (`#7`)、supplemental materialization (`#16`)、feature-builder runtime (`#17`) は close 済みである。loader runtime の small safe cuts を進めた `#18` も wrap-up 段階にあり、current operational anchor は引き続き `r20260329_tighter_policy_ratio003_abs90` である。

この時点の current reading は次である。`#49` の jockey-trainer-combo family では first child `#50` が `pass / promote` に到達したが、`#53` の actual-date role split read で family anchor にはならないことが確認された。したがって直近の execution queue は `#52` の NAR separate-universe baseline formalization に寄せる。

Primary active issue:

- `#60`
- <https://github.com/Npieju/nr-learn/issues/60>
- `#49`
- <https://github.com/Npieju/nr-learn/issues/49>

Primary issue draft:

- `docs/nar_model_transfer_strategy.md`
- `docs/nar_bet_denominator_standard.md`
- `docs/next_issue_local_nankan_baseline_formalization.md`
- `docs/next_issue_nar_post_formal_read.md`
- `docs/next_issue_nar_wf_runtime_followup.md`
- `docs/next_issue_nar_class_rest_surface_replay.md`
- `docs/next_issue_nar_class_rest_surface_availability_audit.md`
- `docs/next_issue_nar_selection_fix_for_buildable_replay.md`
- `docs/next_issue_nar_jockey_trainer_combo_replay.md`
- `docs/next_issue_nar_wf_summary_path_alignment.md`
- `docs/next_issue_nar_gate_frame_course_replay.md`
- `configs/model_local_baseline_wf_runtime_narrow.yaml`
- `configs/features_local_baseline_class_rest_surface_replay.yaml`
- `#60` で long-running job の live progress を repo 内 log file に tee し、VS Code から確認できる operator path を作る
- `docs/next_issue_jockey_trainer_combo_regime_extension.md`

GitHub issue:

- `#60`
- <https://github.com/Npieju/nr-learn/issues/60>
- `#49`
- <https://github.com/Npieju/nr-learn/issues/49>

Primary next execution order:

1. NAR の next family は `docs/next_issue_nar_owner_signal_replay.md` で owner signal replay を narrow に試す
2. JRA の `#49` family は promoted-but-non-anchor read として一段止め、次の widening は NAR next family formal read の後に再判断する

Side universe track:

1. `#52` で NAR を `separate universe` baseline として formalize する
2. `bets / races / bet-rate` を mandatory read にする
3. JRA の process / gating を移植し、threshold や feature priority は NAR 専用に再検証する

### 3.1 [experiment] Tighter policy search frontier refinement

GitHub issue:

- `#2`
- <https://github.com/Npieju/nr-learn/issues/2>

Template:

- `Model Experiment`

Recommended labels:

- `experiment`
- `policy`
- `jra`

Body draft:

```md
Universe
JRA

Category
Policy

Objective
`tighter policy search` family の support frontier をさらに明確化し、`ROI`、`feasible folds`、`drawdown` のバランスが最もよい policy 設定帯を整理する。新しい exotic family を増やすのではなく、既存の strongest defensive family を formal に詰める。

Hypothesis
if `tighter policy search` family の threshold frontier を `ratio`, `min_bets_abs`, `min_prob`, `odds_max`, `min_expected_value` 周辺で狭く再探索する, then we can preserve defensive behavior while improving support clarity and possibly edge toward the ROI>1.20 north-star band, while keeping drawdown and role interpretation stable.

In-Scope Surface
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_minprob005.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_odds25.yaml`
- `scripts/run_revision_gate.py`
- `scripts/run_wf_threshold_sweep.py`
- compare / dashboard artifacts

Non-Goals
- 新しい feature family の導入
- staged family の大型追加
- broad baseline replacement の即時判断
- NAR policy への展開

Success Metrics
- `5/5` を維持する strictest anchor が `abs90` として説明できる
- drawdown / bankroll / bet volume を壊さない narrow frontier が見つかる
- next revision gate candidate を 1 本に絞れる

Eval Plan
- smoke: threshold sweep と existing compare artifact の読み直し、candidate shortlist 化
- formal: 有望候補のみ revision gate に載せ、September/December role を compare で再確認する

Validation Commands
- `python scripts/run_wf_threshold_sweep.py ...`
- `python scripts/run_revision_gate.py ... --dry-run`
- `python scripts/run_revision_gate.py ...`
- `python scripts/run_serving_profile_compare.py ...`

Expected Artifacts
- threshold frontier summary
- candidate shortlist
- revision gate artifact
- compare dashboard summary

Stop Condition
- support を増やすと drawdown / bankroll が悪化する
- role が曖昧になり baseline より説明しづらくなる
- same-family refinement より別 family 比較のほうが有望と判明する
```

Primary references:

- `docs/next_issue_tighter_policy_frontier.md`
- `docs/tighter_policy_frontier_execution.md`
- `docs/tighter_policy_candidate_matrix.md`

### 3.2 [ops] Revision gate duplicate-run prevention and artifact collision guard

GitHub issue:

- `#1`
- <https://github.com/Npieju/nr-learn/issues/1>

Template:

- `Model Experiment`

Recommended labels:

- `ops`
- `automation`
- `reliability`

Body draft:

```md
Universe
JRA

Category
Evaluation

Objective
同一 `revision` / artifact path に対する duplicate formal run を防ぎ、artifact collision や結果誤読を避ける。現在は同一 revision の `run_revision_gate.py` が並走しうるため、formal result の解釈リスクがある。

Hypothesis
if duplicate revision runs are explicitly blocked or surfaced early, then formal evaluation artifacts become easier to trust and operate, while preserving normal experiment throughput.

In-Scope Surface
- `scripts/run_revision_gate.py`
- related manifest / artifact naming rules
- 必要なら docs の execution standard

Non-Goals
- model policy の再設計
- benchmark 指標の変更
- GitHub Actions 全面導入

Success Metrics
- 同一 revision の並走が operator に明確に見える
- artifact collision を未然に防げる
- `planned / running / completed / failed` の解釈が曖昧でなくなる

Eval Plan
- smoke: same revision の duplicate invocation path を再現・検証する
- formal: collision guard と operator-facing message を docs と manifest に反映する

Validation Commands
- `python scripts/run_revision_gate.py ... --dry-run`
- `python scripts/run_revision_gate.py ...`
- `ps -af | rg 'run_revision_gate.py|run_evaluate.py'`

Expected Artifacts
- updated revision gate manifest behavior
- guardrail log messages
- docs update

Stop Condition
- 既存 operator flow を大きく壊す
- uniqueness rule が厳しすぎて legitimate rerun を阻害する
```

Why now:

- `r20260329_tighter_policy_ratio003_abs90` の formal run で、同一 revision の並走疑いが実際に発生した

### 3.3 [ops] Progress instrumentation real-run validation and message quality pass

GitHub issue:

- `#3`
- <https://github.com/Npieju/nr-learn/issues/3>

Template:

- `Model Experiment`

Recommended labels:

- `ops`
- `automation`
- `observability`

Body draft:

```md
Universe
JRA

Category
Evaluation

Objective
`scripts/run_*.py` へ入れた progress instrumentation を real run で spot check し、message quality と failure-phase logging を整える。dry-run では改善が見えているため、次は実ジョブでも stalled に見えないことを確認したい。

Hypothesis
if progress instrumentation is validated on real runs and message quality is tightened, then operators can distinguish healthy long-running execution from stalls more reliably, while keeping logs readable.

In-Scope Surface
- `scripts/run_*.py` progress instrumentation
- `docs/development_operational_cautions.md`
- `docs/progress_coverage_audit.md`

Non-Goals
- benchmark policy の変更
- NAR/JRA artifact の再定義
- unrelated logging refactor

Success Metrics
- representative real runs で start / phase / completion / failure point が読める
- message quality が coarse すぎる箇所を特定・修正できる
- operator が stuck / running を誤認しにくくなる

Eval Plan
- smoke: dry-run spot check の継続
- formal: representative real run を選んで progress と manifest exit を確認する

Validation Commands
- `python scripts/run_mixed_universe_readiness.py ...`
- `python scripts/run_mixed_universe_numeric_compare.py ...`
- `python scripts/run_revision_gate.py ...`

Expected Artifacts
- updated progress messages
- audit doc update
- spot-check notes

Stop Condition
- log volume が増えすぎて読みにくくなる
- progress 粒度の追加が処理本体より複雑になる
```

Primary references:

- `docs/development_operational_cautions.md`
- `docs/progress_coverage_audit.md`

## 4. Execution Order

今の順番は次で固定する。

1. source-of-truth docs と queue summary が current reading と矛盾しないかを保守する
2. JRA 本線は active issue が空のまま維持されているので、再開時だけ draft library から次の 1 measurable hypothesis issue を選ぶ
3. 並列で新しい hypothesis issue を足さない
4. NAR separate-universe は current JRA queue の外に置き、必要時だけ別 issue で再開する

## 5. Note

`gh` は利用可能であり、issue を先に立ててから実行する運用を前提にする。

- この文書は current queue の要約と draft library の境界を示す正本である。
- 実行開始前には `3.0` と `## 4. Execution Order` を見て、active issue が本当に current queue かを確認する。
- `3.1+` にある historical draft は再利用してよいが、そのまま active queue とは見なさない。
