# Next Issue: NAR Value-Blend Architecture Bootstrap

## Summary

parent completion gate:

- `#123 [nar] JRA-equivalent trust completion gate`
- `#103` の role は `Stage 1 architecture parity` を解く blocker issue であり、これ自体は NAR solved を意味しない

NAR を JRA と同水準の model architecture で比較できるようにするには、strict `pre_race_only` benchmark が成立した後に `value_blend` family を持ち込む必要がある。

current NAR は dataset readiness corrective が先行しており、LightGBM baseline line までは formalized されているが、JRA 本線の

- CatBoost win
- LightGBM ROI
- stack / `value_blend`

までは揃っていない。

したがって post-readiness の first measurable architecture issue は、NAR `value_blend` bootstrap である。

## Objective

strict `pre_race_only` NAR benchmark 上で、JRA current standard と同型の `CatBoost win + LightGBM ROI + value_blend stack` を bootstrap し、NAR baseline を architecture parity まで引き上げる。

## Hypothesis

if NAR strict `pre_race_only` benchmark 上で JRA-style `value_blend` architecture を train / evaluate できる, then NAR は LightGBM baseline ではなく JRA と同じ model-development surface で feature / policy compare を進められる。

## In Scope

- NAR 用 win / ROI / stack config
- component retrain 導線
- stack bundle 導線
- strict `pre_race_only` benchmark 上の formal compare

## Non-Goals

- provenance 不足の backfilled benchmark で architecture parity を主張すること
- JRA baseline 更新
- mixed-universe compare
- serving default promotion

## Success Metrics

- NAR strict `pre_race_only` benchmark で
  - CatBoost win retrain
  - LightGBM ROI retrain
  - `value_blend` stack bundle
  - formal evaluation
  が end-to-end で通る
- artifact / gate / actual-date compare の surface が JRA と揃う

## Validation Plan

1. strict `pre_race_only` benchmark rerun を baseline freeze として固定する
2. NAR 用 component config を用意する
3. component retrain
4. stack bundle
5. formal evaluation / promotion gate

## Stop Condition

- `#101` formal rerun artifact が失効し、benchmark reference を一次参照として維持できない
- strict `pre_race_only` benchmark の row / race support が architecture compare に足りない
- narrowed scaffold から child command を起動しても component train / stack / evaluate のどこかが再現不能
- その場合は benchmark reference を保守したまま `hold` か implementation corrective に切り分ける

## First Read

`#102` は negative read で close しており、`#101` formal rerun `r20260415_local_nankan_pre_race_ready_formal_v1` も trust-ready historical corpus 上で完了した。したがって `#103` の current role は blocked issue ではなく、completed benchmark reference を起点にした Stage 1 architecture bootstrap である。

current benchmark reference:

- pointer:
  - [evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](/workspaces/nr-learn/artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json)
- evaluation summary:
  - [evaluation_summary_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json)
- train metrics:
  - [train_metrics_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1.json](/workspaces/nr-learn/artifacts/reports/train_metrics_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1.json)

current read:

- train:
  - `auc=0.8649528380`
  - `logloss=0.2159883155`
  - `best_iteration=158`
- evaluate:
  - `auc=0.8625051675`
  - `logloss=0.2171821084`
  - `top1_roi=0.8402165859`
  - `ev_threshold_1_0_roi=2.8603751213`
  - `ev_threshold_1_2_roi=4.5726544989`
- nested WF:
  - `wf_nested_test_roi_weighted=3.9660920371`
  - `wf_nested_test_bets_total=778`
  - `wf_nested_completed=true`

ただし architecture 側は、JRA current standard をそのまま NAR に落とせば済むわけではない。`rich_high_coverage_diag` を local Nankan に対して feature-gap した結果、bootstrap の block は sample support だけでなく feature schema mismatch も含むと確定した。

実行:

- data config: `configs/data_local_nankan.yaml`
- feature config: `configs/features_catboost_rich_high_coverage_diag.yaml`
- model config: `configs/model_catboost_win_high_coverage_diag.yaml`
- `max_rows=50000`
- output:
  - [feature_gap_summary_nar_value_blend_bootstrap_first_read.json](/workspaces/nr-learn/artifacts/reports/feature_gap_summary_nar_value_blend_bootstrap_first_read.json)
  - [feature_gap_feature_coverage_nar_value_blend_bootstrap_first_read.csv](/workspaces/nr-learn/artifacts/reports/feature_gap_feature_coverage_nar_value_blend_bootstrap_first_read.csv)
  - [feature_gap_raw_column_coverage_nar_value_blend_bootstrap_first_read.csv](/workspaces/nr-learn/artifacts/reports/feature_gap_raw_column_coverage_nar_value_blend_bootstrap_first_read.csv)

confirmed read:

- template raw columns:
  - `25 / 25 present`
  - `priority_missing_raw_columns=[]`
- feature frame:
  - `feature_columns_total=87`
  - `selected_feature_count=52`
  - `categorical_feature_count=9`
- missing force-include features:
  - `course_gate_bucket_last_100_avg_rank`
  - `course_gate_bucket_last_100_win_rate`
  - `horse_class_change`
  - `horse_class_down_long_layoff`
  - `horse_class_down_short_turnaround`
  - `horse_class_up_long_layoff`
  - `horse_class_up_short_turnaround`
  - `horse_is_class_down`
  - `horse_is_class_up`
  - `horse_last_class_score`
  - `horse_surface_switch`
  - `horse_surface_switch_long_layoff`
  - `horse_surface_switch_short_turnaround`
  - `race_class_score`
- low coverage force-include features:
  - `[]`

一方で、次は rich config の中でも NAR にそのまま乗る。

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `gate_ratio`
- `frame_ratio`
- `owner_last_50_win_rate`

meaning:

- `#103` の main question は benchmark readiness ではなく architecture parity bootstrap に移った
- current local Nankan では、JRA `rich_high_coverage_diag` の full parity config をそのまま one-shot で持ち込むと force-include mismatch が残る
- したがって post-readiness の first bootstrap は
  - JRA stack / gate / artifact discipline を再利用しつつ
  - NAR buildable subset に narrowed した win / ROI / stack config
  から始めるのが妥当である

next cut:

- completed `#101` benchmark reference を baseline freeze として固定する
- NAR `value_blend` bootstrap 用 narrowed feature / win / ROI / stack config を実行面に乗せる
- first execution では component retrain -> stack bundle -> formal evaluation / promotion gate を revision 固有 artifact で閉じる

## Current Run

current execution revision:

- `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1`

current artifacts:

- win train:
  - [train_metrics_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.json](/workspaces/nr-learn/artifacts/reports/train_metrics_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.json)
- roi train:
  - [train_metrics_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.json](/workspaces/nr-learn/artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.json)
- stack train:
  - [train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.json](/workspaces/nr-learn/artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1.json)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_model_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_full_nested.json)
- fast feasibility summary:
  - [wf_feasibility_diag_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_fast_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_wf_fast_nested.json)
- promotion gate:
  - [promotion_gate_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_fast_gate.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1_fast_gate.json)

current read:

- component train は end-to-end で completed
- formal evaluation は `status=completed`, `stability_assessment=representative`
- evaluate:
  - `auc=0.7386067382`
  - `logloss=0.3906596985`
  - `top1_roi=0.7955827871`
  - `ev_threshold_1_0_roi=0.1506567164`
  - `wf_nested_test_roi_weighted=0.6579177603`
  - `wf_nested_test_bets_total=3429`
- fast feasibility gate read:
  - `wf_fold_count=3`
  - `wf_feasible_fold_count=0`
  - `dominant_failure_reason=min_bets`
  - `formal_benchmark_weighted_roi=null`
- promotion gate:
  - `status=block`
  - `decision=hold`
  - `blocking_reasons=[min_feasible_folds, formal_benchmark_weighted_roi_missing]`

meaning:

- `#103` の first execution は train / stack / formal evaluate まで完走したが、current policy/support では walk-forward feasible fold を作れなかった
- baseline `#101` に対して `auc`, `top1_roi`, `wf_nested_test_roi_weighted` のいずれも下回り、Stage 1 candidate として promote しない
- next cut は family promote ではなく、`min_bets` failure を直接減らす selection/support corrective へ narrowed する

## Support Corrective Follow-up

first hold の直後に、retrain は行わず stack artifact を再利用した policy-only support corrective を 1 hypothesis として切り出した。目的は `dominant_failure_reason=min_bets` を直接減らし、feasible fold が回復するかを見ることだけである。

current corrective artifacts:

- fast feasibility summary:
  - [wf_feasibility_diag_r20260416_local_nankan_value_blend_bootstrap_support_corrective_v1_wf_fast_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_r20260416_local_nankan_value_blend_bootstrap_support_corrective_v1_wf_fast_nested.json)
- bounded candidate runtime config:
  - [model_value_stack_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_support_corrective_candidate_v1.yaml](/workspaces/nr-learn/artifacts/runtime_configs/model_value_stack_local_nankan_value_blend_bootstrap_r20260416_local_nankan_value_blend_bootstrap_support_corrective_candidate_v1.yaml)
- bounded candidate evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_022c1411e21f36ad_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_022c1411e21f36ad_wf_full_nested.json)
- bounded candidate evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_022c1411e21f36ad_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_022c1411e21f36ad_wf_full_nested.json)

corrective hypothesis:

- reuse trained stack artifact from `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1`
- lower policy gate to `min_bet_ratio=0.03`, `min_bets_abs=90`
- crystallize the best fast-feasible candidate to
  - `strategy_kind=portfolio`
  - `blend_weight=0.8`
  - `min_prob=0.03`
  - `odds_max=25.0`
  - `top_k=1`
  - `min_expected_value=0.95`

current corrective read:

- fast feasibility:
  - `wf_fold_count=3`
  - `wf_feasible_fold_count=2`
  - fold `1/3` と `3/3` で feasible candidate が回復した
- bounded formal evaluate:
  - `status=completed`
  - `stability_assessment=representative`
  - `auc=0.7235740748`
  - `top1_roi=0.8005172090`
  - `ev_threshold_1_0_roi=0.1727832934`
  - `wf_nested_test_roi_weighted=0.8160725261`
  - `wf_nested_test_bets_total=3254`
  - `wf_nested_actual_folds=5`
  - `no_bet folds=2` (`fold 2`, `fold 5`)

meaning:

- support corrective によって first hold の `wf_feasible_fold_count=0` は崩せたため、`min_bets` block が policy/support surface に強く依存していたことは確認できた
- ただし bounded formal read でも `wf_nested_test_roi_weighted=0.8161 < 1.0` で、`#101` baseline `3.9661` と first `#103` run `0.6579` の間にとどまる
- `top1_roi` も first `#103` run の `0.7956` から微増に留まり、2/5 fold が `no_bet` のままなので Stage 1 candidate としてはなお promote しない
- next cut は support-only corrective の継続ではなく、fold 2/5 の no-bet を減らしつつ ROI を上げる新しい measurable hypothesis に切り替えるべきである

## First Cut

result-ready benchmark 到着前に、post-readiness rerun でそのまま使う narrowed scaffold config を追加した。

- feature config:
  - [features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- win config:
  - [model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- roi config:
  - [model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- stack config:
  - [model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml)

scaffold rule:

- JRA stack / gate / artifact discipline は再利用する
- force-include mismatch が出た class / surface / course-gate-bucket 群は入れない
- current local Nankan で buildable と確認できた subset に narrow する
- policy / policy_search は current local Nankan formal line に近い conservative setting を使う

included bootstrap subset:

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `gate_ratio`
- `frame_ratio`
- `owner_last_50_win_rate`

remaining blocker:

- strict `pre_race_only` benchmark reference 自体は `#101` formal rerun で確立済みだが、`#103` child execution 自体はまだ走らせていない
- remaining work は blocked-state 監視ではなく、この scaffold を起点に component retrain と stack bundle を実行し、formal read まで閉じることだけである

validation:

- config parse:
  - all 4 scaffold YAML files loaded successfully
- narrowed feature-gap:
  - [feature_gap_summary_nar_value_blend_bootstrap_narrowed_v1.json](/workspaces/nr-learn/artifacts/reports/feature_gap_summary_nar_value_blend_bootstrap_narrowed_v1.json)
  - [feature_gap_feature_coverage_nar_value_blend_bootstrap_narrowed_v1.csv](/workspaces/nr-learn/artifacts/reports/feature_gap_feature_coverage_nar_value_blend_bootstrap_narrowed_v1.csv)
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`

meaning:

- `#103` の first cut scaffold は local Nankan schema に整合している
- benchmark reference は存在するため、remaining uncertainty は config mismatch ではなく first formal execution の support / ROI / concentration read に narrowed された

## Ready-Run Plan

current `#103` の実行は、`#101` formal rerun artifact を benchmark reference として固定した上で進める。future-only operator path の board / wrapper は引き続き `#122` 側の live readiness surface として有効だが、historical trust-ready corpus 上の `#103` current blocker ではない。

current first read:

- benchmark reference:
  - [evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](/workspaces/nr-learn/artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json)
  - [evaluation_summary_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json)
  - [evaluation_manifest_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json)
- scaffold inputs:
  - [data_local_nankan_pre_race_ready.yaml](/workspaces/nr-learn/configs/data_local_nankan_pre_race_ready.yaml)
  - [features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
  - [model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
  - [model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
  - [model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml)

current execution options:

1. generic profile path
   - `run_train.py --profile local_nankan_value_blend_bootstrap_pre_race_ready`
   - component / stack / evaluate を revision 固有 suffix で実行する
2. wrapper-derived runtime-config path
   - [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json) の `runtime_configs` と `bootstrap_command_plan` を bootstrap scaffold の canonical command plan として再利用する

rule:

- current benchmark judgment は board の `await_result_arrival` ではなく `#101` pointer / summary / manifest を一次参照にする
- wrapper / status board が `not_ready` のままでも、それは `#122` future-only operator path の read であり、historical trust-ready corpus の `#103` blocker ではない
- child run は revision 固有 artifact を出し、baseline compare は必ず `#101` formal rerun reference に対して行う

future-only operator surfaces を見る必要があるときの canonical read は次である。

- board:
  - [local_nankan_data_status_board.json](/workspaces/nr-learn/artifacts/reports/local_nankan_data_status_board.json)
- primary fields:
  - `status`
  - `current_phase`
  - `recommended_action`
  - `readiness_surfaces.capture_loop`
  - `readiness_surfaces.readiness_probe`
  - `readiness_surfaces.pre_race_handoff`
  - `readiness_surfaces.bootstrap_handoff`
  - `readiness_surfaces.readiness_watcher`
  - `readiness_surfaces.followup_entrypoint`

rule:

- `#122` operator path を確認するときだけ、まず board を見る
- board が `status=partial`, `current_phase=future_only_readiness_track` でも、これは live capture/readiness の current read であり `#103` の historical benchmark blocker と混同しない
- refresh 完了直後の readiness-only 再確認は `readiness_surfaces.followup_entrypoint` の preview から follow-up oneshot に降りる
- 詳細が必要なときだけ個別 manifest に降りる

default surface:

- data config:
  - [data_local_nankan_pre_race_ready.yaml](/workspaces/nr-learn/configs/data_local_nankan_pre_race_ready.yaml)
- feature config:
  - [features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- win config runtime materialize source:
  - [model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- roi config runtime materialize source:
  - [model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- stack config runtime materialize source:
  - [model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml)
- wrapper:
  - [run_local_nankan_result_ready_bootstrap_handoff.py](/workspaces/nr-learn/scripts/run_local_nankan_result_ready_bootstrap_handoff.py)

operator command:

```bash
cd /workspaces/nr-learn
PYTHONPATH=/workspaces/nr-learn/src /workspaces/nr-learn/.venv/bin/python \
  scripts/run_local_nankan_result_ready_bootstrap_handoff.py \
  --wait-for-results \
  --poll-interval-seconds 300 \
  --run-bootstrap
```

expected phase contract:

1. benchmark reference fixed
  - `#101` pointer / summary / manifest が current baseline freeze として読める
2. scaffold ready
  - `runtime_configs` または profile-based command が child execution 可能である
3. bootstrap completed
  - win / ROI / stack / revision gate / local evaluate まで完走し、revision artifact を formal read に進める

primary artifacts to read:

- [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json)
- [local_nankan_pre_race_benchmark_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json)
- [benchmark_gate_local_nankan_pre_race_ready.json](/workspaces/nr-learn/artifacts/reports/benchmark_gate_local_nankan_pre_race_ready.json)
- [local_nankan_primary_pre_race_ready_materialize_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json)

go / no-go read:

- go:
  - `#101` benchmark reference が `completed` で読める
  - scaffold config / runtime config が current workspace で再現できる
  - `#103` child command が revision 固有 artifact を出せる
- no-go:
  - `#101` benchmark reference が壊れている
  - child command が non-zero exit で止まる
  - support / concentration read が Stage 1 compare に使えない

meaning:

- `#103` は now active issue であり、resume path ではなく first architecture bootstrap execution を閉じる段階にある
- future-only wrapper 自体は `#122` 側の operator surface として維持しつつ、historical benchmark compare は `#101` artifact reference に固定する
- `#103` 完了は `#123` に対する Stage 1 exit であり、Stage 2/3 parity と operator trust が残る限り NAR completion ではない

## Formal Read Contract

bootstrap 完走後の判定は、top-line ROI だけで読まず、[nar_formal_read_template.md](/workspaces/nr-learn/docs/nar_formal_read_template.md) と current NAR promotion threshold に揃える。

minimum read order:

1. `policy_bets`
2. `race_count`
3. `row_count`
4. `feasible_race_count`
5. `bet_rate_races`
6. `bet_rate_rows`
7. `bet_rate_feasible_races`
8. `weighted_roi`
9. `total_net`
10. `feasible_folds / total_folds`
11. `zero_bet_dates / total_dates`
12. concentration / low-support warning
13. `promote / hold / reject`

required compare:

- current NAR baseline line と bootstrap challenger を次で比較する
  - `bets / races`
  - `weighted_roi`
  - `total_net`
  - `wf_feasible_fold_count`
- read は suppression なのか real improvement なのかを明示する

decision floor:

- current NAR promotion gate では held-out formal `weighted_roi < 1.0` は `hold` で読む
- したがって `#103` でも、support があっても held-out formal `weighted_roi < 1.0` line は `promote` にしない
- `status=pass` と operational / issue decision を混同せず、必要なら `formal pass but hold` を明示する

minimum artifact set after completion:

- revision gate summary
- promotion gate report
- wf feasibility summary
- evaluation summary
- wrapper manifest / handoff manifest

issue summary template:

```text
Formal read:
- policy_bets: <bets>
- race_count: <races>
- row_count: <rows>
- feasible_race_count: <feasible_races>
- bet_rate_races: <bets/races>
- bet_rate_rows: <bets/rows>
- bet_rate_feasible_races: <bets/feasible_races>
- weighted_roi: <weighted_roi>
- total_net: <total_net>
- feasible_folds: <feasible_folds>/<total_folds>
- zero_bet_dates: <zero_bet_dates>/<total_dates>

Compare:
- baseline: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- challenger: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- delta_bets: <delta>
- delta_net: <delta>
- delta_weighted_roi: <delta>

Interpretation:
- <support / concentration / suppression read>

Decision:
- <promote|hold|reject>
- <if needed: formal promoted but operational hold>
```

meaning:

- current `#103` は execution と judgment surface の両方が fixed 済みである
- 次に必要なのは external result arrival ではなく、bootstrap revision 実行 -> formal artifact read -> issue decision summary である

## Issue Thread Templates

`#103` の issue thread には、phase ごとに次の template をそのまま使う。

### 1. Benchmark Reference Fixed, Execution Pending

```text
Current read:
- #101 benchmark reference: completed
- #103 scaffold status: ready
- recommended action: run first architecture bootstrap revision

Artifacts:
- evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json
- evaluation_summary_local_nankan_baseline_wf_full_nested.json
- evaluation_manifest_local_nankan_baseline_wf_full_nested.json
- local_nankan_result_ready_bootstrap_handoff_manifest.json

Read:
- Stage 0 benchmark reference is fixed on historical trust-ready corpus
- next work is component retrain / stack / formal read, not readiness waiting
```

### 2. Bootstrap Running / First Read

```text
Current read:
- #101 benchmark reference: completed
- #103 revision: <revision>
- current phase: <train_win|train_roi|build_stack|evaluate|gate>

Artifacts:
- benchmark reference pointer / summary / manifest
- runtime config set or profile name
- child log files

Execution:
- win: <config or runtime path>
- roi: <config or runtime path>
- stack: <config or runtime path>

Read:
- benchmark reference is fixed
- first architecture bootstrap is in progress
- next action is to finish child commands and read formal artifacts
```

### 3. Completed First Read

```text
Current read:
- wrapper status: completed
- benchmark handoff status: completed
- revision: <revision>

Formal read:
- policy_bets: <bets>
- race_count: <races>
- row_count: <rows>
- feasible_race_count: <feasible_races>
- bet_rate_races: <bets/races>
- bet_rate_rows: <bets/rows>
- bet_rate_feasible_races: <bets/feasible_races>
- weighted_roi: <weighted_roi>
- total_net: <total_net>
- feasible_folds: <feasible_folds>/<total_folds>
- zero_bet_dates: <zero_bet_dates>/<total_dates>

Compare:
- baseline: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- challenger: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- delta_bets: <delta>
- delta_net: <delta>
- delta_weighted_roi: <delta>

Interpretation:
- <support / concentration / suppression / real improvement read>

Decision:
- <promote|hold|reject>
- <if needed: formal pass but hold>

Artifacts:
- revision gate summary: <path>
- promotion gate report: <path>
- wf feasibility summary: <path>
- evaluation summary: <path>
- wrapper manifest: artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json
```

meaning:

- issue thread update も phase ごとに fixed された
- unblock 後は runbook, formal read, issue comment まで同じ template で進められる
