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

## ROI Weight Zero Diagnostic

baseline control 側の geometry 差分を切り分けた後、`#103` 本線では stack composition の 1 hypothesis だけを追加で切った。目的は `roi_weight=0.12` の ROI signal injection 自体が formal degradation の主要因かを、component retrain なしで直接検証することだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_roiweight0_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_roiweight0_diag.yaml)
- stack train metrics:
  - [train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_roiweight0_diag.json](/workspaces/nr-learn/artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_roiweight0_diag.json)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d2e1336206bff09d_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d2e1336206bff09d_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d2e1336206bff09d_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d2e1336206bff09d_wf_full_nested.json)
- wf progress:
  - [evaluation_wf_progress_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d2e1336206bff09d_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_wf_progress_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d2e1336206bff09d_wf_full_nested.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_roiweight0_diag_v1`
- component runtime configs は `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1` の win / ROI artifact を再利用
- stack だけ `roi_weight: 0.0` に変更
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- `auc=0.7374694358`
- `logloss=0.3906664924`
- `top1_roi=0.7957765745`
- `ev_threshold_1_0_roi=0.1519409038`
- `wf_nested_test_roi_weighted=0.6581096849`
- `wf_nested_test_bets_total=3428`
- nested 5 folds は completed、fold 4 だけ `no_bet`

meaning:

- first formal `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1` の `wf_nested_test_roi_weighted=0.6579177603` に対して実質不変であり、simple な `roi_weight` 除去だけでは主要 residual は解けなかった
- component quality と stack quality を切り分ける狙い自体は満たしたが、仮説の結果は negative read である
- 次の measurable hypothesis は `roi_weight` 単独 sweep を広げるのではなく、stack composition の別軸か train-time artifact 差へ 1 本に絞る

## Market Blend 0.1 Diagnostic

`roi_weight=0.0` が不変だった後、次の 1 hypothesis は artifact-level market anchoring そのものだった。目的は stack manifest 上の `market_blend_weight=0.97` が formal degradation の主要因かを、component retrain なしで直接切ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_diag.yaml)
- stack train metrics:
  - [train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_diag.json](/workspaces/nr-learn/artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_diag.json)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e97daaf7c33f5a4c_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e97daaf7c33f5a4c_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e97daaf7c33f5a4c_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e97daaf7c33f5a4c_wf_full_nested.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_marketblend010_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_marketblend010_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_marketblend010_diag_v1`
- component runtime configs は `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1` の win / ROI artifact を再利用
- stack だけ `market_blend_weight: 0.1` に変更し、`roi_weight: 0.12` は維持
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- `auc=0.8521846114`
- `logloss=0.2227738866`
- `top1_roi=0.7930065546`
- `ev_threshold_1_0_bets=189`
- `wf_nested_test_bets_total=0`
- `wf_nested_test_roi_weighted=null`
- nested 5 folds は completed、winner は `['no_bet', 'no_bet', 'no_bet', 'no_bet', 'no_bet']`
- revision gate は `evaluation_nested_all_no_bet_short_circuit` で `decision=hold`

meaning:

- post-inference ranking quality 自体は first formal や `roiweight0` diagnostic より大きく改善したが、それだけでは nested WF の support / feasibility を 1 fold も回復できなかった
- このため residual は simple な market anchoring の過多ではなく、train-time artifact 差、probability-market coupling、または current policy surface と score shape の structural mismatch に残っているとみなすべきである
- 次の measurable hypothesis は stack weight の追加 sweep ではなく、上の residual のどれか 1 本へさらに狭めて切る

## WF Blend Low Diagnostic

artifact-level `market_blend_weight=0.1` でも all-no-bet が続いたため、次の 1 hypothesis は nested WF 自身の market re-blend grid だった。目的は evaluate-time `blend_weights=[0.2, 0.4]` が support recovery を阻害しているのかを、同じ built artifact のまま直接切ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_wfblendlow_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_wfblendlow_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_dc52be1d94b9621e_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_dc52be1d94b9621e_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_dc52be1d94b9621e_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_dc52be1d94b9621e_wf_full_nested.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_marketblend010_wfblendlow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_marketblend010_wfblendlow_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_marketblend010_wfblendlow_diag_v1`
- built artifact は `marketblend010` diagnostic の stack bundle をそのまま再利用
- stack params は固定し、evaluation `policy_search.full.blend_weights` だけを `[0.0, 0.1, 0.2]` に変更
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- `auc=0.8521846114`
- `logloss=0.2227738866`
- `top1_roi=0.7930065546`
- `ev_threshold_1_0_bets=189`
- `wf_nested_test_bets_total=0`
- `wf_nested_test_roi_weighted=null`
- nested 5 folds は completed、winner は再び `['no_bet', 'no_bet', 'no_bet', 'no_bet', 'no_bet']`
- all folds の chosen `blend_weight` は `0.0` に落ちても、selection reason は全 fold `no_feasible_candidate`
- revision gate は `evaluation_nested_all_no_bet_short_circuit` で `decision=hold`

meaning:

- evaluate-time WF blend grid を 0.0 側へ広げても support / feasibility は 1 fold も回復しなかった
- このため current residual は「nested WF が market_prob と再 blend しすぎている」ことでは説明できず、train-time artifact 差か score shape / coupling の別の structural mismatch に残っている
- 次の measurable hypothesis は artifact weight や WF blend grid の単純 sweep をやめ、train-time artifact 差か probability-market coupling の別形へ 1 本に狭めるべきである

## Market Blend 0.1 Plus Support Corrective Diagnostic

WF blend low でも all-no-bet が続いたため、次の 1 hypothesis は「rank recovery 済み artifact に support-corrective surface を載せれば feasible fold が戻るか」だった。目的は policy strictness 自体が残差なのかを、same artifact / same components のまま直接切ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_supportcorrective_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_marketblend010_supportcorrective_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_5c8c85c698afa2ba_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_5c8c85c698afa2ba_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_5c8c85c698afa2ba_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_5c8c85c698afa2ba_wf_full_nested.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_marketblend010_supportcorrective_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_marketblend010_supportcorrective_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_marketblend010_supportcorrective_diag_v1`
- built artifact は `marketblend010` diagnostic の stack bundle をそのまま再利用
- stack params は固定し、policy surface だけ support-corrective candidate に変更
- `min_bet_ratio=0.03`, `min_bets_abs=90`
- `blend_weight=0.8`, `min_prob=0.03`, `odds_max=25.0`, `top_k=1`, `min_expected_value=0.95`
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- `auc=0.8521846114`
- `logloss=0.2227738866`
- `top1_roi=0.7930065546`
- `ev_threshold_1_0_bets=189`
- `wf_nested_test_bets_total=0`
- `wf_nested_test_roi_weighted=null`
- nested 5 folds は completed、winner は再び `['no_bet', 'no_bet', 'no_bet', 'no_bet', 'no_bet']`
- fold family diagnostics 上の best portfolio valid bets は `24/5/4/2/57` で、`min_bets_abs=90` を 1 fold も満たせなかった
- revision gate は `evaluation_nested_all_no_bet_short_circuit` で `decision=hold`

meaning:

- marketblend010 で rank を回復させても、support-corrective surface は nested WF feasible fold を 1 つも作れなかった
- したがって current residual は simple な support strictness ではなく、train-time artifact 差か score shape / probability-market coupling の別の structural mismatch に残っている
- 次の measurable hypothesis は policy-only corrective を続けるのではなく、train-time artifact 差か coupling 形状差へ 1 本に絞るべきである

## Support-Preserving Residual Path Diagnostic

marketblend010 + support-corrective でも all-no-bet が続いたため、次の 1 hypothesis は probability-market coupling の形状自体だった。目的は direct な final market blend をやめ、market residual だけを注入する `support_preserving_residual_path` が support recovery を戻せるかを、component retrain なしで直接切ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag.yaml)
- stack train metrics:
  - [train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag.json](/workspaces/nr-learn/artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag.json)
- model manifest:
  - [catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag_model.manifest.json](/workspaces/nr-learn/artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag_model.manifest.json)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json)
- wf feasibility summary:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1`
- component runtime configs は `r20260416_local_nankan_value_blend_bootstrap_pre_race_ready_formal_v1` の win / ROI artifact を再利用
- stack を rebuild し、`probability_path_mode: support_preserving_residual_path` に変更
- `market_residual_weight: 0.08`, `market_residual_scale: 0.75`, `market_blend_weight: 0.1`, `roi_weight: 0.12` を使用
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- `auc=0.6867873251`
- `logloss=0.6647491770`
- `top1_roi=0.8046622970`
- `ev_threshold_1_0_roi=0.1544464075`
- `ev_threshold_1_0_bets=3396`
- `wf_nested_test_roi_weighted=0.7245602799`
- `wf_nested_test_bets_total=6482`
- nested 5 folds は completed、winner は `['kelly', 'kelly', 'portfolio', 'portfolio', 'portfolio']`
- fold test ROI は `0.7380 / 0.3355 / 0.8234 / 0.8197 / 0.6227`

meaning:

- current branch で初めて nested all-no-bet を解消し、support-preserving path によって feasibility 自体は回復した
- 一方で post-inference quality は大きく悪化しており、weighted ROI も `0.7245602799` に留まるため、`#101` baseline `3.9660920371` との parity には遠い
- initial sidecar completion は後続 run が `evaluation_manifest.json` alias を上書きした後に promotion gate を読んだため、`wf_summary_matches_evaluation_config=false` の stale tuple mismatch block を返した。matching versioned manifest を使って promotion gate を再実行すると `status=pass`, `decision=promote`, `wf_feasible_fold_count=1`, `formal_benchmark_weighted_roi=0.8920618287`, `formal_benchmark_bets_total=3817` で整合した
- ただしこの `promote` は artifact-level gate が `min_feasible_folds=1` だけを要求し、formal weighted ROI threshold を持っていないためである。issue-level read としてはなお `0.7245602799` が baseline parity に遠く、この shape だけで `#103` を閉じられるわけではない
- したがって「support 不足は legacy coupling 形状にも起因する」という読みは得られたが、この shape だけで benchmark-parity 候補にはならない
- 次の measurable hypothesis は support-preserving artifact を固定し、policy surface を 1 本だけ再探索して recovered support をより高い weighted ROI へ変換できるかを見ることに絞る

## Support-Preserving Policy-Tight Diagnostic

support-preserving path で feasibility が戻った後、次の 1 hypothesis は selection surface の tightening だった。目的は recovered support が「過剰に広い policy surface」で薄まっているだけなら、same artifact 上で tighter な portfolio surface に絞ることで weighted ROI を押し上げられるかを見ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_policytight_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_policytight_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e5d8a0ce938f4d36_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e5d8a0ce938f4d36_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e5d8a0ce938f4d36_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e5d8a0ce938f4d36_wf_full_nested.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_policytight_diag_v1`
- built artifact は `supportpreserve_diag` stack bundle をそのまま再利用
- stack params は固定し、evaluation `policy_search.full` だけを tightened surface に変更
- `min_probabilities=[0.05]`, `odds_maxs=[25.0]`, `min_expected_values=[1.05]`
- `blend_weights=[0.2, 0.4]`, `min_edges=[0.01, 0.03, 0.05]` は維持
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- `auc=0.6867873251`
- `logloss=0.6647491770`
- `top1_roi=0.8046622970`
- `ev_threshold_1_0_roi=0.1544464075`
- `wf_nested_test_roi_weighted=0.7740222115`
- `wf_nested_test_bets_total=6213`
- nested 5 folds は completed、winner は `['portfolio', 'portfolio', 'portfolio', 'portfolio', 'portfolio']`
- fold test ROI は `0.8144 / 1.1538 / 0.9542 / 0.8350 / 0.6227`

meaning:

- support-preserving の first feasible read (`0.7245602799`) から weighted ROI は小幅改善し、late folds を含めて all-portfolio に収束した
- 一方で fold 1/3/4/5 は依然として 1.0 を下回り、`#101` baseline `3.9660920371` と比べると parity gap はなお大きい
- したがって recovered support を tighter policy で少し改善できることは確認できたが、残差はまだ policy tightening 1 本では埋まらない
- 次の measurable hypothesis も support-preserving artifact を固定しつつ、portfolio family の selective surface の別 residual 1 本だけを切るべきである

## Support-Preserving Odds-Tight Diagnostic

support-preserving artifact を固定した次の 1 hypothesis では、late folds の large-bet / low-ROI 残差を「高オッズ露出の多さ」とみなし、`odds_max` だけを `25.0 -> 15.0` へ絞った。目的は policytight candidate のまま high-odds side を cut すれば、support を大きく落とさず weighted ROI をもう一段引き上げられるかを確認することだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_394a8a16e7e757d0_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_394a8a16e7e757d0_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_394a8a16e7e757d0_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_394a8a16e7e757d0_wf_full_nested.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_v1.json)
- wf feasibility summary:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_v1.json)
- nested WF progress:
  - [evaluation_wf_progress.json](/workspaces/nr-learn/artifacts/reports/evaluation_wf_progress.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddstight_diag_v1`
- built artifact は `supportpreserve_diag` stack bundle をそのまま再利用
- stack params は固定し、evaluation `policy_search.full.odds_maxs` と serving `odds_max` だけを `15.0` に変更
- `min_prob=0.05`, `min_expected_value=1.05`, `blend_weights=[0.2, 0.4]`, `min_edges=[0.01, 0.03, 0.05]` は維持
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- post-inference は不変
  - `auc=0.6867873251`
  - `logloss=0.6647491770`
  - `top1_roi=0.8046622970`
- nested WF:
  - `wf_nested_test_roi_weighted=0.7670015985`
  - `wf_nested_test_bets_total=4412`
  - winners は `['portfolio', 'kelly', 'portfolio', 'kelly', 'portfolio']`
  - fold test ROI は `0.5205 / 0.8113 / 0.8451 / 0.7911 / 0.7721` で、fold 1 と fold 4 が特に悪化した
  - fold 4 は `test_final_bankroll=0.2449234672`, `test_max_drawdown=0.8067115091` まで崩れた

meaning:

- policytight candidate (`0.7740222115`) から weighted ROI はむしろ悪化し、all-portfolio の安定も失った
- 特に fold 1 `test_roi=0.5205` と fold 4 `kelly` 化は、`odds_max=15.0` が selective tightening ではなく family destabilization を起こしたことを示す
- sidecar 完了後の formal read では、WF feasibility summary は `wf_feasible_fold_count=0`, `dominant_failure_reason=min_bets`, `binding_min_bets_source_counts={'ratio': 5}`, `wf_max_infeasible_bets_observed=0` を返した。5 folds x 12 candidates の全 60 candidate が `min_bets` で落ちており、`odds_max=15.0` に絞った surface は support を 1 fold も回復できていない
- promotion gate も config tuple 一致のまま `status=block`, `decision=hold`, blocking reason `Walk-forward feasible fold count is below threshold: 0 < 3` で閉じたため、ここでの悪化は stale sidecar や path mismatch ではなく oddstight candidate 固有の結果として確定した
- したがって support-preserving artifact の current local optimum は high-odds tighten 側ではなく `odds_max=25.0` 近傍に残ると読むべきで、この residual は reject で閉じる
- 次の measurable hypothesis は `odds_max` の追加 tighten ではなく、別の policy/surface residual 1 本に絞るべきである

## Support-Preserving Min-Prob Relax Diagnostic

`odds_max=15` tighten が negative read だった後、次の 1 hypothesis は `policytight` surface のうち `min_prob` だけを `0.05 -> 0.03` へ戻すことだった。目的は `odds_max=25.0`, `min_expected_value=1.05` を維持したまま lower-probability tail を少しだけ reopen すれば、support-preserving artifact の recovered support を ROI 悪化なしに増やせるかを見ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d3b241202a1bcb87_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d3b241202a1bcb87_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d3b241202a1bcb87_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_d3b241202a1bcb87_wf_full_nested.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_v1`
- built artifact は `supportpreserve_diag` stack bundle をそのまま再利用
- `odds_max=25.0`, `min_expected_value=1.05`, `blend_weights=[0.2, 0.4]`, `min_edges=[0.01, 0.03, 0.05]` は維持
- `policy_search.full.min_probabilities` と serving `min_prob` だけを `0.03` に変更
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- post-inference は不変
  - `auc=0.6867873251`
  - `logloss=0.6647491770`
  - `top1_roi=0.8046622970`
- nested WF:
  - `wf_nested_test_roi_weighted=0.7764314432`
  - `wf_nested_test_bets_total=7213`
  - winners は `['portfolio', 'portfolio', 'portfolio', 'portfolio', 'portfolio']`
  - fold test ROI は `0.8792 / 0.9434 / 0.8198 / 0.8342 / 0.6227`
  - fold winners の params は early folds で `blend_weight=0.4`, late folds で `blend_weight=0.2` へ分かれたが、全 fold とも `portfolio` family を維持した

meaning:

- `supportpreserve_policytight_diag` (`0.7740222115`) に対して weighted ROI は `0.7764314432` へ微増し、`odds_max=15` tighten よりは明確に良い
- 一方で改善幅は `+0.0024092317` と小さく、fold 3-5 は依然として 1.0 未満が並び、`#101` baseline `3.9660920371` との parity gap は事実上不変である
- したがって current read は「support-preserving path では `min_prob=0.03` が `0.05` よりわずかに良いが、主要 residual を解くほどではない」であり、`min_prob` をさらに broad に sweep する優先度は高くない

formal sidecar read:

- [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_wf_full_nested.json) では `wf_feasible_fold_count=0`, `dominant_failure_reason=min_bets`, `failure_reason_counts_total={'min_bets': 60}`, `binding_min_bets_source_counts={'ratio': 5}`, `wf_max_infeasible_bets_observed=1` を返した
- 各 fold の `min_bets_required` は `456 / 463 / 478 / 485 / 490` で、closest infeasible はほぼ全 fold で `bets=0`、例外的に fold 2-3 の kelly fallback が `bets=1` を出しただけだった。つまり `min_prob` を `0.03` まで緩めても support surface は full-table history 上ほとんど開いていない
- [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_v1.json) は tuple-consistent に `status=block`, `decision=hold`, blocking reason `Walk-forward feasible fold count is below threshold: 0 < 1` を返した
- よって final read は「`min_prob` relax は nested evaluation の weighted ROI をわずかに押し上げるが、formal support recovery には全くつながらない」である。次の measurable hypothesis は `min_prob` sweep 継続ではなく、support を直接増やせる別の single policy residual に切るべきである

## Support-Preserving Min-EV Relax Diagnostic

`min_prob` relax でも full-table support が 0 fold のままだったため、次の 1 hypothesis は `policytight` surface のうち `min_expected_value` だけを `1.05 -> 1.00` へ緩めることだった。目的は EV floor のみを下げれば portfolio family の candidate density が増え、support-preserving artifact の formal feasibility を回復できるかを見ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_8fe1f14aefd07c20_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_8fe1f14aefd07c20_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_8fe1f14aefd07c20_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_8fe1f14aefd07c20_wf_full_nested.json)
- wf feasibility:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_v1.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_v1`
- built artifact は `supportpreserve_diag` stack bundle をそのまま再利用
- `blend_weights=[0.2, 0.4]`, `min_edges=[0.01, 0.03, 0.05]`, `min_prob=0.05`, `odds_max=25.0` は維持
- `policy_search.full.min_expected_values` と serving `min_expected_value` だけを `1.0` に変更
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- post-inference は不変
  - `auc=0.6867873251`
  - `logloss=0.6647491770`
  - `top1_roi=0.8046622970`
- nested WF:
  - `wf_nested_test_roi_weighted=0.7825910322`
  - `wf_nested_test_bets_total=6646`
  - `wf_nested_test_roi_mean=0.9008288830`
  - winners は `['portfolio', 'portfolio', 'portfolio', 'portfolio', 'portfolio']`
  - fold test ROI は `0.7993 / 1.3082 / 0.9535 / 0.8197 / 0.6234`
- formal sidecar:
  - `wf_feasible_fold_count=0`
  - `dominant_failure_reason=min_bets`
  - `failure_reason_counts_total={'min_bets': 60}`
  - `binding_min_bets_source_counts={'ratio': 5}`
  - `wf_max_infeasible_bets_observed=1`
  - promotion gate は `status=block`, `decision=hold`, blocking reason `Walk-forward feasible fold count is below threshold: 0 < 1`

meaning:

- `min_expected_value` relax は nested evaluation では `policytight` (`0.7740222115`) と `minproblow` (`0.7764314432`) の両方を上回り、scalar threshold loosen の中では最良だった
- それでも formal support は 1 fold も回復せず、closest / fallback candidate も最大 `bets=1` に留まった。つまり EV floor だけ下げても support geometry 自体はほぼ変わっていない
- したがって current read は「support-preserving path の failure mode は single-threshold loosening では解けず、bet count を構造的に増やす別軸 residual が必要」である
- 次の measurable hypothesis は `min_prob` や `min_expected_value` の追加 scalar sweep ではなく、`top_k` など portfolio density を直接変える single residual に切るべきである

## Support-Preserving Top-K Expansion Diagnostic

scalar threshold loosen が formal support を全く回復しなかったため、次の 1 hypothesis は `policytight` surface のうち `top_k` だけを `1 -> 2` へ増やすことだった。目的は portfolio family の bet density を直接増やせば、`min_bets` に支配された support failure を崩せるかを見ることだけである。

diagnostic config / artifacts:

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_9bdb31b7100187a0_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_9bdb31b7100187a0_wf_full_nested.json)
- evaluation manifest:
  - [evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_9bdb31b7100187a0_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_9bdb31b7100187a0_wf_full_nested.json)
- wf feasibility:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_v1.json)
- revision manifest:
  - [revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_v1.json)

diagnostic setup:

- revision: `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_v1`
- built artifact は `supportpreserve_diag` stack bundle をそのまま再利用
- `blend_weights=[0.2, 0.4]`, `min_edges=[0.01, 0.03, 0.05]`, `min_prob=0.05`, `odds_max=25.0`, `min_expected_value=1.05` は維持
- `policy_search.full.top_ks` と serving `top_k` だけを `2` に変更
- evaluate-only で `wf_mode=full`, `wf_scheme=nested`, `max_rows=200000` を実行

diagnostic read:

- post-inference は不変
  - `auc=0.6867873251`
  - `logloss=0.6647491770`
  - `top1_roi=0.8046622970`
- nested WF:
  - `wf_nested_test_roi_weighted=0.7665700950`
  - `wf_nested_test_bets_total=6213`
  - `wf_nested_test_roi_mean=0.8625876105`
  - winners は `['portfolio', 'portfolio', 'portfolio', 'portfolio', 'portfolio']`
  - fold test ROI は `0.8292 / 1.2403 / 0.7822 / 0.7647 / 0.6965`
- formal sidecar:
  - `wf_feasible_fold_count=0`
  - `dominant_failure_reason=min_bets`
  - `failure_reason_counts_total={'min_bets': 60}`
  - `binding_min_bets_source_counts={'ratio': 5}`
  - `wf_max_infeasible_bets_observed=1`
  - promotion gate は `status=block`, `decision=hold`, blocking reason `Walk-forward feasible fold count is below threshold: 0 < 1`

meaning:

- `top_k=2` は bet density を直接増やす意図だったが、nested evaluation 自体が `policytight` と `evlow` の両方より悪化し、support も 1 fold も回復しなかった
- closest infeasible / fallback も最大 `bets=1` に留まり、`top_k` expansion は candidate density を meaningful に増やしていない
- したがって current read は「support-preserving path の failure mode は threshold loosen だけでなく `top_k` expansion でも解けない」である
- 次の measurable hypothesis は `top_k` の追加 expansion ではなく、候補 universe 側を広げる `odds_max` relax など別軸 single residual に切るべきである

## Support-Preserving Odds Relax Diagnostics

`top_k` expansion も negative read だったため、次の hypothesis 群は candidate universe 側で `odds_max` を緩めることだった。狙いは low-probability / high-odds tail を少しだけ reopen すれば `min_bets` block が崩れるかを見ることである。

### Odds-Max 40 Diagnostic

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_b6fa31010aec680e_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_b6fa31010aec680e_wf_full_nested.json)
- wf feasibility:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag_v1.json)

read:

- nested WF:
  - `wf_nested_test_roi_weighted=0.7262348949`
  - `wf_nested_test_bets_total=3907`
  - winners は `['kelly', 'kelly', 'portfolio', 'portfolio', 'no_bet']`
- fold 5 は `no_bet` へ崩れ、fold 2 `test_roi=0.3355`, fold 4 `test_roi=0.7163` も悪い
- formal sidecar は `wf_feasible_fold_count=0`, `dominant_failure_reason=min_bets`, `status=block`, `decision=hold`

meaning:

- `odds_max=40.0` は universe expansion が過剰で、support を回復するどころか evaluation 自体を崩した
- この cut は reject とみなしてよい

### Odds-Max 30 Diagnostic

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_85c7d5b733231feb_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_85c7d5b733231feb_wf_full_nested.json)
- wf feasibility:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag_v1.json)

read:

- nested WF:
  - `wf_nested_test_roi_weighted=0.8165625528`
  - `wf_nested_test_bets_total=5923`
  - `wf_nested_test_roi_mean=0.9709020893`
  - winners は 5/5 `portfolio`
  - fold test ROI は `1.0580 / 1.2604 / 1.0340 / 0.8177 / 0.6939`
- formal sidecar:
  - `wf_feasible_fold_count=0`
  - `dominant_failure_reason=min_bets`
  - `failure_reason_counts_total={'min_bets': 60}`
  - `binding_min_bets_source_counts={'ratio': 5}`
  - `wf_max_infeasible_bets_observed=1`
  - promotion gate は `status=block`, `decision=hold`, blocking reason `Walk-forward feasible fold count is below threshold: 0 < 1`

meaning:

- `odds_max=30.0` はこの residual family で最良の evaluation read を返した。`policytight` `0.7740222115`, `minproblow` `0.7764314432`, `evlow` `0.7825910322`, `topk2` `0.7665700950`, `odds40` `0.7262348949` をすべて上回る
- それでも formal support は 1 fold も回復していないため、current blocker は依然 `min_bets` にある
- したがって current read は「support-preserving path の current evaluation local optimum は `odds_max=30.0` だが、formal support recovery には別の 1 変数がまだ必要」である
- 次の measurable hypothesis は `odds_max=30.0` を起点に `min_expected_value` か `min_prob` を再緩和する single residual である

### Odds-Max 30 + EV-Low Diagnostic

- config:
  - [configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag.yaml)
- evaluation summary:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e60f285b1d2a0e7f_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e60f285b1d2a0e7f_wf_full_nested.json)
- wf feasibility:
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_wf_full_nested.json)
- promotion gate:
  - [promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1.json)
- revision manifest:
  - [revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1.json)

read:

- nested WF:
  - `wf_nested_test_roi_weighted=0.8225407438`
  - `wf_nested_test_bets_total=6138`
  - `wf_nested_test_roi_mean=0.9688709062`
  - winners は 5/5 `portfolio`
- formal sidecar:
  - `wf_feasible_fold_count=0`
  - `wf_dominant_failure_reason=min_bets`
  - `wf_binding_min_bets_source_counts={'ratio': 5}`
  - `wf_max_infeasible_bets_observed=1`
  - fold 4 fallback は portfolio `bets=1`
  - fold 5 closest infeasible は `bets=0`
  - promotion gate は `status=block`, `decision=hold`, blocking reason `Walk-forward feasible fold count is below threshold: 0 < 1`

meaning:

- `odds30_evlow` は `odds30` `0.8165625528` を上回り、この residual family の current best evaluation read を更新した
- ただし formal support はなお 5 folds 全滅で、failure mode は依然として `min_bets` に固定されている
- `policytight`, `minproblow`, `evlow`, `topk2`, `odds40`, `odds30`, `odds30_evlow` の 7 本はいずれも `wf_feasible_fold_count=0` で閉じており、評価だけが少しずつ動いても formal gate は一度も開いていない
- したがって current read は「support-preserving path の local threshold / density / universe residual は exhausted であり、これ以上の近傍 sweep を続けても惰性になりやすい」である
- 次 action は single residual を追加することではなく、`#103` を structural residual へ切り直すか human review へ上げることである。少なくとも `odds30` 近傍の ad-hoc sweep はここで停止する
- その structural cut として [issue_library/next_issue_nar_value_blend_train_time_artifact_control.md](issue_library/next_issue_nar_value_blend_train_time_artifact_control.md) で定義した `supportpreserve_winonly_control` も formal 実行した。versioned summary は [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json)、WF summary は [../artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_wf_full_nested.json](../artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_wf_full_nested.json)、promotion gate は [../artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](../artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json)、revision manifest は [../artifacts/reports/revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](../artifacts/reports/revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json) である
- nested evaluation は `wf_nested_test_roi_weighted=0.7282222669`, `wf_nested_test_bets_total=5847`, winners `['kelly', 'kelly', 'portfolio', 'portfolio', 'portfolio']` を返し、support-preserving base `0.7245602799` はわずかに上回ったが、current best eval `odds30_evlow=0.8225407438` には届かなかった
- formal sidecar は `status=pass`, `decision=promote`, `wf_feasible_fold_count=1`, `formal_benchmark_weighted_roi=0.8562282347`, `formal_benchmark_bets_total=3733` を返したが、feasible fold は fold 5 のみで、`supportpreserve_diag` corrected sidecar の `formal_benchmark_weighted_roi=0.8920618287` を下回った。failure mode も引き続き `wf_dominant_failure_reason=min_bets`, `wf_binding_min_bets_source_counts={'ratio': 5}` だった
- したがって pure win-centric simplification だけでは current residual を縮められていない。train-time artifact family を 1 本切った価値はあったが、issue-level read としては `advance` ではなく `hold/reject` 寄りで閉じるべきで、次 action は別の structural residual を足す前に human review で `#103` の再定義境界を明文化することにある

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

latest diagnostic update:

- baseline model を使った evaluate-only compare `r20260419_baseline_default_surface_october_override_full_rows200k_v1_retry1` では、same runtime config / `max_rows=200000` のまま `wf_full_nested` を回すと `wf_nested_test_roi_weighted=3.9660920371`, `wf_nested_test_bets_total=778`, `wf_nested_completed=true` で `#101` formal baseline と一致した
- このため `r20260418_baseline_default_surface_october_override_fast_rows200k_v1` の `3.0151249975` は baseline artifact 劣化ではなく、主に `fast 3-fold` と `full 5-fold nested` の geometry 差に由来していたとみなしてよい
- current `#103` 読みでは、baseline control 側の追加 inference-time tweak を掘るより、`fast rows200k` を smoke/control 扱いに下げて、value-blend architecture bootstrap 本線の問いと切り分ける方を優先する

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
