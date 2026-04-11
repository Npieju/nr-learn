# Next Issue: Pruning Stage 8 Calendar Recent-History Gate/Frame/Course Track/Weather Class/Rest/Surface JT-ID Combo Dispatch-Meta Condition-Quartet Bundle Audit

## Summary

stage-7 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core + jockey/trainer/combo core + dispatch metadata` bundle は `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1` で formal `pass / promote` と actual-date Sep/Dec equivalence まで完了した。

これにより current remaining race-condition family は次の quartet に縮んだ。

- `競争条件`
- `リステッド・重賞競走`
- `障害区分`
- `sex`

full `race-condition / dispatch context` 6 列を stage-1 に早期追加した stage-2 line は `0/5 feasible folds` で `hold` だったが、stage-7 で dispatch metadata 2 列だけを分離すると `3/5 feasible folds` まで戻った。したがって next measurable hypothesis は、remaining quartet も supported branch の eighth block として分離可能かを測ることにある。

## Objective

stage-7 supported bundle に `競争条件`, `リステッド・重賞競走`, `障害区分`, `sex` を追加除外した stage-8 bundle を formal compare し、remaining quartet まで剥がしても supported branch が維持できるかを判定する。

## Hypothesis

if stage-7 で支持された 45-column bundle に `競争条件`, `リステッド・重賞競走`, `障害区分`, `sex` を追加除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then current race-condition family は dispatch metadata だけでなく remaining condition quartet まで staged simplification 上で safely separable である。

## Current Read

- `docs/issue_library/next_issue_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_bundle.md` は supported seventh block として formal `pass / promote`、`feasible_fold_count=3/5`、actual-date Sep/Dec equivalence まで完了した
- `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md` は full 6 列を second block として早期追加すると `0/5 feasible folds` に崩れることを示した
- `docs/issue_library/next_issue_race_condition_dispatch_context_ablation_audit.md` は same 6 列の individual ablation では formal `pass / promote` と actual-date equivalence を確認している
- したがって current ambiguity は remaining quartet 自体の必要性ではなく、「supported branch の終点が quartet 手前なのか、quartet 後なのか」である

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `race_year`
- `race_month`
- `race_dayofweek`
- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`
- `track`
- `weather`
- `ground_condition`
- `馬場状態2`
- `芝・ダート区分`
- `芝・ダート区分2`
- `右左回り・直線区分`
- `内・外・襷区分`
- `horse_days_since_last_race`
- `horse_days_since_last_race_log1p`
- `horse_is_short_turnaround`
- `horse_is_long_layoff`
- `horse_weight_change`
- `horse_weight_change_abs`
- `horse_distance_change`
- `horse_distance_change_abs`
- `horse_surface_switch`
- `horse_surface_switch_short_turnaround`
- `horse_surface_switch_long_layoff`
- `race_class_score`
- `horse_last_class_score`
- `horse_class_change`
- `horse_is_class_up`
- `horse_is_class_down`
- `horse_class_up_short_turnaround`
- `horse_class_down_short_turnaround`
- `horse_class_up_long_layoff`
- `horse_class_down_long_layoff`
- `jockey_id`
- `trainer_id`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `発走時刻`
- `東西・外国・地方区分`
- `競争条件`
- `リステッド・重賞競走`
- `障害区分`
- `sex`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet.yaml`

## In Scope

- stage-8 bundle 49 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- one-shot pruning bundle の再実行
- owner signal keep decision の再審
- baseline config の即時 rewrite
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 49 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、少なくとも stage-7 からの deterioration が許容範囲かを説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-8 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. formal compare
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 49 columns が十分に抜けず no-op が濃厚
- formal support が stage-7 より明確に悪化し、quartet 追加 block の説明価値が薄い
- actual-date で September または December に regression が出る

## Notes

- この issue は stage-7 supported branch の上に eighth block を足せるかを見る execution source である
- current race-condition family の最終 boundary を閉じることが目的である
- artifact suffix は長くなりやすいので、actual-date compare label は `ps8_condq` のような短縮 alias を前提にする
- 現時点では baseline rewrite の承認を意味しない

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
	- `artifacts/reports/feature_gap_summary_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.json`
	- `artifacts/reports/feature_gap_feature_coverage_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.csv`
	- `artifacts/reports/feature_gap_raw_column_coverage_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.csv`
- log:
	- `artifacts/logs/feature_gap_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.log`
- summary:
	- `priority_missing_raw_columns=[]`
	- `missing_force_include_features=[]`
	- `empty_force_include_features=[]`
	- `low_coverage_force_include_features=[]`
	- `selected_feature_count=60`
	- `categorical_feature_count=21`
	- `force_include_total=1`

read:

- stage-8 quartet bundle config は clean に buildable
- target 49 columns を除外しても selected feature set は `60` まで縮み、no-op ではない
- force include blocker も見えていない
- したがって next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
	- `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260411_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.json`
	- `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260411_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.json`
- metrics:
	- `auc=0.8397600195171315`
	- `logloss=0.20279964650857166`
	- `best_iteration=501`
	- `feature_count=60`
	- `categorical_feature_count=21`

### ROI Component

- artifact:
	- `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260411_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.json`
	- `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260411_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_v1.json`
- metrics:
	- `top1_roi=0.8474963820549922`
	- `best_iteration=129`
	- `feature_count=60`
	- `categorical_feature_count=21`

### Stack Build

- artifact:
	- `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260411_pruning_stage8_condq_v1.json`
	- `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260411_pruning_stage8_condq_v1.json`
- metrics:
	- `component_count=2`
	- `feature_count=60`
	- `categorical_feature_count=21`

## Formal Compare

- revision:
	- `r20260411_pruning_stage8_condq_v1`
- evaluation summary:
	- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260411_pruning_stage8_condq_v1_wf_full_nested.json`
	- `auc=0.8422288519737056`
	- `top1_roi=0.8064556962025317`
	- `ev_top1_roi=0.7503567318757192`
	- `wf_nested_test_roi_weighted=0.9622002820874471`
	- `wf_nested_test_bets_total=709`
	- `stability_assessment=representative`
- WF feasibility:
	- `artifacts/reports/wf_feasibility_diag_pruning_stage8_condq_v1.json`
	- `fold_count=5`
	- `feasible_fold_count=0`
	- `dominant_failure_reason=min_bets`
	- `min_bets_required_range=947..995`
- promotion gate:
	- `artifacts/reports/promotion_gate_r20260411_pruning_stage8_condq_v1.json`
	- `status=block`
	- `decision=hold`

read:

- top-line evaluate 自体は強く、AUC/ROI は stage-7 比でも崩れていない
- それでも WF feasibility は `0/5` で、best fallback も `232..616 bets` に留まり ratio-bound `min_bets` を満たせない
- current gate 設定では stage-8 quartet block は supported staged simplification と読めない

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260411_pruning_stage8_condq_v1` を replay-existing で読んだ。artifact 名は path 長制限を避けるため `ps8_condq` の短縮ラベルを使った。

### Broad September 2025

- compare summary:
	- `artifacts/reports/serving_smoke_compare_sep25_ps8_condq_base_vs_sep25_ps8_condq_cand.json`
- bankroll sweep:
	- `artifacts/reports/serving_stateful_bankroll_sweep_sep25_ps8_condq_base_vs_sep25_ps8_condq_cand.json`
- result:
	- `shared_ok_dates=8`
	- `differing_score_source_dates=[]`
	- `differing_policy_dates=[]`
	- baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
	- candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare summary:
	- `artifacts/reports/serving_smoke_compare_dec25_ps8_condq_base_vs_dec25_ps8_condq_cand.json`
- bankroll sweep:
	- `artifacts/reports/serving_stateful_bankroll_sweep_dec25_ps8_condq_base_vs_dec25_ps8_condq_cand.json`
- result:
	- `shared_ok_dates=8`
	- `differing_score_source_dates=[]`
	- `differing_policy_dates=[]`
	- baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
	- candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260411_pruning_stage8_condq_v1` は actual-date replay では broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- それでも WF feasibility は `0/5` で、promotion gate は `status=block`, `decision=hold` だった
- したがって staged simplification の defendable boundary は current reading では stage-7 で止まり、remaining quartet `競争条件`, `リステッド・重賞競走`, `障害区分`, `sex` を eighth block として足すのはまだ早い
- この結果により current race-condition family の remaining quartet は narrow hold boundary artifact として retain する