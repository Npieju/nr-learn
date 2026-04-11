# Next Issue: Pruning Stage 7 Calendar Recent-History Gate/Frame/Course Track/Weather Class/Rest/Surface JT-ID Combo Dispatch-Meta Bundle Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

stage-6 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core + jockey/trainer/combo core` bundle は `r20260411_pruning_stage6_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_v1` で formal `pass / promote` と actual-date Sep/Dec equivalence まで完了した。

一方で、`race-condition / dispatch context` 6 列を stage-1 に早期追加した stage-2 line は actual-date 同値を保ちながらも `0/5 feasible folds` で `hold` へ戻った。また stage-6 に同 family 全体を足すと、実質的には one-shot bundle hold に近い終点へ寄るため、full 6 列再実行の説明価値は薄い。

したがって next measurable hypothesis は、remaining family を narrower subgroup に分解し、まず dispatch metadata 側だけを supported branch の seventh block として足しても defendability が残るかを測ることに置く。

candidate add-on block:

- `発走時刻`
- `東西・外国・地方区分`

defer block:

- `競争条件`
- `リステッド・重賞競走`
- `障害区分`
- `sex`

## Objective

stage-6 supported bundle に `発走時刻`, `東西・外国・地方区分` を追加除外した narrower stage-7 bundle を formal compare し、dispatch metadata 2 列だけなら supported branch を壊さずにさらに一段進められるかを判定する。

## Hypothesis

if stage-6 で支持された 43-column bundle に `発走時刻`, `東西・外国・地方区分` を追加除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then remaining race-condition/dispatch family のうち dispatch metadata 2 列は seventh block として safely separable である。

## Current Read

- `docs/issue_library/next_issue_pruning_stage6_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_bundle.md` は supported sixth block として formal `pass / promote`、`feasible_fold_count=2/5`、actual-date Sep/Dec equivalence まで完了した
- `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md` は full race-condition/dispatch 6 列を second block として足すと `0/5 feasible folds` で `hold` に戻ることを示した
- `docs/issue_library/next_issue_race_condition_dispatch_context_ablation_audit.md` は same 6 列を baseline から単独除外した individual audit では formal `pass / promote` と actual-date equivalence を確認している
- したがって current ambiguity は「full 6 列 family が常に危険」ではなく、「supported branch 上でどの subgroup までなら support を維持できるか」である

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

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta.yaml`

## In Scope

- stage-7 bundle 45 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- `競争条件`, `リステッド・重賞競走`, `障害区分`, `sex` の同時追加
- full race-condition/dispatch 6 列 block の再実行
- one-shot pruning bundle の再実行
- owner signal keep decision の再審
- baseline config の即時 rewrite
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 45 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、少なくとも stage-6 からの deterioration が許容範囲かを説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-7 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. formal compare
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 45 columns が十分に抜けず no-op が濃厚
- formal support が stage-6 より明確に悪化し、dispatch metadata 追加 block の説明価値が薄い
- actual-date で September または December に regression が出る

## Notes

- この issue は stage-6 supported branch の上に seventh block を足せるかを見る execution source である
- branch-sensitive な failure 切り分けが目的であり、full race-condition family の再審ではない
- artifact suffix は長くなりやすいので、actual-date compare label は `ps7_dispatch_meta` のような短縮 alias を前提にする
- 現時点では baseline rewrite の承認を意味しない

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
	- `artifacts/reports/feature_gap_summary_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
	- `artifacts/reports/feature_gap_feature_coverage_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.csv`
	- `artifacts/reports/feature_gap_raw_column_coverage_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.csv`
- log:
	- `artifacts/logs/feature_gap_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.log`
- summary:
	- `priority_missing_raw_columns=[]`
	- `missing_force_include_features=[]`
	- `empty_force_include_features=[]`
	- `low_coverage_force_include_features=[]`
	- `selected_feature_count=64`
	- `categorical_feature_count=25`
	- `force_include_total=1`

read:

- stage-7 dispatch-meta bundle config は clean に buildable
- target 45 columns を除外しても selected feature set は `64` まで縮み、no-op ではない
- force include blocker も見えていない
- したがって next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
	- `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
	- `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
- metrics:
	- `auc=0.8394851021251964`
	- `logloss=0.2028767935630605`
	- `best_iteration=522`
	- `feature_count=64`
	- `categorical_feature_count=25`

read:

- win component は clean に完走した
- manifest 上も target 45 columns は actual used set に残っていない
- leakage audit でも suspicious feature は検出されなかった

### ROI Component

- artifact:
	- `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
	- `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
- metrics:
	- `top1_roi=0.9124457308248909`
	- `best_iteration=144`
	- `feature_count=64`
	- `categorical_feature_count=25`

read:

- ROI component も clean に完走した
- ROI 側でも target 45 columns は actual used set に残っていない
- leakage audit でも blocker は見えていない

### Stack Build

- artifact:
	- `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
	- `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1.json`
- metrics:
	- `component_count=2`
	- `feature_count=64`
	- `categorical_feature_count=25`

read:

- stack も clean に bundle できた
- stage-7 bundle は end-to-end で no-op ではなく、formal compare へ進める状態になった

## Formal Compare

- revision:
	- `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1`
- evaluation summary:
	- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1_wf_full_nested.json`
	- `auc=0.842043836749433`
	- `top1_roi=0.8029459148446491`
	- `ev_top1_roi=0.7295512082853856`
	- `wf_nested_test_roi_weighted=1.1925373134328359`
	- `wf_nested_test_bets_total=402`
	- `stability_assessment=representative`
- WF feasibility:
	- `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`
	- `fold_count=5`
	- `feasible_fold_count=3`
	- `dominant_failure_reason=min_bets`
	- `min_bets_required_range=947..995`
- promotion gate:
	- `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`
	- `status=pass`
	- `decision=promote`

read:

- stage-7 も formal `pass / promote` を維持した
- supported branch は dispatch metadata 2 列まで延長できる
- support margin は stage-6 `2/5 feasible folds` から stage-7 `3/5 feasible folds` へ戻った

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1` を replay-existing で読んだ。artifact 名は path 長制限を避けるため `ps7_dmeta` の短縮ラベルを使った。

### Broad September 2025

- compare summary:
	- `artifacts/reports/serving_smoke_compare_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`
- bankroll sweep:
	- `artifacts/reports/serving_stateful_bankroll_sweep_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`
- result:
	- `shared_ok_dates=8`
	- `differing_score_source_dates=[]`
	- `differing_policy_dates=[]`
	- baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
	- candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare summary:
	- `artifacts/reports/serving_smoke_compare_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
- bankroll sweep:
	- `artifacts/reports/serving_stateful_bankroll_sweep_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
- result:
	- `shared_ok_dates=8`
	- `differing_score_source_dates=[]`
	- `differing_policy_dates=[]`
	- baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
	- candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Summary

- `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- したがって `発走時刻`, `東西・外国・地方区分` は supported stage-6 branch の上でも separable な seventh block と読める
- current remaining race-condition family は `競争条件`, `リステッド・重賞競走`, `障害区分`, `sex` の quartet にまで縮んだ