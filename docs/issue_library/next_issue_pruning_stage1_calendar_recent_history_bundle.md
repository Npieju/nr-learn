# Next Issue: Pruning Stage 1 Calendar Recent-History Bundle Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

`calendar context` 3 列と `recent-history core` 2 列は、それぞれ individual ablation では pruning candidate judgment まで完了した。

- calendar context:
  - `race_year`
  - `race_month`
  - `race_dayofweek`
- recent-history core:
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`

一方で、one-shot pruning bundle は `0/5 feasible folds` で `hold` になったため、「個別では安全」「全部まとめると formal support が崩れる」の間を埋める staged simplification read がまだ無い。

human review が staged path を許可するなら、最初の measurable hypothesis は semantic risk が最も低い最小 block から始めるべきである。

## Objective

calendar context 3 列と recent-history core 2 列を current JRA high-coverage baseline から同時に外した stage-1 bundle を formal compare し、pruning package の最初の段階導入 block が standalone で defend できるかを判定する。

## Hypothesis

if `race_year`, `race_month`, `race_dayofweek`, `horse_last_3_avg_rank`, `horse_last_5_win_rate` を同時除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then pruning package の staged simplification は最小 block から安全に開始できる。

## Current Read

- `docs/issue_library/next_issue_calendar_context_ablation_audit.md` は formal `pass / promote` と actual-date equivalence まで完了した
- `docs/issue_library/next_issue_recent_history_core_ablation_audit.md` は formal `pass / promote` と actual-date equivalence まで完了した
- `docs/issue_library/next_issue_pruning_bundle_ablation_audit.md` は same-day one-shot simplification を formal `hold` で閉じた
- `docs/jra_pruning_package_review_20260410.md` では、review が staged path を許可するなら `calendar + recent-history` から始めるのが自然と固定した

したがって next measurable hypothesis は、「support が最も強く semantic risk が最も低い 2 groups を先に束ねても still viable か」を narrow に読むことである。

## Candidate Definition

keep current JRA high-coverage baseline core and exclude only:

- `race_year`
- `race_month`
- `race_dayofweek`
- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_stage1_calendar_recent_history.yaml`

## In Scope

- stage-1 bundle 5 columns だけを外した high-coverage config 1 本
- feature-gap / coverage read
- no-op risk の事前確認
- buildable なら JRA true component retrain flow
- formal compare
- broad September 2025 と December control 2025 の actual-date compare

## Non-Goals

- full pruning package の再実行
- baseline config の即時 rewrite
- gate/frame/course 以降の他 group を同時追加すること
- owner signal keep decision の再審
- policy rewrite
- NAR work

## Success Metrics

- selected feature set から target 5 columns が消える
- win / ROI / stack の true retrain が clean に完走する
- formal compare で `status=pass` を維持できるか、または少なくとも one-shot bundle より support deterioration が軽いかを説明できる
- September / December actual-date compare で baseline との operational delta を説明できる

## Validation Plan

1. stage-1 bundle config を追加する
2. feature-gap / first read で buildability を確認する
3. true component retrain
4. stack rebuild
5. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...` または evaluate + wf feasibility + promotion gate の分離実行
6. broad September 2025 と December control 2025 の actual-date compare

## Stop Condition

- selected set から target 5 columns が十分に抜けず no-op が濃厚
- formal support が one-shot bundle と同様に崩れ、stage-1 narrowing の説明価値が薄い
- actual-date で September または December に regression が出る

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
  - `artifacts/reports/feature_gap_summary_pruning_stage1_calendar_recent_history_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_pruning_stage1_calendar_recent_history_v1.csv`
  - `artifacts/reports/feature_gap_raw_column_coverage_pruning_stage1_calendar_recent_history_v1.csv`
- log:
  - `artifacts/logs/feature_gap_pruning_stage1_calendar_recent_history_v1.log`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `empty_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=104`
  - `categorical_feature_count=37`
  - `force_include_total=29`

read:

- stage-1 bundle config は clean に buildable
- target 5 columns を同時除外しても selected feature set は `104` まで縮み、no-op ではない
- low coverage / missing force include blocker は見えていない
- したがって next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_pruning_stage1_calendar_recent_history_v1.json`
  - `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_pruning_stage1_calendar_recent_history_v1.json`
- metrics:
  - `auc=0.8400222863382493`
  - `logloss=0.20271735503207888`
  - `best_iteration=537`
  - `feature_count=104`
  - `categorical_feature_count=37`

read:

- win component は clean に完走した
- manifest 上も target 5 columns は残っていない
- leakage audit でも suspicious feature は検出されなかった

### ROI Component

- artifact:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_pruning_stage1_calendar_recent_history_v1.json`
  - `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_pruning_stage1_calendar_recent_history_v1.json`
- metrics:
  - `top1_roi=0.8868306801736612`
  - `best_iteration=100`
  - `feature_count=104`
  - `categorical_feature_count=37`

read:

- ROI component も clean に完走した
- ROI 側でも target 5 columns は actual used set に残っていない
- leakage audit でも blocker は見えていない

### Stack Build

- artifact:
  - `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_pruning_stage1_calendar_recent_history_v1.json`
  - `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_pruning_stage1_calendar_recent_history_v1.json`
- metrics:
  - `component_count=2`

read:

- stack も clean に bundle できた
- stage-1 bundle は end-to-end で no-op ではなく、formal compare へ進める状態になった

## Formal Compare

- revision:
  - `r20260410_pruning_stage1_calendar_recent_history_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_stage1_calendar_recent_history_v1_wf_full_nested.json`
  - `auc=0.8421756998485714`
  - `top1_roi=0.8017491369390104`
  - `ev_top1_roi=0.7465017261219793`
  - `wf_nested_test_roi_weighted=0.9736842105263156`
  - `wf_nested_test_bets_total=988`
  - `stability_assessment=representative`
- WF feasibility:
  - `artifacts/reports/wf_feasibility_diag_pruning_stage1_calendar_recent_history_v1.json`
  - `fold_count=5`
  - `feasible_fold_count=3`
  - `feasible_candidates_by_fold=[0, 0, 4, 4, 4]`
  - `dominant_failure_reason=min_bets`
  - `min_bets_required_range=947..995`
- promotion gate:
  - `artifacts/reports/promotion_gate_r20260410_pruning_stage1_calendar_recent_history_v1.json`
  - `status=pass`
  - `decision=promote`

read:

- one-shot pruning bundle の `0/5 feasible folds` と違い、stage-1 は `3/5 feasible folds` まで改善した
- fold 1-2 は依然として formal support が弱いが、mass removal をやめることで defendability は明確に回復した
- current gate 設定では stage-1 block は `pass / promote` を通過した

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_pruning_stage1_calendar_recent_history_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_sep25_pruning_stage1_calendar_recent_history_base_vs_sep25_pruning_stage1_calendar_recent_history_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_sep25_pruning_stage1_calendar_recent_history_base_vs_sep25_pruning_stage1_calendar_recent_history_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_pruning_stage1_calendar_recent_history_base_vs_sep25_pruning_stage1_calendar_recent_history_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
  - candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
  - `artifacts/reports/serving_smoke_profile_compare_dec25_pruning_stage1_calendar_recent_history_base_vs_dec25_pruning_stage1_calendar_recent_history_cand.json`
- compare summary:
  - `artifacts/reports/serving_smoke_compare_dec25_pruning_stage1_calendar_recent_history_base_vs_dec25_pruning_stage1_calendar_recent_history_cand.json`
- bankroll sweep:
  - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_pruning_stage1_calendar_recent_history_base_vs_dec25_pruning_stage1_calendar_recent_history_cand.json`
- result:
  - `shared_ok_dates=8`
  - `differing_score_source_dates=[]`
  - `differing_policy_dates=[]`
  - baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
  - candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision Rule

- `pass / promote` かつ actual-date equivalence を維持すれば、staged simplification の first executable block として retain する
- `hold` でも one-shot bundle より support が明確に改善するなら、bundle hold の主因が mass removal である可能性を強める reference として retain する
- `hold` かつ support 改善も無いなら、staged simplification は最小 block でも formal 支持が弱いと判断し、package 全体を documentation-only に寄せる

## Decision Summary

- `r20260410_pruning_stage1_calendar_recent_history_v1` は formal `pass / promote` を通過した
- actual-date role split でも broad September 2025 と December control 2025 の両方で baseline と完全同値だった
- one-shot pruning bundle の `hold` と違い、最小 staged block では `feasible_fold_count=3/5` まで support が改善した
- したがって `calendar context + recent-history core` は staged simplification の first executable block として supported と読める
- baseline config への実反映はなお human review 前提だが、review が staged path を許可するなら最初に採る block はこの組み合わせでよい

## Artifacts To Compare Against

- `artifacts/reports/promotion_gate_r20260408_calendar_context_ablation_v1.json`
- `artifacts/reports/promotion_gate_r20260408_recent_history_core_ablation_v1.json`
- `artifacts/reports/promotion_gate_r20260410_pruning_bundle_ablation_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_bundle_ablation_v1.json`

## Notes

- この issue は `docs/jra_pruning_package_review_20260410.md` の human review で staged path が許可された後の first execution source として使う
- 現時点では baseline rewrite の承認を意味しない