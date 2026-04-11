# Next Issue: Pruning Bundle Ablation Audit

Historical note:

- この draft は artifact-based judgment まで完了している。
- current active queue の source draft ではなく、historical decision reference として扱う。

## Summary

`calendar context`, `gate/frame/course core`, `recent history core`, `jockey/trainer/combo core`, `class/rest/surface core`, `track/weather/surface context`, `race-condition/dispatch context`, `jockey/trainer ID core` は個別 audit ではすべて formal `pass / promote` と actual-date equivalence を通過し、pruning candidate judgment まで到達した。

次に問うべきは、これら individual equivalence が bundle でも保たれるかである。個別では harmless でも、同時除外で interaction が崩れるなら current serving family の簡約はそこで止めるべきである。

## Objective

個別に pruning candidate と判定済みの列群を current JRA high-coverage baseline から同時に外した bundle ablation を formal compare し、simplified family が still viable か、それとも interaction loss により hold へ戻るかを判定する。

## Hypothesis

if individually equivalent と判定された列群を bundle で同時除外しても formal gate と September / December actual-date compare が baseline 同等に保たれる, then these groups can advance from separate pruning candidates to a bundled simplification candidate for later human review.

## In Scope

- pruning candidate judgment 済み 8 groups の同時除外
- JRA true component retrain flow
- formal compare
- September difficult window と December control window の actual-date compare

## Non-Goals

- baseline default promotion
- owner signal keep decision の再審
- new add-on family
- NAR work
- human-review なしの serving default 差し替え

## Candidate Definition

keep current high-coverage baseline and exclude only the already-cleared pruning groups:

- calendar context 3 columns
- gate/frame/course core 4 columns
- recent history core 2 columns
- jockey/trainer/combo core 4 columns
- class/rest/surface core 20 columns
- track/weather/surface context 8 columns
- race-condition/dispatch context 6 columns
- jockey/trainer ID core 2 columns

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_pruning_bundle_ablation.yaml`

## Success Metrics

- component actual used feature set から target bundle columns が消える
- formal compare を 1 本作れる
- September / December actual-date compare で bundle simplification の可否を説明できる

## Validation Plan

1. bundle config を追加
2. feature-gap / first read で buildability を確認
3. true component retrain
4. stack rebuild
5. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`
6. September / December actual-date compare

## Stop Condition

- buildability はあるが selected set から想定 bundle が十分に抜けない
- formal support が崩れて simplification candidate として defend できない
- actual-date で September or December に明確な regression が出る

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
	- `artifacts/reports/feature_gap_summary_pruning_bundle_ablation_v1.json`
	- `artifacts/reports/feature_gap_feature_coverage_pruning_bundle_ablation_v1.csv`
	- `artifacts/reports/feature_gap_raw_column_coverage_pruning_bundle_ablation_v1.csv`
- log:
	- `artifacts/logs/feature_gap_pruning_bundle_ablation_v1.log`
- summary:
	- `priority_missing_raw_columns=[]`
	- `missing_force_include_features=[]`
	- `low_coverage_force_include_features=[]`
	- `selected_feature_count=60`
	- `categorical_feature_count=21`
	- `force_include_total=1`

read:

- bundle ablation config は clean に buildable
- selected feature set は `60` まで縮み、bundle 同時除外は明確に no-op ではない
- next step は true component retrain に進めてよい

## Acceptance Points

### Win Component

- artifact:
	- `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260410_pruning_bundle_ablation_v1.json`
	- `artifacts/models/catboost_win_high_coverage_diag_model.manifest_r20260410_pruning_bundle_ablation_v1.json`
- metrics:
	- `auc=0.8397600195171315`
	- `logloss=0.20279964650857166`
	- `best_iteration=501`
	- `feature_count=60`
	- `categorical_feature_count=21`

### ROI Component

- artifact:
	- `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260410_pruning_bundle_ablation_v1.json`
	- `artifacts/models/lightgbm_roi_high_coverage_diag_model.manifest_r20260410_pruning_bundle_ablation_v1.json`
- metrics:
	- `top1_roi=0.8474963820549922`
	- `best_iteration=129`
	- `feature_count=60`
	- `categorical_feature_count=21`

### Stack Build

- artifact:
	- `artifacts/reports/train_metrics_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_r20260410_pruning_bundle_ablation_v1.json`
	- `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model.manifest_r20260410_pruning_bundle_ablation_v1.json`
- metrics:
	- `component_count=2`
	- `feature_count=60`
	- `categorical_feature_count=21`

read:

- win / ROI / stack は end-to-end で完走した
- actual used set は owner signal を残した 60 features まで縮み、個別 pruning candidate 群は bundle としても selected set から消えた

## Formal Compare

- evaluation summary:
	- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_bundle_ablation_v1_wf_full_nested.json`
	- `auc=0.8422288519737056`
	- `top1_roi=0.8064556962025317`
	- `ev_top1_roi=0.7503567318757192`
	- `wf_nested_test_roi_weighted=0.9622002820874471`
	- `wf_nested_test_bets_total=709`
	- `stability_assessment=representative`
- wf feasibility:
	- `artifacts/reports/wf_feasibility_diag_pruning_bundle_ablation_v1.json`
	- `feasible_fold_count=0/5`
	- dominant gate failure reason: `min_bets`
	- fold-wise required minimum bets were ratio-bound: `947` to `995`
- promotion gate:
	- `artifacts/reports/promotion_gate_r20260410_pruning_bundle_ablation_v1.json`
	- `status=block`
	- `decision=hold`

read:

- top-line summary 自体は baseline 近辺を保った
- ただし formal gate では `5/5` folds が `min_bets` 不達で infeasible だった
- best fallback は fold ごとに高 ROI を持つが、`232` から `544` bets 程度しか出ず、current gate の ratio-bound minimum に届かない
- したがって bundled simplification candidate は formal promotion に進めない

## Actual-Date Role Split

actual-date compare は `current_recommended_serving_2025_latest` を baseline にし、right side だけ `r20260410_pruning_bundle_ablation_v1` の true retrain suffix を replay-existing で読んだ。

### Broad September 2025

- compare manifest:
	- `artifacts/reports/serving_smoke_profile_compare_sep25_pruning_bundle_base_vs_sep25_pruning_bundle_cand.json`
- compare summary:
	- `artifacts/reports/serving_smoke_compare_sep25_pruning_bundle_base_vs_sep25_pruning_bundle_cand.json`
- bankroll sweep:
	- `artifacts/reports/serving_stateful_bankroll_sweep_sep25_pruning_bundle_base_vs_sep25_pruning_bundle_cand.json`
- result:
	- `shared_ok_dates=8`
	- `differing_score_source_dates=[]`
	- `differing_policy_dates=[]`
	- baseline `33 bets / -20.0 / pure bankroll 0.3931722898269604`
	- candidate `33 bets / -20.0 / pure bankroll 0.3931722898269604`

### December Control 2025

- compare manifest:
	- `artifacts/reports/serving_smoke_profile_compare_dec25_pruning_bundle_base_vs_dec25_pruning_bundle_cand.json`
- compare summary:
	- `artifacts/reports/serving_smoke_compare_dec25_pruning_bundle_base_vs_dec25_pruning_bundle_cand.json`
- bankroll sweep:
	- `artifacts/reports/serving_stateful_bankroll_sweep_dec25_pruning_bundle_base_vs_dec25_pruning_bundle_cand.json`
- result:
	- `shared_ok_dates=8`
	- `differing_score_source_dates=[]`
	- `differing_policy_dates=[]`
	- baseline `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`
	- candidate `17 bets / -5.199999999999999 / pure bankroll 0.7886889523160848`

## Decision

pruning bundle simplification hypothesis は hold で close する。

理由:

- actual-date Sep/Dec は baseline と完全同値で、operational replay 上の regression は見えない
- それでも formal gate では `min_bets` 制約で `0/5 feasible folds` となり、bundle を single promoted simplification candidate として defend できない
- current reading では、individual pruning candidates をそのまま one-shot bundle promotion へ畳むのは早い

固定 wording:

- individual pruning candidate judgments はそのまま保持する
- bundled simplification は `analysis reference / hold` に留める
- 次手は bundle promotion ではなく、human review による pruning package judgment か、別 family の 1 measurable hypothesis 再選定とする