# Next Issue: NAR Value-Blend Support-Preserving Residual Calibration

Current queue note:

- `#103` の support-preserving residual sweep は `odds30_evlow` まで掘って formal `wf_feasible_fold_count=0` が 7 本続き、ad-hoc policy residual としては exhausted で閉じている。
- train-time artifact family の first control `supportpreserve_winonly_control` も 2026-04-21 に formal 実行済みで、gate-level `promote` に対して issue-level read は negative/hold だった。
- したがって current next cut は family 追加ではなく、`support_preserving_residual_path` 内の bounded calibration 1 本に絞る。

## Summary

`#103` value-blend architecture bootstrap の current read は次で固定する。

- legacy blend から `support_preserving_residual_path` へ切り替えると、nested all-no-bet は解消され、corrected sidecar では `wf_feasible_fold_count=1`, `formal_benchmark_weighted_roi=0.8920618287` まで回復した
- その後の policy-only residual `policytight -> minproblow -> evlow -> topk2 -> odds40 -> odds30 -> odds30_evlow` は 7 本連続で `wf_feasible_fold_count=0` だった
- train-time artifact simplification `supportpreserve_winonly_control` は `wf_feasible_fold_count=1` を維持した一方、`formal_benchmark_weighted_roi=0.8562282347` で `supportpreserve_diag` を下回った

meaning は狭い。

- support-preserving family 自体は existence proof を持つ
- local threshold sweep と pure simplification は issue-level advance を作れなかった
- したがって next measurable hypothesis は、same family の中で symmetric market residual を bounded calibration する 1 軸だけである

## Objective

strict `pre_race_only` NAR benchmark 上で、`support_preserving_residual_path` を維持したまま market residual を `positive_only + lower_weight` に再較正した candidate を 1 本だけ定義し、support を落とさずに `supportpreserve_diag` より高い formal weighted ROI を作れるかを判定する。

## Hypothesis

if current `supportpreserve_diag` の弱さが support-preserving path 自体ではなく symmetric market residual の drag にある, then `market_residual_positive_only: true` と `market_residual_weight: 0.08 -> 0.05` の bounded calibration を入れれば、`wf_feasible_fold_count>=1` を維持したまま `formal_benchmark_weighted_roi` を `0.8920618287` より改善できる。

## In Scope

- `probability_path_mode: support_preserving_residual_path` は固定する
- calibration 差分は `market_residual_positive_only` と `market_residual_weight` に限定する
- win / ROI component artifact は current `#103` formal runtime config を再利用する
- full nested evaluate / wf feasibility / promotion gate までの formal compare を取る

## Non-Goals

- policy-only residual の追加 sweep
- `roi_weight`, component family, stack composition の再変更
- pure win-centric simplification の追加
- `#101` baseline freeze の変更
- issue definition 自体の再切り直しを同じ issue で進めること

## Why This Issue

current code path では `support_preserving_residual_path` が既に次をサポートしている。

- [../../src/racing_ml/evaluation/scoring.py](../../src/racing_ml/evaluation/scoring.py)
  - `market_signal = tanh((market_logit - win_logit) / market_residual_scale)`
  - `market_residual_positive_only` が true のとき residual を `max(signal, 0.0)` に clip できる
  - `market_residual_weight` を独立に調整できる

このため next cut を broad redesign にする必要はない。current residual はすでに実装済みレバーの bounded calibration で 1 本切れる。

さらに empirical read もその 1 hypothesis を支持している。

1. symmetric residual `0.08` では `wf_feasible_fold_count=1` を作れた
2. residual を完全に切った `winonly_control` では support は維持したが ROI が悪化した
3. つまり current bottleneck は「market residual があるかないか」ではなく、「どの符号と強度で入れるか」に narrowed されている

## Fixed Compare Surface

primary compare references は次で固定する。

1. baseline issue reference:
   - [next_issue_nar_value_blend_architecture_bootstrap.md](next_issue_nar_value_blend_architecture_bootstrap.md)
2. support-preserving base:
   - [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json)
   - [../artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json](../artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json)
3. best evaluation-only residual:
   - [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e60f285b1d2a0e7f_wf_full_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e60f285b1d2a0e7f_wf_full_nested.json)
4. train-time artifact control:
   - [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json)
   - [../artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](../artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json)

## Proposed Candidate

first candidate theme は `supportpreserve_positive_only_market_calibration` で固定する。

candidate config:

- [../../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_positiveonly_calibration_control.yaml](../../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_positiveonly_calibration_control.yaml)

minimal parameter delta from `supportpreserve_diag`:

- `market_residual_positive_only: true`
- `market_residual_weight: 0.08 -> 0.05`
- other path / alpha / ROI / policy surfaces unchanged

狙い:

- support を作っている support-preserving path は残す
- negative market residual の drag だけを抑える
- `winonly_control` のように signal を切りすぎず、`supportpreserve_diag` より高い formal weighted ROI を狙う

## Entry Read

current fixed read:

- `supportpreserve_diag`
  - `wf_nested_test_roi_weighted=0.7245602799`
  - `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8920618287`
- `odds30_evlow`
  - `wf_nested_test_roi_weighted=0.8225407438`
  - `wf_feasible_fold_count=0`
- `supportpreserve_winonly_control`
  - `wf_nested_test_roi_weighted=0.7282222669`
  - `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8562282347`

entry meaning:

1. success metric の primary target は `odds30_evlow` の evaluation read ではなく `supportpreserve_diag` の formal weighted ROI である
2. candidate が `wf_feasible_fold_count=0` に落ちた時点で policy-only family と同じ failure に戻る
3. candidate が `0.8920618287` を超えられないなら、bounded calibration の headroom は薄い

## Success Metrics

1. full retrain -> full nested evaluate -> wf feasibility -> promotion gate まで end-to-end 完走する
2. `wf_feasible_fold_count>=1` を維持する
3. `formal_benchmark_weighted_roi > 0.8920618287` を返す
4. `supportpreserve_winonly_control` の `0.8562282347` を下回らない

## Validation Plan

1. candidate config を 1 本だけ materialize する
2. same strict benchmark / same full nested gate で formal revision を実行する
3. compare は `supportpreserve_diag`, `odds30_evlow`, `supportpreserve_winonly_control` の 3 点で閉じる
4. `advance / hold / reject` をこの issue 単体で決める

## Stop Condition

- candidate が policy family や component family の変更を必要とする
- `wf_feasible_fold_count` が 0 に戻り、policy-only residual family と同じ block に落ちる
- positive-only lower-weight calibration でも `supportpreserve_diag` を超えられず、bounded calibration 仮説が弱いと判定できる

## Exit Rule

- `advance`: `wf_feasible_fold_count>=1` を維持しつつ `formal_benchmark_weighted_roi` が `0.8920618287` を上回る
- `hold`: feasible fold は維持するが ROI 改善が弱い
- `reject`: feasible fold を失うか、`supportpreserve_diag` を超えられない

## Follow-Up Boundary

- この issue が閉じるまでは `odds30` 近傍の policy residual を再開しない
- train-time artifact simplification の別 candidate を同時に足さない
- current source-of-truth の meaning は変えず、human review 前の next measurable hypothesis だけを固定する

## First Read

- current queue:
  - [../github_issue_queue_current.md](../github_issue_queue_current.md)
- current `#103` source:
  - [next_issue_nar_value_blend_architecture_bootstrap.md](next_issue_nar_value_blend_architecture_bootstrap.md)
- executed train-time control:
  - [next_issue_nar_value_blend_train_time_artifact_control.md](next_issue_nar_value_blend_train_time_artifact_control.md)
- review snapshot:
  - [../nar_value_blend_supportpreserve_review_20260421.md](../nar_value_blend_supportpreserve_review_20260421.md)