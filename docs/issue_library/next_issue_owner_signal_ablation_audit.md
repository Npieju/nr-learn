# Next Issue: Owner Signal Ablation Audit

## Summary

JRA current baseline には `owner_last_50_win_rate` が既に入っている。したがって owner family の次仮説は add-on ではなく、owner を外したときに formal / actual-date でどれだけ劣化するかを測る ablation である。

`feature_family_ranking.md` では owner signal は Tier C で、pedigree 全体よりは実務的だが、Tier A family ほどの優先度ではない。ここで owner を外した selective audit を 1 本作れば、current baseline に対する owner の marginal contribution を pedigree 混入なしで読める。

## Objective

`owner_last_50_win_rate` を current JRA high-coverage baseline から外した selective ablation を formal compare し、owner signal が current baseline に対して実質的な alpha source か、削ってもよい weak contributor かを判定する。

## Hypothesis

if `owner_last_50_win_rate` を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then owner signal は current serving family の必須要素ではなく pruning 候補になる。

## In Scope

- `owner_last_50_win_rate` を明示除外した high-coverage config 1 本
- JRA true component retrain flow
- formal compare
- September difficult window と December control window の actual-date role split

## Non-Goals

- pedigree family の再開
- owner feature の broad 拡張
- policy rewrite
- NAR work

## Candidate Definition

keep current high-coverage baseline core and exclude only:

- `owner_last_50_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_owner_ablation.yaml`

## Success Metrics

- win / roi component の actual used feature set から `owner_last_50_win_rate` が消える
- formal compare を 1 本作れる
- actual-date read で owner の marginal contribution を baseline と比較して説明できる

## Validation Plan

1. owner ablation config を追加
2. true component retrain
3. stack rebuild
4. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`
5. September / December actual-date compare

## Stop Condition

- actual selected set から owner が外れない
- component retrain が no-op に近い
- formal support が baseline 比で明確に崩れ、analysis value も薄い

## Actual Execution Read

first acceptance point:

- win component:
  - `artifacts/reports/train_metrics_catboost_win_high_coverage_diag_r20260403_owner_signal_ablation_audit_v1.json`
  - `auc=0.8346368290924392`
  - `best_iteration=527`
- roi component:
  - `artifacts/reports/train_metrics_lightgbm_roi_high_coverage_diag_r20260403_owner_signal_ablation_audit_v1.json`
  - `top1_roi=0.8901018922852993`
  - `best_iteration=111`

actual selected set:

- win / roi の両 component で selected features は `108`
- `owner_last_50_win_rate` は used features から消えた
- したがって owner ablation は no-op ではない

stack rebuild:

- `artifacts/models/catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260403_owner_signal_ablation_audit_v1.joblib`
- stack feature count `108`

## Final Read

- revision:
  - `r20260403_owner_signal_ablation_audit_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260403_owner_signal_ablation_audit_v1.json`
  - `auc=0.8449770387983483`
  - `top1_roi=0.4597201921069117`
  - `ev_top1_roi=0.8105412402909629`
  - nested WF: `no_bet / no_bet / no_bet`
  - `wf_nested_test_bets_total=0`
- revision gate:
  - `artifacts/reports/revision_gate_r20260403_owner_signal_ablation_audit_v1.json`
  - `decision=hold`
  - `error_code=evaluation_nested_all_no_bet_short_circuit`

baseline refresh compare:

- baseline refresh:
  - `auc=0.8400959298075428`
  - `top1_roi=0.8070328660078143`
  - `ev_top1_roi=0.5568030337853367`
  - `wf_nested_test_roi_weighted=0.7628366750468021`
  - `wf_nested_test_bets_total=544`
- owner ablation:
  - `auc` は上
  - `ev_top1_roi` も上
  - ただし nested WF は `3/3 no_bet`, `bets_total=0`

decision:

- owner signal を外すと policy viability が壊れる
- AUC / EV top-line だけでは baseline 置換根拠にならない
- owner signal は current baseline で prune しない
- owner family の次手は widening ではなく、baseline keep decision で一旦固定する
