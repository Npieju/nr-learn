# Next Issue: Kelly-Centered Runtime Family

## 1. Title

`[experiment] Kelly-centered runtime family baseline defensive fallback formalization`

## 2. Objective

runtime の `kelly` family を、promoted anchor に対する次の defensive comparison family として formal に整理する。

## 3. Why This Is Next

- `tighter policy search` family の first-wave frontier は完了した
- promoted anchor は `r20260329_tighter_policy_ratio003_abs90`
- formal benchmark の fold winners は繰り返し `kelly` に寄っている
- next shortlist rank は `kelly-centered runtime family`

## 4. In-Scope Surface

- `docs/kelly_runtime_candidate_matrix.md`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_kelly_runtime_base25.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_kelly_runtime_minprob003.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_kelly_runtime_edge005.yaml`
- `scripts/run_revision_gate.py`

## 5. Acceptance Criteria

- candidate matrix が明文化されている
- 3 candidate の dry-run が通る
- first formal candidate を 1 本に絞れる
- promoted A に対する success / hold / reject の読みが固定されている

## 6. Non-Goals

- 新しい feature family の追加
- tighter family の再拡張
- seasonal override family の同時展開

## 7. Validation Commands

- `python scripts/run_revision_gate.py ... --dry-run`
- `python scripts/run_revision_gate.py ...`

## 8. References

- `docs/policy_family_shortlist.md`
- `docs/kelly_runtime_candidate_matrix.md`
- `docs/policy_challenger_decision_checklist.md`
