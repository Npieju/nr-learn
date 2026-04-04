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

## 9. First Read

- issue:
  - `#92`
- dry-run revisions:
  - `r20260404_kelly_runtime_base25_dryrun_v1`
  - `r20260404_kelly_runtime_minprob003_dryrun_v1`
  - `r20260404_kelly_runtime_edge005_dryrun_v1`
- dry-run result:
  - 3 candidate とも current codepath の `run_revision_gate.py --dry-run` を通過
  - `skip-train + evaluate-model-artifact-suffix=r20260326_tighter_policy_ratio003` の threshold-only compare 導線も正常
  - versioned `wf_summary` / `promotion_output` の planned path も candidate ごとに分離されている

historical formal read:

- `promotion_gate_r20260329_kelly_runtime_base25.json`
  - `weighted_roi=1.1038859989058847`
  - `bets_total=598`
  - `feasible_fold_count=5`
- `promotion_gate_r20260329_kelly_runtime_minprob003.json`
  - `weighted_roi=1.1038859989058847`
  - `bets_total=598`
  - `feasible_fold_count=5`
- `promotion_gate_r20260329_kelly_runtime_edge005.json`
  - `weighted_roi=1.1038859989058847`
  - `bets_total=598`
  - `feasible_fold_count=5`

interpretation:

- historical top-line は 3 candidate で同値だった
- したがって first formal candidate は最も解釈しやすい `K1 = kelly_runtime_base25` に固定する
- `K2` と `K3` は contrast 候補として残すが、initial rerun の優先順位は下げる
