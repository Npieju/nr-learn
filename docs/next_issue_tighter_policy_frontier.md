# Next Issue: Tighter Policy Frontier Refinement

## 1. Purpose

この文書は、次に着手する最有力 issue

`[experiment] Tighter policy search frontier refinement`

の実務下書きである。

## 2. Why This Is Next

現時点で最も evidence が強い policy family は `tighter policy search` である。

特に次の 2 本が重要である。

- `r20260326_tighter_policy_ratio003`
- `r20260327_tighter_policy_ratio003_abs80`

この 2 本は同一 family 上の frontier 調整であり、`abs80` によって feasible folds が `4/5 -> 5/5` に改善した。さらに narrow threshold sweep により、`min_bet_ratio=0.03` では `min_bets_abs=90` でも `5/5` feasible folds を維持できることが確認できた。つまり今は、新しい family を増やすより、この family の strict frontier を `abs90` anchor で詰めるほうが期待値が高い。

## 3. Ready-To-Use Issue Draft

### Title

`[experiment] Tighter policy search frontier refinement`

### Universe

`JRA`

### Category

`Policy`

### Objective

`tighter policy search` family の support frontier をさらに明確化し、`ROI`、`feasible folds`、`drawdown` のバランスが最もよい policy 設定帯を整理する。新しい exotic family を増やすのではなく、既存の strongest defensive family を formal に詰めたい。

### Hypothesis

if `tighter policy search` family の threshold frontier を `ratio`, `min_bets_abs`, `min_prob`, `odds_max`, `min_expected_value` 周辺で狭く再探索する, then we can preserve defensive behavior while improving support clarity and possibly edge toward the ROI>1.20 north-star band, while keeping drawdown and role interpretation stable.

### In-Scope Surface

- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_minprob005.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_odds25.yaml`
- `scripts/run_revision_gate.py`
- `scripts/run_wf_threshold_sweep.py`
- related compare / dashboard artifacts

### Non-Goals

- 新しい feature family の導入
- staged family の大型追加
- broad baseline replacement の即時判断
- NAR policy への展開

### Success Metrics

- feasible folds の境界が今より明確になる
- `5/5` を維持する strictest anchor が `abs90` として説明できる
- drawdown / bankroll / bet volume を壊さない narrow frontier が見つかる
- next revision gate candidate を 1 本に絞れる

### Eval Plan

- smoke:
  - threshold sweep と existing compare artifact の読み直し
  - candidate 数本に絞る
- formal:
  - 有望候補のみ revision gate に載せる
  - baseline と September/December role を compare で再確認する

### Validation Commands

- `python scripts/run_wf_threshold_sweep.py ...`
- `python scripts/run_revision_gate.py ... --dry-run`
- `python scripts/run_revision_gate.py ...`
- `python scripts/run_serving_profile_compare.py ...`

### Expected Artifacts

- threshold frontier summary
- candidate shortlist
- revision gate artifact
- compare dashboard summary

### Stop Condition

- support を増やすと drawdown / bankroll が悪化する
- role が曖昧になり baseline より説明しづらくなる
- same-family refinement より別 family 比較のほうが有望と判明する

## 4. Recommended Search Focus

最初に絞る軸は次のとおりである。

1. `min_bets_abs` around `80-100`
2. `min_prob` around `0.03-0.05`
3. `odds_max` around `18-25`
4. `min_expected_value` around `1.0-1.05`

まずはこの narrow sweep だけで十分である。

## 5. Baseline References

この issue が最低限参照すべき baseline / candidate artifact:

- `docs/jra_baseline_artifact_inventory.md`
- `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
- `artifacts/reports/promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
- `docs/policy_family_shortlist.md`
