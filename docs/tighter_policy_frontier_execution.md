# Tighter Policy Frontier Execution

## 1. Purpose

この文書は、`tighter policy search frontier refinement` を実際に進めるための実行手順書である。

## 2. Baseline Inputs

最初に使う入力は次で固定する。

- baseline promotion:
  - `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- candidate promotion:
  - `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
  - `artifacts/reports/promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
- candidate evaluation:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260326_tighter_policy_ratio003_wf_full_nested.json`
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260327_tighter_policy_ratio003_abs80_wf_full_nested.json`

## 3. Narrow Sweep Focus

今回は新しい family を増やさず、次の軸だけを狭く見る。

1. `min_bets_abs`: `80,90,100`
2. `min_bet_ratio`: `0.03,0.04,0.05`
3. `min_prob`: `0.03,0.04,0.05`
4. `odds_max`: `18,25`
5. `min_expected_value`: `1.0,1.05`

まず最初は `min_bets_abs` と `min_bet_ratio` だけで十分である。

## 4. Step 1: Threshold Sweep

### 4.1 Existing `ratio003` Candidate

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_sweep.py \
  --wf-summary artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_wf_full_nested.json \
  --min-bet-ratio-values 0.05,0.04,0.03 \
  --min-bets-abs-values 100,90,80 \
  --min-feasible-folds 3 \
  --target-feasible-fold-counts 3,4,5 \
  --output artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_frontier_narrow.json \
  --summary-csv artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_frontier_narrow.csv
```

### 4.2 Existing `ratio003_abs80` Candidate

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_sweep.py \
  --wf-summary artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs80_wf_full_nested.json \
  --min-bet-ratio-values 0.05,0.04,0.03 \
  --min-bets-abs-values 100,90,80 \
  --min-feasible-folds 3 \
  --target-feasible-fold-counts 3,4,5 \
  --output artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_abs80_frontier_narrow.json \
  --summary-csv artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_abs80_frontier_narrow.csv
```

## 5. Step 2: Candidate Matrix

threshold sweep の結果、`min_bet_ratio=0.03` では `min_bets_abs=90` と `80` の両方が `5/5` feasible folds を満たした。したがって、実際に revision gate 候補へ進める matrix は、より strict な `90` anchor の次の 3 本から始める。

### Candidate A

- config base:
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90.yaml`
- idea:
  - keep `min_bet_ratio=0.03`
  - keep `min_bets_abs=90`
  - keep `min_prob=0.06`
  - keep `odds_max=18`
  - keep `min_expected_value=1.0`
  - result:
    - promoted anchor as of `2026-03-29`
    - formal benchmark weighted ROI `1.1042287961989103`
    - feasible folds `5/5`

### Candidate B

- config base:
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_minprob005.yaml`
- idea:
  - keep `min_bet_ratio=0.03`
  - keep `min_bets_abs=90`
  - relax serving `min_prob` from `0.06` to `0.05`
  - keep `odds_max=18`
  - role:
    - first challenger against promoted A

### Candidate C

- config base:
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_odds25.yaml`
- idea:
  - keep `min_bet_ratio=0.03`
  - keep `min_bets_abs=90`
  - keep `min_prob=0.06`
  - widen `odds_max` from `18` to `25`
  - role:
    - second challenger if B does not clearly win

最初は 3 本で止める。広げすぎない。

## 6. Step 3: Revision Gate Dry Run

候補を作ったら、まず dry-run で command 解決だけ確認する。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --config <candidate-config> \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --revision <revision_slug> \
  --train-artifact-suffix <revision_slug> \
  --skip-train \
  --evaluate-model-artifact-suffix r20260326_tighter_policy_ratio003 \
  --evaluate-max-rows 120000 \
  --evaluate-pre-feature-max-rows 300000 \
  --evaluate-wf-mode full \
  --evaluate-wf-scheme nested \
  --promotion-min-feasible-folds 3 \
  --dry-run
```

## 7. Step 4: Formal Candidate Run

dry-run が通った candidate だけ本実行する。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --config <candidate-config> \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --revision <revision_slug> \
  --train-artifact-suffix <revision_slug> \
  --skip-train \
  --evaluate-model-artifact-suffix r20260326_tighter_policy_ratio003 \
  --evaluate-max-rows 120000 \
  --evaluate-pre-feature-max-rows 300000 \
  --evaluate-wf-mode full \
  --evaluate-wf-scheme nested \
  --promotion-min-feasible-folds 3 \
  --challenger-anchor-evaluation-summary artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260329_tighter_policy_ratio003_abs90_wf_full_nested.json
```

`2026-03-29` update:

- challenger run では promoted anchor の evaluation summary を必ず渡す
- `run_revision_gate.py` は evaluate 完了後に anchor との equivalence check を行う
- `--stop-on-equivalent-challenger` を付けると、evaluation summary が anchor と実質同値だった時点で `wf_feasibility` と `promotion_gate` へ進まず `hold` で止められる

## 8. Step 5: Operational Compare

formal support が出た candidate だけ compare に進める。

最低限見る window:

1. September difficult window
2. December control window

## 9. Decision Rule

次の 3 択で判断する。

- `promote`: support と role がともに改善
- `keep as candidate`: support はあるが role が narrow
- `reject`: support か drawdown が悪化

2026-03-29 update:

Candidate A はすでに `promote` まで通過した。したがって今後の意思決定は、

- `B/C promote`: A を上回るか、または A と異なる明確な運用 role を示す
- `B/C keep as candidate`: A を超えないが補助候補として説明可能
- `B/C reject`: A を上回る理由がなく、support / guardrail も弱い

の読みで進める。

2026-03-29 close-out:

- Candidate B:
  - evaluation-summary layer では A と実質同値
  - promoted anchor 置換候補にはしない
- Candidate C:
  - equivalence guard では `different`
  - final gate は `pass / promote`
  - ただし formal benchmark weighted ROI は A をわずかに下回る
- operational decision:
  - A を default anchor のまま維持する
  - C は near-par challenger reference として残す
  - tighter family の first-wave frontier issue は完了

## 10. Challenger Launch Order

`2026-03-29` 時点の実行順は次で固定する。

1. Candidate B: `ratio003_abs90_minprob005`
2. Candidate C: `ratio003_abs90_odds25`

この順にする理由:

- B は promoted A からの widening が最小で、差分解釈が最も明確
- C は odds 帯の widening を含むため、B より role の変化が大きい

Candidate C を起動する条件:

- B が明確に A を上回らない
- あるいは B が support を広げても role を改善しない
- あるいは B の結果から wider odds band の検証価値が明確になる

Candidate C 結果:

- wider odds band は no-op ではなかった
- ただし formal replacement には至らなかった

## 10. Companion Docs

- `docs/next_issue_tighter_policy_frontier.md`
- `docs/policy_family_shortlist.md`
- `docs/revision_gate_candidate_checklist.md`
- `docs/policy_challenger_decision_checklist.md`
