# NAR Value-Blend Support-Preserving Review 2026-04-21

## Quick Read

- role: 2026-04-21 時点の `#103` support-preserving residual sweep を閉じる human review 用 snapshot
- current read: support-preserving path は evaluation を `0.7246 -> 0.8225` まで押し上げたが、formal gate は 7 本連続 `wf_feasible_fold_count=0`
- current use: current source-of-truth の代替ではなく、停止判断と次の structural re-scope をまとめた review package
- latest boundary update は [github_issue_queue_current.md](github_issue_queue_current.md) と [issue_library/next_issue_nar_value_blend_architecture_bootstrap.md](issue_library/next_issue_nar_value_blend_architecture_bootstrap.md) を優先する

## Purpose

この文書は、`#103` の support-preserving residual family をこれ以上惰性で掘らないために、2026-04-21 時点の停止判断を human review 用に 1 か所へ固定する snapshot である。

ここで固定したいのは次の 3 点である。

1. support-preserving path が何を改善し、何を改善できなかったか
2. なぜ local threshold / density / universe residual を続ける優先度が下がったか
3. 次を single residual ではなく structural residual / review judgment に切り替える根拠

## Non-Goals

- `#103` の current source-of-truth をこの文書へ移すこと
- 新しい policy residual を正当化すること
- support-preserving artifact の promote を主張すること
- `#101` baseline の更新や否定

## Executive Read

2026-04-21 時点の current reading は次で固定する。

- support-preserving path への切替で all-no-bet は脱したが、issue-level parity gap は大きく残った
- その後の local residual sweep は evaluation を少しずつ押し上げた一方、formal feasibility は 1 回も回復しなかった
- 追加で切った train-time artifact control `supportpreserve_winonly_control` も gate-level では pass したが、issue-level では support-preserving base を上回れなかった
- よって current next step は `odds30` 近傍の追加 sweep でも pure win-centric simplification の継続でもなく、structural residual の再定義か human review である

最短の結論は次の 4 行で足りる。

- build-required support-preserving diagnostic 自体は `wf_feasible_fold_count=1` まで戻し、legacy coupling が一部ボトルネックだったことは確認できた
- しかし policy-only residual `policytight -> minproblow -> evlow -> topk2 -> odds40 -> odds30 -> odds30_evlow` は 7 本連続で `wf_feasible_fold_count=0` だった
- train-time artifact control `supportpreserve_winonly_control` は `wf_feasible_fold_count=1`, `decision=promote` まで通ったが、`formal_benchmark_weighted_roi=0.8562282347` で `supportpreserve_diag=0.8920618287` を下回った
- よってこの family の ad-hoc residual sweep も pure win-centric simplification も停止し、次は structural re-scope か review judgment に切り替える

## Included Evidence

### 1. Support-Preserving Base Diagnostic

- revision:
  - `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1`
- source artifacts:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_359b5539f96738e9_wf_full_nested.json)
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_diag_wf_full_nested.json)
  - [promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260419_local_nankan_value_blend_bootstrap_supportpreserve_diag_v1.json)
- fixed read:
  - `wf_nested_test_roi_weighted=0.7245602799`
  - corrected sidecar では `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8920618287`
- meaning:
  - support surface 単独ではなく legacy の probability-market coupling 形状が残差の一部だったことは確認できた
  - ただし ROI parity はなお `#101` baseline `3.9660920371` に遠い

### 2. Exhausted Policy-Only Residuals

この snapshot では次の 7 residual を 1 family として固定する。

| residual | revision | nested weighted ROI | bets total | feasible folds | read |
| --- | --- | ---: | ---: | ---: | --- |
| `policytight` | `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_policytight_diag_v1` | `0.7740222115` | `6213` | `0` | winners 5/5 `portfolio` |
| `minproblow` | `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_minproblow_diag_v1` | `0.7764314432` | `7213` | `0` | slight eval up only |
| `evlow` | `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_evlow_diag_v1` | `0.7825910322` | `6646` | `0` | threshold loosen only |
| `topk2` | `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_topk2_diag_v1` | `0.7665700950` | `6213` | `0` | density increase worsened |
| `odds40` | `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_oddsrelax_diag_v1` | `0.7262348949` | `3907` | `0` | broad relax reject |
| `odds30` | `r20260419_local_nankan_value_blend_bootstrap_supportpreserve_odds30_diag_v1` | `0.8165625528` | `5923` | `0` | previous best eval read |
| `odds30_evlow` | `r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1` | `0.8225407438` | `6138` | `0` | current best eval read |

common fixed read:

- formal gate は 7 本すべて `status=block`, `decision=hold`
- `wf_dominant_failure_reason=min_bets`
- `wf_binding_min_bets_source_counts={'ratio': 5}`
- `wf_max_infeasible_bets_observed=1` が続き、fold 4/5 は `bets=1` または `0` の近傍に張り付く

meaning:

- 評価改善は作れているが、support geometry 自体は動いていない
- したがって local threshold / density / universe tweak は informative ではあっても promotable path を開いていない

### 3. Current Best Eval Read Is Still a Formal Reject

- revision:
  - `r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1`
- source artifacts:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e60f285b1d2a0e7f_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e60f285b1d2a0e7f_wf_full_nested.json)
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_wf_full_nested.json)
  - [promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_odds30_evlow_diag_v1.json)
- fixed read:
  - `wf_nested_test_roi_weighted=0.8225407438`
  - `wf_nested_test_bets_total=6138`
  - winners 5/5 `portfolio`
  - `wf_feasible_fold_count=0`
  - blocking reason: `Walk-forward feasible fold count is below threshold: 0 < 1`
- meaning:
  - current best evaluation candidate 自体が formal reject なので、近傍追加の期待値は低い

### 4. Win-Only Structural Control Also Missed Issue-Level Advance

- revision:
  - `r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1`
- source artifacts:
  - [evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json)
  - [wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_wf_full_nested.json)
  - [promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json)
  - [revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](/workspaces/nr-learn/artifacts/reports/revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json)
- fixed read:
  - `wf_nested_test_roi_weighted=0.7282222669`
  - `wf_nested_test_bets_total=5847`
  - `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8562282347`
  - `formal_benchmark_bets_total=3733`
  - gate result `status=pass`, `decision=promote`
- meaning:
  - fold 5 だけが feasible で gate-level pass にはなったが、issue-level の主指標は `supportpreserve_diag` corrected sidecar の `0.8920618287` を下回った
  - pure win-centric simplification は support-preserving base を超える explanatory candidate ではなく、artifact family の単純化だけでは主要 residual を縮められないことを示した
  - よって next step を「さらに単純な train-time control の追加」へは進めず、review judgment か structural re-scope の再設計へ留めるべきである

## Review Decision Boundary

この snapshot で human に判断してほしい境界は次の 2 択である。

1. `#103` を support-preserving residual family と win-only structural control まで含めて `hold` で閉じ、single residual sweep や pure simplification の追加は再開しない
2. `#103` を新しい 1 hypothesis に切り直し、train-time artifact 差、stack family 差、または probability-market coupling の structural residual だけを狙う

次をやってはいけないラインも固定する。

- `odds30` 近傍で `min_prob`, `min_expected_value`, `top_k`, `odds_max` を順番に足していく惰性 sweep
- `win only -> even simpler control` のように issue-level negative read を無視して simplification を足し続けること
- formal gate が 0 feasible のまま current source-of-truth を更新し続けること
- `#101` baseline と `#103` residual family を混同して support-preserving 評価改善を parity 改善と読み替えること

## Recommended Next Step

current best recommendation は次の通りである。

- `#103` の next measurable hypothesis は single residual ではなく structural residual に切り替える
- review なしに続けるなら、候補は train-time artifact / stack composition / probability-market coupling のどれか 1 本だけに絞る
- immediate draft は [issue_library/next_issue_nar_value_blend_support_preserving_residual_calibration.md](issue_library/next_issue_nar_value_blend_support_preserving_residual_calibration.md) を使う
- review を先に通すなら、この snapshot を境界として support-preserving residual sweep を close 扱いにする

latest boundary note:

- `supportpreserve_winonly_control` 完了後の read でも current recommendation は変わらない
- gate-level `promote` を issue-level `advance` と読み替えず、`supportpreserve_diag` を超えられなかった点を優先して review judgment を取るべきである

## Source-of-Truth Pointers

- current queue:
  - [github_issue_queue_current.md](/workspaces/nr-learn/docs/github_issue_queue_current.md)
- current issue source:
  - [issue_library/next_issue_nar_value_blend_architecture_bootstrap.md](/workspaces/nr-learn/docs/issue_library/next_issue_nar_value_blend_architecture_bootstrap.md)
- benchmark reference:
  - [evaluation_summary_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json)
