# Next Issue: NAR Value-Blend Train-Time Artifact Control

Current queue note:

- この draft の first candidate `supportpreserve_winonly_control` は 2026-04-21 に formal 実行済みである。
- current active issue は引き続き `#103` で、train-time artifact family の first control result をこの文書で固定する。
- 結果は gate-level では `promote` だが、issue-level read では support-preserving base を上回れず、human review 前の negative/hold read として扱う。

## Summary

`#103` value-blend architecture bootstrap では、first formal が baseline `#101` を大きく下回った後、support-preserving path と policy-only residual を順に切った。

その結果、support-preserving path 自体は `all-no-bet` を脱し、corrected sidecar では `wf_feasible_fold_count=1` まで戻したが、その後の `policytight -> minproblow -> evlow -> topk2 -> odds40 -> odds30 -> odds30_evlow` は 7 本連続で `wf_feasible_fold_count=0` だった。

したがって current next step は、同じ policy family の近傍 residual を継ぎ足すことではない。残差が policy surface ではなく train-time artifact / stack family 側にあるかを 1 本で判定する control issue が必要である。

## Execution Result

first candidate `supportpreserve_winonly_control` は次の artifact で formal 完走した。

- evaluation summary:
  - [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json)
- evaluation manifest:
  - [../artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json](../artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blen_e2ca5daeab6dd133_wf_full_nested.json)
- wf feasibility summary:
  - [../artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_wf_full_nested.json](../artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_wf_full_nested.json)
- promotion gate:
  - [../artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](../artifacts/reports/promotion_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json)
- revision manifest:
  - [../artifacts/reports/revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json](../artifacts/reports/revision_gate_r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.json)

fixed read:

- nested evaluation:
  - `wf_nested_test_roi_weighted=0.7282222669`
  - `wf_nested_test_bets_total=5847`
  - winners `['kelly', 'kelly', 'portfolio', 'portfolio', 'portfolio']`
- formal sidecar:
  - `status=pass`, `decision=promote`
  - `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8562282347`
  - `formal_benchmark_bets_total=3733`
  - dominant failure は引き続き `min_bets`
  - `wf_binding_min_bets_source_counts={'ratio': 5}`

meaning:

- gate-level では `min_feasible_folds=1` を満たして `promote` になったが、これは fold 5 だけが feasible だったためであり、issue-level の parity 改善を意味しない
- nested evaluation は support-preserving base `0.7245602799` をわずかに上回った一方、current best eval `odds30_evlow=0.8225407438` より明確に低い
- formal weighted ROI も `supportpreserve_diag` corrected sidecar の `0.8920618287` を下回っており、pure win-centric simplification は current residual を縮めていない
- したがってこの 1 control の conclusion は「train-time artifact family の first cut としては informative だが positive read ではない」であり、`advance` ではなく `hold/reject` 寄りで閉じるのが妥当である

## Objective

strict `pre_race_only` NAR benchmark 上で、`#103` value-blend lineの残差が policy-only ではなく train-time artifact / stack family 側にあるかを、simplified train-time control 1 本で formal に判定する。

## Hypothesis

if current `#103` residual の主要因が local threshold / density / odds residual ではなく train-time artifact / stack family 差にある, then same strict benchmark と same full nested gate の下で simplified stack-family control を retrain すれば、support-preserving policy-only residual 群とは異なる feasibility / weighted ROI response が観測される。

より具体的には、`#103` value-blend train-time stack を簡約した control line が

- `wf_feasible_fold_count`
- `formal_benchmark_weighted_roi`
- fold winner family

の少なくとも 1 つで support-preserving residual family と有意に異なる read を返すはずである。

## Current Read

- benchmark reference は `#101` formal rerun で固定済み:
  - `wf_nested_test_roi_weighted=3.9660920371`
  - `wf_nested_test_bets_total=778`
- `#103` first formal:
  - `wf_nested_test_roi_weighted=0.6579177603`
  - fast feasibility `wf_feasible_fold_count=0`
- support-preserving base diagnostic:
  - `wf_nested_test_roi_weighted=0.7245602799`
  - corrected sidecar `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8920618287`
- executed train-time control `supportpreserve_winonly_control`:
  - `wf_nested_test_roi_weighted=0.7282222669`
  - `wf_feasible_fold_count=1`
  - `formal_benchmark_weighted_roi=0.8562282347`
  - `decision=promote`
- exhausted policy-only residual family:
  - best evaluation read は `odds30_evlow` `wf_nested_test_roi_weighted=0.8225407438`
  - しかし `policytight`, `minproblow`, `evlow`, `topk2`, `odds40`, `odds30`, `odds30_evlow` の 7 本はすべて `wf_feasible_fold_count=0`
  - `wf_dominant_failure_reason=min_bets`
  - `wf_binding_min_bets_source_counts={'ratio': 5}`

meaning:

- local threshold / density / universe residual は evaluation を少し動かすが、formal support geometry を変えられていない
- pure win-centric train-time simplification は `wf_feasible_fold_count` を増やさず、formal weighted ROI も support-preserving baseより悪化したため、artifact family 単純化だけでは主要 residual を説明できなかった
- current residual はなお stack family の別形、probability-market coupling の structural mismatch、または issue definition 側に残っていると読むべきである
- 次 action は train-time control をさらに足す前に、human review で `#103` の re-scope 境界を固定することが妥当である

## Proposed Control

この issue で切る control は broad redesign ではなく、次の条件を満たす simplified stack-family candidate 1 本に限る。

- same dataset freeze:
  - `configs/data_local_nankan_pre_race_ready.yaml`
- same feature surface:
  - `configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml`
- same formal evaluation discipline:
  - full nested WF
  - `min_bet_ratio=0.05`
  - `min_bets_abs=100`
  - `max_drawdown=0.45`
  - `min_final_bankroll=0.85`
- changed layer:
  - train-time stack family / coupling のみ

first candidate は抽象候補のままにしない。current recommended control は次の win-centric single-score stack で固定する。

- config:
  - [../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control.yaml](../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control.yaml)
- shape:
  - `probability_path_mode: support_preserving_residual_path`
  - `market_residual_weight: 0.0`
  - `roi_weight: 0.0`
  - component は `win` のみを使い、same strict benchmark / same policy constraints / same full nested gate を維持する

meaning:

- evaluate-only `roiweight0` は first formal (`0.6579177603`) と実質同値で、ROI injection 単独 removal は主要残差を説明しなかった
- 一方、support-preserving path は coupling 形状の差で support を動かせることを示した
- したがって next cut は「ROI branch を少し弱める」ではなく、「support-preserving scaffold のまま ROI branch と market residual branch を両方ゼロにした純 win-centric control」が適切である

## In Scope

- new narrowed model config 1 本
- 必要なら stack train config 1 本
- `scripts/run_revision_gate.py` による true retrain
- formal evaluate / wf feasibility / promotion gate
- compare artifacts:
  - `#101` baseline reference
  - `#103` first formal
  - support-preserving base diagnostic
  - support-preserving best eval residual (`odds30_evlow`)

## Non-Goals

- policy-only residual の追加 sweep
- `odds30` 近傍の threshold 再探索
- broad track split を同じ issue で進めること
- JRA baseline 更新
- NAR separate-universe の role 変更

## Success Metrics

- simplified control 1 本が full retrain -> full nested evaluate -> wf feasibility -> promotion gate まで end-to-end 完走する
- 次のどちらか 1 つを明確に判定できる
  - `wf_feasible_fold_count` が support-preserving policy-only family と有意に異なる
  - `formal_benchmark_weighted_roi` が support-preserving base (`0.8920618287`) から意味のある差を返す
  - fold winner family / no-bet pattern が明確に変わる
- その結果、residual を「train-time artifact family にある」か「そこにもない」かの 2 択へ絞れる

result:

- first candidate は完走し、`wf_feasible_fold_count=1` と `formal_benchmark_weighted_roi=0.8562282347` を返した
- ただし support-preserving base (`0.8920618287`) を上回れず、issue-level read は train-time artifact simplification を positive candidate として支持しなかった

## Validation Plan

1. dataset freeze
   - `#101` benchmark reference と current `#103` support-preserving review snapshot を一次参照に固定する

2. control config
  - first candidate は `supportpreserve_winonly_control` 1 本に固定する
  - changed surface は `market_residual_weight=0.0` と `roi_weight=0.0` のみとし、policy surface は broad default のまま維持する

3. formal execution
   - true retrain
   - full nested evaluate
   - wf feasibility
   - promotion gate

4. decision read
   - `#101`
   - `#103` first formal
   - support-preserving base
   - `odds30_evlow`
   の 4 点 compare で `advance / hold / reject` を閉じる

## Stop Condition

- simplified control が broad redesign になり、1 measurable hypothesis を超える
- compare のために policy surface まで同時変更が必要になる
- true retrain 後も support-preserving base を明確に上回れず、artifact family 仮説が弱いと判断できる

## Execution Note

- revision 名は `r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1` を第一候補とする
- heavy run に入るときは `scripts/run_revision_gate.py` を terminal 実行し、外部ログは `artifacts/logs/r20260421_local_nankan_value_blend_bootstrap_supportpreserve_winonly_control_v1.log` に保存する

execution outcome:

- run は `--skip-train --evaluate-no-model-artifact-suffix` workaround で完了した
- revision manifest は `decision=promote` を返したが、issue-level recommendation は `promote_candidate` ではなく `hold/reject for #103 scope` と読む

## First Read

- current queue:
  - [../github_issue_queue_current.md](../github_issue_queue_current.md)
- current `#103` source:
  - [next_issue_nar_value_blend_architecture_bootstrap.md](next_issue_nar_value_blend_architecture_bootstrap.md)
- review snapshot:
  - [../nar_value_blend_supportpreserve_review_20260421.md](../nar_value_blend_supportpreserve_review_20260421.md)
- transfer strategy:
  - [../nar_model_transfer_strategy.md](../nar_model_transfer_strategy.md)
