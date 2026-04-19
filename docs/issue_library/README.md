# Issue Library Index

この directory は current source-of-truth ではない。GitHub issue thread、current queue、tagged snapshot/reference を補助する local reference library である。

current priority を追うときは、まず [../github_issue_queue_current.md](../github_issue_queue_current.md) を読む。この index は queue や thread から参照が必要になったときだけ使う。

## First Rules

- raw listing を入口にしない。
- `next_issue_*.md` を current queue として直接読まない。
- GitHub issue thread に source-of-truth が移った local draft は削除候補とする。
- file 自体に「古い情報」と書き足して回るのではなく、version/tag 付き snapshot として扱う。

## Entry Points

- JRA pruning reference: [jra_pruning_index.md](jra_pruning_index.md)
- JRA architecture rebuild reference: [next_issue_model_architecture_rebuild_track_split.md](next_issue_model_architecture_rebuild_track_split.md)
- JRA prediction foundation support diagnostics reference: [next_issue_jra_prediction_foundation_probability_support_diagnostics.md](next_issue_jra_prediction_foundation_probability_support_diagnostics.md)
- JRA broader composition support-preserving probability path reference: [next_issue_jra_broader_composition_support_preserving_probability_path.md](next_issue_jra_broader_composition_support_preserving_probability_path.md)
- JRA broader composition bounded residual calibration reference: [next_issue_jra_broader_composition_bounded_residual_calibration.md](next_issue_jra_broader_composition_bounded_residual_calibration.md)
- JRA market deviation candidate reference: [next_issue_jra_market_deviation_formal_candidate.md](next_issue_jra_market_deviation_formal_candidate.md)
- JRA market-aware probability path reference: [next_issue_jra_market_deviation_market_aware_probability_path.md](next_issue_jra_market_deviation_market_aware_probability_path.md)
- JRA market deviation coverage recovery reference: [next_issue_jra_market_deviation_lightgbm_coverage_recovery.md](next_issue_jra_market_deviation_lightgbm_coverage_recovery.md)
- JRA market deviation target-clip reference: [next_issue_jra_market_deviation_lightgbm_target_clip_compression.md](next_issue_jra_market_deviation_lightgbm_target_clip_compression.md)
- JRA market deviation trade-off merge reference: [next_issue_jra_market_deviation_lightgbm_tradeoff_merge.md](next_issue_jra_market_deviation_lightgbm_tradeoff_merge.md)
- JRA market deviation target redesign reference: [next_issue_jra_market_deviation_target_redefinition.md](next_issue_jra_market_deviation_target_redefinition.md)
- JRA market deviation policy reintegration reference: [next_issue_jra_market_deviation_policy_reintegration.md](next_issue_jra_market_deviation_policy_reintegration.md)
- JRA operator/policy reference: [jra_operator_policy_index.md](jra_operator_policy_index.md)
- NAR reference: [nar_issue_index.md](nar_issue_index.md)
- ops/runtime reference: [ops_runtime_index.md](ops_runtime_index.md)

## Current Review Package

- [../jra_pruning_staged_decision_summary_20260411.md](../jra_pruning_staged_decision_summary_20260411.md)
- [../jra_pruning_package_review_20260410.md](../jra_pruning_package_review_20260410.md)
- [../jra_pruning_stage7_implementation_review_checklist.md](../jra_pruning_stage7_implementation_review_checklist.md)
- [../jra_pruning_stage7_rollback_checklist.md](../jra_pruning_stage7_rollback_checklist.md)

## Maintenance Rule

- この index は entrypoint が変わったときだけ更新する。
- 個別 doc の本文更新をここへ平行反映しない。