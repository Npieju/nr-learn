# Ops And Runtime Index

data / loader / runtime / residual cleanup 系の historical reference index である。current operator entrypoint ではなく、過去の residual trail を辿るときだけ使う。

## First Read

- [../command_reference.md](../command_reference.md)
- [../development_operational_cautions.md](../development_operational_cautions.md)
- [../artifact_guide.md](../artifact_guide.md)

それでも過去の residual issue source が必要なときだけ下の一覧に降りる。

## Historical References

- [next_issue_append_external_residual.md](next_issue_append_external_residual.md)
- [next_issue_append_loader_logic_residual.md](next_issue_append_loader_logic_residual.md)
- [next_issue_append_parser_residual.md](next_issue_append_parser_residual.md)
- [next_issue_deque_trim_exact_safe_followup.md](next_issue_deque_trim_exact_safe_followup.md)
- [next_issue_deque_trim_promotion_decision.md](next_issue_deque_trim_promotion_decision.md)
- [next_issue_evaluate_post_inference_progress_gap.md](next_issue_evaluate_post_inference_progress_gap.md)
- [next_issue_feature_builder_runtime.md](next_issue_feature_builder_runtime.md)
- [next_issue_materialized_supplementals.md](next_issue_materialized_supplementals.md)
- [next_issue_minimum_columns_residual.md](next_issue_minimum_columns_residual.md)
- [next_issue_netkeiba_race_result_append_preslim_source.md](next_issue_netkeiba_race_result_append_preslim_source.md)
- [next_issue_post_runtime_benchmark_refresh.md](next_issue_post_runtime_benchmark_refresh.md)
- [next_issue_primary_source_shaping.md](next_issue_primary_source_shaping.md)
- [next_issue_primary_tail_cache_default_promotion.md](next_issue_primary_tail_cache_default_promotion.md)
- [next_issue_primary_tail_cache_refresh_automation.md](next_issue_primary_tail_cache_refresh_automation.md)
- [next_issue_race_card_load_residual.md](next_issue_race_card_load_residual.md)
- [next_issue_race_result_keys_preslim_source.md](next_issue_race_result_keys_preslim_source.md)
- [next_issue_racecard_preslim_source.md](next_issue_racecard_preslim_source.md)
- [next_issue_read_csv_tail_residual_again.md](next_issue_read_csv_tail_residual_again.md)
- [next_issue_read_tail_dominant_phase.md](next_issue_read_tail_dominant_phase.md)
- [next_issue_supplemental_dominant_phase.md](next_issue_supplemental_dominant_phase.md)
- [next_issue_supplemental_merge_residual.md](next_issue_supplemental_merge_residual.md)
- [next_issue_tail_equivalence_harness.md](next_issue_tail_equivalence_harness.md)
- [next_issue_tail_load_phase_budget.md](next_issue_tail_load_phase_budget.md)
- [next_issue_tail_loader_runtime.md](next_issue_tail_loader_runtime.md)
- [next_issue_training_table_load_residuals.md](next_issue_training_table_load_residuals.md)

## Reading Order

1. current runtime/operator 判断は first read の正本 docs
2. loader / append residual の過去経緯だけ見たいときは append / racecard / race-result 系
3. tail loader 系の過去経緯だけ見たいときは tail / primary tail cache 系
4. feature builder / progress gap は runtime historical reference として読む