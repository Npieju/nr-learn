# JRA Pruning Index

JRA pruning / simplification 系の local reference と tagged snapshot をまとめた index である。current source-of-truth は GitHub issue `#124` であり、この file は二次参照に留める。

## Quick Read

- current JRA source-of-truth は GitHub issue `#124` `[jra] pruning stage-7 rollout guardrails`
- current review status は `awaiting reviewer decision`
- current defendable stopping point は `stage-7`
- human review package は version/date 付き docs を正本にする
- family-level completed audits と staged boundary references は tagged snapshot/reference として読む

## First Read

- GitHub issue `#124`
- [../jra_pruning_staged_decision_summary_20260411.md](../jra_pruning_staged_decision_summary_20260411.md)
- [../jra_pruning_stage7_implementation_review_checklist.md](../jra_pruning_stage7_implementation_review_checklist.md)

local transfer snapshot が必要なときだけ [next_issue_pruning_stage7_rollout_guardrails.md](next_issue_pruning_stage7_rollout_guardrails.md) を開く。

## Current Source Thread

- GitHub issue `#124`
  - status: open / mechanically review-ready / awaiting reviewer decision
  - expected bounded decision: `approve implementation-candidate review package` or `keep docs-only`

## Current Human Review Package

- [../jra_pruning_staged_decision_summary_20260411.md](../jra_pruning_staged_decision_summary_20260411.md)
- [../jra_pruning_package_review_20260410.md](../jra_pruning_package_review_20260410.md)
- [../jra_pruning_stage7_implementation_review_checklist.md](../jra_pruning_stage7_implementation_review_checklist.md)
- [../jra_pruning_stage7_rollback_checklist.md](../jra_pruning_stage7_rollback_checklist.md)

## Family-Level Completed Audits

- [next_issue_calendar_context_ablation_audit.md](next_issue_calendar_context_ablation_audit.md)
- [next_issue_recent_history_core_ablation_audit.md](next_issue_recent_history_core_ablation_audit.md)
- [next_issue_gate_frame_course_core_ablation_audit.md](next_issue_gate_frame_course_core_ablation_audit.md)
- [next_issue_track_weather_surface_context_ablation_audit.md](next_issue_track_weather_surface_context_ablation_audit.md)
- [next_issue_class_rest_surface_core_ablation_audit.md](next_issue_class_rest_surface_core_ablation_audit.md)
- [next_issue_jockey_trainer_id_core_ablation_audit.md](next_issue_jockey_trainer_id_core_ablation_audit.md)
- [next_issue_jockey_trainer_combo_core_ablation_audit.md](next_issue_jockey_trainer_combo_core_ablation_audit.md)
- [next_issue_race_condition_dispatch_context_ablation_audit.md](next_issue_race_condition_dispatch_context_ablation_audit.md)
- [next_issue_owner_signal_ablation_audit.md](next_issue_owner_signal_ablation_audit.md)

## Staged Boundary References

- [next_issue_pruning_bundle_ablation_audit.md](next_issue_pruning_bundle_ablation_audit.md)
- [next_issue_pruning_stage1_calendar_recent_history_bundle.md](next_issue_pruning_stage1_calendar_recent_history_bundle.md)
- [next_issue_pruning_stage2_calendar_recent_history_gate_frame_course_bundle.md](next_issue_pruning_stage2_calendar_recent_history_gate_frame_course_bundle.md)
- [next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md](next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md)
- [next_issue_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_bundle.md](next_issue_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_bundle.md)
- [next_issue_pruning_stage4_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_bundle.md](next_issue_pruning_stage4_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_bundle.md)
- [next_issue_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_bundle.md](next_issue_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_bundle.md)
- [next_issue_pruning_stage6_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_bundle.md](next_issue_pruning_stage6_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_bundle.md)
- [next_issue_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_bundle.md](next_issue_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_bundle.md)
- [next_issue_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_bundle.md](next_issue_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_bundle.md)

## Reading Order

1. current issue を追うなら GitHub issue `#124`
2. human review 全体像を確認するなら [../jra_pruning_staged_decision_summary_20260411.md](../jra_pruning_staged_decision_summary_20260411.md)
3. local transfer snapshot が必要なときだけ [next_issue_pruning_stage7_rollout_guardrails.md](next_issue_pruning_stage7_rollout_guardrails.md)
4. pruning の根拠を遡るなら family-level audits と staged boundary references

## Maintenance Rule

- この index は current active source、human review package、または category entrypoint が変わったときだけ更新する。
- 各 family audit や stage doc の本文更新を毎回ここへ平行反映しない。