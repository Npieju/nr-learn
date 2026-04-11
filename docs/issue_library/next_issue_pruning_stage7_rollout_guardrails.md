# Next Issue: Stage-7 Pruning Rollout Guardrails

## Quick Read

- role: GitHub issue #124 transfer snapshot
- current status: GitHub issue #124 へ転記済みの transfer snapshot
- decision boundary: `stage-7` は review-ready implementation candidate、`stage-8` は hold boundary
- use this doc when: GitHub thread 正本に対する local reference / artifact entrypoint を短時間で確認したいとき

## Summary

`stage-7` staged simplification は current JRA pruning branch の natural stopping point まで到達した。

- supported branch end:
  - `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1`
- formal read:
  - `status=pass`
  - `decision=promote`
  - `feasible_fold_count=3/5`
- actual-date read:
  - broad September 2025 baseline 完全同値
  - December control 2025 baseline 完全同値

一方で `stage-8 condition quartet` は top-line を維持しても `0/5 feasible folds` で `hold` に戻った。

したがって current question は、さらに別 family を足すことではない。`stage-7` を human review に掛ける前提で、実装候補として扱うための diff / rollback / post-change validation guardrail を 1 issue で固定できるかにある。

## Local Status

- current role: GitHub issue #124 transfer snapshot
- source-of-truth: GitHub issue #124 `[jra] pruning stage-7 rollout guardrails`
- local handling rule: thread 側の objective / rollback / validation が正本であり、この local draft は削除または historical summary への畳み込み候補として扱う

## Canonical References

- GitHub source-of-truth:
  - issue `#124` `[jra] pruning stage-7 rollout guardrails`
  - mechanical validation comment: `#issuecomment-4228262330`
- local review package:
  - `docs/jra_pruning_staged_decision_summary_20260411.md`
  - `docs/jra_pruning_package_review_20260410.md`
  - `docs/jra_pruning_stage7_implementation_review_checklist.md`
  - `docs/jra_pruning_stage7_rollback_checklist.md`
- boundary references:
  - `docs/issue_library/next_issue_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_bundle.md`
  - `docs/issue_library/next_issue_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_bundle.md`

## Snapshot Facts

- defended stopping point:
  - `stage-7`
- hold boundary:
  - `stage-8 condition quartet`
- effective retained anchors:
  - `owner_last_50_win_rate`
  - `競争条件`
  - `リステッド・重賞競走`
  - `障害区分`
  - `sex`
- declared-but-excluded dispatch metadata:
  - `発走時刻`
  - `東西・外国・地方区分`
- bounded removal set:
  - 45 columns
- first mechanical re-check:
  - `python scripts/run_pruning_rollout_guardrails_check.py`
  - report: `artifacts/reports/pruning_rollout_guardrails_stage7_check.json`
  - accepted result: `status=pass`, `decision=review_ready`

## Snapshot Intent

- keep `stage-7` as an internal implementation-candidate review package only
- do not use this surface as a public benchmark update reason
- prefer GitHub thread updates over expanding this local snapshot

## Retirement Rule

- if all ongoing review state is readable from issue `#124` plus the existing checklists, delete this file
- if a local residue is still useful, keep only this compact snapshot and avoid restoring duplicated issue body text here