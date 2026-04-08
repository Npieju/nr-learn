# Next Issue: Surface Plus Class-Layoff Interaction Candidate

Historical note:

- この draft は `#44` として formal candidate execution まで完了している。
- surface-plus-class-layoff interaction candidate は completed feature reference として扱い、この文書は historical issue source / artifact reference として使う。

## Summary

`#43` の `r20260330_surface_interaction_only_v1` は、`#41` より support を改善したが、formal result は `hold` だった。

- nested summary の support shape は改善
- matching WF feasibility は `1/5 -> 2/5`
- しかし nested weighted ROI は `1.1003 -> 0.7462` まで低下

したがって次の candidate は、surface-only と full 6-feature variant の中間に寄せる。

## Objective

surface-only で回復した support をできるだけ残しつつ、full variant が持っていた ROI / ranking signal を一部取り戻す。

## Current Read

- `#41`: ROI summary は強いが support は `1/5`
- `#43`: support は `2/5` まで改善したが ROI が弱い
- short-turnaround class interaction 2 本は win/ROI の両面で最も弱かった
- long-layoff class interaction 2 本は、surface-only で削りすぎた signal の戻し候補として最も自然

## Candidate Definition

keep:

- `horse_surface_switch_short_turnaround`
- `horse_surface_switch_long_layoff`
- `horse_class_up_long_layoff`
- `horse_class_down_long_layoff`

drop:

- `horse_class_up_short_turnaround`
- `horse_class_down_short_turnaround`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_surface_plus_class_layoff_interactions.yaml`

## Execution Standard

前回と同じ true component retrain flow を使う。

1. win component retrain
2. roi component retrain
3. stack rebuild
4. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`

## Success Criteria

- matching WF feasibility が `3/5` 以上に回復する
- nested weighted ROI が surface-only より明確に改善する
- `#41` と `#43` のどちらより次に進める candidate か判断できる
