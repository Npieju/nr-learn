# JRA Pruning Stage-7 Rollback Checklist

## 1. Purpose

このチェックリストは、`stage-7` pruning implementation candidate を rollback するときの最小 runbook である。

goal は「どこまで戻すか」を group 単位で即決できる状態にすることであり、新しい評価実行を設計することではない。

## 2. Default Rule

- default first action は narrow rollback `stage-7 -> stage-6`
- full baseline rollback は broad shape drift が見えたときだけ使う
- `stage-8` に進んで切り分けるのは rollback ではなく out-of-scope とする

## 3. Rollback Order

1. `stage-7 -> stage-6`
   - restore:
     - `発走時刻`
     - `東西・外国・地方区分`
2. `stage-6 -> stage-5`
   - restore:
     - `jockey_last_30_win_rate`
     - `trainer_last_30_win_rate`
     - `jockey_trainer_combo_last_50_win_rate`
     - `jockey_trainer_combo_last_50_avg_rank`
3. `stage-5 -> stage-4`
   - restore:
     - `jockey_id`
     - `trainer_id`
4. `stage-4 -> stage-3`
   - restore class/rest/surface core 20 columns
5. `stage-3 -> stage-2 supported branch`
   - restore track/weather/surface context 8 columns
6. `stage-2 supported branch -> stage-1`
   - restore gate/frame/course core 4 columns
7. `stage-1 -> baseline`
   - restore calendar context 3 columns
   - restore recent-history core 2 columns

## 4. Trigger Mapping

### Use `stage-7 -> stage-6`

- [ ] suspicion is limited to dispatch metadata removal
- [ ] broader structural groups still look acceptable
- [ ] review concern is about operator-facing metadata context only

### Use `stage-6 -> stage-5` or `stage-5 -> stage-4`

- [ ] concern extends into combo core or ID core
- [ ] issue is no longer isolated to dispatch metadata
- [ ] narrower rollback is insufficient to explain the drift

### Use Full Baseline Rollback

- [ ] broad shape drift is observed
- [ ] supported branch assumption itself is no longer trusted
- [ ] rollback needs to minimize decision ambiguity immediately

## 5. Verification After Rollback

- [ ] restored config matches the intended stage file or baseline file exactly
- [ ] rollback target is documented as one of:
  - `stage-6`
  - `stage-5`
  - `stage-4`
  - `stage-3`
  - `stage-2 supported branch`
  - `stage-1`
  - `baseline`
- [ ] review note explains why the narrower rollback level was or was not sufficient
- [ ] no public benchmark explanation is attached to the rollback

## 6. First References

- `docs/issue_library/next_issue_pruning_stage7_rollout_guardrails.md`
- `docs/jra_pruning_staged_decision_summary_20260411.md`
- `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`
- `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`

## 7. Output Requirement

rollback note には最低限次を残す。

- rollback start point
- rollback destination point
- restored group set
- reason classification
- whether full baseline rollback was avoided or required