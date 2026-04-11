# JRA Pruning Stage-7 Implementation Review Checklist

## 1. Purpose

このチェックリストは、`stage-7` pruning simplification を implementation candidate として human review に掛ける前の標準確認項目である。

benchmark 正本更新の判定ではなく、bounded rollout candidate として review できる状態かを確認する。

## 2. Required Inputs

- `docs/jra_pruning_staged_decision_summary_20260411.md`
- `docs/jra_pruning_package_review_20260410.md`
- `docs/issue_library/next_issue_pruning_stage7_rollout_guardrails.md`
- baseline feature config:
  - `configs/features_catboost_rich_high_coverage_diag.yaml`
- stage-7 feature config:
  - `configs/features_catboost_rich_high_coverage_diag_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta.yaml`

## 3. Scope Readiness

- [ ] universe が JRA only で固定されている
- [ ] change surface が feature config diff に限定されている
- [ ] `stage-8` や別 family の追加 removal を混ぜていない
- [ ] public benchmark update ではなく internal implementation candidate review として扱っている

## 4. Exact Diff Check

- [ ] baseline と candidate の差分が removal 45 columns だけで説明できる
- [ ] retained anchors が次の 5 columns で固定されている
  - `owner_last_50_win_rate`
  - `競争条件`
  - `リステッド・重賞競走`
  - `障害区分`
  - `sex`
- [ ] group 単位の説明が可能である
  - calendar context
  - recent-history core
  - gate/frame/course core
  - track/weather/surface context
  - class/rest/surface core
  - jockey/trainer ID core
  - jockey/trainer/combo core
  - dispatch metadata

## 5. Artifact Check

- [ ] first formal reference として次を開いた
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1_wf_full_nested.json`
  - `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`
  - `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`
- [ ] formal read が `status=pass`, `decision=promote`, `feasible_fold_count=3/5` であることを確認した
- [ ] first actual-date reference として次を開いた
  - `artifacts/reports/serving_smoke_compare_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`
  - `artifacts/reports/serving_smoke_compare_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
- [ ] broad September / December control が baseline 完全同値であることを確認した

## 6. Rollback Readiness

- [ ] narrow rollback `stage-7 -> stage-6` が `発走時刻`, `東西・外国・地方区分` の復帰だけで説明できる
- [ ] broader rollback が `stage-6 -> stage-5 -> stage-4 -> stage-3 -> stage-2 -> stage-1 -> baseline` の順で記述されている
- [ ] rollback を 1 change unit として実行できる
- [ ] rollback reason が dispatch metadata / combo core / ID core / broad shape drift のどれかに分類できる

## 7. Do Not Approve Yet If

- [ ] diff が 45 columns を超えて広がっている
- [ ] retained anchors 5 columns 以外まで同時変更が必要になっている
- [ ] rollback を group 単位で説明できない
- [ ] stage-7 review を public benchmark update reason に使おうとしている
- [ ] `stage-8 condition quartet` まで同時に進めないと成立しない

## 8. Review Outcome

### Approve For Implementation Candidate Review

- [ ] exact diff が bounded である
- [ ] artifact read が stage-7 stopping point を支持している
- [ ] rollback path が具体的である

### Keep Docs-Only

- [ ] stage-7 は supported だが、implementation unit としての diff / rollback がまだ粗い
- [ ] human review を超える justification が揃っていない

### Reject

- [ ] bounded rollout candidate として説明できない
- [ ] stage-7 stopping point 以外の broad strategy change に流れている