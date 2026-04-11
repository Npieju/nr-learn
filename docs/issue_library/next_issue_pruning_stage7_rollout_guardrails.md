# Next Issue: Stage-7 Pruning Rollout Guardrails

## Quick Read

- role: current JRA primary next issue draft
- current status: GitHub issue thread へ転記する前の local source draft
- decision boundary: `stage-7` は review-ready implementation candidate、`stage-8` は hold boundary
- use this doc when: exact diff / rollback / validation gate を短時間で確認したいとき

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

- current role: GitHub issue thread へ転記する前の local source draft
- once transferred: GitHub thread を source-of-truth にし、この local draft は削除候補として再評価する

## Objective

`stage-7` を baseline rewrite 候補として即 promote するのではなく、human review に出せる bounded implementation candidate として扱うための rollout guardrail package を固定する。

## Hypothesis

if `current_recommended_serving_2025_latest` baseline と `stage-7` simplification candidate の差分、rollback point、post-change validation を 1 本の issue で具体化できる, then `stage-7` は benchmark 正本更新ではなく review-ready implementation candidate として safely handoff できる。

## Dataset Scope

- JRA only
- current high-coverage latest baseline line
- no NAR / no mixed universe

## In Scope

- `stage-7` removal set の exact diff 整理
- rollback point の固定
- post-change validation checklist の固定
- review thread に貼る decision summary / risk summary の整形
- implementation に進む前提条件の明文化

## Non-Goals

- baseline feature config の即時 rewrite
- new train / evaluate / serving replay
- `stage-8` 再実行
- owner signal keep decision の再審
- public benchmark docs の更新
- benchmark 正本の更新判断

## Starting Context

current accepted read:

- stage-7 supported decision summary:
  - `docs/jra_pruning_staged_decision_summary_20260411.md`
- human review package:
  - `docs/jra_pruning_package_review_20260410.md`
- supported execution source:
  - `docs/issue_library/next_issue_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_bundle.md`
- hold boundary source:
  - `docs/issue_library/next_issue_pruning_stage8_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_condition_quartet_bundle.md`

current accepted stopping point:

- natural stopping point は `stage-7`
- `stage-8` は narrow hold boundary artifact として retain
- implementation を検討するなら group-wise rollback point を先に固定する

## Candidate Surface

source configs:

- baseline feature config:
   - `configs/features_catboost_rich_high_coverage_diag.yaml`
- stage-7 candidate feature config:
   - `configs/features_catboost_rich_high_coverage_diag_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta.yaml`

candidate removal groups are only the groups already supported on the stage-7 branch:

1. calendar context
   - `race_year`
   - `race_month`
   - `race_dayofweek`
2. recent-history core
   - `horse_last_3_avg_rank`
   - `horse_last_5_win_rate`
3. gate/frame/course core
   - `gate_ratio`
   - `frame_ratio`
   - `course_gate_bucket_last_100_win_rate`
   - `course_gate_bucket_last_100_avg_rank`
4. track/weather/surface context
   - `track`
   - `weather`
   - `ground_condition`
   - `馬場状態2`
   - `芝・ダート区分`
   - `芝・ダート区分2`
   - `右左回り・直線区分`
   - `内・外・襷区分`
5. class/rest/surface core
   - `horse_days_since_last_race`
   - `horse_days_since_last_race_log1p`
   - `horse_is_short_turnaround`
   - `horse_is_long_layoff`
   - `horse_weight_change`
   - `horse_weight_change_abs`
   - `horse_distance_change`
   - `horse_distance_change_abs`
   - `horse_surface_switch`
   - `horse_surface_switch_short_turnaround`
   - `horse_surface_switch_long_layoff`
   - `race_class_score`
   - `horse_last_class_score`
   - `horse_class_change`
   - `horse_is_class_up`
   - `horse_is_class_down`
   - `horse_class_up_short_turnaround`
   - `horse_class_down_short_turnaround`
   - `horse_class_up_long_layoff`
   - `horse_class_down_long_layoff`
6. jockey/trainer ID core
   - `jockey_id`
   - `trainer_id`
7. jockey/trainer/combo core
   - `jockey_last_30_win_rate`
   - `trainer_last_30_win_rate`
   - `jockey_trainer_combo_last_50_win_rate`
   - `jockey_trainer_combo_last_50_avg_rank`
8. dispatch metadata
   - `発走時刻`
   - `東西・外国・地方区分`

## Exact Diff

config-level exact diff は次で固定する。

- baseline `force_include_columns` は 34 columns を要求する
- stage-7 candidate `force_include_columns` は `owner_last_50_win_rate` だけを残す
- baseline `force_categorical_columns` は 16 columns を要求する
- stage-7 candidate config は `force_categorical_columns` に 6 columns を宣言するが、このうち `発走時刻`, `東西・外国・地方区分` は同時に `exclude_columns` に入っており effective selected surface には残らない
- effective retained categorical anchors は次の 4 columns で固定される
  - `競争条件`
  - `リステッド・重賞競走`
  - `障害区分`
  - `sex`
- effective retained anchor は `owner_last_50_win_rate` を加えた 5 columns であり、stage-7 candidate は current baseline から 45 columns を removal set として扱う
- mechanical re-check command:
   - `python scripts/run_pruning_rollout_guardrails_check.py`

grouped exact diff:

1. remove 3 calendar context columns
2. remove 2 recent-history core columns
3. remove 4 gate/frame/course core columns
4. remove 8 track/weather/surface context columns
5. remove 20 class/rest/surface core columns
6. remove 2 jockey/trainer ID core columns
7. remove 4 jockey/trainer/combo core columns
8. remove 2 dispatch metadata columns
9. keep as remaining non-removal anchors:
   - `owner_last_50_win_rate`
   - `競争条件`
   - `リステッド・重賞競走`
   - `障害区分`
   - `sex`

current artifact read では、この exact diff を持つ `stage-7` candidate は next stage の `condition quartet` を混ぜないかぎり supported branch の stopping point として読める。

## Rollback Plan

rollback は group 単位で次の順に固定する。

1. immediate full rollback:
   - feature config を `configs/features_catboost_rich_high_coverage_diag.yaml` に戻す
2. narrow rollback from stage-7 to stage-6:
   - `発走時刻`, `東西・外国・地方区分` を復帰する
3. stage-6 to stage-5:
   - `jockey_last_30_win_rate`, `trainer_last_30_win_rate`, `jockey_trainer_combo_last_50_win_rate`, `jockey_trainer_combo_last_50_avg_rank` を復帰する
4. stage-5 to stage-4:
   - `jockey_id`, `trainer_id` を復帰する
5. stage-4 to stage-3:
   - class/rest/surface core 20 columns を復帰する
6. stage-3 to stage-2 supported branch:
   - `track`, `weather`, `ground_condition`, `馬場状態2`, `芝・ダート区分`, `芝・ダート区分2`, `右左回り・直線区分`, `内・外・襷区分` を復帰する
7. stage-2 supported branch to stage-1:
   - `gate_ratio`, `frame_ratio`, `course_gate_bucket_last_100_win_rate`, `course_gate_bucket_last_100_avg_rank` を復帰する
8. stage-1 to baseline:
   - `race_year`, `race_month`, `race_dayofweek`, `horse_last_3_avg_rank`, `horse_last_5_win_rate` を復帰する

rollback decision rule:

- dispatch metadata のみが原因なら `stage-7 -> stage-6` rollback で止める
- combo or ID family まで疑わしいなら `stage-6 -> stage-5` または `stage-5 -> stage-4` へ戻す
- broad shape drift が出た場合だけ full rollback を使う
- `stage-8 condition quartet` へ進んで切り分けることは rollback ではなく out-of-scope とする

## Post-Change Validation Gate

implementation candidate review 前に確認する gate は次の 4 層で固定する。

1. config diff gate
   - baseline config と `stage-7` config の diff が removal 45 columns と retained anchors 5 columns だけで説明できる
2. shape gate
   - selected feature set が `stage-7` artifact read と同じ `feature_count=64`, `categorical_feature_count=25` の期待に沿っているかを見る
3. formal artifact gate
   - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1_wf_full_nested.json`
   - `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`
   - `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`
   を first reference として再確認する
4. actual-date gate
   - `artifacts/reports/serving_smoke_compare_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`
   - `artifacts/reports/serving_stateful_bankroll_sweep_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`
   - `artifacts/reports/serving_smoke_compare_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
   - `artifacts/reports/serving_stateful_bankroll_sweep_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
   を first reference として broad September / December control equivalence を再確認する

implementation candidate を reject する条件は次で固定する。

- removal diff が 45 columns を超えて広がる
- retained anchors 5 columns 以外まで同時に変える必要が出る
- rollback が 1 change unit で記述できない
- public benchmark update reason として説明しないと進められない

## GitHub Issue Body

```text
Objective:
- stage-7 pruning を benchmark update ではなく review-ready implementation candidate として扱うため、exact diff / rollback / post-change validation を固定する

Current read:
- supported stopping point: `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1`
- formal: `status=pass`, `decision=promote`, `feasible_fold_count=3/5`
- actual-date: broad September 2025 / December control 2025 とも baseline 完全同値
- hold boundary: `stage-8 condition quartet` は `0/5 feasible folds`, `status=block`, `decision=hold`

Exact diff:
- baseline feature config: `configs/features_catboost_rich_high_coverage_diag.yaml`
- candidate feature config: `configs/features_catboost_rich_high_coverage_diag_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta.yaml`
- removal set: 45 columns across calendar, recent-history, gate/frame/course, track/weather/surface, class/rest/surface, jockey/trainer ID, combo core, dispatch metadata
- retained anchors: `owner_last_50_win_rate`, `競争条件`, `リステッド・重賞競走`, `障害区分`, `sex`

Rollback order:
- narrow rollback: `stage-7 -> stage-6` by restoring `発走時刻`, `東西・外国・地方区分`
- then `stage-6 -> stage-5 -> stage-4 -> stage-3 -> stage-2 -> stage-1 -> baseline` only if broader drift is observed

Validation gate:
- first formal references: `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1_wf_full_nested.json`, `artifacts/reports/wf_feasibility_diag_pruning_stage7_dispatch_meta_v1.json`, `artifacts/reports/promotion_gate_r20260411_pruning_stage7_dispatch_meta_v1.json`
- first actual-date references: `artifacts/reports/serving_smoke_compare_sep25_ps7_dmeta_base_vs_sep25_ps7_dmeta_cand.json`, `artifacts/reports/serving_smoke_compare_dec25_ps7_dmeta_base_vs_dec25_ps7_dmeta_cand.json`
- reject if diff widens beyond the supported 45-column removal set or if rollback cannot be expressed as a single change unit

Decision boundary:
- keep this as internal implementation-candidate review material only
- do not use it as a public benchmark update reason
```

## Success Metrics

- baseline vs `stage-7` の exact diff が docs 上で再利用可能な形に固定される
- rollback point が group 単位で明文化される
- implementation 実行前に必要な smoke / compare / formal artifact check が明文化される
- public benchmark update reason には使わないことが issue 上で固定される
- review outcome が `approve for implementation candidate review` / `keep docs-only` の二択まで狭まる

## Validation Plan

1. current baseline config と `stage-7` config の feature diff を group 単位で整理する
2. rollback order を `stage-7 -> stage-6 -> stage-5 ...` の group 単位で固定する
3. post-change validation を次の 3 層で固定する
   - config diff review
   - reduced smoke / shape check
   - existing September / December actual-date compare references と formal gate references の再確認
4. implementation を許可しない条件も同じ issue に書く

## Expected Output Paths

- review-ready rollout memo:
   - `docs/issue_library/next_issue_pruning_stage7_rollout_guardrails.md`
- stage-7 implementation checklist:
   - `docs/jra_pruning_stage7_implementation_review_checklist.md`
- rollback checklist:
   - `docs/jra_pruning_stage7_rollback_checklist.md`
- issue thread draft for human review:
   - embedded in this issue

## Stop Condition

- exact diff が group 単位で簡潔に説明できない
- rollback point が曖昧で、1 change unit で戻せない
- implementation candidate の説明が public benchmark update や broad strategy change に流れてしまう
- stage-7 を超える removal を同じ issue に混ぜないと完結しない

## Current Best Read

default expectation は affirmative である。

- `stage-7` は supported branch の natural stopping point として十分に defend できる
- ただし benchmark 正本更新の根拠にはまだ使わない
- したがって next measurable issue は、新しい model family ではなく `stage-7` を review-ready implementation candidate として safely package できるかを narrow に切るのが妥当である