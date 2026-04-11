# JRA Pruning Package Review 2026-04-10

## Quick Read

- role: 2026-04-10 時点の human review 用 snapshot
- current read: individual pruning candidate judgments は成立、one-shot bundle simplification は `hold`
- current use: baseline rewrite の正本ではなく、review package の versioned snapshot/reference
- latest boundary update は [jra_pruning_staged_decision_summary_20260411.md](jra_pruning_staged_decision_summary_20260411.md) を優先する

## Purpose

この文書は、2026-04-10 時点で individual pruning candidate judgment まで完了した JRA high-coverage baseline feature groups を、human review 用に 1 か所へ束ねる internal memo である。

ここで固定したいのは次の 3 点である。

1. どの groups が individual artifact ベースで pruning candidate になったか
2. one-shot bundle simplification がどこで止まったか
3. baseline config へ実反映する前に何を人が判断すべきか

## Non-Goals

- public benchmark message の更新
- baseline config の即時書き換え
- pruning package の自動 promote
- owner keep decision の再審

## Executive Read

2026-04-10 時点の current reading は次で固定する。

- individual pruning candidate judgments は artifact ベースで十分に成立している
- ただし same-day one-shot bundle simplification は formal gate で `hold` になった
- したがって current next step は baseline rewrite ではなく、human review による pruning package judgment である

最短の結論は次の 4 行で足りる。

- 8 groups は individual issue ではすべて actual-date equivalence まで完了した
- bundle 同時除外は actual-date では Sep/Dec 完全同値だった
- それでも formal 側は `min_bets` 制約で `0/5 feasible folds` となり promoted simplification candidate にはならなかった
- よって「各 group を独立 pruning candidate として保持し、baseline rewrite は human review で段階導入判断する」が current best read である

## Included Groups

### 1. Calendar Context

- columns:
  - `race_year`
  - `race_month`
  - `race_dayofweek`
- source:
  - `docs/issue_library/next_issue_calendar_context_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September 2025 と December control 2025 で baseline 完全同値
  - individual pruning candidate

### 2. Gate/Frame/Course Core

- columns:
  - `gate_ratio`
  - `frame_ratio`
  - `course_gate_bucket_last_100_win_rate`
  - `course_gate_bucket_last_100_avg_rank`
- source:
  - `docs/issue_library/next_issue_gate_frame_course_core_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

### 3. Recent-History Core

- columns:
  - `horse_last_3_avg_rank`
  - `horse_last_5_win_rate`
- source:
  - `docs/issue_library/next_issue_recent_history_core_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

### 4. Jockey/Trainer/Combo Core

- columns:
  - `jockey_last_30_win_rate`
  - `trainer_last_30_win_rate`
  - `jockey_trainer_combo_last_50_win_rate`
  - `jockey_trainer_combo_last_50_avg_rank`
- source:
  - `docs/issue_library/next_issue_jockey_trainer_combo_core_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

### 5. Class/Rest/Surface Core

- columns:
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
- source:
  - `docs/issue_library/next_issue_class_rest_surface_core_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

### 6. Track/Weather/Surface Context

- columns:
  - `track`
  - `weather`
  - `ground_condition`
  - `馬場状態2`
  - `芝・ダート区分`
  - `芝・ダート区分2`
  - `右左回り・直線区分`
  - `内・外・襷区分`
- source:
  - `docs/issue_library/next_issue_track_weather_surface_context_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

### 7. Race-Condition/Dispatch Context

- columns:
  - `競争条件`
  - `リステッド・重賞競走`
  - `障害区分`
  - `発走時刻`
  - `sex`
  - `東西・外国・地方区分`
- source:
  - `docs/issue_library/next_issue_race_condition_dispatch_context_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

### 8. Jockey/Trainer ID Core

- columns:
  - `jockey_id`
  - `trainer_id`
- source:
  - `docs/issue_library/next_issue_jockey_trainer_id_core_ablation_audit.md`
- reading:
  - formal `pass / promote`
  - broad September / December control で baseline 完全同値
  - individual pruning candidate

## Bundle Audit Read

bundle source:

- `docs/issue_library/next_issue_pruning_bundle_ablation_audit.md`
- feature config:
  - `configs/features_catboost_rich_high_coverage_diag_pruning_bundle_ablation.yaml`

bundle candidate は上の 8 groups を同時除外し、`owner_last_50_win_rate` を残した simplified stack として評価した。

### What Held

- feature-gap は clean buildable
- win / ROI / stack は end-to-end で完走
- actual used set は `60 features / 21 categorical features` まで縮小
- broad September 2025 actual-date compare は baseline 完全同値
- December control 2025 actual-date compare も baseline 完全同値

### What Broke

- formal evaluation summary は top-line 自体は baseline 近辺を維持した
- ただし WF feasibility は `0/5 feasible folds`
- dominant blocking reason は `min_bets`
- ratio-bound required minimum bets は fold ごとに `947` から `995`
- best fallback candidate は fold ごとに positive ROI を出すが、`232` から `544` bets` で gate を満たさない
- promotion gate は `status=block`, `decision=hold`

### Interpretation

bundle 同時除外は operational replay 上は harmless に見えるが、formal support を満たす simplification candidate にはなっていない。

この差は次のように読む。

- actual-date equivalence は「current serving path を壊していない」証拠である
- formal infeasibility は「benchmark gate を通る簡約 revision としては support が薄い」証拠である
- よって individual candidate の寄せ集めを 1 本の promoted simplification として扱うのは早い

## Decision Boundary For Human Review

human review で決めるべき論点は次の 3 点である。

1. baseline feature config を one-shot rewrite するか、group 単位で段階導入するか
2. actual-date equivalence を重視して軽量化を進めるか、formal support 欠如を重視して現状維持するか
3. pruning package を benchmark 正本更新ではなく serving simplification backlog として扱うか

current recommendation は次である。

- one-shot baseline rewrite はしない
- individual pruning candidates は retain する
- 実反映するなら group 単位の staged removal と rollback point を明示する

## Current Staged Follow-Up

stage-1 `calendar + recent-history` block は `r20260410_pruning_stage1_calendar_recent_history_v1` で formal `pass / promote` と actual-date Sep/Dec equivalence まで確認できた。

その次の staged candidate として、stage-2 `calendar + recent-history + race-condition/dispatch context` を次の source に固定した。

- issue source:
  - `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_race_condition_dispatch_bundle.md`
- feature config:
  - `configs/features_catboost_rich_high_coverage_diag_pruning_stage2_calendar_recent_history_race_condition_dispatch.yaml`

first read はすでに clean である。

- `priority_missing_raw_columns=[]`
- `missing_force_include_features=[]`
- `empty_force_include_features=[]`
- `low_coverage_force_include_features=[]`
- `selected_feature_count=98`
- `categorical_feature_count=31`

stage-2 `race-condition / dispatch context` true retrain も完了したため、まず first failed second-block read は次で固定する。

- stage-2 revision:
  - `r20260410_pruning_stage2_calendar_recent_history_race_condition_dispatch_v1`
- formal summary:
  - `auc=0.8422432477925081`
  - `top1_roi=0.8084810126582278`
  - `ev_top1_roi=0.7635097813578826`
  - `wf_nested_test_roi_weighted=1.0147154471544715`
  - `wf_nested_test_bets_total=1230`
- gate read:
  - `feasible_fold_count=0/5`
  - `dominant_failure_reason=min_bets`
  - `status=block`
  - `decision=hold`
- actual-date read:
  - broad September 2025 baseline 完全同値
  - December control 2025 baseline 完全同値

その後、alternative second-block として `gate/frame/course core` も実行した。

- issue source:
  - `docs/issue_library/next_issue_pruning_stage2_calendar_recent_history_gate_frame_course_bundle.md`
- revision:
  - `r20260410_pruning_stage2_calendar_recent_history_gate_frame_course_v1`
- formal summary:
  - `auc=0.8420815082342571`
  - `top1_roi=0.8077445339470656`
  - `ev_top1_roi=0.7973993095512083`
  - `wf_nested_test_roi_weighted=0.9506373117033606`
  - `wf_nested_test_bets_total=863`
- gate read:
  - `feasible_fold_count=3/5`
  - `dominant_failure_reason=min_bets`
  - `status=pass`
  - `decision=promote`
- actual-date read:
  - broad September 2025 baseline 完全同値
  - December control 2025 baseline 完全同値

したがって current staged boundary read は次に更新される。

- `stage-1` は supported first block
- `stage-2 race-condition/dispatch` は hold boundary artifact
- `stage-2 gate/frame/course` は supported second block

つまり current reading は「stage-1 が hard ceiling」ではなく、「second block の選び方で support が分岐する」である。

## Recommended Review Outcome

最も defensible な review outcome は次である。

- decision:
  - `approve for staged simplification review, not for one-shot baseline rewrite`
- rationale:
  - individual evidence は強い
  - bundle formal support は弱い
  - actual-date replay は harmless なので、implementation risk は低く見える
  - それでも benchmark 正本更新としては hold を維持すべき

## If Review Approves A Staged Path

staged path を切るなら順序は次が自然である。

1. calendar + recent-history のような最小 block から始める
2. categorical context groups を次段に回す
3. class/rest/surface core と ID core は最後に回す

first execution source:

- `docs/issue_library/next_issue_pruning_stage1_calendar_recent_history_bundle.md`

current artifact read:

- stage-1 `calendar + recent-history` bundle `r20260410_pruning_stage1_calendar_recent_history_v1` は formal `pass / promote` と actual-date Sep/Dec equivalence まで完了した
- one-shot bundle `hold` の直後でも、最小 block では `feasible_fold_count=3/5` まで改善した
- supported branch の延長として stage-3 `calendar + recent-history + gate/frame/course core + track/weather/surface context` bundle `r20260410_pruning_stage3_calendar_recent_history_gate_frame_course_track_weather_v1` も formal `pass / promote`、`feasible_fold_count=3/5`、actual-date Sep/Dec equivalence まで完了した
- supported branch の延長として stage-4 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core` bundle `r20260410_pruning_stage4_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_v1` も formal `pass / promote`、`feasible_fold_count=2/5`、actual-date Sep/Dec equivalence まで完了した
- supported branch の延長として stage-5 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core` bundle `r20260410_pruning_stage5_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_v1` も formal `pass / promote`、`feasible_fold_count=3/5`、actual-date Sep/Dec equivalence まで完了した
- supported branch の延長として stage-6 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core + jockey/trainer/combo core` bundle `r20260411_pruning_stage6_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_v1` も formal `pass / promote`、`feasible_fold_count=2/5`、actual-date Sep/Dec equivalence まで完了した
- supported branch の延長として stage-7 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core + jockey/trainer/combo core + dispatch metadata` bundle `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1` も formal `pass / promote`、`feasible_fold_count=3/5`、actual-date Sep/Dec equivalence まで完了した
- narrow hold boundary として stage-8 `calendar + recent-history + gate/frame/course core + track/weather/surface context + class/rest/surface core + jockey/trainer ID core + jockey/trainer/combo core + dispatch metadata + condition quartet` bundle `r20260411_pruning_stage8_condq_v1` は actual-date Sep/Dec equivalence を保ったが、`feasible_fold_count=0/5`、`status=block`, `decision=hold` に戻った

理由:

- semantic risk が比較的低い block から先に剥がせる
- rollback と diff explanation を簡単に保てる
- bundle hold の原因が specific subgroup interaction なのか、単純な mass removal なのかを切り分けやすい

## Stage-7 vs Stage-8 Boundary Read

human review で最も見やすい差分は、stage-7 と stage-8 を次のように並べることである。

- stage-7 `dispatch metadata` add-on:
  - revision: `r20260411_pruning_stage7_calendar_recent_history_gate_frame_course_track_weather_class_rest_surface_jt_id_combo_dispatch_meta_v1`
  - formal: `auc=0.842043836749433`, `top1_roi=0.8029459148446491`, `ev_top1_roi=0.7295512082853856`, `wf_nested_test_roi_weighted=1.1925373134328359`
  - gate: `feasible_fold_count=3/5`, `status=pass`, `decision=promote`
  - actual-date: broad September / December control とも baseline 完全同値
- stage-8 `condition quartet` add-on:
  - revision: `r20260411_pruning_stage8_condq_v1`
  - formal: `auc=0.8422288519737056`, `top1_roi=0.8064556962025317`, `ev_top1_roi=0.7503567318757192`, `wf_nested_test_roi_weighted=0.9622002820874471`
  - gate: `feasible_fold_count=0/5`, `status=block`, `decision=hold`
  - actual-date: broad September / December control とも baseline 完全同値

この比較から読めることは次の 3 点である。

1. stage-8 の失敗は top-line 劣化ではなく、formal support の `min_bets` 制約である
2. stage-7 までの staged removal は defendable だが、remaining quartet を足すと one-shot bundle hold と同じ failure class に戻る
3. したがって staged simplification を実反映候補として人が検討するなら、natural stopping point は stage-7 である

## Artifacts To Review First

- `artifacts/reports/promotion_gate_r20260410_pruning_bundle_ablation_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_bundle_ablation_v1.json`
- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260410_pruning_bundle_ablation_v1_wf_full_nested.json`
- `artifacts/reports/serving_smoke_compare_sep25_pruning_bundle_base_vs_sep25_pruning_bundle_cand.json`
- `artifacts/reports/serving_smoke_compare_dec25_pruning_bundle_base_vs_dec25_pruning_bundle_cand.json`
- `artifacts/reports/promotion_gate_r20260410_pruning_stage1_calendar_recent_history_v1.json`
- `artifacts/reports/wf_feasibility_diag_pruning_stage1_calendar_recent_history_v1.json`

## Final Reading

2026-04-11 時点では、pruning package は「one-shot bundle promotion は hold だが、staged simplification は supported branch 上で stage-7 まで通る」と読むのが正しい。stage-8 quartet は actual-date では harmless でも formal gate では `0/5 feasible folds` に戻るため、current defendable boundary は stage-7 で止まる。したがって次の変更単位は benchmark 正本更新ではなく human review decision であり、remaining quartet は hold boundary artifact として扱うのが妥当である。