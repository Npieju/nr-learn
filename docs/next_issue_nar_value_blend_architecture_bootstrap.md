# Next Issue: NAR Value-Blend Architecture Bootstrap

## Summary

NAR を JRA と同水準の model architecture で比較できるようにするには、strict `pre_race_only` benchmark が成立した後に `value_blend` family を持ち込む必要がある。

current NAR は dataset readiness corrective が先行しており、LightGBM baseline line までは formalized されているが、JRA 本線の

- CatBoost win
- LightGBM ROI
- stack / `value_blend`

までは揃っていない。

したがって post-readiness の first measurable architecture issue は、NAR `value_blend` bootstrap である。

## Objective

strict `pre_race_only` NAR benchmark 上で、JRA current standard と同型の `CatBoost win + LightGBM ROI + value_blend stack` を bootstrap し、NAR baseline を architecture parity まで引き上げる。

## Hypothesis

if NAR strict `pre_race_only` benchmark 上で JRA-style `value_blend` architecture を train / evaluate できる, then NAR は LightGBM baseline ではなく JRA と同じ model-development surface で feature / policy compare を進められる。

## In Scope

- NAR 用 win / ROI / stack config
- component retrain 導線
- stack bundle 導線
- strict `pre_race_only` benchmark 上の formal compare

## Non-Goals

- provenance 不足の backfilled benchmark で architecture parity を主張すること
- JRA baseline 更新
- mixed-universe compare
- serving default promotion

## Success Metrics

- NAR strict `pre_race_only` benchmark で
  - CatBoost win retrain
  - LightGBM ROI retrain
  - `value_blend` stack bundle
  - formal evaluation
  が end-to-end で通る
- artifact / gate / actual-date compare の surface が JRA と揃う

## Validation Plan

1. strict `pre_race_only` benchmark rerun を baseline freeze として固定する
2. NAR 用 component config を用意する
3. component retrain
4. stack bundle
5. formal evaluation / promotion gate

## Stop Condition

- `#101` / `#102` が未完了で provenance-defensible benchmark が無い
- strict `pre_race_only` benchmark の row / race support が architecture compare に足りない
- その場合はこの issue は blocked 扱いで維持する

## First Read

`#102` は negative read で close したため、current blocker は `#101` の result-ready strict `pre_race_only` benchmark rerun のみである。

ただし architecture 側は、JRA current standard をそのまま NAR に落とせば済むわけではない。`rich_high_coverage_diag` を local Nankan に対して feature-gap した結果、bootstrap の block は sample support だけでなく feature schema mismatch も含むと確定した。

実行:

- data config: `configs/data_local_nankan.yaml`
- feature config: `configs/features_catboost_rich_high_coverage_diag.yaml`
- model config: `configs/model_catboost_win_high_coverage_diag.yaml`
- `max_rows=50000`
- output:
  - [feature_gap_summary_nar_value_blend_bootstrap_first_read.json](/workspaces/nr-learn/artifacts/reports/feature_gap_summary_nar_value_blend_bootstrap_first_read.json)
  - [feature_gap_feature_coverage_nar_value_blend_bootstrap_first_read.csv](/workspaces/nr-learn/artifacts/reports/feature_gap_feature_coverage_nar_value_blend_bootstrap_first_read.csv)
  - [feature_gap_raw_column_coverage_nar_value_blend_bootstrap_first_read.csv](/workspaces/nr-learn/artifacts/reports/feature_gap_raw_column_coverage_nar_value_blend_bootstrap_first_read.csv)

confirmed read:

- template raw columns:
  - `25 / 25 present`
  - `priority_missing_raw_columns=[]`
- feature frame:
  - `feature_columns_total=87`
  - `selected_feature_count=52`
  - `categorical_feature_count=9`
- missing force-include features:
  - `course_gate_bucket_last_100_avg_rank`
  - `course_gate_bucket_last_100_win_rate`
  - `horse_class_change`
  - `horse_class_down_long_layoff`
  - `horse_class_down_short_turnaround`
  - `horse_class_up_long_layoff`
  - `horse_class_up_short_turnaround`
  - `horse_is_class_down`
  - `horse_is_class_up`
  - `horse_last_class_score`
  - `horse_surface_switch`
  - `horse_surface_switch_long_layoff`
  - `horse_surface_switch_short_turnaround`
  - `race_class_score`
- low coverage force-include features:
  - `[]`

一方で、次は rich config の中でも NAR にそのまま乗る。

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `gate_ratio`
- `frame_ratio`
- `owner_last_50_win_rate`

meaning:

- `#103` の block は strict `pre_race_only` benchmark readiness だけではない
- current local Nankan では、JRA `rich_high_coverage_diag` の full parity config をそのまま one-shot で持ち込むと force-include mismatch が残る
- したがって post-readiness の first bootstrap は
  - JRA stack / gate / artifact discipline を再利用しつつ
  - NAR buildable subset に narrowed した win / ROI / stack config
  から始めるのが妥当である

next cut:

- `#101` result-ready benchmark ができたら
  - NAR `value_blend` bootstrap 用の narrowed feature config
  - NAR 用 win / ROI / stack config
  を scaffold する

## First Cut

result-ready benchmark 到着前に、post-readiness rerun でそのまま使う narrowed scaffold config を追加した。

- feature config:
  - [features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- win config:
  - [model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- roi config:
  - [model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- stack config:
  - [model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml)

scaffold rule:

- JRA stack / gate / artifact discipline は再利用する
- force-include mismatch が出た class / surface / course-gate-bucket 群は入れない
- current local Nankan で buildable と確認できた subset に narrow する
- policy / policy_search は current local Nankan formal line に近い conservative setting を使う

included bootstrap subset:

- `horse_last_3_avg_rank`
- `horse_last_5_win_rate`
- `horse_days_since_last_race`
- `horse_weight_change`
- `horse_distance_change`
- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`
- `gate_ratio`
- `frame_ratio`
- `owner_last_50_win_rate`

remaining blocker:

- まだ strict `pre_race_only` result-ready benchmark が無いので、train / evaluate 実行はしていない
- `#101` が ready になったら、この scaffold を起点に component retrain と stack bundle に入る

validation:

- config parse:
  - all 4 scaffold YAML files loaded successfully
- narrowed feature-gap:
  - [feature_gap_summary_nar_value_blend_bootstrap_narrowed_v1.json](/workspaces/nr-learn/artifacts/reports/feature_gap_summary_nar_value_blend_bootstrap_narrowed_v1.json)
  - [feature_gap_feature_coverage_nar_value_blend_bootstrap_narrowed_v1.csv](/workspaces/nr-learn/artifacts/reports/feature_gap_feature_coverage_nar_value_blend_bootstrap_narrowed_v1.csv)
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`

meaning:

- `#103` の first cut scaffold は local Nankan schema に整合している
- post-readiness rerun の blocker は config mismatch ではなく、strict `pre_race_only` result-ready benchmark 未到着だけになった

## Ready-Run Plan

`#101` が result-ready になった後の再開導線は、ad hoc に train / stack / gate を個別起動するのではなく、`run_local_nankan_result_ready_bootstrap_handoff.py` を entrypoint に固定する。

blocked 中の canonical read は次である。

- board:
  - [local_nankan_data_status_board.json](/workspaces/nr-learn/artifacts/reports/local_nankan_data_status_board.json)
- primary fields:
  - `status`
  - `current_phase`
  - `recommended_action`
  - `readiness_surfaces.readiness_probe`
  - `readiness_surfaces.pre_race_handoff`
  - `readiness_surfaces.bootstrap_handoff`
  - `readiness_surfaces.readiness_watcher`

rule:

- blocked 中は、まず board を見る
- board が `status=partial`, `current_phase=await_result_arrival` なら current blocker は external result arrival のままと読む
- 詳細が必要なときだけ個別 manifest に降りる

default surface:

- data config:
  - [data_local_nankan_pre_race_ready.yaml](/workspaces/nr-learn/configs/data_local_nankan_pre_race_ready.yaml)
- feature config:
  - [features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- win config runtime materialize source:
  - [model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- roi config runtime materialize source:
  - [model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml)
- stack config runtime materialize source:
  - [model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml](/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml)
- wrapper:
  - [run_local_nankan_result_ready_bootstrap_handoff.py](/workspaces/nr-learn/scripts/run_local_nankan_result_ready_bootstrap_handoff.py)

operator command:

```bash
cd /workspaces/nr-learn
PYTHONPATH=/workspaces/nr-learn/src /workspaces/nr-learn/.venv/bin/python \
  scripts/run_local_nankan_result_ready_bootstrap_handoff.py \
  --wait-for-results \
  --poll-interval-seconds 300 \
  --run-bootstrap
```

expected phase contract:

1. `status=not_ready`
   - external result arrival 待ちなので keep blocked
   - read:
     - [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json)
     - [local_nankan_pre_race_ready_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_ready_summary.json)
2. `status=benchmark_ready`
   - strict `pre_race_only` benchmark freeze は通ったが child execution は未実行
   - `bootstrap_command_plan` と `runtime_configs` を確認して rerun policy を最終確認する
3. `status=completed`
   - win / ROI / stack / revision gate まで完走
   - revision artifact を formal read に進める

primary artifacts to read:

- [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json)
- [local_nankan_pre_race_benchmark_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json)
- [benchmark_gate_local_nankan_pre_race_ready.json](/workspaces/nr-learn/artifacts/reports/benchmark_gate_local_nankan_pre_race_ready.json)
- [local_nankan_primary_pre_race_ready_materialize_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_primary_pre_race_ready_materialize_manifest.json)

go / no-go read:

- go:
  - handoff manifest `status=completed`
  - wrapper manifest `status=completed` or at least `benchmark_ready`
  - result-ready race support が 0 ではない
- no-go:
  - wrapper manifest `status=not_ready`
  - benchmark handoff failed
  - runtime config materialization は通るが child command が non-zero exit

meaning:

- `#103` は now blocked issue だが、resume path 自体は fixed されている
- unblock 後は `#101` handoff 完了確認と同時に、同じ entrypoint から architecture bootstrap を再現できる

## Formal Read Contract

bootstrap 完走後の判定は、top-line ROI だけで読まず、[nar_formal_read_template.md](/workspaces/nr-learn/docs/nar_formal_read_template.md) と current NAR promotion threshold に揃える。

minimum read order:

1. `policy_bets`
2. `race_count`
3. `row_count`
4. `feasible_race_count`
5. `bet_rate_races`
6. `bet_rate_rows`
7. `bet_rate_feasible_races`
8. `weighted_roi`
9. `total_net`
10. `feasible_folds / total_folds`
11. `zero_bet_dates / total_dates`
12. concentration / low-support warning
13. `promote / hold / reject`

required compare:

- current NAR baseline line と bootstrap challenger を次で比較する
  - `bets / races`
  - `weighted_roi`
  - `total_net`
  - `wf_feasible_fold_count`
- read は suppression なのか real improvement なのかを明示する

decision floor:

- current NAR promotion gate では held-out formal `weighted_roi < 1.0` は `hold` で読む
- したがって `#103` でも、support があっても held-out formal `weighted_roi < 1.0` line は `promote` にしない
- `status=pass` と operational / issue decision を混同せず、必要なら `formal pass but hold` を明示する

minimum artifact set after completion:

- revision gate summary
- promotion gate report
- wf feasibility summary
- evaluation summary
- wrapper manifest / handoff manifest

issue summary template:

```text
Formal read:
- policy_bets: <bets>
- race_count: <races>
- row_count: <rows>
- feasible_race_count: <feasible_races>
- bet_rate_races: <bets/races>
- bet_rate_rows: <bets/rows>
- bet_rate_feasible_races: <bets/feasible_races>
- weighted_roi: <weighted_roi>
- total_net: <total_net>
- feasible_folds: <feasible_folds>/<total_folds>
- zero_bet_dates: <zero_bet_dates>/<total_dates>

Compare:
- baseline: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- challenger: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- delta_bets: <delta>
- delta_net: <delta>
- delta_weighted_roi: <delta>

Interpretation:
- <support / concentration / suppression read>

Decision:
- <promote|hold|reject>
- <if needed: formal promoted but operational hold>
```

meaning:

- unblock 後の `#103` は execution だけでなく judgment surface も fixed された
- result-ready 到着後は、wrapper 完走 -> formal artifact read -> issue decision summary まで同じ手順で閉じられる

## Issue Thread Templates

`#103` の issue thread には、phase ごとに次の template をそのまま使う。

### 1. Still Blocked

```text
Current read:
- wrapper status: not_ready
- blocker: external result arrival for `#101`
- recommended action: keep blocked

Artifacts:
- local_nankan_pre_race_ready_summary.json
- local_nankan_pre_race_benchmark_handoff_manifest.json
- local_nankan_result_ready_bootstrap_handoff_manifest.json

Read:
- result-ready race support is still zero
- `#103` execution surface remains fixed and does not need further command reconstruction
```

### 2. Benchmark Ready, Bootstrap Pending

```text
Current read:
- wrapper status: benchmark_ready
- benchmark handoff status: completed
- current phase: bootstrap_pending

Artifacts:
- local_nankan_pre_race_ready_summary.json
- local_nankan_pre_race_benchmark_handoff_manifest.json
- benchmark_gate_local_nankan_pre_race_ready.json
- local_nankan_result_ready_bootstrap_handoff_manifest.json

Runtime configs:
- win: <runtime win config path>
- roi: <runtime roi config path>
- stack: <runtime stack config path>

Read:
- strict `pre_race_only` benchmark freeze is complete
- bootstrap command plan is ready
- next action is to run child commands or confirm auto-run completion
```

### 3. Completed First Read

```text
Current read:
- wrapper status: completed
- benchmark handoff status: completed
- revision: <revision>

Formal read:
- policy_bets: <bets>
- race_count: <races>
- row_count: <rows>
- feasible_race_count: <feasible_races>
- bet_rate_races: <bets/races>
- bet_rate_rows: <bets/rows>
- bet_rate_feasible_races: <bets/feasible_races>
- weighted_roi: <weighted_roi>
- total_net: <total_net>
- feasible_folds: <feasible_folds>/<total_folds>
- zero_bet_dates: <zero_bet_dates>/<total_dates>

Compare:
- baseline: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- challenger: <bets>/<races> = <bet_rate>, net=<net>, weighted_roi=<weighted_roi>
- delta_bets: <delta>
- delta_net: <delta>
- delta_weighted_roi: <delta>

Interpretation:
- <support / concentration / suppression / real improvement read>

Decision:
- <promote|hold|reject>
- <if needed: formal pass but hold>

Artifacts:
- revision gate summary: <path>
- promotion gate report: <path>
- wf feasibility summary: <path>
- evaluation summary: <path>
- wrapper manifest: artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json
```

meaning:

- issue thread update も phase ごとに fixed された
- unblock 後は runbook, formal read, issue comment まで同じ template で進められる
