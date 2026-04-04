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
