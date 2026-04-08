# Next Issue: Jockey Trainer Combo Closing-Time Selective Candidate

## Summary

`jockey / trainer / combo` family の first child `r20260330_jockey_trainer_combo_style_distance_v1` は formal `pass / promote` まで到達したが、actual-date role は `analysis-first promoted candidate` に留まった。

次に同 family を再開するなら、前回の broad add-on をそのまま繰り返すのではなく、より narrow な child hypothesis に切り直すべきである。

今回の first child は、`closing_time_3f` 系 2 本だけを current JRA high-coverage line に追加する selective candidate とする。

## Objective

`jockey_last_30_avg_closing_time_3f` と `trainer_last_30_avg_closing_time_3f` を current JRA high-coverage line に narrow add-on し、`jockey / trainer / combo` family の second child として formal compare に載せる価値があるかを判定する。

## Hypothesis

if `jockey / trainer / combo` family の richer side のうち `closing_time_3f` pair だけを selective に追加する, then broad `style + track-distance` child より redundancy を抑えつつ、current baseline に対して support を壊さない second child candidate を作れる。

## Current Read

- `feature_family_ranking.md` では `jockey / trainer / combo` family は Tier A
- first child `r20260330_jockey_trainer_combo_style_distance_v1` は formal `pass / promote` まで到達した
- ただし actual-date role split では family anchor にならず、`analysis-first promoted candidate` に留まった
- broad child には
  - corner-gain pair
  - closing-time pair
  - track-distance quartet
  が同時に入っており、どこが効いたか切り分けが弱い
- `closing_time_3f` pair は builder に既に存在し、pace/closing family の recent read とも意味が接続する

## Candidate Definition

keep current JRA high-coverage core:

- `jockey_last_30_win_rate`
- `trainer_last_30_win_rate`
- `jockey_trainer_combo_last_50_win_rate`
- `jockey_trainer_combo_last_50_avg_rank`

add selective pair:

- `jockey_last_30_avg_closing_time_3f`
- `trainer_last_30_avg_closing_time_3f`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_closing_time_selective.yaml`

## In Scope

- feature-gap / coverage read
- true component retrain ready な selective config
- formal compare へ進めるかどうかの execution decision

## Non-Goals

- broad `style + track-distance` child の再実行
- new builder implementation
- NAR work
- serving role split の再議論

## Success Criteria

- `closing_time_3f` pair の coverage / selection risk を独立に読める
- narrow child として true component retrain に載せる価値があるか判断できる
- `jockey / trainer / combo` family の second child hypothesis が 1 本に固定される

## Validation Plan

1. feature-gap / coverage を読み、2 本が `present=True` かつ low-coverage ではないことを確認する
2. win / roi component retrain に進める場合は true component flow を使う
3. formal compare では
   - `auc`
   - `top1_roi`
   - `ev_top1_roi`
   - nested WF shape
   - held-out formal `weighted_roi`
   - `bets / races / bet_rate`
   を baseline と比較する

## Stop Condition

- `closing_time_3f` pair のどちらかが low coverage か missing
- selected set に乗らず no-op の可能性が高い
- broad first child と比べて hypothesis としての独立性が弱い

## First Read

- feature-gap:
  - `artifacts/reports/feature_gap_summary_jockey_trainer_combo_closing_time_selective_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_jockey_trainer_combo_closing_time_selective_v1.csv`
- summary:
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - `selected_feature_count=109`
  - `categorical_feature_count=37`
- focal coverage:
  - `jockey_last_30_avg_closing_time_3f`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.99727`
  - `trainer_last_30_avg_closing_time_3f`
    - `selected=True`
    - `present=True`
    - `non_null_ratio=0.99635`

interpretation:

- closing-time pair は high coverage で clean に build / select される
- no-op でも low-coverage でもない
- second child として true component retrain に進めてよい

## Final Read

- revision:
  - `r20260404_jockey_trainer_combo_closing_time_selective_v1`
- evaluation summary:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260404_jockey_trainer_combo_closing_time_selective_v1.json`
  - `auc=0.8426169492248933`
  - `top1_roi=0.8082087836284203`
  - `ev_top1_roi=0.8213612324672338`
  - `wf_nested_test_roi_weighted=0.7632385120350111`
  - `wf_nested_test_bets_total=457`
  - nested WF shape は `portfolio / portfolio / portfolio`
- promotion / formal:
  - `artifacts/reports/promotion_gate_r20260404_jockey_trainer_combo_closing_time_selective_v1.json`
  - `status=pass`
  - `decision=promote`
  - held-out formal `weighted_roi=1.2311149102465346`
  - `formal_benchmark_bets_total=9806`
  - `formal_benchmark_feasible_fold_count=3`
  - `bets / races / bet_rate = 9806 / 57620 = 17.02%`

interpretation:

- narrow closing-time pair child は no-op ではなく、formal candidate として成立した
- top-line は current JRA promoted candidates と比較可能な水準まで到達した
- ただし serving default や seasonal fallback に上げるかは actual-date read が必要
- したがって `#90` は feature child hypothesis issue として完了し、次は role split issue に進む
