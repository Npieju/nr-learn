# Next Issue: JRA Market Deviation LightGBM Trade-Off Merge

## Summary

LightGBM alpha の single-lever probes は 2 本まで完了した。

- `r20260418_jra_lightgbm_alpha_leaf40_v1` は `alpha_pred_corr=0.2079891249` まで改善したが、`positive_signal_rate=0.0061666667` で coverage recovery は起きなかった
- `r20260418_jra_lightgbm_alpha_clip6_v1` は `positive_signal_rate=0.0069466667` まで改善し、`alpha_pred_corr=0.1844760397` も維持した

したがって next hypothesis は broad tuning ではない。`leaf40` の corr-side lever と `clip6` の coverage-side lever を同時に入れたとき、trade-off front を前に動かせるかである。

## Objective

JRA `market_deviation` の LightGBM challenger について、`leaf40` と `clip6` を同時に入れた merge variant が corr-side / coverage-side の両 read を base challenger より前進させられるかを判定する。

## Hypothesis

if `min_data_in_leaf: 80 -> 40` と `target_clip: 8.0 -> 6.0` を同時に入れる, then LightGBM alpha は `leaf40` の corr-side gain を一部保ちながら `clip6` の coverage-side gain も取り込み、base challenger `r20260418_jra_lightgbm_alpha_challenger_v1` より良い trade-off point に移れる。

## In Scope

- JRA universe 固定
- `market_deviation` track 固定
- LightGBM alpha challenger 固定
- dataset / feature set / evaluation surface 固定
- レバーは `min_data_in_leaf` と `target_clip` の 2 点だけ

## Non-Goals

- CatBoost incumbent の即時置換
- feature / dataset / policy の同時変更
- LightGBM parameter sweep への拡張
- probability-centric gate への回帰

## Baseline And Comparator

right reference baseline:

- baseline freeze tag: `strategy-baseline-20260418-pre-track-split`
- market deviation incumbent candidate:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json)

LightGBM compare set:

- base challenger:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json)
- corr-side variant:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json)
- coverage-side variant:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_clip6_model_r20260418_jra_lightgbm_alpha_clip6_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_clip6_model_r20260418_jra_lightgbm_alpha_clip6_v1_wf_off_nested.json)

## Current Read

base challenger:

- `alpha_pred_corr=0.1831187046`
- `positive_signal_rate=0.0064266667`
- `pred_mean=-4.8852264273`
- `pred_std=1.1604059493`

leaf40:

- `alpha_pred_corr=0.2079891249`
- `positive_signal_rate=0.0061666667`
- `pred_mean=-4.8257131395`
- `pred_std=1.3345624348`

clip6:

- `alpha_pred_corr=0.1844760397`
- `positive_signal_rate=0.0069466667`
- `pred_mean=-4.5976170647`
- `pred_std=1.4571416610`

read:

- `leaf40` は corr-side を押し上げた
- `clip6` は coverage-side を押し上げた
- まだどちらも CatBoost incumbent の coverage には遠い
- 次は 2 レバー merge が front を少しでも前進させるかを見る価値がある

## Candidate Definition

first candidate config は次とする。

- [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag_leaf40_clip6.yaml](../configs/model_lightgbm_alpha_cpu_diag_leaf40_clip6.yaml)

変更点は次の 2 つだけである。

- `min_data_in_leaf: 80 -> 40`
- `target_clip: 8.0 -> 6.0`

## Primary Metrics

1. `alpha_pred_corr`
2. `positive_signal_rate`
3. `pred_mean` / `pred_std`
4. market-relative OOS stability
5. secondary read として `top1_roi` / `ev_threshold_1_0_roi`

## Success Metrics

- `alpha_pred_corr` が base challenger `0.1831187046` を上回る
- `positive_signal_rate` が base challenger `0.0064266667` を上回る
- できれば corr-side variant `leaf40` と coverage-side variant `clip6` の中間以上に入る
- `stability_assessment=representative` を維持する
- compare read を `keep as candidate / reject` のどちらかで更新できる

## Validation Plan

1. candidate config で fresh train / evaluate を実行する
2. incumbent CatBoost / base challenger / leaf40 / clip6 / merge candidate の 5 点を compare rule で読む
3. merge candidate が trade-off front を押し出せるかを確認する

## Result

first candidate `r20260418_jra_lightgbm_alpha_leaf40_clip6_v1` の train / evaluate は完了した。

- train artifact:
  - [/workspaces/nr-learn/artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_leaf40_clip6_r20260418_jra_lightgbm_alpha_leaf40_clip6_v1.json](../artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_leaf40_clip6_r20260418_jra_lightgbm_alpha_leaf40_clip6_v1.json)
- evaluate summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_clip6_model_r20260418_jra_lightgbm_alpha_leaf40_clip6_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_clip6_model_r20260418_jra_lightgbm_alpha_leaf40_clip6_v1_wf_off_nested.json)

confirmed OOS read:

- `alpha_pred_corr=0.1740577483`
- `positive_signal_rate=0.0062400000`
- `pred_mean=-4.6064838189`
- `pred_std=1.4206430946`
- `top1_roi=0.4322693812`
- `ev_threshold_1_0_roi=1.1418518519`
- `stability_assessment=representative`

compare read:

- vs base challenger: `alpha_pred_corr` は悪化 (`0.1831187046 -> 0.1740577483`)、`positive_signal_rate` も悪化 (`0.0064266667 -> 0.0062400000`)
- vs leaf40: coverage はわずかに改善したが、corr は大きく悪化
- vs clip6: coverage / corr の両方で劣後した

したがって merge candidate は trade-off front を押し出せなかった。

## Decision

current decision は `reject` とする。

reason:

- base challenger を corr / coverage の両 primary metric で上回れなかった
- clip6 coverage-side variant と leaf40 corr-side variant を単純結合しても front improvement は起きなかった
- 以後は連想的に multi-lever probe を足すのではなく、architecture-level roadmap に戻して次 stage を選ぶべきである

## Stop Condition

- corr / coverage のどちらかしか取れず、merge の意味がない
- base challenger をどちらの primary metric でも上回れない
- これ以上は parameter sweep 化しないと説明できない

## Follow-Up Boundary

- この issue では `leaf40 + clip6` merge 1 本だけを扱う
- 追加の multi-lever sweep は別 issue に分離する