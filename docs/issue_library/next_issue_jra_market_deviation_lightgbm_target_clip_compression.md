# Next Issue: JRA Market Deviation LightGBM Target Clip Compression

## Summary

`r20260418_jra_lightgbm_alpha_challenger_v1` と `r20260418_jra_lightgbm_alpha_leaf40_v1` により、LightGBM alpha は JRA `market_deviation` track で一貫して次の shape を示した。

- `alpha_pred_corr` は CatBoost incumbent を上回る
- ただし `positive_signal_rate` は `0.0064` 前後で collapse したままである
- `min_data_in_leaf: 80 -> 40` でも coverage recovery は起きなかった

したがって next lever は tree sparsity ではなく、target 生成側の負側 tail を少し圧縮できるかに置く。

## Objective

JRA `market_deviation` の LightGBM challenger について、higher corr を大きく壊さずに negative tail compression で positive signal coverage を戻せるかを narrow hypothesis で判定する。

## Hypothesis

if `configs/model_lightgbm_alpha_cpu_diag.yaml` の `target_clip` だけを `8.0 -> 6.0` に下げる, then extreme negative market-deviation target が圧縮され、LightGBM alpha の score center がわずかに持ち上がり、`alpha_pred_corr` を大きく失わずに `positive_signal_rate` を回復できる。

## In Scope

- JRA universe 固定
- `market_deviation` track 固定
- LightGBM alpha challenger 固定
- dataset / feature set / evaluation surface 固定
- single lever として `target_clip` のみ変更

## Non-Goals

- CatBoost incumbent をこの issue 単体で置換すること
- features / dataset scope を同時に変えること
- policy tuning で secondary ROI を押し上げること
- LightGBM family の multi-lever tune に広げること

## Baseline And Comparator

right reference baseline:

- baseline freeze tag: `strategy-baseline-20260418-pre-track-split`
- operational baseline profile: `current_recommended_serving_2025_latest`
- market deviation incumbent candidate:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json)

current LightGBM references:

- base challenger:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json)
- rejected leaf40 variant:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json)

## Current Read

CatBoost incumbent candidate:

- `alpha_pred_corr=0.1645068790`
- `positive_signal_rate=0.0655266667`
- `pred_mean=-4.7689093875`
- `pred_std=1.5306004124`

LightGBM challenger:

- `alpha_pred_corr=0.1831187046`
- `positive_signal_rate=0.0064266667`
- `pred_mean=-4.8852264273`
- `pred_std=1.1604059493`

LightGBM leaf40:

- `alpha_pred_corr=0.2079891249`
- `positive_signal_rate=0.0061666667`
- `pred_mean=-4.8257131395`
- `pred_std=1.3345624348`

read:

- `min_data_in_leaf` を緩めると corr は上がったが coverage は戻らなかった
- したがって current bottleneck は split sparsity だけではなく、negative target tail が強すぎる可能性がある
- `market_deviation` target は `observed_logit - market_logit` を `±target_clip` で clip しているため、ここを 1 段圧縮する価値がある

## Candidate Definition

first candidate config は次とする。

- [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag_clip6.yaml](../configs/model_lightgbm_alpha_cpu_diag_clip6.yaml)

変更点は 1 つだけである。

- `target_clip: 8.0 -> 6.0`

それ以外の `data_config`, `feature_config`, `odds_clip`, `policy constraints`, `wf_mode` は固定する。

## Primary Metrics

この issue の primary read は次の順で扱う。

1. `positive_signal_rate`
2. `alpha_pred_corr`
3. `pred_mean` / `pred_std`
4. market-relative OOS stability
5. secondary read として `top1_roi` / `ev_threshold_1_0_roi`

## Success Metrics

- `positive_signal_rate` が current LightGBM challenger `0.0064266667` と rejected leaf40 `0.0061666667` の両方を上回る
- `alpha_pred_corr` が current LightGBM challenger `0.1831187046` を大きく下回らない
- `pred_mean` が負側へさらに悪化しない
- `stability_assessment=representative` を維持する
- compare read を `keep as candidate / reject` のどちらかで更新できる

## Validation Plan

1. candidate config で fresh train / evaluate を実行する
2. incumbent CatBoost / current LightGBM challenger / rejected leaf40 / clip6 challenger の 4 点を compare rule で読む
3. `higher corr / lower coverage` が `higher corr / slightly recovered coverage` に変わるかを確認する

## Result

first candidate `r20260418_jra_lightgbm_alpha_clip6_v1` の train / evaluate は完了した。

- train artifact:
  - [/workspaces/nr-learn/artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_clip6_r20260418_jra_lightgbm_alpha_clip6_v1.json](../artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_clip6_r20260418_jra_lightgbm_alpha_clip6_v1.json)
- evaluate summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_clip6_model_r20260418_jra_lightgbm_alpha_clip6_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_clip6_model_r20260418_jra_lightgbm_alpha_clip6_v1_wf_off_nested.json)
- evaluate manifest:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_clip6_model_r20260418_jra_lightgbm_alpha_clip6_v1_wf_off_nested.json](../artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_clip6_model_r20260418_jra_lightgbm_alpha_clip6_v1_wf_off_nested.json)

confirmed OOS read:

- `top1_roi=0.4466643662`
- `ev_threshold_1_0_roi=1.1215088283`
- `ev_threshold_1_2_roi=1.1457504521`
- `alpha_pred_corr=0.1844760397`
- `positive_signal_rate=0.0069466667`
- `pred_mean=-4.5976170647`
- `pred_std=1.4571416610`
- `stability_assessment=representative`

compare read vs current LightGBM challenger:

- `alpha_pred_corr` は `0.1831187046 -> 0.1844760397` で小幅改善
- `positive_signal_rate` は `0.0064266667 -> 0.0069466667` で小幅改善
- `pred_mean` は `-4.8852264273 -> -4.5976170647` で負側バイアスが緩和
- `pred_std` は `1.1604059493 -> 1.4571416610` で拡大
- `top1_roi` は `0.4286749482 -> 0.4466643662` で改善
- `ev_threshold_1_0_roi` は `0.9521551724 -> 1.1215088283` で `1.0` を上回った

compare read vs rejected leaf40:

- `alpha_pred_corr` は leaf40 `0.2079891249` より低い
- ただし `positive_signal_rate` は leaf40 `0.0061666667` より改善している
- したがって clip6 は `corr max` ではなく `coverage side` の better contrast variant と読める

compare read vs CatBoost incumbent:

- `alpha_pred_corr` はなお CatBoost `0.1645068790` を上回る
- ただし `positive_signal_rate` は CatBoost `0.0655266667` に対して依然として大幅に低い

したがって `target_clip: 8.0 -> 6.0` は coverage collapse の root fix ではないが、single-lever としては初めて OOS coverage を改善した。

## Decision

current decision は `keep as candidate` とする。

reason:

- objective だった coverage recovery は小幅ながら達成した
- `alpha_pred_corr` は current LightGBM challenger を下回らず、identity を維持した
- `pred_mean` の負側バイアスも緩和しており、target-tail compression 仮説には一定の説明力がある
- ただし改善幅は小さく、CatBoost incumbent 置換を主張できる水準ではない

meaning:

- clip6 は current LightGBM line の `best coverage-side contrast variant` として保持する
- leaf40 は `corr-side`, clip6 は `coverage-side` の contrast read として役割が分かれた
- 次段は両者の trade-off を 1 本の follow-up hypothesis にまとめる余地がある

## Stop Condition

- coverage が回復しない
- `alpha_pred_corr` の下落が大きい
- `pred_mean` が改善せず、tail compression 仮説の説明力がない
- `target_clip` 単独では足りず multi-lever へ進まないと説明できなくなる

## Follow-Up Boundary

- この issue では LightGBM alpha の target-tail compression 1 本だけを扱う
- `odds_clip` / `market_prob_floor` / feature / policy の multi-lever tuning は次 issue に分離する