# Next Issue: JRA Market Deviation LightGBM Coverage Recovery

## Summary

`r20260418_jra_lightgbm_alpha_challenger_v1` により、LightGBM alpha は JRA `market_deviation` track で first challenger compare まで到達した。

current read は明確である。

- `alpha_pred_corr` は CatBoost incumbent を上回った
- ただし `positive_signal_rate` は `0.0655266667 -> 0.0064266667` まで低下した
- したがって current role は incumbent replacement ではなく `higher corr / lower coverage` の contrast challenger である

ここで次に見るべき論点は broad な model rewrite ではない。LightGBM challenger の coverage collapse を、1 本のレバーでどこまで戻せるかである。

## Objective

JRA `market_deviation` の LightGBM challenger について、higher corr を大きく壊さずに positive signal coverage を回復できるかを narrow hypothesis で判定する。

## Hypothesis

if `configs/model_lightgbm_alpha_cpu_diag.yaml` の `min_data_in_leaf` だけを `80 -> 40` に緩める, then LightGBM alpha の score sparsity が緩和され、`alpha_pred_corr` を大きく失わずに `positive_signal_rate` を回復できる。

## In Scope

- JRA universe 固定
- `market_deviation` track 固定
- LightGBM alpha challenger 固定
- dataset / feature set / evaluation surface 固定
- single lever として `min_data_in_leaf` のみ変更

## Non-Goals

- CatBoost incumbent をこの issue 単体で置換すること
- features / dataset scope を同時に変えること
- policy tuning で secondary ROI を押し上げること
- probability-centric gate へ戻すこと

## Baseline And Comparator

right reference baseline:

- baseline freeze tag: `strategy-baseline-20260418-pre-track-split`
- operational baseline profile: `current_recommended_serving_2025_latest`
- market deviation incumbent candidate:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json)

current LightGBM challenger reference:

- [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json)

## Current Read

CatBoost incumbent candidate:

- `alpha_pred_corr=0.1645068790`
- `positive_signal_rate=0.0655266667`
- `top1_roi=0.3864619278`
- `ev_threshold_1_0_roi=0.8931734317`

LightGBM challenger:

- `alpha_pred_corr=0.1831187046`
- `positive_signal_rate=0.0064266667`
- `top1_roi=0.4286749482`
- `ev_threshold_1_0_roi=0.9521551724`

train-side read も同方向である。

- train `alpha_pred_corr=0.2305806236`
- train `positive_signal_rate=0.0059801272`

したがって current bottleneck は OOS の偶然ではなく、LightGBM alpha 自体がかなり sparse な score distribution を学習していることにある。

## Candidate Definition

first candidate config は次とする。

- [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag_leaf40.yaml](../configs/model_lightgbm_alpha_cpu_diag_leaf40.yaml)

変更点は 1 つだけである。

- `min_data_in_leaf: 80 -> 40`

それ以外の `data_config`, `feature_config`, `target_clip`, `odds_clip`, `policy constraints`, `wf_mode` は固定する。

## Primary Metrics

この issue の primary read は次の順で扱う。

1. `positive_signal_rate`
2. `alpha_pred_corr`
3. market-relative OOS stability
4. secondary read として `top1_roi` / `ev_threshold_1_0_roi`

## Success Metrics

- `positive_signal_rate` が current LightGBM challenger `0.0064266667` から有意に回復する
- `alpha_pred_corr` が current LightGBM challenger `0.1831187046` から大きく崩れない
- `stability_assessment=representative` を維持する
- compare read を `keep as candidate / reject` のどちらかで更新できる

## Validation Plan

1. candidate config で train artifact reuse なしの fresh evaluate を実行する
2. incumbent CatBoost / current LightGBM challenger / leaf40 challenger の 3 点を compare rule で読む
3. `higher corr / lower coverage` が `higher corr / partial coverage recovery` に変わるかを確認する

## Result

first candidate `r20260418_jra_lightgbm_alpha_leaf40_v1` の train / evaluate は完了した。

- train artifact:
  - [/workspaces/nr-learn/artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_leaf40_r20260418_jra_lightgbm_alpha_leaf40_v1.json](../artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_leaf40_r20260418_jra_lightgbm_alpha_leaf40_v1.json)
- evaluate summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json)
- evaluate manifest:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json](../artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_leaf40_model_r20260418_jra_lightgbm_alpha_leaf40_v1_wf_off_nested.json)

confirmed OOS read:

- `top1_roi=0.4253278123`
- `ev_threshold_1_0_roi=1.1487654321`
- `ev_threshold_1_2_roi=1.1635135135`
- `alpha_pred_corr=0.2079891249`
- `positive_signal_rate=0.0061666667`
- `stability_assessment=representative`

compare read vs current LightGBM challenger:

- `alpha_pred_corr` は `0.1831187046 -> 0.2079891249` で改善
- `positive_signal_rate` は `0.0064266667 -> 0.0061666667` で改善せず、coverage recovery は失敗
- `top1_roi` は `0.4286749482 -> 0.4253278123` で横ばい以下
- `ev_threshold_1_0_roi` は `0.9521551724 -> 1.1487654321` で改善したが、primary hypothesis の判定軸ではない

compare read vs CatBoost incumbent:

- `alpha_pred_corr` はなお CatBoost `0.1645068790` を上回る
- ただし `positive_signal_rate` は CatBoost `0.0655266667` に対して依然として 10 分の 1 未満である

したがって、`min_data_in_leaf: 80 -> 40` だけでは `higher corr / lower coverage` の identity は変わらない。

## Decision

current decision は `reject` とする。

reason:

- この issue の objective は coverage recovery であり、`positive_signal_rate` は回復しなかった
- `alpha_pred_corr` は改善したが、single-lever 変更で coverage collapse を解消できないことが確認できた
- ROI の改善は見えるが、primary read を overturn する理由にはならない

meaning:

- LightGBM alpha は引き続き `higher corr / lower coverage` の contrast challenger として保持する
- `min_data_in_leaf` 単独調整は current bottleneck の root fix ではない
- 次段は multi-lever hypothesis を別 issue に分離して扱う

## Execution Plan

current execution suffix は次で固定する。

- train suffix: `r20260418_jra_lightgbm_alpha_leaf40_v1`
- evaluate suffix: `r20260418_jra_lightgbm_alpha_leaf40_v1`

train completion 後は、次の evaluate をそのまま実行する。

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_evaluate.py \
  --config configs/model_lightgbm_alpha_cpu_diag_leaf40.yaml \
  --data-config configs/data.yaml \
  --feature-config configs/features_catboost_rich.yaml \
  --max-rows 150000 \
  --wf-mode off \
  --artifact-suffix r20260418_jra_lightgbm_alpha_leaf40_v1
```

もし evaluate 側で suffix 付き model artifact を探しに行く場合は、existing train artifact naming に合わせて `--model-artifact-suffix __NO_MODEL_ARTIFACT_SUFFIX__` を付ける。

## Stop Condition

- coverage がほぼ回復しない
- `alpha_pred_corr` の下落が大きく、LightGBM challenger の identity を失う
- `market_deviation` 以外の layer を同時に動かさないと説明できなくなる

## Follow-Up Boundary

- この issue では LightGBM alpha の single-lever coverage recovery まで扱う
- feature / target / policy の multi-lever tuning は次 issue に分離する