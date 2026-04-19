# Next Issue: JRA Market Deviation Target Redefinition

## Summary

Stage 1 の parameter probes は current LightGBM `market_deviation` line の trade-off front をほぼ出し切った。

- base challenger は `alpha_pred_corr=0.1831187046`, `positive_signal_rate=0.0064266667`
- `leaf40` は corr-side を押し上げたが coverage は改善しなかった
- `clip6` は coverage-side を小幅改善したが incumbent replacement には遠い
- `leaf40 + clip6` merge も trade-off front を押し出せず reject になった

したがって current bottleneck は parameter の局所最適化ではなく、`market_deviation` target 自体の shape にある可能性が高い。

## Objective

JRA `market_deviation` track について、current `observed_logit - market_logit` target が coverage collapse を生んでいないかを検証し、race-normalized residual target へ置き換える価値があるかを 1 issue で判定する。

## Hypothesis

if current `market_deviation` target を clipped logit residual ではなく race-normalized residual target に置き換える, then negative tail が緩和され、market-relative corr を大きく壊さずに positive signal coverage を改善できる。

## In Scope

- current `market_deviation` target shape の診断
- race-normalized residual target 1 種の定義
- target implementation surface の最小変更
- fixed JRA compare surface での first formal compare

## Non-Goals

- policy rewrite
- feature redesign
- model family の同時変更
- target variant を複数本同時に試すこと

## Why This Stage

1. tree sparsity の緩和だけでは coverage は戻らなかった
2. target tail compression には一定の効果が見えた
3. ただし parameter probe の足し算では trade-off front を押し出せなかった

このため Stage 2 の first issue は architecture-level 論点の中でも最小の root cause として target redesign を先に切る。

## Success Metrics

- target shape の before/after を artifact で説明できる
- base challenger より `positive_signal_rate` が改善する
- `alpha_pred_corr` が catastrophic に崩れない
- result を `advance / hold / reject` で閉じられる

## Validation Plan

1. current target の distribution / clipping surface を確認する
2. race-normalized residual target 1 種を定義する
3. implementation / tests を追加する
4. fixed JRA compare surface で fresh train / evaluate を実行する
5. CatBoost incumbent / current LightGBM base / Stage 1 best coverage-side variant / new target candidate を compare する

## Diagnostic Read

current target の first diagnostic artifact は次を正本として読む。

- diagnostic artifact:
	- [/workspaces/nr-learn/artifacts/reports/market_deviation_target_diagnostic_model_lightgbm_alpha_cpu_diag.json](../artifacts/reports/market_deviation_target_diagnostic_model_lightgbm_alpha_cpu_diag.json)
- diagnostic log:
	- [/workspaces/nr-learn/artifacts/logs/r20260418_market_deviation_target_diag_base_v1.log](../artifacts/logs/r20260418_market_deviation_target_diag_base_v1.log)

confirmed read:

- current raw target は `mean=-4.9035042404`, `std=4.5357945866`
- current clipped target は `mean=-4.7924701738`, `std=3.8329459907`
- `lower_clip_rate=0.03702`, `upper_clip_rate=0.06808` で clip hit が無視できない
- `raw_target positive_rate=0.0725365417`, `clipped_target positive_rate=0.0680866667`
- race-normalized preview は `positive_rate=0.0815066667`, `std=0.9402446490`

meaning:

- current target は負側に大きく偏りつつ、上側 clip hit も多い
- parameter probe だけではなく target shape 自体が sparse coverage を作っている読みを支持する
- Stage 2 first candidate は race-normalized residual 1 本でよい

## Implemented Candidate

- config:
	- [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag_race_norm.yaml](../configs/model_lightgbm_alpha_cpu_diag_race_norm.yaml)
- implementation:
	- [/workspaces/nr-learn/src/racing_ml/models/trainer.py](../src/racing_ml/models/trainer.py)
- test:
	- [/workspaces/nr-learn/tests/test_market_deviation_target.py](../tests/test_market_deviation_target.py)

candidate definition:

- base residual は current と同じ `observed_logit - market_logit`
- first candidate は clipped residual を race ごとに z-score 化した `race_normalized_residual`
- default mode は維持し、new config だけ `target_mode: race_normalized_residual` を使う

smoke:

- `pytest tests/test_market_deviation_target.py tests/test_run_evaluate.py -q`
- result: `11 passed`

## First Compare Read

- run id: `r20260418_jra_lightgbm_alpha_race_norm_v1`
- train metrics:
	- [/workspaces/nr-learn/artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_race_norm_r20260418_jra_lightgbm_alpha_race_norm_v1.json](../artifacts/reports/train_metrics_lightgbm_alpha_cpu_diag_race_norm_r20260418_jra_lightgbm_alpha_race_norm_v1.json)
- evaluation summary:
	- [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_race_norm_model_r20260418_jra_lightgbm_alpha_race_norm_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_race_norm_model_r20260418_jra_lightgbm_alpha_race_norm_v1_wf_off_nested.json)
- evaluation manifest:
	- [/workspaces/nr-learn/artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_race_norm_model_r20260418_jra_lightgbm_alpha_race_norm_v1_wf_off_nested.json](../artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_race_norm_model_r20260418_jra_lightgbm_alpha_race_norm_v1_wf_off_nested.json)

confirmed read:

- train: `alpha_pred_corr=0.2728496055`, `positive_signal_rate=0.1776093519`
- evaluate: `alpha_pred_corr=0.2232080870`, `positive_signal_rate=0.2132800000`
- evaluate: `pred_mean=0.0560925649`, `pred_std=0.3274352437`
- evaluate: `top1_roi=0.4675178284`, `ev_threshold_1_0_roi=1.1144751854`, `ev_threshold_1_2_roi=1.1666951762`
- evaluate: `stability_assessment=representative`

compare read:

- vs CatBoost incumbent `alpha_pred_corr=0.1645068790 -> 0.2232080870`
- vs CatBoost incumbent `positive_signal_rate=0.0655266667 -> 0.2132800000`
- vs LightGBM base `alpha_pred_corr=0.1831187046 -> 0.2232080870`
- vs LightGBM base `positive_signal_rate=0.0064266667 -> 0.2132800000`
- vs LightGBM clip6 `alpha_pred_corr=0.1844760397 -> 0.2232080870`
- vs LightGBM clip6 `positive_signal_rate=0.0069466667 -> 0.2132800000`

## Stop Condition

- target を変えても coverage が改善しない
- corr の劣化が大きい
- target redesign 単独では説明できず、feature / architecture branch を同時に触らないと仮説が書けない

## Exit Rule

- `advance`: target redesign が意味を持ち、次に architecture branch issue へ進める
- `hold`: target redesign は partial だが contrast candidate として保持する
- `reject`: 次は target ではなく market feature routing / dual-head architecture へ進む

## Current Decision

current decision は `advance` とする。

reason:

- diagnostic artifact が current target shape の歪みを説明できた
- first candidate は `alpha_pred_corr` と `positive_signal_rate` を同時に大きく改善した
- representative support を維持し、secondary ROI read も `ev_threshold_1_0_roi > 1.0` を満たした
- この issue の measurable hypothesis は満たしたので、次は Stage 3 の formal candidate read 更新へ進む