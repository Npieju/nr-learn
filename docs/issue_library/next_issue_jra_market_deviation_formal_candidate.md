# Next Issue: JRA Market Deviation Formal Candidate

## Summary

`market_deviation` task 自体はすでに実装されており、JRA では `configs/model_catboost_alpha.yaml` と `configs/model_alpha.yaml` を使って train / evaluate artifact も出ている。

ただし current read では、これはまだ mainline compare surface に乗っていない。

理由は 2 つある。

- evaluation 側で `score_is_probability=false` となり、calibration と walk-forward が skip される
- compare の中心がなお `classification -> blend -> gate_then_roi` 系にあり、market-relative signal を primary metric として判定していない

そのため、alpha 系は implementation asset としては存在するが、formal compare 候補としては未整備のままである。

## Objective

JRA universe で `market_deviation` task を current mainline baseline と比較可能な formal candidate に引き上げる。

ここでの目的は、policy tuning で ROI を盛ることではなく、市場に対する residual / deviation signal を OOS で安定して計測できる評価面を定義することである。

## Hypothesis

if JRA `market_deviation` task に対して market-relative primary metrics と OOS compare surface を明示し、policy / probability calibration と切り離して formal candidate 化する, then `classification` mainline と別 track で signal quality を比較できるようになり、後段 policy tuning に依存しない structural gap を判定できる。

## In Scope

- JRA `market_deviation` baseline artifact の freeze
- market-relative primary metrics の定義
- `market_deviation` 用 evaluation / compare surface の formalization
- current alpha config を candidate として再評価するための issue / artifact / decision surface 整備

## Non-Goals

- この issue で execution policy を最適化すること
- `classification` mainline を即時置換すること
- NAR readiness queue を巻き込むこと
- value stack 全体を一気に promote すること

## Current Read

baseline right reference:

- profile: `current_recommended_serving_2025_latest`
- revision: `r20260325_current_recommended_serving_2025_latest_benchmark_refresh`
- baseline freeze tag: `strategy-baseline-20260418-pre-track-split`
- public reference:
  - [/workspaces/nr-learn/artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json](../artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json)
- promotion gate:
  - [/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)
- evaluation summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)

current implemented references:

- config:
  - [/workspaces/nr-learn/configs/model_catboost_alpha.yaml](../configs/model_catboost_alpha.yaml)
  - [/workspaces/nr-learn/configs/model_alpha.yaml](../configs/model_alpha.yaml)
- trainer implementation:
  - [/workspaces/nr-learn/src/racing_ml/models/trainer.py](../src/racing_ml/models/trainer.py)

current artifacts:

- train metrics:
  - [/workspaces/nr-learn/artifacts/reports/train_metrics_catboost_alpha.json](../artifacts/reports/train_metrics_catboost_alpha.json)
- evaluation summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json)
- evaluation by date:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_by_date_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.csv](../artifacts/reports/evaluation_by_date_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.csv)

confirmed read:

- train:
  - `alpha_pred_corr=0.2370363399`
  - `positive_signal_rate=0.0074425164`
  - `top1_roi=0.7757278020`
- evaluate:
  - `top1_roi=0.3864619278`
  - `ev_threshold_1_0_roi=0.8931734317`
  - `ev_threshold_1_2_roi=0.9037656904`
  - `alpha_target_mean=-4.4821970729`
  - `alpha_target_std=4.5397814214`
  - `pred_mean=-4.7689093875`
  - `pred_std=1.5306004124`
  - `alpha_pred_corr=0.1645068790`
  - `positive_signal_rate=0.0655266667`
  - `stability_assessment=representative`
  - `score_is_probability=false`
  - `wf_enabled=false`
  - `wf_skipped_reason=non_probability_task_or_score`

meaning update:

- baseline refresh により `market_deviation` の market-relative summary fields は evaluation summary / by-date summary へ出力されるようになった
- したがって current bottleneck は「指標が見えないこと」から「この signal をどの compare rule で formal candidate として判定するか」へ移った
- 一方で `score_is_probability=false` と `wf_skipped_reason=non_probability_task_or_score` は維持されており、current probability-centric formal gate にそのまま載せる構造ではない

meaning:

- alpha 系は train/evaluate 自体は動いている
- 現状の summary には `market_deviation` の primary metric が出るようになったが、compare の中心にはまだ置かれていない
- non-probability task として calibration / nested WF が skip されており、current formal gate surface にそのままは載らない
- したがって first task は policy optimization ではなく、signal evaluation surface の formalization である

## Primary Metrics

この track では、primary metric を次の順で扱う。

1. `alpha_pred_corr`
2. positive signal coverage
3. market-relative OOS stability
4. secondary read として realized ROI

`top1_roi` は参考値として残すが、track の主判定にはしない。

## Compare Rule

`market_deviation` track は current probability-centric benchmark と同じ gate で直接 promote / reject しない。

この track の compare rule は次の順で固定する。

1. baseline freeze が固定されていること
2. `stability_assessment=representative` を満たすこと
3. primary metric を `alpha_pred_corr` と positive signal coverage で読むこと
4. by-date の market-relative 指標で OOS stability を確認すること
5. `top1_roi` / `ev_threshold_*_roi` は secondary read として扱うこと

比較時の禁止事項も固定する。

- `classification` mainline の AUC と `market_deviation` の `alpha_pred_corr` を同列比較しない
- `wf_skipped_reason=non_probability_task_or_score` を failure と誤読しない
- short window の ROI 改善だけで signal-source 採用を主張しない

## Decision Gate

この track の暫定 3 択は次で固定する。

### `promote`

- `stability_assessment=representative`
- `alpha_pred_corr` が baseline candidate を上回る
- positive signal coverage が極端な sparse collapse に入っていない
- by-date の market-relative 指標に catastrophic drift がない
- secondary ROI read が極端な破綻を示していない

### `keep as candidate`

- representative ではある
- signal は見えるが、coverage か OOS stability に不確実性が残る
- secondary ROI は参考値として読むが、まだ stack / policy へ統合する段階ではない

### `reject`

- representative でない
- `alpha_pred_corr` が baseline と同等未満で改善仮説を支えない
- positive signal coverage が実運用で扱えないほど崩れる
- by-date の drift が大きく、signal-source として安定しない

## Compare Outputs

first formal compare では、少なくとも次を artifact read として揃える。

- versioned evaluation summary
- versioned by-date CSV
- train metrics
- decision summary

読み出す列は次を基本形にする。

- `alpha_pred_corr`
- `positive_signal_rate`
- `alpha_target_mean`
- `alpha_target_std`
- `pred_mean`
- `pred_std`
- `stability_assessment`
- `top1_roi`
- `ev_threshold_1_0_roi`
- `ev_threshold_1_2_roi`

## Formal Rerun Read

current formal candidate read は次の run を正本として扱う。

- run id: `r20260418_jra_catboost_alpha_baseline_refresh_v1`
- manifest:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json](../artifacts/reports/evaluation_manifest_catboost_alpha_model_r20260418_jra_catboost_alpha_baseline_refresh_v1_wf_off_nested.json)

この run は次を満たしている。

- JRA universe 固定
- `configs/model_catboost_alpha.yaml` 固定
- `configs/data.yaml` 固定
- `configs/features_catboost_rich.yaml` 固定
- `requested_max_rows=150000`
- `n_dates=559`
- `stability_assessment=representative`
- versioned summary / by-date / manifest が揃っている

したがって、この revision は current compare rule に基づく first formal candidate read として使ってよい。

## Challenger Compare Read

first challenger compare として次の run を追加した。

- challenger run id: `r20260418_jra_lightgbm_alpha_challenger_v1`
- config:
  - [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag.yaml](../configs/model_lightgbm_alpha_cpu_diag.yaml)
- manifest:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json](../artifacts/reports/evaluation_manifest_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json)
- summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_model_r20260418_jra_lightgbm_alpha_challenger_v1_wf_off_nested.json)

confirmed read:

- `top1_roi=0.4286749482`
- `ev_threshold_1_0_roi=0.9521551724`
- `ev_threshold_1_2_roi=0.9687402799`
- `alpha_target_mean=-4.4821970729`
- `alpha_target_std=4.5397814214`
- `pred_mean=-4.8852264273`
- `pred_std=1.1604059493`
- `alpha_pred_corr=0.1831187046`
- `positive_signal_rate=0.0064266667`
- `stability_assessment=representative`
- `score_is_probability=false`
- `wf_enabled=false`
- `wf_skipped_reason=non_probability_task_or_score`

Stage 2 first redesign candidate として次の run を追加した。

- challenger run id: `r20260418_jra_lightgbm_alpha_race_norm_v1`
- config:
  - [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag_race_norm.yaml](../configs/model_lightgbm_alpha_cpu_diag_race_norm.yaml)
- summary:
  - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_race_norm_model_r20260418_jra_lightgbm_alpha_race_norm_v1_wf_off_nested.json](../artifacts/reports/evaluation_summary_lightgbm_alpha_cpu_diag_race_norm_model_r20260418_jra_lightgbm_alpha_race_norm_v1_wf_off_nested.json)

confirmed read:

- `top1_roi=0.4675178284`
- `ev_threshold_1_0_roi=1.1144751854`
- `ev_threshold_1_2_roi=1.1666951762`
- `alpha_target_mean=-4.4821970729`
- `alpha_target_std=4.5397814214`
- `pred_mean=0.0560925649`
- `pred_std=0.3274352437`
- `alpha_pred_corr=0.2232080870`
- `positive_signal_rate=0.2132800000`
- `stability_assessment=representative`
- `score_is_probability=false`
- `wf_enabled=false`
- `wf_skipped_reason=non_probability_task_or_score`

compare read vs CatBoost baseline candidate:

- `alpha_pred_corr` は `0.1645068790 -> 0.1831187046` で改善
- `positive_signal_rate` は `0.0655266667 -> 0.0064266667` で大きく低下
- `top1_roi` は `0.3864619278 -> 0.4286749482` で小幅改善
- `ev_threshold_1_0_roi` は `0.8931734317 -> 0.9521551724` で改善したが、なお `1.0` 未満
- representative support は維持しているが、signal coverage は CatBoost より明確に sparse

compare read vs race-normalized residual candidate:

- `alpha_pred_corr` は base LightGBM `0.1831187046 -> 0.2232080870` で改善
- `positive_signal_rate` は base LightGBM `0.0064266667 -> 0.2132800000` で大幅改善
- `ev_threshold_1_0_roi` は `0.9521551724 -> 1.1144751854` で `1.0` 超へ回復
- CatBoost incumbent に対しても corr / coverage / EV threshold read で上回った

## Tentative Decision

current decision は `promote` とする。

reason:

- representative support は満たしている
- race-normalized residual candidate は CatBoost incumbent と Stage 1 LightGBM variants の両方を corr / coverage で上回った
- `ev_threshold_1_0_roi=1.1144751854`, `ev_threshold_1_2_roi=1.1666951762` で secondary ROI read も破綻していない
- `score_is_probability=false` と `wf_skipped_reason=non_probability_task_or_score` は変わらないため probability mainline への即時統合はしないが、market_deviation track の current best candidate としては十分に説明可能である

meaning:

- signal-source 候補としては 1 段進めてよい
- current best read は CatBoost alpha ではなく race-normalized residual LightGBM である
- Stage 1 variants は contrast read として閉じ、次段は architecture / policy 再統合の順序を Stage roadmap に沿って切る

## Residual Risks

- `alpha_pred_corr` の absolute level は見えているが、current issue 単体では challenger baseline との差分比較がまだない
- positive signal coverage が薄く、secondary ROI read は support-sensitive である
- by-date summary は market-relative 診断を出すが、non-probability task なので current nested WF gate と同一の安定性 surface ではない
- したがって、この issue の read は「formal candidate 化までは到達、promotion judgement はまだ先」と読むべきである

## Next Decision Boundary

この issue の次判断は次のどちらかに限定する。

1. baseline freeze を tag / artifact pointer で固定し、次の challenger run を同一 compare rule でぶつける
2. current candidate を queue 上では exploration track に留め、mainline benchmark judgement へは混ぜない

current status では 1 のうち baseline freeze tag と right-reference artifact pointer までは固定済みである。残るのは challenger run の execution のみである。

## Success Metrics

- JRA `market_deviation` candidate に対して baseline artifact が固定されている
- compare surface に次が含まれる
  - train-time `alpha_pred_corr`
  - OOS market-relative metrics
  - support / coverage diagnostics
  - realized ROI の secondary read
- decision を `promote / keep as candidate / reject` の 3 択で書ける
- execution policy を大きく変えなくても signal quality の比較が成立する

## Validation Plan

1. baseline freeze
   - current `market_deviation` artifacts を baseline candidate として固定する

2. metric formalization
   - market-relative primary metrics を docs / issue 上で明示する
   - `classification` metrics と混ぜない compare rule を定義する

3. evaluation surface check
   - current `run_evaluate` surface で何が見えて、何が skip されるかを整理する
   - 必要なら `market_deviation` 用の summary fields を minimal に追加する

4. formal rerun candidate
   - same universe, frozen dataset, fixed baseline compare で rerun し、signal quality と secondary ROI を読む

## Stop Condition

- market-relative signal 指標が OOS で一貫して弱く、compare 候補として成立しない
- compare のために `classification` / `policy` / `market_deviation` の 3 layer を同時に大きく動かす必要が出る
- baseline freeze を保てず、current read が run ごとに揺れる

## Proposed First Implementation Cut

最初の実装単位は narrow に保つ。

- code change が必要なら `market_deviation` summary / evaluation surface の最小補強だけ行う
- policy surface は固定する
- feature set と dataset scope も first cut では固定する

最初の measurable hypothesis は次である。

`if JRA market_deviation candidate を market-relative metrics で formal compare できるようにする, then classification mainline とは別に signal-source として採用可否を判定できる`

## Follow-Up Boundary

- この issue で扱うのは signal-source の formal candidate 化まで
- value stack への再統合は次 issue とする
- policy conversion の最適化はさらにその後段とする