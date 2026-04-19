# Next Issue: JRA Prediction Foundation Probability Support Diagnostics

## Summary

`market_aware_probability_path` first candidate は `reject` で閉じた。

current read の重要点は次である。

- representative evaluate は `auc=0.8410248775`, `logloss=0.2026480271` で catastrophic ではなかった
- しかし `ev_threshold_1_0_bets=1` で stability guardrail は `probe_only` だった
- `2025-09-06/07/13/14/20/21/27/28` の actual-date compare では same policy / same visible score-source surface のまま `policy_bets=0` まで support が痩せた

これは、current bottleneck が直ちに execution policy や market deviation track の別 composition probe であることを示していない。むしろ先に、`classification` baseline 自体の probability support / calibration / ranking surface を ROI と切り離して formalize する必要がある。

したがって next issue は prediction foundation track に戻し、fixed baseline 上で「どこで support collapse が起きるか」を診断面として固定する。

## Objective

JRA `classification` baseline を fixed compare surface として、probability support / calibration / ranking diagnostics を ROI と切り離して formalize し、next architecture branch が prediction foundation 側にあるのか、broader composition 側にあるのかを 1 issue で判定する。

## Hypothesis

if `current_recommended_serving_2025_latest` と rejected market-aware candidate を fixed policy / fixed compare surface で読み、probability distribution・calibration・ranking・market anchor dependence を formalizeする, then support collapse の主因が policy family ではなく prediction foundation 側にあるかどうかを measurable に切り分けられる。

## In Scope

- JRA baseline `classification` surface の prediction-foundation diagnostics 定義
- baseline / bounded sidecar / rejected market-aware candidate を fixed compare surface で並べる
- `auc`, `logloss`, race-level quality, calibration, probability support, date-local support cliff を main read に昇格する
- next child issue を prediction foundation continuation にするか、broader architecture branch にするかの decision rule を固定する

## Non-Goals

- 新しい policy sweep を足すこと
- 新しい market deviation target を再設計すること
- 新しい probability-path candidate を追加実装すること
- baseline model family をこの issue の中で置き換えること

## Why This Issue

first market-aware probability-path candidate の reject により、次が確認された。

1. representative top-line は壊れていないのに actual-date exposure が zero まで痩せうる
2. same policy / same visible score-source surface でも final support は大きく崩れる
3. この崩れを late-fusion vs market-aware の構造差だけで説明するには、prediction foundation diagnostics が不足している

したがって next question は「どの architecture candidate を次に足すか」ではなく、「baseline `classification` probability surface のどこが support cliff を作っているか」である。

## Fixed Compare Surface

current issue では次を固定する。

1. baseline profile は `current_recommended_serving_2025_latest`
2. bounded sidecar reference は `current_recommended_serving_alpha_sidecar_race_norm_2025_latest`
3. rejected architecture reference は `current_recommended_serving_market_aware_prob_race_norm_2025_latest`
4. policy family と serving default は compare のための reading surface として扱い、ここでは最適化しない
5. next action は diagnostics read からだけ決める

## Primary Metrics

1. representative evaluate の `auc` / `logloss`
2. race-level quality read
   - model race logloss
   - pseudo-$R^2$
   - public-vs-model gap
3. calibration / support read
   - calibration eval rows / races
   - top probability mass の圧縮度
   - EV threshold support
4. actual-date support read
   - `policy_bets`
   - selected rows
   - zero-bet dates
   - date-local support cliff

guardrail:

- ROI は reference として読むが、この issue の主 KPI にはしない
- policy family difference を root cause と誤読しないため、same visible surface comparison を優先する

## Baseline References

- parent architecture issue:
  - [next_issue_model_architecture_rebuild_track_split.md](next_issue_model_architecture_rebuild_track_split.md)
- baseline representative reference:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)
- rejected candidate representative reference:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json)
- baseline vs rejected candidate actual-date compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_base_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_base_vs_cand.json)
- sidecar vs rejected candidate actual-date compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_sidecar_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_sidecar_vs_cand.json)

## First Empirical Read

2026-04-19 時点の first empirical read は次で固定する。

representative baseline read:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)
- `auc=0.8400959298`
- `logloss=0.2038718179`
- `top1_roi=0.8070328660`
- `ev_threshold_1_0_bets=4872`
- `market_prob_corr=0.9867135690`
- run geometry:
  - `loaded_rows=300000`
  - `data_load_strategy=tail_training_table`
  - `wf_mode=fast`
  - `feature_count=103`

representative rejected-candidate read:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json)
- `auc=0.8410248775`
- `logloss=0.2026480271`
- `top1_roi=0.7893301105`
- `ev_threshold_1_0_bets=1`
- `market_prob_corr=0.9999304630`
- run geometry:
  - `loaded_rows=1730857`
  - `data_load_strategy=full_training_table`
  - `wf_mode=off`
  - `feature_count=123`

meaning:

1. top-line `auc` / `logloss` だけを見ると rejected candidate は catastrophic ではない
2. しかし EV support は baseline `4872 bets` に対して candidate `1 bet` まで崩れている
3. さらにこの compare 自体が like-for-like ではない
   - baseline は `tail_training_table + wf_fast + 103 features`
   - candidate は `full_training_table + wf_off + 123 features`
4. したがって current bottleneck は「candidate が弱い」で閉じず、baseline `classification` probability surface を固定 geometry で formalize し直す必要がある

September difficult-window support read:

- baseline smoke summary:
  - [../../artifacts/reports/serving_smoke_current_recommended_serving_2025_latest_sep25_market_aware_prob_v1_base.json](../../artifacts/reports/serving_smoke_current_recommended_serving_2025_latest_sep25_market_aware_prob_v1_base.json)
- sidecar smoke summary:
  - [../../artifacts/reports/serving_smoke_current_recommended_serving_alpha_sidecar_race_norm_2025_latest_sep25_market_aware_prob_v1_sidecar.json](../../artifacts/reports/serving_smoke_current_recommended_serving_alpha_sidecar_race_norm_2025_latest_sep25_market_aware_prob_v1_sidecar.json)
- rejected candidate compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_base_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_base_vs_cand.json)

aggregated support read:

- baseline:
  - `34 bets / 34 selected rows / total net -13.6`
- bounded sidecar:
  - `35 bets / 35 selected rows / total net -9.9`
- rejected candidate:
  - `0 bets / 0 selected rows / total net 0.0`

date-local read:

- baseline は 8/8 日で selection を維持した
- bounded sidecar も 8/8 日で selection を維持した
- rejected candidate は 8/8 日すべて `policy_bets=0`

structural read:

- `differing_score_source_dates=[]`
- `differing_policy_dates=[]`
- つまり actual-date support collapse は visible な policy/surface switch ではなく、same visible surface 上で probability support が閾値を跨げなくなった結果として読むべきである

## Success Metrics

- prediction foundation diagnostics の primary read が 1 issue として固定される
- support collapse を policy ではなく probability foundation として説明できるか判定できる
- next child issue が prediction foundation continuation か broader composition redesign かのどちらか 1 本に絞られる
- Stage 4 bounded reintegration を連想的に再開しない

## Validation Plan

1. baseline / sidecar / rejected candidate の fixed compare surface を確定する
2. representative evaluate から probability-quality metrics を抽出する
3. actual-date compare から support cliff metrics を抽出する
4. `advance / hold / reject` で prediction foundation hypothesis を閉じる
5. 次の child issue を 1 本だけ切る

## Stop Condition

- diagnostics だけではなく新しい model retrain を同時にやらないと hypothesis が書けない
- policy search を動かさないと差が読めない
- prediction foundation ではなく市場 target redesign が先だと判断された

## Exit Rule

- `advance`: support collapse の主因が prediction foundation 側だと判定でき、次 child issue を prediction foundation continuation に固定できる
- `hold`: signal はあるが prediction foundation と broader composition のどちらが主因かまだ弱い
- `reject`: probability support cliff は prediction foundation issue ではなく、broader architecture branch へ戻すべきと判断される

## Follow-Up Boundary

- この issue が閉じるまで、新しい probability-path candidate や policy probe は追加しない
- next child issue はこの diagnostics read を根拠に 1 本だけ切る