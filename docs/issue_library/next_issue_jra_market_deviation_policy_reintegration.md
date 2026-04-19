# Next Issue: JRA Market Deviation Policy Reintegration Boundary

## Summary

Stage 2 の target redesign は完了し、race-normalized residual candidate `r20260418_jra_lightgbm_alpha_race_norm_v1` は current market_deviation track の best read になった。

- `alpha_pred_corr=0.2232080870`
- `positive_signal_rate=0.2132800000`
- `ev_threshold_1_0_roi=1.1144751854`
- `stability_assessment=representative`

ただし、この run はまだ non-probability task の signal-source read であり、execution policy へどう戻すかは未固定である。

Stage 4 の first issue では、「市場 signal を本線へ戻す」と言っても broad な policy rewrite や model family 差し替えにしないための boundary を先に固定する。

## Objective

JRA mainline について、race-normalized residual alpha を execution policy へ再統合する最小の compare surface を 1 本に固定し、mainline baseline を壊さない bounded reintegration hypothesis を formal compare できる状態にする。

## Hypothesis

if race-normalized residual alpha を baseline win-probability mainline の置換ではなく bounded sidecar score source として再統合する, then market-relative signal を使った policy compare を baseline probability path と分離したまま評価でき、September difficult window 向けの de-risk / selective EV 改善余地を mainline drift なしで判定できる。

## In Scope

- Stage 4 で使う policy reintegration boundary の固定
- alpha sidecar を使う first bounded compare 1 本の定義
- compare で使う primary / secondary metrics の固定
- baseline probability path を壊さない rollback boundary の明文化

## Non-Goals

- baseline win model の置換
- broad policy family rewrite
- target / architecture / policy の再同時変更
- multiple reintegration variants の同時探索

## Why This Stage

1. Stage 2 で signal-source 自体の quality は大きく改善した
2. ただし `market_deviation` は `score_is_probability=false` のため、そのまま execution mainline に置けない
3. 現段階で必要なのは、policy へ戻すときの bounded entry point を 1 本に固定すること

## First Candidate Boundary

Stage 4 first candidate は次に固定する。

1. baseline probability source は現行 mainline のまま維持する
2. race-normalized residual alpha は sidecar raw score としてのみ使う
3. reintegration は evaluation / replay compare surface に限定し、serving default はまだ差し替えない
4. first compare は `compose_value_blend_probabilities` に既存で存在する `alpha_raw` integration surface を使い、`alpha_weight` と `alpha_positive_only` の bounded read に留める

meaning:

- Stage 4 first issue は新しい policy family を増やす話ではない
- alpha を probability mainline の外側からどう効かせるかを 1 本に固定する話である
- success / failure の責任範囲を `alpha sidecar bounded reintegration` に限定する

## First Candidate Surface

first candidate config は次を正本にする。

- bounded sidecar config:
	- [/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_alpha_sidecar_race_norm.yaml](../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_alpha_sidecar_race_norm.yaml)
- fixed alpha component reference:
	- [/workspaces/nr-learn/configs/model_lightgbm_alpha_cpu_diag_race_norm_component_ref.yaml](../configs/model_lightgbm_alpha_cpu_diag_race_norm_component_ref.yaml)

bounded settings:

- `alpha_weight: 0.05`
- `alpha_scale: 0.5`
- `alpha_positive_only: true`
- `roi_weight`, `market_blend_weight`, policy search surface, serving policy は baseline と同一に保つ

meaning:

- first compare は alpha sidecar 以外の差分を作らない
- bundle build は current baseline win / roi component に race-normalized alpha component だけを追加する

## Success Metrics

- compare surface が `baseline probability path` と `alpha sidecar path` の 2 本で明確に説明できる
- actual-date / representative compare の primary metric が先に固定されている
- baseline default を変えずに `advance / hold / reject` を判定できる

## Primary Metrics

1. actual-date compare での `total net` / bankroll deterioration の抑制
2. support を伴う selection rate / bet count の維持
3. secondary read として EV threshold ROI

market_deviation track 単体の `alpha_pred_corr` はこの issue では entry condition として扱い、主判定には使わない。

## Validation Plan

1. Stage 4 first candidate を bounded sidecar reintegration に固定する
2. baseline probability source と alpha sidecar source の artifact pointer を固定する
3. compare 用 config か score composition parameter を 1 本だけ追加する
4. actual-date compare と representative evaluate を実行する
5. `advance / hold / reject` のいずれかで閉じる

## Current Read

implemented surface:

- bounded sidecar config:
	- [/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_alpha_sidecar_race_norm.yaml](../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_alpha_sidecar_race_norm.yaml)
- profile:
	- `current_recommended_serving_alpha_sidecar_race_norm_2025_latest`

bundle build read:

- `component_count=3`
- `alpha_weight=0.05`
- `market_blend_weight=0.97`

representative evaluate read:

- artifact:
	- [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_dc6450998fe6cd17_wf_off_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_dc6450998fe6cd17_wf_off_nested.json)
- `auc=0.8384159536`
- `logloss=0.2037916115`
- `top1_roi=0.7973504431`
- `ev_threshold_1_0_roi=0.3598014471`
- `stability_assessment=representative`

baseline same-surface representative compare:

- artifact:
	- [/workspaces/nr-learn/artifacts/reports/evaluation_summary.json](../artifacts/reports/evaluation_summary.json)
- baseline `auc=0.8383868804`
- baseline `logloss=0.2039192355`
- baseline `top1_roi=0.7989290990`
- baseline `ev_threshold_1_0_roi=0.3601650665`
- read:
	- representative surface では sidecar は calibration / auc をわずかに改善した一方、`top1_roi` と `ev_threshold_1_0_roi` は baseline をごく小さく下回る
	- したがって representative read だけでは broad default change の根拠にはならない

September difficult-window actual-date compare:

- sidecar artifact:
	- [/workspaces/nr-learn/artifacts/reports/serving_smoke_current_recommended_serving_alpha_sidecar_race_norm_2025_latest_20250906_20250928_8d.json](../artifacts/reports/serving_smoke_current_recommended_serving_alpha_sidecar_race_norm_2025_latest_20250906_20250928_8d.json)
- sidecar read:
	- `policy_bets=35`
	- `total_policy_net=-4.7458333333`
	- `pure_bankroll=0.4352668690`
- baseline reference read:
	- `policy_bets=32`
	- `total_policy_net=-27.3`
	- `pure_bankroll=0.2959`
- compare read:
	- bounded alpha sidecar は September difficult window で baseline より損失を大きく圧縮した
	- ただし exposure は `32 -> 35` へ増えており、pure bankroll も既存 defensive candidate の `0.8395` には届かない
	- current role は broad replacement ではなく analysis-first reintegration candidate に留まる

lower-alpha-weight follow-up (`alpha_weight: 0.05 -> 0.02`):

- sidecar artifact:
	- [/workspaces/nr-learn/artifacts/reports/serving_smoke_current_recommended_serving_alpha_sidecar_race_norm_low_weight_2025_latest_20250906_20250928_8d.json](../artifacts/reports/serving_smoke_current_recommended_serving_alpha_sidecar_race_norm_low_weight_2025_latest_20250906_20250928_8d.json)
- low-weight read:
	- `policy_bets=35`
	- `total_policy_net=-14.6`
	- `pure_bankroll=0.3314214738`
- compare read:
	- `alpha_weight` を下げても September difficult window の exposure は `35 bets` のままで減らなかった
	- aggregate net / bankroll は original sidecar `-4.7458333333 / 0.4352668690` より明確に悪化した
	- したがって `lower alpha weight` 単独では support drift を抑えるレバーにならず、この single-lever follow-up は `reject` とする

meaning:

- bounded alpha sidecar candidate 自体は build / evaluate 可能であることを確認した
- representative / actual-date の両 read から、race-normalized alpha sidecar は bounded reintegration としては成立する
- ただし `alpha_weight` を単純に下げる follow-up は exposure を減らせず、September difficult window の actual-date read も悪化した
- 一方で representative top line の優位は弱く、September difficult window でも既存 defensive line を更新するほどではない
- current decision は `hold` とし、serving default は変えない

## Stop Condition

- reintegration に baseline probability path の差し替えが必要になる
- alpha sidecar 以外の policy rewrite を同時に入れないと仮説が立たない
- actual-date の improvement が見えても representative support が壊れる

## Exit Rule

- `advance`: bounded sidecar reintegration が有望で、次の formal compare issue へ進む
- `hold`: signal はあるが baseline default を動かすには弱く、analysis-first candidate として保持する
- `reject`: alpha sidecar reintegration ではなく architecture / score composition の別 branch へ戻る