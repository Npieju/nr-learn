# Next Issue: JRA Broader Composition Support-Preserving Probability Path

## Summary

`prediction_foundation_probability_support_diagnostics` の current read は、prediction foundation continuation を `advance` するより、broader composition branch へ戻る方が自然であることを示した。

fixed control read は次で揃っている。

- aligned baseline は `full_training_table + wf_off` でも `ev_threshold_1_0_bets=5009` を維持した
- aligned sidecar も `123 features + alpha component` のまま `ev_threshold_1_0_bets=5014` を維持した
- rejected market-aware candidate だけが同条件で `ev_threshold_1_0_bets=1`、September difficult window でも `8/8` 日 `policy_bets=0` だった

したがって current bottleneck は baseline `classification` foundation の一般劣化ではなく、`market_aware_alpha_branch` に代表される probability composition path 側の構造にあるとみなす。

next issue は prediction foundation diagnostics の継続ではなく、support-preserving な broader composition candidate を 1 本だけ定義し、late-fusion sidecar と rejected branch の間にある設計空間を measurable hypothesis として切る。

## Objective

JRA mainline について、baseline win probability の support を壊さないことを first constraint に置いた support-preserving probability composition candidate を 1 本だけ定義し、late-fusion sidecar より構造比較しやすく、`market_aware_alpha_branch` より support collapse しにくい branch を作れるか判定する。

## Hypothesis

if baseline win probability を final support carrier として固定し、market/alpha signal を `market_aware_alpha_branch` のような branch replacement ではなく bounded residual composition として注入する, then representative support と actual-date exposure を維持したまま、market deviation signal を probability layer の architecture candidate として読める compare surface を作れる。

## In Scope

- baseline / sidecar / rejected market-aware candidate の fixed compare surface を継承する
- support-preserving probability composition candidate を 1 本だけ定義する
- candidate の compare rule を representative evaluate と actual-date compare で固定する
- next action を policy sweep ではなく architecture candidate compare に限定する

## Non-Goals

- baseline win model を再学習すること
- alpha target 自体を再定義すること
- policy family や serving default を同時に最適化すること
- 複数の probability-path candidate を同時に追加すること

## Why This Issue

current read では次が確定している。

1. geometry mismatch は support collapse の主因ではない
2. `123 features + alpha component` 自体も support collapse の主因ではない
3. rejected candidate の主要差分は `market_aware_alpha_branch` の composition logic に残っている

meaning は明確である。

- prediction foundation を追加で掘るより先に、composition path そのものを切り分けるべきである
- ただし first market-aware candidate と同じ branch replacement をもう 1 本足すのは悪い進め方である
- next compare surface は「baseline support を carrier として残す」ことを hard constraint に置かなければならない

## Fixed Compare Surface

1. baseline profile は `current_recommended_serving_2025_latest`
2. sidecar reference は `current_recommended_serving_alpha_sidecar_race_norm_2025_latest`
3. rejected architecture reference は `current_recommended_serving_market_aware_prob_race_norm_2025_latest`
4. baseline canonical artifact は `r20260325_current_recommended_serving_2025_latest_benchmark_refresh`
5. next candidate 差分は probability composition path だけに限定する

## Candidate Constraint

next candidate は次の constraint を満たすものだけを許す。

1. baseline `win_prob` が final support carrier であり続けること
2. market probability を branch 置換の primary carrier にしないこと
3. alpha signal は bounded residual としてだけ入ること
4. compare は baseline policy / same visible surface で読めること

## Candidate Theme

first candidate theme は `support_preserving_residual_path` とする。

狙いは次である。

- `legacy_blend` のように後段 additive sidecar へ退避しすぎない
- `market_aware_alpha_branch` のように market branch へ carrier を置き換えない
- baseline win logit を中心に残したまま、alpha / market を bounded residual として加える

この issue の成功条件は、この theme が最終勝者になることではない。support-preserving constraint の下で architecture compare が成立する surface を 1 本作れるかどうかである。

## Primary Metrics

1. representative `auc` / `logloss`
2. `ev_threshold_1_0_bets` と stability assessment
3. September difficult window の `policy_bets` / `selected_rows` / `total net`
4. baseline / sidecar と同一 policy surface 上で compare が成立すること

guardrail:

- `ev_threshold_1_0_bets` が sidecar / baseline に対して catastrophic に細る candidate は即 reject とする
- actual-date `8/8 zero-bet` を再発させた時点で deeper sweep に進まない

## Baseline References

- parent architecture issue:
  - [next_issue_model_architecture_rebuild_track_split.md](next_issue_model_architecture_rebuild_track_split.md)
- prediction support diagnostics:
  - [next_issue_jra_prediction_foundation_probability_support_diagnostics.md](next_issue_jra_prediction_foundation_probability_support_diagnostics.md)
- rejected market-aware candidate:
  - [next_issue_jra_market_deviation_market_aware_probability_path.md](next_issue_jra_market_deviation_market_aware_probability_path.md)
- score composition implementation:
  - [../../src/racing_ml/evaluation/scoring.py](../../src/racing_ml/evaluation/scoring.py)

## Entry Read

2026-04-19 時点の entry read は次で固定する。

aligned baseline control:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_af0b3a870abf79f4_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_af0b3a870abf79f4_wf_off_nested.json)
- `auc=0.8403234192`
- `logloss=0.2035570661`
- `ev_threshold_1_0_bets=5009`
- `stability_assessment=representative`

aligned sidecar control:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_ee346cbcc6343bc3_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_ee346cbcc6343bc3_wf_off_nested.json)
- `auc=0.8403593818`
- `logloss=0.2034040069`
- `ev_threshold_1_0_bets=5014`
- `stability_assessment=representative`

rejected market-aware candidate:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json)
- `auc=0.8410248775`
- `logloss=0.2026480271`
- `ev_threshold_1_0_bets=1`
- `stability_assessment=probe_only`

September difficult window compare:

- baseline:
  - `34 bets / total net -13.6`
- sidecar:
  - `35 bets / total net -9.9`
- rejected candidate:
  - `0 bets / total net 0.0`

entry meaning:

1. broader composition branch に戻る根拠は十分に揃っている
2. 次 candidate は baseline support を carrier として維持する constraint を持つべきである
3. next issue の役割は candidate を量産することではなく、support-preserving compare surface を 1 本作ることである

## First Candidate Read

2026-04-19 に `support_preserving_residual_path` の first candidate を実装した。

implementation surface:

- probability path:
  - [../../src/racing_ml/evaluation/scoring.py](../../src/racing_ml/evaluation/scoring.py)
- model config:
  - [../../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_race_norm.yaml](../../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_race_norm.yaml)
- profile:
  - [../../src/racing_ml/common/model_profiles.py](../../src/racing_ml/common/model_profiles.py)

implementation rule:

1. baseline `win_logit` を final carrier として維持する
2. market は `market_logit - win_logit` の bounded residual を `tanh` で圧縮して注入する
3. alpha は従来どおり bounded residual の additive signal として残す
4. legacy の post-sigmoid market blend はこの mode では無効化する

representative evaluate read:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_da9a84b67bd0d1c4_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_da9a84b67bd0d1c4_wf_off_nested.json)
- profile:
  - `current_recommended_serving_support_preserving_prob_race_norm_2025_latest`
- `auc=0.8402890052`
- `logloss=0.2034577245`
- `ev_threshold_1_0_bets=5128`
- `ev_threshold_1_2_bets=3092`
- `stability_assessment=representative`

first read meaning:

1. representative support は preserved とみなしてよい
2. baseline / sidecar に近い compare surface は確保できた
3. first candidate は `market_aware_alpha_branch` のような catastrophic support collapse を再発させていない
4. next judgment には September difficult window compare がまだ必要である

## Success Metrics

- next candidate が `baseline support carrier` constraint を満たす
- representative read で `probe_only` を回避できる
- September difficult window で `8/8 zero-bet` を再発させない
- next issue を `advance / hold / reject` のいずれかで閉じられる

## Validation Plan

1. support-preserving composition candidate を 1 本だけ実装する
2. representative evaluate を aligned geometry で読む
3. September difficult window actual-date compare を読む
4. baseline / sidecar / candidate の 3 面 compare で `advance / hold / reject` を閉じる

## Stop Condition

- candidate を書くために policy family rewrite が必要になる
- target redesign と composition redesign を同時にやらないと仮説が書けない
- support-preserving constraint を守ると candidate 差分が消えて compare にならない

## Exit Rule

- `advance`: support-preserving candidate が representative / actual-date の両面で compare surface を成立させる
- `hold`: support は守れるが architecture winner を主張するには弱い
- `reject`: broader composition branch でも support-preserving compare surface を作れず、signal-return path 自体の前提を見直すべきと判定する

## Follow-Up Boundary

- この issue が閉じるまで policy sweep は追加しない
- prediction foundation docs は current read reference として維持し、継続 issue にはしない
- next candidate は 1 本だけに限定する