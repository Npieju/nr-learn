# Next Issue: JRA Broader Composition Bounded Residual Calibration

## Summary

`support_preserving_residual_path` の first candidate は `hold` で閉じた。

current read は次である。

- representative surface では `ev_threshold_1_0_bets=5128`, `stability_assessment=representative` で support-preserving constraint を通過した
- September difficult window では `37 bets / total net -11.9` で `8/8 zero-bet` を回避した
- ただし same window の sidecar reference は `35 bets / total net -9.9` で、actual-date net / mean ROI ともに first candidate を上回った

したがって broader composition branch で次に切るべき論点は「support-preserving path が成立するか」ではなく、「bounded residual をどう calibrate すれば sidecar に近い return surface まで戻せるか」である。

この issue は broader composition を継続するとしても 1 仮説だけに絞るための follow-up draft である。current queue の正本を即更新するのではなく、human review 後に切る next measurable hypothesis を固定する。

## Objective

`support_preserving_residual_path` を維持したまま、market residual を positive-only かつ lower-weight に再較正した candidate を 1 本だけ定義し、support を保ったまま September difficult window の actual-date net を sidecar に近づけられるか判定する。

## Hypothesis

if `support_preserving_residual_path` で market residual を symmetric residual ではなく `positive_only + lower_weight` の bounded calibration に絞る, then representative support を維持したまま September difficult window の過剰 exposure / mean ROI 劣化を抑え、sidecar に近い return surface を作れる。

## In Scope

- base path は `support_preserving_residual_path` のまま維持する
- calibration 差分は market residual に限定する
- first candidate との差分は `market_residual_weight` と `market_residual_positive_only` に限る
- representative evaluate と September difficult-window compare で判定する

## Non-Goals

- sidecar path を同時に変更すること
- alpha target や alpha component を再設計すること
- policy family を変更すること
- 複数の calibration candidate を同時に足すこと

## Why This Issue

first candidate から読めることは次である。

1. support-preserving branch 自体は成立する
2. current weakness は catastrophic support collapse ではなく actual-date return quality に寄っている
3. symmetric な market residual は September difficult window で negative residual もそのまま通しており、sidecar 比較では return quality のノイズ源になっている可能性がある

したがって次の 1 本は architecture family を切り替えるのではなく、bounded residual calibration の 1 軸だけを検証するべきである。

## Candidate Constraint

next candidate は次の constraint を満たすものだけを許す。

1. baseline `win_prob` は final carrier のままとする
2. `probability_path_mode` は `support_preserving_residual_path` から変えない
3. `alpha_weight` と alpha signal path は変更しない
4. market residual calibration 以外の差分を入れない

## Fixed Compare Surface

1. baseline profile は `current_recommended_serving_2025_latest`
2. sidecar reference は `current_recommended_serving_alpha_sidecar_race_norm_2025_latest`
3. support-preserving first candidate reference は `current_recommended_serving_support_preserving_prob_race_norm_2025_latest`
4. difficult window は `2025-09-06/07/13/14/20/21/27/28`

## Proposed Candidate

candidate theme は `support_preserving_residual_path_positive_only_market_gate` とする。

minimal parameter delta:

- `market_residual_positive_only: true`
- `market_residual_weight: 0.08 -> 0.05`
- `market_residual_scale` は現状維持から開始する

狙い:

- negative market residual の drag を抑える
- support を支えている baseline carrier と alpha residual は維持する
- September difficult window の downside を first candidate より絞る

## Baseline References

- broader composition first issue:
  - [next_issue_jra_broader_composition_support_preserving_probability_path.md](next_issue_jra_broader_composition_support_preserving_probability_path.md)
- representative first candidate read:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_da9a84b67bd0d1c4_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_da9a84b67bd0d1c4_wf_off_nested.json)
- baseline compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_v1_base_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_v1_base_vs_cand.json)
- sidecar compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_v1_sidecar_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_v1_sidecar_vs_cand.json)

## Entry Read

first candidate fixed read:

- representative:
  - `ev_threshold_1_0_bets=5128`
  - `stability_assessment=representative`
- September difficult window:
  - `37 bets / total net -11.9 / mean_policy_roi 0.7027777778`

compare references:

- baseline:
  - `34 bets / total net -13.6 / mean_policy_roi 0.85`
- sidecar:
  - `35 bets / total net -9.9 / mean_policy_roi 0.9152777778`

entry meaning:

1. support-preserving family は `reject` ではなく calibration follow-up の余地がある
2. next issue は path redesign ではなく residual calibration の 1 軸だけに絞るべきである
3. winner claim の条件は sidecar に近い September return surface を取り戻せるかどうかである

## Success Metrics

1. representative `stability_assessment=representative` を維持する
2. representative `ev_threshold_1_0_bets` が first candidate 比で catastrophic に落ちない
3. September difficult window で `8/8 zero-bet` を再発させない
4. September difficult window の `total_policy_net` が first candidate `-11.9` を上回る
5. sidecar `-9.9` に近づくか、少なくとも差分を縮める

## Validation Plan

1. config 差分を market residual calibration だけに限定した candidate を 1 本作る
2. representative evaluate を `full_training_table + wf_off` で読む
3. same 8-date September difficult window compare を baseline / sidecar と取る
4. `advance / hold / reject` の 3 択で閉じる

## Stop Condition

- `market_residual_positive_only` だけでは差分がほぼ消えて compare surface にならない
- weight 調整だけでは actual-date read が安定せず、policy family 変更が必要になる
- support-preserving path を維持したままでは sidecar に近づく余地がないと読める

## Exit Rule

- `advance`: representative / actual-date の両面で first candidate より明確に改善し、sidecar との差を縮める
- `hold`: support は維持するが return quality 改善が弱い
- `reject`: calibration follow-up でも sidecar に近づかず、support-preserving residual family 自体の headroom が薄い

## Follow-Up Boundary

- human review 前に current queue の正本へ昇格しない
- first candidate issue の `hold` judgment を書き換えない
- next code change はこの issue が正本化された後に 1 candidate だけ実装する

## First Execution Read

2026-04-19 に proposed candidate `support_preserving_residual_path_positive_only_market_gate` を 1 本だけ実装した。

implementation surface:

- config:
  - [../../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_race_norm_positive_only.yaml](../../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_support_preserving_prob_race_norm_positive_only.yaml)
- profile:
  - [../../src/racing_ml/common/model_profiles.py](../../src/racing_ml/common/model_profiles.py)

parameter delta from first candidate:

- `market_residual_positive_only: true`
- `market_residual_weight: 0.05`
- other path / alpha / policy surfaces unchanged

representative evaluate read:

- source:
  - [../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_04dd633acf0ad297_wf_off_nested.json](../../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_04dd633acf0ad297_wf_off_nested.json)
- `auc=0.8402890052`
- `logloss=0.2034577245`
- `ev_threshold_1_0_bets=5128`
- `stability_assessment=representative`

September difficult-window read:

- source:
  - [../../artifacts/reports/serving_smoke_current_recommended_serving_support_preserving_prob_race_norm_positive_only_2025_latest_sep25_support_preserving_prob_positive_only_v1.json](../../artifacts/reports/serving_smoke_current_recommended_serving_support_preserving_prob_race_norm_positive_only_2025_latest_sep25_support_preserving_prob_positive_only_v1.json)
- `policy_bets=37`
- `policy_selected_rows=37`
- `total_policy_net=-11.9`
- `mean_policy_roi=0.7027777778`

compare reads:

- baseline compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_positive_only_v1_base_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_positive_only_v1_base_vs_cand.json)
- sidecar compare:
  - [../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_positive_only_v1_sidecar_vs_cand.json](../../artifacts/reports/serving_smoke_compare_sep25_support_preserving_prob_positive_only_v1_sidecar_vs_cand.json)

first execution meaning:

1. representative support は preserved のままだった
2. September difficult window でも `8/8 zero-bet` は再発しなかった
3. ただし aggregate read は first candidate と実質同一で、sidecar との差も縮まらなかった
4. `positive_only + lower_weight` だけでは measurable improvement が出ない

## Decision

- current decision: `reject`

reason:

1. representative / actual-date ともに first candidate 比で有意な改善がなかった
2. sidecar reference との gap も縮まず、bounded residual calibration の 1 軸としては headroom が薄い
3. current family をこれ以上 ad-hoc に掘るより、human review で broader composition branch 継続可否を判断する方が自然である

exit meaning:

1. support-preserving residual family の existence proof は first candidate で足りている
2. positive-only calibration follow-up は winner search としては前進を作れなかった
3. next action は queue の current read どおり human review / next issue recut を優先する