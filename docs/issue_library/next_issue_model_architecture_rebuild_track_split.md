# Next Issue: Model Architecture Rebuild Track Split

## Summary

current mainline は `classification` を基盤にし、その後段で market blend と `gate_then_roi` policy を重ねる構造になっている。

この構造は short-term tuning には向くが、次の 3 layer が同時に動きやすい。

- win-probability prediction quality
- market deviation expression
- execution policy / bankroll conversion

その結果、`top1_roi` や `min_edge` などの後段 surface を sweep しても、何が改善したかを layer 単位で説明しにくい。

repo standard では problem definition / dataset / feature / model / policy を混ぜないことが明示されているため、ここで比較軸を固定し、track を分離し直す必要がある。

## Objective

current mainline を snapshot tag で baseline freeze したうえで、model development を次の 3 track に分離する。

- prediction foundation track
- market deviation track
- execution policy track

これにより、以後の experiment を `1 issue = 1 measurable hypothesis` で進められる状態へ戻す。

## Hypothesis

if current mainline を baseline snapshot として固定し、prediction foundation / market deviation / execution policy を別 track に分離する, then improvement の所在を layer 単位で説明できるようになり、policy-only tuning の反復では埋まらない structural gap を正面から検証できる。

## In Scope

- current mainline の baseline freeze 定義
- snapshot tag 命名と参照先の固定
- 3 track の objective / non-goals / success metrics / validation order の定義
- follow-up issue を切る順番の固定

## Non-Goals

- この issue 単体で新しい model family を promote すること
- JRA benchmark の即時更新
- NAR readiness queue を current queue から下ろすこと
- broad な docs 再編

## Success Metrics

- current mainline baseline を参照する snapshot tag が 1 つに固定されている
- prediction foundation / market deviation / execution policy の各 track に対して
  - objective
  - non-goals
  - primary metric
  - guardrails
  - validation plan
  が定義されている
- 次に着手する issue が execution policy tuning ではなく、layer 固有の measurable hypothesis として 1 本ずつ切り出されている

## Validation Plan

1. baseline freeze
   - current mainline config / artifact / public reading を snapshot tag で固定する

2. track split
   - prediction foundation track の primary metric を ranking / calibration 系へ限定する
   - market deviation track の primary metric を market-relative expression へ限定する
   - execution policy track の primary metric を bankroll / coverage / realized ROI conversion に限定する

3. issue cut
   - 最初の child issue を 1 本だけ選ぶ
   - dataset freeze, baseline artifact, success metrics, stop condition を明記する

4. formal comparison rule
   - cross-track compare を禁止し、同一 track 内だけで baseline compare する

## Stop Condition

- baseline snapshot を 1 つに固定できず compare surface が揺れる
- 3 track のうち 2 つ以上を同時に動かさないと hypothesis が書けない
- current queue で優先中の blocker issue を壊す broad change が必要になる

## Proposed Baseline Freeze

current structural baseline は、少なくとも次の性質を持つ run / config 群として定義する。

- task center: `classification`
- label: `is_win`
- market information は後段の blend / edge / EV 計算で使う
- execution selection は `gate_then_roi` / family selection による

representative references:

- JRA mainline config:
  - [/workspaces/nr-learn/configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml](../configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml)
- market deviation candidate config:
  - [/workspaces/nr-learn/configs/model_catboost_alpha.yaml](../configs/model_catboost_alpha.yaml)
- policy implementation:
  - [/workspaces/nr-learn/src/racing_ml/evaluation/policy.py](../src/racing_ml/evaluation/policy.py)
- training target implementation:
  - [/workspaces/nr-learn/src/racing_ml/models/trainer.py](../src/racing_ml/models/trainer.py)

baseline artifact entrypoint:

- public reference:
   - [/workspaces/nr-learn/artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json](../artifacts/reports/public_benchmark_reference_current_recommended_serving_2025_latest.json)

baseline canonical artifact set:

- promotion gate:
   - [/workspaces/nr-learn/artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)
- revision gate:
   - [/workspaces/nr-learn/artifacts/reports/revision_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../artifacts/reports/revision_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)
- evaluation manifest:
   - [/workspaces/nr-learn/artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../artifacts/reports/evaluation_manifest_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)
- evaluation summary:
   - [/workspaces/nr-learn/artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json)

baseline identity read:

- profile: `current_recommended_serving_2025_latest`
- revision: `r20260325_current_recommended_serving_2025_latest_benchmark_refresh`
- universe: `jra`
- status: `pass / promote`
- stability: `representative`

baseline reading order:

1. public benchmark reference
2. promotion gate
3. revision gate
4. evaluation manifest / summary

operating rule:

- baseline compare では latest pointer より versioned artifact を優先する
- baseline identity は profile だけでなく revision まで書く
- tag 作成前でも、この canonical artifact set を baseline freeze surface として扱う

baseline freeze tag:

- `strategy-baseline-20260418-pre-track-split`

anchor commit:

- commit: `2245ed32d1b25a5a38c46501585273c81cdcf4ab`
- tag status: created

この tag は source release ではなく compare surface freeze を指す補助ラベルとして使う。

## Track Definitions

### 1. Prediction Foundation Track

purpose:

- 馬の相対的な勝ちやすさを安定して表現できているかを見る

primary metrics:

- `auc`
- calibration / logloss
- ranking quality

guardrails:

- train / evaluation leakage なし
- sample support 維持
- cross-date degradation が catastrophic でない

non-goals:

- 後段 policy で ROI を作ること自体をこの track の成功条件にしない

### 2. Market Deviation Track

purpose:

- 市場が付けた価格に対して、model がどの方向へどれだけズレた signal を出せるかを見る

primary metrics:

- market-relative target correlation
- positive signal coverage
- market-relative OOS stability

candidate surfaces:

- `market_deviation`
- alpha / residual style target

non-goals:

- execution threshold の細かい探索で見かけの ROI を最大化すること

### 3. Execution Policy Track

purpose:

- fixed model output を、どの exposure / bankroll / candidate family で換金するかを見る

primary metrics:

- realized ROI
- drawdown
- final bankroll
- bet volume / feasible folds

guardrails:

- `min_bets` を満たす support
- operator 運用に耐える coverage
- family diagnostics が説明可能

non-goals:

- model layer の弱さを policy sweep だけで埋めたと主張すること

## Proposed Execution Order

1. baseline freeze を snapshot tag と参照 artifact で固定する
2. prediction foundation track の current issue を 1 本切る
3. market deviation track の current issue を 1 本切る
4. execution policy track は上のどちらかで signal source を固定した後にだけ進める

## Current Progress Update

2026-04-19 時点の current read は次で固定する。

1. baseline freeze は完了している
   - tag は `strategy-baseline-20260418-pre-track-split`
   - anchor commit は `2245ed32d1b25a5a38c46501585273c81cdcf4ab`
2. market deviation track は Stage 3 まで完了している
   - current best signal-source candidate は `r20260418_jra_lightgbm_alpha_race_norm_v1`
   - read は `alpha_pred_corr=0.2232080870`, `positive_signal_rate=0.2132800000`, `ev_threshold_1_0_roi=1.1144751854`
3. execution policy track は Stage 4 first pass まで到達したが current decision は `hold`
   - bounded sidecar compare は September difficult window で baseline より損失を圧縮した
   - ただし representative compare の優位は弱く、support drift も残った
   - `alpha_weight: 0.05 -> 0.02` follow-up は `reject` で、blind weight sweep が root fix ではないことが確認された

meaning:

- track split 自体は未着手ではなく、market deviation signal-source の再設計までは完了している
- いま未解決なのは「その signal をどう probability path へ戻すか」であり、ここで policy-only probe を継ぎ足す段階ではない

## Current Structural Read

current bottleneck は Stage 4 の parameter surface ではなく、現 mainline が `classification` の確率を基準にした late-fusion 構造のまま、`market_deviation` signal を後段の score composition でしか扱えていないことにある。

現状の bounded sidecar は `compose_value_blend_probabilities` 上で `win_prob logit + alpha_raw + roi_raw + market anchor` を足し込む設計である。この path は narrow compare には向くが、次の限界が確認された。

1. signal-source 自体の改善と execution support の改善が分離している
2. `alpha_weight` のような scalar 調整では final bet exposure を制御できない
3. market signal が probability layer ではなく後段 composition layer に留まるため、mainline replacement の判断面を作りにくい

したがって next branch は Stage 4 の別レバー探索ではなく、market-aware probability path を 1 issue として切り出す architecture / score-composition branch に戻す。

## Locked Next Child Issue

current next child issue は次で固定する。

- [next_issue_jra_market_deviation_market_aware_probability_path.md](next_issue_jra_market_deviation_market_aware_probability_path.md)

この issue の role は次である。

1. race-normalized residual alpha を post-hoc sidecar ではなく probability path 側へ戻す最小構造を定義する
2. policy family や serving default を動かさず、probability layer の改善有無だけを判定する
3. bounded Stage 4 compare をこれ以上継ぎ足さず、late-fusion の構造限界を直接検証する

resume rule:

- execution policy track へ戻るのは、この child issue が `advance` で閉じた後だけにする
- それまでは Stage 4 bounded reintegration issue に新しい parameter probe を追加しない

## First Child Issue Candidates

1. prediction foundation
   - `classification` baseline の calibration / ranking diagnostics を ROI と切り離して formalize する

2. market deviation
   - `market_deviation` task を JRA か NAR のどちらか 1 universe で formal compare できる状態まで持っていく

3. execution policy
   - fixed model artifact に対する family-aware feasibility / bankroll optimization を narrow hypothesis で回す

## Decision Rule

- current bottleneck が model signal にあるときは policy issue を先に開かない
- current bottleneck が support / bankroll conversion にあるときだけ execution policy track を優先する
- Stage 4 first pass が `hold` に留まった後は、bounded policy probe を連想的に継ぎ足さず、architecture / score-composition branch の child issue を先に固定する
- public reporting はこの split 後の current truth に追従させるが、この doc 自体は current source-of-truth ではなく issue draft として扱う