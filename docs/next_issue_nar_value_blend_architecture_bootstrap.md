# Next Issue: NAR Value-Blend Architecture Bootstrap

## Summary

NAR を JRA と同水準の model architecture で比較できるようにするには、strict `pre_race_only` benchmark が成立した後に `value_blend` family を持ち込む必要がある。

current NAR は dataset readiness corrective が先行しており、LightGBM baseline line までは formalized されているが、JRA 本線の

- CatBoost win
- LightGBM ROI
- stack / `value_blend`

までは揃っていない。

したがって post-readiness の first measurable architecture issue は、NAR `value_blend` bootstrap である。

## Objective

strict `pre_race_only` NAR benchmark 上で、JRA current standard と同型の `CatBoost win + LightGBM ROI + value_blend stack` を bootstrap し、NAR baseline を architecture parity まで引き上げる。

## Hypothesis

if NAR strict `pre_race_only` benchmark 上で JRA-style `value_blend` architecture を train / evaluate できる, then NAR は LightGBM baseline ではなく JRA と同じ model-development surface で feature / policy compare を進められる。

## In Scope

- NAR 用 win / ROI / stack config
- component retrain 導線
- stack bundle 導線
- strict `pre_race_only` benchmark 上の formal compare

## Non-Goals

- provenance 不足の backfilled benchmark で architecture parity を主張すること
- JRA baseline 更新
- mixed-universe compare
- serving default promotion

## Success Metrics

- NAR strict `pre_race_only` benchmark で
  - CatBoost win retrain
  - LightGBM ROI retrain
  - `value_blend` stack bundle
  - formal evaluation
  が end-to-end で通る
- artifact / gate / actual-date compare の surface が JRA と揃う

## Validation Plan

1. strict `pre_race_only` benchmark rerun を baseline freeze として固定する
2. NAR 用 component config を用意する
3. component retrain
4. stack bundle
5. formal evaluation / promotion gate

## Stop Condition

- `#101` / `#102` が未完了で provenance-defensible benchmark が無い
- strict `pre_race_only` benchmark の row / race support が architecture compare に足りない
- その場合はこの issue は blocked 扱いで維持する
