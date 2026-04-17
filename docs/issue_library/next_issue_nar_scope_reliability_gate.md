# Next Issue: NAR South-Kanto Scope And ROI Reliability Gate

## Summary

current NAR line は `local Nankan`、すなわち South Kanto local track だけを対象にしている。

一方で historical diagnostics では、current high ROI の読みについて次が残っている。

- `#69` evaluation integrity audit:
  - high AUC / high EV ROI の大部分は `odds / popularity` 依存
- `#70` policy optimism audit:
  - evaluation phase と promotion phase の間に大きな optimism gap が残る

したがって現時点の local Nankan result を `NAR 全体` や `JRA 相当の信頼性を持つ ROI` として扱うのは不適切である。

## Objective

current local Nankan line を `South-Kanto-only readiness track` として明示し、universe coverage gap と ROI reliability gap が解消されるまで、NAR result の読み方と queue 上の位置づけを誤解なく固定する。

## Hypothesis

if South-Kanto-only scope と ROI reliability unresolved state を docs / queue / issue thread で明示し、NAR current result の interpretation gate を固定する, then operators will stop over-reading local Nankan ROI as full-NAR or JRA-equivalent evidence, while preserving the readiness path for `#101` and `#103`.

## In Scope

- local Nankan only である current universe scope の固定
- `#69` / `#70` diagnostic read の queue 反映
- NAR current result を readiness track として読む wording の固定
- next action を `#101`, `#103`, future universe expansion / reliability audit に整理する

## Non-Goals

- full NAR multi-region ingestion の即時実装
- current local Nankan architecture bootstrap の即時実行
- JRA と NAR の mixed-universe train
- ROI を改善したと主張する新規 benchmark rerun

## Success Metrics

- current NAR line が `South-Kanto-only` と明記される
- current local Nankan ROI が `unresolved reliability` を含むと明記される
- `#101` / `#103` / future universe expansion の関係が queue 上で混乱なく読める
- NAR result を JRA-equivalent evidence と誤読しない guardrail が source-of-truth docs に入る

## Validation Plan

1. `nar_model_transfer_strategy.md` に scope / reliability guardrail を入れる
2. `nar_jra_parity_issue_ladder.md` に `local Nankan != full NAR` の注意書きを入れる
3. `github_issue_queue_current.md` の NAR section に unresolved state を反映する
4. GitHub issue thread に objective / stop condition / next actions を残す

## Stop Condition

- wording fix だけでは current misread risk が十分に下がらない
- その場合は次 issue を `universe expansion` または `reliability corrective rerun` のどちらか 1 本に絞って切る

## Current Read

- current NAR corpus は `local Nankan` only
- current benchmark reference は `#101` formal rerun `r20260415_local_nankan_pre_race_ready_formal_v1`
- current next measurable hypothesis は `#103` value-blend architecture bootstrap
- current interpretation blocker は South-Kanto-only scope と ROI reliability unresolved state である

## Decision Rule

- `#101` benchmark reference は current baseline freeze として保守し、その上で Stage 1 architecture parity を読む
- `#103` は architecture parity issue として保持するが、local Nankan line を full-NAR representative と主張しない
- universe coverage と ROI reliability が解消されるまで、current local Nankan result は readiness / audit track として扱う
