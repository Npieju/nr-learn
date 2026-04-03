# Next Issue: Exposure Widening For Surface Plus Class-Layoff Promoted Line

## Summary

`r20260330_surface_plus_class_layoff_interactions_v1` は formal では `pass / promote` に到達したが、actual-date compare と bankroll sweep では broad replacement ではなく low-frequency conservative candidate と読むのが妥当だった。

- September: `8 / 216 races = 3.70%`, total net `-8.0`
- December: `13 / 264 races = 4.92%`, total net `+20.0`
- December best hybrid: `1.7395`, promoted を使うのは `2025-12-06` のみ

## Objective

promoted feature line の gain を壊さずに、actual-date の bet rate と date concentration を少し広げられるかを検証する。

## Hypothesis

if promoted feature config を固定したまま policy / serving 側だけを再調整する, then September / December の actual-date bet rate を少し広げつつ、analysis-first promoted role から一段説明しやすい selective override candidate に近づける可能性がある。

## Current Read

- feature line 自体は formal に勝っている
- serving role の block は主に low bet-rate / narrow exposure
- 次の改善余地は feature widening ではなく policy / serving 側の recalibration にある
- 既存 widening probe は suppressive 方向だった
  - September promoted: `8 / 216 races = 3.70%`, total net `-8.0`
  - exposure widening v1: `1 / 216 races = 0.46%`, total net `-1.0`
  - exposure widening v2 lower blend: `0 / 216 races = 0.00%`, total net `0.0`

## In Scope

- promoted feature config を固定したままの policy-side compare
- actual-date September / December compare
- denominator と bankroll sweep を伴う role read

## Non-Goals

- feature family の作り直し
- unrelated runtime work
- NAR work

## Success Criteria

- September / December の両方で role が説明しやすい
- bet rate が少し広がっても fragility が悪化しない
- next serving reading が `default / selective override / analysis-first` のどこに置くべきか整理できる

## Validation Plan

- existing artifact reread で promoted / widening v1 / widening v2 の denominator と date concentration を並べ直す
- September / December compare を baseline と promoted の両方に対して読む
- bankroll sweep / hybrid read を添えて、broad replacement ではなく selective override が成立するかを判定する

## Validation Commands

- `python scripts/run_serving_profile_compare.py ...`
- `python scripts/run_serving_stage_path_compare.py ...`
- `python scripts/run_serving_compare_dashboard.py ...`

## Expected Artifacts

- September / December compare read
- widening vs promoted denominator compare
- role decision summary

## Stop Condition

- widening candidate が promoted よりさらに suppressive になる
- bet rate を広げると December control の gain が崩れる
- selective override としても説明価値がなく、analysis-first promoted role 維持が妥当と確定する
