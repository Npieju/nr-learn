# Next Issue: Exposure Widening For Surface Plus Class-Layoff Promoted Line

## Summary

`r20260330_surface_plus_class_layoff_interactions_v1` は formal では `pass / promote` に到達したが、actual-date compare と bankroll sweep では broad replacement ではなく low-frequency conservative candidate と読むのが妥当だった。

- September: `8 / 216 races = 3.70%`, total net `-8.0`
- December: `13 / 264 races = 4.92%`, total net `+20.0`
- December best hybrid: `1.7395`, promoted を使うのは `2025-12-06` のみ

## Objective

promoted feature line の gain を壊さずに、actual-date の bet rate と date concentration を少し広げられるかを検証する。

## Current Read

- feature line 自体は formal に勝っている
- serving role の block は主に low bet-rate / narrow exposure
- 次の改善余地は feature widening ではなく policy / serving 側の recalibration にある

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
