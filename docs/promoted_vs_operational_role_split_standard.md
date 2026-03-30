# Promoted Vs Operational Role Split Standard

## Purpose

formal gate で `pass / promote` した candidate が、そのまま serving default になるとは限らない場面を標準化する。

この文書は特に、`r20260330_surface_plus_class_layoff_interactions_v1` のように formal benchmark では勝つ一方、actual-date serving では low-frequency / concentrated shape を示す candidate をどう扱うかの正本である。

## Core Rule

次の 2 つは明示的に分けて扱う。

- formal promoted line
- operational default line

formal promoted line は、revision gate / promotion gate で最も強い根拠を持つ line を指す。  
operational default line は、actual-date compare と role simplicity まで含めて日常運用の基準に据える line を指す。

## Default Reading

candidate が次を同時に満たすなら、`formal promoted but not operational default` と読む。

- formal benchmark では baseline を上回る
- actual-date compare では bet rate が著しく低い
- date-level concentration が高い
- simple policy-side widening でも broad replacement に育たない

## Current Applied Reading

`r20260330_surface_plus_class_layoff_interactions_v1` の current reading は次で固定する。

- formal promoted line: `r20260330_surface_plus_class_layoff_interactions_v1`
- operational default line: `current_recommended_serving_2025_latest`
- operational role: analysis-first conservative promoted candidate

根拠:

- September actual-date: `8 / 216 races = 3.70%`, total net `-8.0`
- December actual-date: `13 / 264 races = 4.92%`, total net `+20.0`
- December best read は promoted 単独ではなく selective hybrid
- widening probes:
  - `1 / 216 = 0.46%`
  - `0 / 216 = 0.00%`

## Operator Template

current conclusion は次の文面で固定する。

- formal promoted line remains `<revision>`
- operational default remains `<profile>`
- current role is `analysis-first conservative promoted candidate`
- do not treat this line as broad replacement unless actual-date bet-rate robustness is demonstrated

## Next-Step Rule

role split が確定したら、次の queue は次のどちらかに限定する。

1. role split を前提にした次 family の experiment reentry
2. candidate の fragility を直接ほどくための narrowly scoped follow-up

low bet-rate candidate を、根拠なしに broad default 昇格へ押し込まない。
