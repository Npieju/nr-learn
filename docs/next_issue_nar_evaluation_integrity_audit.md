# Next Issue: NAR Evaluation Integrity Audit

## Objective

local Nankan line の高すぎる AUC / ROI を、feature leakage と evaluation / policy optimism に分解して監査する。

## Why Now

current NAR line は次のように見える。

- baseline narrow:
  - `auc=0.8775`
  - `formal_benchmark_weighted_roi=3.6903`
  - `bets / races = 3525 / 28997 = 12.16%`
- combo promoted:
  - `auc=0.8766`
  - `formal_benchmark_weighted_roi=4.3245`
  - `bets / races = 3725 / 28997 = 12.85%`

bet rate 自体は低すぎないが、ROI は高すぎる。まず leak / optimism を疑うべき水準である。

## First Read

現時点の first read は次である。

1. feature leak の直接証拠はまだない  
   baseline feature は [features_local_baseline.yaml](../configs/features_local_baseline.yaml) の 13 本で、露骨な target 列は含まない。

2. leakage audit は弱い  
   現行 audit は keyword / high-corr / exact-match しか見ていない。時系列ずれ、同日混入、policy evaluation optimism は拾えない。

3. evaluation summary と promotion gate は同じものを測っていない  
   `evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_baseline_wf_runtime_narrow_v1.json` では `wf_nested_test_bets_total=0` だが、promotion gate は `formal_benchmark_bets_total=3525` を出している。  
   したがって「モデルがそのまま 3.69 ROI を出した」と読むのは誤りで、まず評価系のズレを疑うべきである。

4. market signal 単独でもかなり強い  
   local Nankan tail `120000` rows の quick sanity check では次だった。
   - `1 / odds` proxy AUC: `0.8554`
   - `-popularity` AUC: `0.8364`

したがって、高 AUC のかなりの部分は `odds / popularity` の market signal で説明できる。

## In-Scope Surface

- `scripts/run_evaluate.py`
- `scripts/run_wf_feasibility_diag.py`
- `scripts/run_revision_gate.py`
- [features_local_baseline.yaml](../configs/features_local_baseline.yaml)
- [features_local_baseline_no_market.yaml](../configs/features_local_baseline_no_market.yaml)

## Non-Goals

- 新しい NAR feature family 実験
- JRA baseline の変更
- いきなり NAR stack / CatBoost 化すること

## Success Metrics

- feature-side leak suspicion を reject するか、具体的な列/時点で証拠化する
- evaluation-side optimism source を具体的な phase に特定する
- corrective action を 1 issue に narrowed できる

## Validation Plan

1. code-path audit
   - `evaluation summary`
   - `wf_feasibility`
   - `promotion gate`
   の分母と最適化対象の違いを明文化する

2. market ablation
   - `odds`
   - `popularity`
   を外した baseline rerun を 1 本回す

3. residual check
   - ablation 後も ROI が異常に高いかを見る

## Stop Condition

- market ablation 後も説明がつかず、さらに raw-timing audit が必要になる
- correction が broad redesign で、1 measurable hypothesis に落ちない

