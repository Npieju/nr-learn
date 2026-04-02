# Public Benchmark-To-Operation Reading Guide

## Purpose

この文書は、対外向けに `nr-learn` の数字を説明するときの読み方を固定する。

特に次の 3 つを混同しないためのガイドである。

1. `evaluation summary`
2. `promotion gate / formal benchmark`
3. `actual-date serving read`

## Core Rule

同じ revision でも、次の 3 つは意味が違う。

| Layer | 何を見るか | 何を意味するか |
| --- | --- | --- |
| evaluation summary | `auc`, `top1_roi`, `ev_top1_roi`, `wf_nested_test_roi_weighted` | モデルと policy の検証上の shape |
| promotion gate | `formal_benchmark_weighted_roi`, `formal_benchmark_bets_total`, `wf_feasible_fold_count` | formal gate 上で採用できるか |
| actual-date / operational read | actual-date compare, `bets / races / bet_rate`, role split | 実運用 default にできるか |

`pass / promote` は「formal gate を通った」という意味であり、そのまま operational default を意味しない。

## Denominator Rule

対外向けには、bet 数だけを単独で出さない。必ず race 分母を併記する。

最低限、次を同時に書く。

- `bets`
- `races`
- `bet_rate = bets / races`

補助的に row 分母を使うことはあるが、運用の読みは race 分母を優先する。

## Low Bet-Rate Caution

高い ROI が見えても、bet rate が低い line は過学習や集中ヒットの可能性を強く疑う。

外向けの標準文面は次で固定する。

- low bet-rate line は formal candidate にはなりうる
- ただし operational default として扱うには actual-date の bet-rate robustness が必要

## Current Public Reading

### JRA

JRA は、formal promoted line と operational default line を分けて説明する。

ここでの race 分母は、JRA current line 同士の比較では matching `wf` の `test_races_total=16215` に揃える。

| line | role | AUC | top1 ROI | EV top1 ROI | formal benchmark weighted ROI | bets / races | reading |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `current_recommended_serving_2025_latest` | operational default | `0.8401` | `0.8070` | `0.5568` | baseline line のため直接比較より nested WF を優先 | nested WF `544 / 16215 = 3.35%` | current default |
| `r20260326_tighter_policy_ratio003` | analysis-first defensive candidate | `0.8401` | `0.8070` | `0.5568` | `1.1728` | `500 / 16215 = 3.08%` | formal pass だが default 置換ではない |
| `r20260330_surface_plus_class_layoff_interactions_v1` | formal promoted but not operational default | `0.8416` | `0.8002` | `0.7100` | `1.1380` | `519 / 16215 = 3.20%` | actual-date bet rate が低く conservative role に留める |

JRA の current public message は次で足りる。

- operational default は `current_recommended_serving_2025_latest`
- tighter policy は difficult regime 向けの defensive candidate
- `surface_plus_class_layoff` は formal promoted line だが、low bet-rate のため operational default ではない

### NAR

NAR は separate universe として扱う。JRA の default 置換とは切り離して読む。

| line | role | AUC | top1 ROI | EV top1 ROI | formal benchmark weighted ROI | bets / races | reading |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| `r20260330_local_nankan_baseline_wf_runtime_narrow_v1` | NAR baseline line | `0.8775` | `0.8382` | `1.9408` | `3.6903` | `3525 / 28997 = 12.16%` | denominator-first baseline |
| `r20260330_local_nankan_jockey_trainer_combo_replay_v1_pathfix` | current NAR formal promoted line | `0.8766` | `0.8330` | `0.9687` | `4.3245` | `3725 / 28997 = 12.85%` | current best formal line |

NAR の current public message は次で固定する。

- NAR は separate universe の benchmark line として運用している
- current formal promoted line は `jockey_trainer_combo_replay_v1_pathfix`
- NAR は race 分母 bet rate が `12%` 台で、low-frequency candidate ではない
- ただし NAR の高 AUC / 高 ROI は market-aware line として読むべきで、non-market skill の純粋な proxy とは扱わない

追加の内部 caution は次である。

- `odds / popularity` を外した no-market audit では、AUC は `0.8775 -> 0.7672`、formal benchmark weighted ROI は `3.6903 -> 0.8104` まで低下した
- したがって、NAR current line の強さの大部分は market signal 依存である
- 対外向けには「market-aware separate-universe benchmark」という説明を優先し、market を切った純モデル性能としては説明しない

## What To Say Publicly

対外向けには、次の順で説明する。

1. universe を分ける
2. operational default と formal promoted line を分ける
3. `bets / races / bet_rate` を出す
4. low bet-rate line には caution を添える

短い説明文は次で足りる。

- JRA は operational default と formal promoted candidate を分けて運用している
- NAR は separate-universe benchmark line として formalized している
- 高 ROI line でも bet rate が低い場合は default 昇格を急がない

## What Not To Say

次の言い方は避ける。

- `pass / promote` したので即運用採用した
- `weighted ROI` が高いのでそのまま live expectation だ
- `AUC` が高いので運用でも強い
- `bets` だけを出して `races` を出さない

## Source Artifacts

### JRA

- baseline evaluation:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- tighter policy evaluation:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260326_tighter_policy_ratio003_wf_full_nested.json`
- tighter policy promotion:
  - `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
- surface-plus-layoff evaluation:
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260330_surface_plus_class_layoff_interactions_v1_wf_full_nested.json`
- surface-plus-layoff promotion:
  - `artifacts/reports/promotion_gate_r20260330_surface_plus_class_layoff_interactions_v1.json`
- operational role split standard:
  - [promoted_vs_operational_role_split_standard.md](promoted_vs_operational_role_split_standard.md)

### NAR

- baseline evaluation:
  - `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_baseline_wf_runtime_narrow_v1.json`
- baseline promotion:
  - `artifacts/reports/promotion_gate_r20260330_local_nankan_baseline_wf_runtime_narrow_v1.json`
- combo evaluation:
  - `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_jockey_trainer_combo_replay_v1_pathfix.json`
- combo promotion:
  - `artifacts/reports/promotion_gate_r20260330_local_nankan_jockey_trainer_combo_replay_v1_pathfix.json`
