# 対外向けベンチマーク概要

## 1. この文書の役割

この文書は、`nr-learn` の現状を対外向けに説明するときに使う benchmark snapshot である。

内部運用の gate 条件、fold ごとの失敗理由、候補の棄却履歴まではここに持ち込まない。そうした内部向け判断材料は [benchmarks.md](benchmarks.md)、[evaluation_guide.md](evaluation_guide.md)、[roadmap.md](roadmap.md) を正本とする。

ここでは次だけを簡潔に示す。

1. 現在の基準線
2. 直近で formal に通過した候補
3. 実運用 window で確認済みの挙動

## 2. 更新ルール

- 新しい `revision gate` が `pass / promote` で通ったら、この文書の数値を更新する。
- operational baseline が変わったら、この文書の「現在の採用位置づけ」を更新する。
- 数値の出典は、原則として `artifacts/reports/` の versioned evaluation / promotion / dashboard artifact に固定する。
- 対外向け文書では、内部 candidate の細かな棄却理由や exploratory な仮説は書かない。

## 3. 現在の snapshot

更新日: 2026-03-26

### 3.1 長期の基準線

2024 年データの nested walk-forward における代表的な benchmark ladder は次のとおりである。

| 区分 | Weighted ROI | Bets | 説明 |
| --- | ---: | ---: | --- |
| 基準モデル | `0.5788` | `603` | 市場情報を切った基準線 |
| 高流動性候補 | `0.9346` | `700` | no-bet を大きく減らした転換点 |
| 単純な運用候補 | `0.9915` | `731` | 構成を抑えたまま ROI 1.0 近辺 |
| 主力候補 | `1.0073` | `713` | 現時点の 2024 benchmark 上位 |

この ladder は「モデルの改善がどこまで進んだか」を見るための長期基準線であり、単一の短期 window をそのまま一般化しないための参照枠として使っている。

### 3.2 最新データでの current baseline

2025 backfill 済みデータを使った latest holdout では、現在の operational baseline は `current_recommended_serving_2025_latest` である。

| 指標 | 値 |
| --- | ---: |
| revision | `r20260325_current_recommended_serving_2025_latest_benchmark_refresh` |
| decision | `pass / promote` |
| stability assessment | `representative` |
| AUC | `0.8401` |
| top1 ROI | `0.8070` |
| nested WF weighted test ROI | `0.7628` |
| nested WF bets total | `544` |

この baseline は、latest 2025 split に対して formal gate を通過しており、現在の運用基準として扱っている。

### 3.3 直近の改善 candidate

2026-03-26 に、2025 regime 向けに policy search を引き締めた candidate も formal に通過した。

| 指標 | 値 |
| --- | ---: |
| revision | `r20260326_tighter_policy_ratio003` |
| decision | `pass / promote` |
| stability assessment | `representative` |
| AUC | `0.8401` |
| top1 ROI | `0.8070` |
| nested WF weighted test ROI | `0.9092` |
| nested WF bets total | `424` |
| formal benchmark weighted ROI | `1.1728` |
| formal benchmark feasible folds | `4 / 5` |

この candidate は support 改善の観点で有望だが、現時点では operational baseline の置換までは行っていない。現在の位置づけは「formal に通過した analysis-first candidate」である。

## 4. 実運用 window で確認済みの挙動

latest 2025 の actual-date compare では、次の 2 点を確認している。

### 4.1 September window では de-risk variant が有効

2025-09 の 8 日 window では、seasonal de-risk variant が baseline より損失を大きく抑えた。

| 指標 | Baseline | Seasonal de-risk |
| --- | ---: | ---: |
| policy bets | `32` | `9` |
| total policy net | `-27.3` | `-4.3` |
| pure path final bankroll | `0.2959` | `0.9996` |

したがって、September regime では defensive variant に意味があることを確認済みである。

### 4.2 December tail では baseline と一致

2025-12 末尾の 8 日 window では、baseline と seasonal de-risk variant は同一挙動だった。

| 指標 | Baseline | Seasonal de-risk |
| --- | ---: | ---: |
| policy bets | `3` | `3` |
| total policy net | `14.9` | `14.9` |
| pure path final bankroll | `1.3691` | `1.3691` |

つまり seasonal de-risk は broad rewrite ではなく、必要な regime だけで差が出る controlled override として機能している。

## 5. 現在の採用位置づけ

- 現在の operational baseline: `current_recommended_serving_2025_latest`
- 現在の seasonal de-risk variant: `current_long_horizon_serving_2025_latest`
- 現在の formal improvement candidate: `current_tighter_policy_search_candidate_2025_latest`

対外向けには、現状を次のように要約できる。

- 長期 benchmark では、基準線 `0.5788` から主力候補 `1.0073` まで改善が進んでいる。
- latest 2025 holdout でも baseline は formal gate を通過済みである。
- 直近では support 改善型の candidate も formal に通過し、latest regime に対する改善余地が確認できた。
- 実運用に近い actual-date compare では、September のような難しい時期にだけ defensive override が意味を持つことを確認している。

## 6. 出典 artifact

- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/revision_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_model_r20260326_tighter_policy_ratio003_wf_full_nested.json`
- `artifacts/reports/revision_gate_r20260326_tighter_policy_ratio003.json`
- `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_sep_full_month_2025_latest_profile.json`
- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_dec_tail_2025_latest_profile.json`