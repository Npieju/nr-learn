# Policy Family Shortlist

## 1. Purpose

この文書は、`high ROI` と `controlled drawdown` を両立するために、既存 artifact から有望な policy family を shortlist 化した初版である。

狙いは、次の policy experiment を思いつきで増やさず、既存 evidence の強い family から順に掘ることである。

## 2. Reading Rule

policy family は次の観点で並べる。

- formal support があるか
- drawdown / bankroll の守りがあるか
- bet volume が極端に細らないか
- operational role が説明しやすいか
- 現行 baseline との比較で位置づけが明確か

## 3. Current Shortlist

### Tier A: Primary Reusable Families

#### 3.1 Tighter Policy Search

代表 config:

- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs80.yaml`

代表 artifact:

- `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
- `artifacts/reports/promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`

読み:

- formal に `pass / promote`
- `ratio003_abs80` では feasible folds `5/5`
- baseline broad replacement ではないが、defensive variant として evidence が強い

位置づけ:

次の policy issue の第一候補。support 改善型の本線 family として扱う。

#### 3.2 Kelly-Centered Runtime Family

代表的な読み:

- best fallback が fold 単位で繰り返し `kelly` に寄る
- `tighter_policy_ratio003` / `ratio003_abs80` の promotion artifact でも feasible fallback の中心が `kelly`

読み:

- 件数を確保しやすい
- drawdown を比較的読みやすい
- policy role が単純で運用解釈しやすい

位置づけ:

portfolio family の最終 fallback というだけでなく、runtime の安定 family として重視する。

### Tier B: Defensive Or Regime-Specific Families

#### 3.3 Long-Horizon / Seasonal De-Risk

代表 profile:

- `current_long_horizon_serving_2025_latest`

読み:

- September difficult window の defensive alias として位置づけが明確
- operational baseline 置換ではなく seasonal de-risk option

位置づけ:

全期間の主役ではないが、regime-conditioned policy family として重要。

#### 3.4 Selected-Rows September Guard Families

代表 config:

- `...sep_selected_rows_guard_candidate.yaml`
- `...sep_selected_rows_kelly_only_candidate.yaml`
- `...sep_selected_rows_ev_only_kelly_candidate.yaml`

読み:

- date-selected rows を使う September 向け override family
- guard / kelly-only / ev-only の variation があり、fallback の説明がしやすい

位置づけ:

seasonal override の探索 family として残す価値が高い。

#### 3.5 Portfolio Lower-Blend / EV-Only Hybrid

代表 config:

- `...serving_portfolio_lower_blend.yaml`
- `...serving_portfolio_lower_blend_hybrid_keep_kelly.yaml`
- `...serving_portfolio_ev_only.yaml`
- `...serving_portfolio_ev_only_hybrid_keep_kelly.yaml`

読み:

- exposure control と EV 純化の両方向を試せる
- ただし現時点では baseline を安定して超える broad evidence は弱い

位置づけ:

secondary defensive family。第一候補ではなく contrast と比較用に使う。

### Tier C: Experimental Families

#### 3.6 Staged Mitigation

代表 config:

- `...staged_mitigation_probe.yaml`
- `...staged_mitigation_ev_guard_probe.yaml`
- `...staged_aug_baseline_stage1_probe.yaml`

読み:

- traceability が高く、どの段で fallback したかが見える
- ただし現時点では runtime capability としては有用でも、主力 family とは言いにくい

位置づけ:

観測・診断に強い family。すぐ promote する本線ではなく、future control family とみなす。

## 4. Policy Ranking

### Rank 1

`tighter policy search`

理由:

- 既存 formal support が最も強い
- ratio と absolute threshold の両軸で frontier を読める
- next issue へ直結しやすい

### Rank 2

`kelly-centered runtime family`

理由:

- fallback の中心で repeatedly feasible
- role が単純で解釈しやすい

### Rank 3

`long-horizon / seasonal de-risk`

理由:

- September difficult regime 向けに operational meaning が明確

### Rank 4

`selected-rows September guard families`

理由:

- seasonal override としては面白いが、broad policy replacement ではない

### Rank 5

`portfolio lower-blend / EV-only hybrid`

理由:

- secondary comparison としては有用
- mainline support はまだ弱い

### Rank 6

`staged mitigation`

理由:

- explainability は高い
- ただし主力 family としての evidence はまだ不足

## 5. Key Evidence

### 5.1 Tighter Policy Search Has The Clearest Formal Support

- `promotion_gate_r20260326_tighter_policy_ratio003.json`
  - feasible folds `4/5`
  - defensive candidate として formal 通過
- `promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
  - feasible folds `5/5`
  - same family の support 境界調整として読める

読み:

まず掘るべきなのは、新しい exotic family ではなく tighter search family の refinement である。

### 5.2 Kelly Is Repeatedly The Feasible Fallback

promotion artifact の `best_fallback_by_fold` を読むと、実務上の feasible family は繰り返し `kelly` に寄る。

読み:

runtime の安定 family として Kelly を軽視しないほうがよい。

### 5.3 Seasonal Families Are Useful But Narrow

`serving_validation_guide.md` の current reading では、long-horizon と selected-row 系は September difficult window の defensive option として意味がある。

読み:

seasonal family は broad replacement ではなく regime-specific role で評価するべきである。

## 6. Default Next Bets

次の policy issue は次の順を推奨する。

1. kelly family を baseline defensive fallback として再整理
2. selected-rows September family の simplest variant 比較
3. lower-blend / EV-only hybrid の取捨選択

2026-03-29 update:

- `tighter policy search family の frontier refinement` は first-wave を完了
- promoted anchor は `r20260329_tighter_policy_ratio003_abs90`
- `ratio003_abs90_odds25` は near-par challenger だったが、anchor replacement には至らなかった
- したがって次の本線は `kelly-centered runtime family` に移す

## 7. Not Recommended Next

当面、次を primary な policy issue にしない。

- staged family の大型拡張
- broad replacement を前提にした seasonal family の昇格
- artifact support のない新規 exotic family の追加

## 8. Operating Rule

新しい policy issue には最低限次を入れる。

- policy family 名
- baseline に対する役割
- primary target
  - ROI improvement
  - drawdown control
  - regime override
- expected support shape
  - more bets
  - fewer bets but safer
  - seasonal / conditional
