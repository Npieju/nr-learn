# ベンチマークと採用基準

## 1. この文書の役割

この文書は、`nr-learn` で何を良い結果とみなすかを定義するための正本である。

日常運用の区切り方や `revision` の切り方は、[development_flow.md](development_flow.md) を参照する。
formal な evaluation 導線は、[evaluation_guide.md](evaluation_guide.md) を参照する。

ここで決めるのは次の 3 点である。

1. 何の指標で評価するか。
2. どの数値を目標とするか。
3. どの条件を満たしたら採用するか。

## 2. 基本姿勢

- 単発の高 ROI は、そのまま採用根拠にしない。
- 同一期間で最適化と評価を兼ねない。
- raw 指標だけでなく、実際に買った場合の件数と純収益まで確認する。
- より単純な構成で同じ observed behavior を再現できるなら、単純な構成を優先する。

## 3. 主要指標

### 3.1 Weighted ROI

時系列検証の各期間で得た ROI を、件数を加味してまとめた指標である。

- `1.0` を超えると、検証全体で回収率が 100% を超えたことを意味する。
- ただし bets が極端に少ない場合は採用しない。

### 3.2 Bets

実際に賭けた件数である。

- 少なすぎる ROI は再現性が低い。
- `Weighted ROI` と必ずセットで解釈する。

### 3.3 Total Net

回収総額から投資総額を引いた値である。

- mean ROI だけでは、件数差による見かけの改善を見落とすことがある。
- serving 比較では `total net` を重要指標として扱う。

### 3.4 AUC / pseudo-R²

予測の並び順品質と、市場オッズに対する追加情報の有無を見る。

- AUC は予測順位の良さを見る。
- pseudo-R² は市場に対する独自情報の強さを見る。

## 4. 外部比較の目安

Benter 系の比較では、重要なのは単純な ROI よりも `ΔR² = R²_combined - R²_public` である。

当面の目安は次のとおりとする。

- 最低到達ライン:
  - `ΔR² > 0.009`
- 強いライン:
  - `ΔR² ≈ 0.018`

また、短期サンプルで `ROI > 1.0` が出ても、それだけで長期運用の目標達成とはみなさない。

## 5. 現在の内部 benchmark ladder

2024 年データの nested walk-forward における主要候補は次のとおりである。

| 候補 | Weighted ROI | Bets | 位置づけ |
| --- | ---: | ---: | --- |
| no-lineage public-free baseline | `0.5788` | `603` | 市場情報を切った基準線 |
| liquidity high-coverage | `0.9346` | `700` | 流動性改善の転換点 |
| regime hybrid | `0.9915` | `731` | 単純な運用候補 |
| regime modelswitch f1 policy may | `1.0073` | `713` | 現在の主力候補 |

この数値だけを見ると最後の候補が最良だが、構成の複雑さも同時に増しているため、運用判断では simpler rollback 候補との比較を必ず行う。

### 5.1 直近の revision comparison

2026-03-22 時点の full revision gate では、次の 2 候補を同条件で通した。

| revision | profile | status | decision | 備考 |
| --- | --- | --- | --- | --- |
| `r20260322a` | `current_best_eval` | `pass` | `promote` | `score_source_count=2`。`may_runtime_liquidity` override を読む |
| `r20260322b` | `current_recommended_serving` | `pass` | `promote` | `score_source_count=1`。単一 source で通過 |

この 2 run では、`evaluate --max-rows 120000 --wf-mode fast --wf-scheme nested` のトップライン指標は実質同一だった。

- `top1_roi`: `0.796298...`
- `ev_top1_roi`: `0.660712...`
- `auc`: `0.834390...`

したがって、現時点の運用判断では `current_recommended_serving` を simpler candidate として優先的に扱ってよい。
ただし、両者とも promotion gate warning として `probe_only` な walk-forward slice が残るため、将来の benchmark 更新では引き続き stricter full evaluation を優先する。

### 5.2 de-risk candidate の formal gate

2026-03-22 に、de-risk 候補 2 本も formal な revision gate に載せた。

| revision | profile | status | decision | 備考 |
| --- | --- | --- | --- | --- |
| `r20260322c` | `current_bankroll_candidate` | `block` | `hold` | representative evaluate は通るが、matching WF で feasible fold `0/5` |
| `r20260322d` | `current_ev_candidate` | `block` | `hold` | representative evaluate は通るが、matching WF で feasible fold `0/5` |

この 2 候補では共通して次が確認できた。

- `evaluate` 側の `stability_assessment` は `representative`
- matching な `wf_feasibility_diag` を付けると promotion gate は `wf_feasible_fold_count=0` で block
- dominant failure reason は `min_bets`
- fold ごとの binding source は全て `min_bets_abs=100` で、ratio 側ではなかった
- `max_infeasible_bets_observed=58` に留まり、threshold 未達が明確だった

同日付の threshold sweep で、support gap も数値化した。

- 両候補とも `min_bets_abs=58` で初めて feasible fold が `1/5` に到達した
- 両候補とも `min_bets_abs=40` で feasible fold が `4/5` になり、`3/5` 条件はここで満たす
- 両候補とも `min_bets_abs=30` まで下げると feasible fold が `5/5` になった
- fold 別の最初の到達 threshold は共通で、fold 3 が `58`、fold 1 が `45`、fold 2 と fold 4 が `40`、fold 5 が `30` だった

さらに threshold compare から mitigation probe を組み立てると、runtime 候補としては次の 2 本に収束した。

- dominant blocked signature は `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95` で、84 blocked occurrence を占めた
- そのうち `74/84` occurrence では `portfolio / blend_weight=0.6 / min_prob=0.03 / top_k=1 / min_ev=1.0` がより高い final bankroll を示した
- 残り `10/84` occurrence では `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=1.0` が pure bankroll で上回った
- mitigation probe からは runtime-ready candidate として `portfolio_lower_blend` と `portfolio_ev_only` の 2 本を書き出せたが、`74/10` の staged hybrid は現行 runtime の単一 policy/date override では直接表現できなかった

つまり、この 2 候補の formal block は「bankroll 側か EV 側か」の違いではなく、同じ support frontier に乗っている問題として読むのが正しい。

したがって、`current_bankroll_candidate` と `current_ev_candidate` は serving 上の de-risk 候補としては有力でも、現時点では benchmark 更新や正式昇格の候補とは扱わない。運用上は rollback / defensive override の候補に留め、昇格判断は support を増やす別の改善が入ってから再評価する。

## 6. serving 比較の判断基準

actual calendar の比較では、次を確認する。

1. `policy_bets`
2. `policy_roi`
3. `total return`
4. `total net`

現在の判断原則は次のとおりである。

- 追加 return があっても `total net` が悪化するなら採用しない。
- 主力候補と observed behavior が一致する単純候補があるなら、rollback ではそちらを優先する。
- representative date だけでなく、複数日 window で再確認する。

### 6.1 直近の複数 window aggregate

2026-03-22 時点で、`current_recommended_serving` を基準に次の 2 系統を複数 window aggregate で比較した。

- `current_bankroll_candidate`
- `current_ev_candidate`

入力 window は次の 4 本である。

- `tail_weekends`
- late-September 5 日 window
- `aug_weekends_20260322`
- `may_weekends_20260322`

aggregate の要点は次のとおりである。

- `current_bankroll_candidate` は pure bankroll delta が `2/4` windows で正、`1/4` は zero、4-window mean は `net -2.6500`, `bankroll +0.1520`
- `current_ev_candidate` も pure bankroll delta が `2/4` windows で正、`1/4` は zero、4-window mean は `net -5.6500`, `bankroll +0.0354`
- 両候補とも net delta が正だったのは late-September 側だけだった
- `tail_weekends` では `current_recommended_serving` が total net で優位で、候補側は pure final bankroll のみ改善した
- `aug_weekends_20260322` では `current_recommended_serving` が net と bankroll の両面で両候補を上回った
- `may_weekends_20260322` では両候補とも `current_recommended_serving` と完全一致した

aggregate JSON の `tradeoff_classification` で言い換えると、window 構成は `++`, `-+`, `--`, `00` が 1 本ずつである。

- late-September は `positive_net_positive_bankroll`
- `tail_weekends` は `negative_net_positive_bankroll`
- `aug_weekends_20260322` は `negative_net_negative_bankroll`
- `may_weekends_20260322` は `zero_net_zero_bankroll`

このため、候補側を「bankroll 改善候補」とだけ要約するのは粗すぎる。現状の evidence は、「一部 regime では効き、別 regime では baseline より悪化し、さらに差分が消える regime もある」である。

したがって、現時点の運用上の位置づけは次のとおりである。

1. `current_bankroll_candidate` は rollback / de-risk の有力候補だが、全 regime で優位とは言えない
2. `current_ev_candidate` は中間候補だが、現時点では `current_recommended_serving` を安定して上回る根拠は弱い
3. `current_recommended_serving` は baseline として維持し、候補置換は regime ごとの裏付けが増えるまで慎重に扱う

## 7. 現在の評価方針

### 7.1 nested evaluation

本命候補の判定は、`run_evaluate.py` による nested walk-forward を基準に行う。

理由は次の 2 点である。

- 同じ期間で最適化と評価を兼ねないため。
- fold ごとの no-bet や gating failure まで記録できるため。

### 7.2 serving validation

運用候補の確認は、`run_serving_smoke.py` と `run_serving_smoke_compare.py` を基準に行う。

理由は次の 2 点である。

- summary から復元した serving ルールが、実日付でも同じように動くか確認できるため。
- nested 上の改善が、actual calendar でも意味を持つかを見られるため。

ただし、短い serving validation は smoke / probe であり、正式な採用判断は `stability_assessment=representative` と promotion gate を通した revision 単位で行う。

## 8. 採用ルール

最終的な採用判断は、次の順序で行う。

1. nested walk-forward で改善しているか。
2. bets が少なすぎないか。
3. actual calendar で total net が悪化していないか。
4. 同等挙動を再現するより単純な候補がないか。

この 4 条件を満たして初めて、本命または rollback 候補として扱う。