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