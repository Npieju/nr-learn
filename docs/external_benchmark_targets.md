# 外部ベンチマーク目標（Benter系を基準にした到達点）

最終更新: 2026-03-08

## 1. このドキュメントの役割
- 公開文献・公開実装から、長期運用で追うべき目標値を repo 側に固定する。
- 単発の高ROIではなく、`fundamental model + public odds` の分離ができているかを先に判定する。
- 当面の採用基準は raw ROI ではなく、**Benter系の情報量指標**を先に満たすことに置く。

## 2. 参考にした公開ソース
- Bill Benter 系の整理: `actamachina.com/posts/annotated-benter-paper`
- 小規模公開実装: Teddy Koker `Beating the Odds: Machine Learning for Horse Racing`
- 公開研究実装: `MartianOak/Hong-Kong-Horse-Racing-ML-Betting-Engine`
- 公開 notebook 実装: `ethan-eplee/HorseRacePrediction`

### 2.1 学習規模の比較上の注意
- これらの公開例は **市場もデータ粒度も運用年次も異なる** ため、厳密な apples-to-apples 比較ではない。
- この repo の現行主データは 1986-01-05 から 2021-07-31 までで、総計 `1,626,811 rows / 121,938 races`。
- 現行 split では train が `1,550,224 rows / 116,442 races`、valid が `76,587 rows / 5,496 races`。
- したがって、公開ベンチマークとの比較は今のところ **情報量指標の到達ライン比較** であり、同一学習規模・同一マーケットの厳密比較ではない。
- 本当に同等規模を狙うなら、JRA 側でも pre-race の質的特徴量を増やすか、外部ソースを結合して feature richness を揃える必要がある。

## 3. まず追うべき構成
1. market を直接食わせない **fundamental win model** を作る。
2. public odds から `p_market` を作る。
3. second-stage で `c_i ∝ f_i^α * p_i^β` を学習する。
4. その combined probability に対して fractional Kelly / gate をかける。
5. exotic / place 系は Harville をそのまま使わず補正付きで扱う。

## 4. 外部ベンチマークの数値目標

### 4.1 情報量の目標
- Benter 系で重要なのは単独モデルの良し悪しより **`ΔR² = R²_combined - R²_public`**。
- 公開例:
  - `R²_public = 0.1218`
  - `R²_fundamental = 0.1245`
  - `R²_combined = 0.1396`
  - `ΔR² = 0.0178`
- 別例では `ΔR² = 0.0090` が有意差のある profitable threshold として扱える。
- したがって当面の内部目標は以下に置く。
  - 最低到達ライン: `ΔR² > 0.009`
  - 典型的な強いライン: `ΔR² ≈ 0.018`

### 4.2 public baseline の目安
- annotated Benter の再現では HKJC public estimate の pseudo-R² は次の通り。
  - 1986-1993: `0.1325`
  - 1996-2003: `0.1437`
  - 2006-2013: `0.1668`
  - 2016-2023: `0.1863`
- これは「市場自体が既にどれだけ強いか」の基準であり、model 単独指標ではない。

### 4.3 実運用 ROI の目安
- Benter の practical constraint として、最大期待利益は total per-race turnover に対して概ね `0.25% - 0.5%`。
- `1.5%` 超は現実的でない上限として扱う。
- startup 運用の現実的目標は `0.1% - 0.2%` of turnover。
- したがって、小サンプルで `ROI > 1.0` が出ても、そのまま長期目標値とは見なさない。

## 5. 公開 GitHub 実装の扱い
- `MartianOak/Hong-Kong-Horse-Racing-ML-Betting-Engine`
  - leak-safe dataset、calibration、band backtest まで揃っていて構成の参考価値は高い。
  - ただし README の strict OOS ROI `~114%` は 7 bets しかなく、**目標値ではなく方向確認用**。
- Teddy Koker
  - 938 races / low-frequency betting で profitable と報告。
  - 小規模かつ bet frequency が低く、目標値というより「最低限の成立例」。
- Ethan Eplee
  - 7/8 models positive backtests と記載があるが、notebook ベースで leak-safe OOS の厳密性は弱い。
  - こちらも構成参考止まり。

## 6. この repo の現状（2026-03-08 時点）

### 6.1 market を含む現行 CatBoost win
- config: `configs/model_catboost.yaml`
- feature profile に `odds` / `popularity` が入っている。
- 100k rows 評価:
  - `public_pseudo_r2 = 0.253913`
  - `model_pseudo_r2 = 0.252391`
  - `benter_combined_pseudo_r2 = 0.253601`
  - `benter_delta_pseudo_r2 = -0.000312`
- 解釈:
  - public 自体は強いが、model が public に追加情報を出せていない。
  - これは Benter 的には **fundamental model 失格**。

### 6.2 value stack
- config: `configs/model_catboost_value_stack.yaml`
- 100k rows 評価:
  - `public_pseudo_r2 = 0.253913`
  - `model_pseudo_r2 = 0.253054`
  - `benter_combined_pseudo_r2 = 0.253954`
  - `benter_delta_pseudo_r2 = 0.000041`
- 解釈:
  - Kelly 系の安定性改善はあるが、情報量としては public をほぼ超えていない。
  - `ΔR²` 観点ではまだ benchmark 未達。

### 6.3 market を抜いた fundamental model
- config: `configs/model_catboost_fundamental.yaml`
- feature profile から `odds` / `popularity` を除外。
- 100k rows 評価:
  - `model_pseudo_r2 = 0.116563`
  - fitted `α = 0.0`, `β = 1.0`
  - `benter_combined_pseudo_r2 = 0.253913`
  - `benter_delta_pseudo_r2 ≈ 0`
- 解釈:
  - 現在の safe feature set だけでは public を補完できない。
  - second-stage fit が public-only を選んでおり、fundamental signal が不足している。

### 6.4 pace-corner enriched fundamental model
- config: `configs/model_catboost_fundamental_enriched.yaml`
- feature config: `configs/features_catboost_fundamental_enriched.yaml`
- 100k rows 評価:
  - `auc = 0.760024`
  - `logloss = 0.229481`
  - `model_pseudo_r2 = 0.139080`
  - `public_pseudo_r2 = 0.253913`
  - `benter_combined_pseudo_r2 = 0.253913`
  - `benter_delta_pseudo_r2 ≈ 0`
  - fitted `α = 0.0`, `β = 1.0`
- 解釈:
  - standalone の fundamental signal は改善しており、`model_pseudo_r2` は `0.116563 -> 0.139080` まで上がった。
  - ただし market を補完するほどではなく、Benter second-stage はまだ public-only を選ぶ。
  - 今回効いたのは `horse_last_3_avg_corner_4_ratio`、`horse_last_3_avg_closing_time_3f`、`horse_last_3_avg_race_pace_back3f`、`jockey_last_30_avg_closing_time_3f` などの pre-race pace / running-style 系だった。

### 6.5 corner_passing_order 補完後の再評価
- config: `configs/model_catboost_fundamental_enriched.yaml`
- data config: `configs/data.yaml` with `corner_passing_order` supplemental enabled
- 100k rows 評価:
  - `auc = 0.759210`
  - `logloss = 0.229734`
  - `top1_roi = 0.743231`
  - `model_pseudo_r2 = 0.137177`
  - `public_pseudo_r2 = 0.253913`
  - `benter_combined_pseudo_r2 = 0.253913`
  - `benter_delta_pseudo_r2 ≈ 0`
  - fitted `α = 0.0`, `β = 1.0`
- 解釈:
  - `corner_passing_order.csv` から horse-level の corner position 補完はできたが、coverage 改善は小さく、benchmark 指標は実質不変だった。
  - `top1_roi` はわずかに上がった一方で、`model_pseudo_r2` は `0.139080 -> 0.137177` と微減した。
  - したがって次の優先課題は corner 補完の深掘りではなく、gap report が示した pedigree / breeder 系 raw columns の投入である。

## 7. 当面の採用基準
1. `public_pseudo_r2` は参考値として記録する。
2. 採用判定は `benter_delta_pseudo_r2` を最優先にする。
3. `ΔR² <= 0` のモデルは raw ROI が一時的に良く見えても本命系にしない。
4. `ΔR² > 0.009` を超えるまで、market-blend の微調整より feature enrichment を優先する。

## 8. 次にやるべき改善
1. 今回入れた pace / corner enriched profile を土台に、pedigree・racecard・owner/breeder の外部 pre-race 情報を足して、market を補完できる水準まで fundamental model を引き上げる。
2. `odds` / `popularity` を含む model は「policy model」として扱い、fundamental benchmark とは分離する。
3. `value_blend_model` を使う場合も、土台 win model は market-free fundamental 系に差し替えて再検証する。
4. 外部CSVを `data/external/...` に受けられる multi-source loader を前提にし、netkeiba 等の将来ソース追加を loader 改修なしで試せる状態を維持する。

### 8.1 実装済みの enriched benchmark profile
- config: `configs/model_catboost_fundamental_enriched.yaml`
- feature config: `configs/features_catboost_fundamental_enriched.yaml`
- 追加対象:
  - horse pace history: `horse_last_3_avg_race_pace_front3f`, `horse_last_3_avg_race_pace_back3f`, `horse_last_3_avg_race_pace_balance_3f`
  - horse corner style history: `horse_last_3_avg_corner_2_ratio`, `horse_last_3_avg_corner_4_ratio`, `horse_last_3_avg_corner_gain_2_to_4`
  - course pace baseline: `course_baseline_race_pace_front3f`, `course_baseline_race_pace_back3f`, `course_baseline_race_pace_balance_3f`
  - jockey / trainer style aggregates: `*_last_30_avg_corner_gain_2_to_4`, `*_last_30_avg_closing_time_3f`