# 2026-03-19 中間総合レビュー

## 1. 要旨

現時点の `nr-learn` は、単なる学習スクリプト集ではなく、データ取り込み、特徴量生成、モデル学習、nested walk-forward 評価、serving ルール生成、実 calendar 上の smoke validation までを一気通貫で回せる研究基盤に育っている。

一方で、主力の改善源はまだ単純なモデル強化ではなく、`regime-dependent` な score / policy routing に強く依存している。特に現在の best は May 専用 score override と June 以降の policy 制御を含むため、成績は最良だが構成はやや複雑である。

中間評価としては、以下の整理が妥当である。

- 研究基盤としてはかなり強い。データから serving 検証までの導線が揃い、比較可能な artifact が残る。
- ベンチマークは明確に前進している。recent 2024 YTD nested WF で `ROI > 1` を達成した。
- 運用面も改善した。best simple fallback より一段よい June-strict rollback 候補が見つかり、actual calendar でも best と一致した。
- ただし「本質的な汎化」が確認できたとはまだ言い切れない。May uplift は追加 runtime score artifact に依存しており、ここが中長期の主課題である。

要するに、プロジェクト全体は「探索段階」から「運用可能なルール設計段階」へ進んだが、最終的な主戦場は raw retrain ではなく、regime selection の一般化と単純化に移っている。

## 2. 現在の構成

### 2.1 システム構成

現在の主要レイヤは以下の 6 層に整理できる。

1. Data Layer
   - Kaggle/JRA 主表をベースに、netkeiba / lap time / corner passing order を multi-source merge で補完する。
   - `horse_key` を優先した履歴再同定により、recent domain でも履歴接続の安定性を高めている。

2. Feature Layer
   - `build_features` で履歴系、pace 系、time 系、recent-domain 系を生成する。
   - `all_safe` / `explicit` による feature selection と coverage summary を備え、low coverage force-include を可視化できる。

3. Model Layer
   - CatBoost を主力の win component とし、LightGBM ROI を組み合わせる `value_blend` stack が中心。
   - 単体モデル群に加え、`win + alpha + roi + time` の bundle 構成も維持している。

4. Evaluation Layer
   - raw AUC / logloss だけでなく、`top1_roi`、EV 系 ROI、Benter benchmark、nested walk-forward weighted ROI を同時に評価する。
   - `gate_then_roi` と policy constraints により、精度と運用 feasibility を分けて判定できる。

5. Serving Layer
   - `serving.score_regime_overrides` と `serving.policy_regime_overrides` により、target date ごとに score source と fixed policy を切り替えられる。
   - `run_export_serving_from_summary.py` により、nested summary から serving block を再生成できる。

6. Validation Layer
   - `run_serving_smoke.py` と `run_serving_smoke_compare.py` により、representative date、短期 calendar window、long tail window を artifact 付きで比較できる。
   - 直近では preset 以外の `--date` でも expected route を config から自動解決できるようになり、任意窓の検証が容易になった。

### 2.2 現在の主力 config の位置づけ

- Primary best nested-eval candidate
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may.yaml`
  - May fold 相当でだけ runtime liquidity score source と Kelly-biased policy search を使う。

- Best simple fallback
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid.yaml`
  - default score のまま、policy search だけを fold regime で切り替える。

- Simpler serving rollback candidate
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml`
  - default score / single-stack を維持しつつ、serving policy だけ May と June を分ける。
  - 現時点では、この config が「単純さ」と「best 追従性」のバランスが最もよい。

## 3. ベンチマーク

### 3.1 モデル評価ベンチマーク

recent 2024 YTD nested WF における benchmark ladder は以下のとおり。

| 候補 | 役割 | Weighted ROI | Bets | 補足 |
| --- | --- | ---: | ---: | --- |
| no-lineage public-free baseline | public-free benchmark | `0.5788` | `603` | deployable ではないが、public を切った基準線 |
| liquidity high-coverage | 改善基準線 | `0.9346` | `700` | late-summer no-bet を解消した転換点 |
| regime hybrid | best simple fallback | `0.9915` | `731` | policy-only regime 切替で `ROI ≈ 1` まで到達 |
| regime modelswitch f1 policy may | current best | `1.0073` | `713` | May 専用 score override と policy bias を含む現行最良 |

現在の best と fallback は raw score metrics では非常に近い。

- `top1_roi = 0.7941`
- `ev_top1_roi = 0.4303`
- `auc = 0.8423`
- `public_pseudo_r2 = 0.2661`
- `model_pseudo_r2 = 0.2728`

差分は主に raw score 自体ではなく、fold ごとの regime selection と serving-time policy の置き方にある。

### 3.2 Serving ベンチマーク

#### A. 10-date May/June actual window

比較対象:

- best: `serving_smoke_best_policy_may_window.json`
- old fallback: `serving_smoke_fallback_hybrid_window.json`
- June-strict rollback: `serving_smoke_fallback_hybrid_june_strict_window.json`

結果:

| 比較 | Shared dates | Bets | Mean policy ROI | Total return | Total net | 所見 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| best | `10` | `17` | `0.2057` | `3.2920` | `-13.7080` | 基準 |
| old fallback | `10` | `24` | `0.1541` | `6.1659` | `-17.8341` | return は多いが net は悪い |
| June-strict rollback | `10` | `17` | `0.2057` | `3.2920` | `-13.7080` | best と完全一致 |

解釈:

- May score source の差は `2024-05-25` と `2024-05-26` の 2 日だけで、realized outcome 差は出ていない。
- 観測された live delta は June policy filtering に集中している。
- June-strict rollback は、この 10 日窓では best の observed behavior をそのまま再現した。

#### B. representative 5-date calendar

比較対象:

- `2024-05-25`, `2024-06-15`, `2024-07-20`, `2024-08-10`, `2024-09-14`

結果:

- June-strict rollback vs best: quantitative outcome は 5 dates すべてで一致。
- June-strict rollback vs old fallback: 差分は `2024-06-15` の 1 bet だけ。
- July / August / September は rollback / fallback / best の routing と outcome がほぼ完全一致している。

解釈:

- June-strict rollback は short window 専用の偶然ではなく、month representative でも best に追従している。

#### C. 24-date tail weekend window (`2024-06-29..2024-09-15`)

比較対象:

- best: `serving_smoke_best_policy_may_tail_weekends.json`
- June-strict rollback: `serving_smoke_fallback_hybrid_june_strict_tail_weekends.json`
- old fallback: `serving_smoke_fallback_hybrid_tail_weekends.json`

結果:

| 比較 | Shared dates | Bets | Mean policy ROI | Total return | Total net | 差分日 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| best | `24` | `82` | `1.0697` | `85.1` | `3.1` | 基準 |
| June-strict rollback | `24` | `82` | `1.0697` | `85.1` | `3.1` | best と完全一致 |
| old fallback | `24` | `85` | `1.0718` | `85.2491` | `0.2491` | `2024-06-29`, `2024-06-30` |

解釈:

- June-strict rollback は June 以降の actual weekend calendar でも best と完全一致した。
- old fallback は June 最終 weekend に 3 extra bets を追加し、わずかな return を拾うが、total net では `+2.8509` units 劣後した。
- したがって June stricter threshold を keep する判断は tail-window でも支持される。

## 4. 中間での総合評価

### 4.1 良い点

1. 研究基盤が end-to-end で閉じた
   - ingest から serving smoke compare まで、同じ repo 内で再現できる。

2. 成果物ベースで比較できる
   - summary JSON, by-date CSV, smoke compare JSON/CSV が揃っており、議論が ad hoc になりにくい。

3. 「良い複雑さ」と「悪い複雑さ」が切り分けられてきた
   - June stricter policy は keep すべき複雑さ。
   - May runtime score override は効果があるが、依然として高コストな複雑さ。

4. 改善の主戦場が見えた
   - raw model retrain だけではなく、regime selection / serving operationalization が主要改善レバーであることが明確になった。

### 4.2 懸念点

1. current best の優位が May runtime score artifact に依存している
   - nested では有効だが、構成は単純ではない。

2. serving calendar は依然 heuristic である
   - valid end month を target-date month へ写像しているため、fold window と runtime window は完全には一致しない。

3. public 市場からの独自情報はまだ薄い
   - `public_pseudo_r2` から `model_pseudo_r2` への改善幅はあるが、支配的とは言えない。

4. データ拡張の潜在力をまだ使い切れていない
   - lineage coverage、course/time normalization、recent-domain 補完は改善したが、まだ粗い部分が残る。

### 4.3 現時点での総合判断

総合的には、「かなり前進しているが、最終形ではない」という評価になる。

- 成績面では、current best が recent nested WF で `1.0073` に達し、ひとつの節目を越えた。
- 運用面では、June-strict rollback が best を実 calendar 上でほぼ完全に代替できる状態に近づいた。
- ただし設計面では、May uplift をより単純な形で再現できていないため、主力 config の複雑性はまだ高い。

したがって本プロジェクトは「精度改善フェーズ」よりも「高成績ルールの単純化と一般化フェーズ」に入ったとみなすのが適切である。

## 5. 今後の目標

### 5.1 直近目標

1. May uplift の単純化
   - `may_runtime_liquidity` に頼らず、default-stack 側または軽量 seasonal proxy 側へ同 uplift を吸収する。

2. rollback 方針の明文化
   - `...regime_hybrid_june_strict_serving.yaml` を probe ではなく正式 rollback target として扱えるか判断する。

3. serving 検証の拡張
   - representative / short window だけでなく、任意期間 actual calendar validation を定例化する。

### 5.2 中期目標

1. regime selection の一般化
   - fold 1 専用のような局所ルールを、より説明可能な seasonal / market / liquidity regime へ置き換える。

2. public benchmark 差分の拡大
   - raw score が市場と高相関なままなので、独自情報の増分を強める。

3. データ基盤の強化
   - lineage / pedigree の coverage 向上、course baseline の細粒度化、time 系の再正規化を進める。

## 6. このあとの改善方針

今後の改善は、以下の原則で進めるのがよい。

1. raw retrain を第一選択にしない
   - 現在の課題は score source と policy の regime design にある。

2. mean ROI ではなく total net を重視する
   - June 最終 weekend のように、tiny extra return より net 悪化のほうが重要な局面がある。

3. best と同じ observed behavior を simpler config で再現できるなら、simpler 側を優先する
   - serving rollback 候補の扱いはこの原則で決める。

4. nested summary と serving config を分断しない
   - summary-driven export と smoke compare を維持し、手書き drift を避ける。

5. 改善案は必ず 2 軸で判定する
   - nested WF での改善
   - actual calendar serving での改善

## 7. 推奨アクション

優先度順では次の 3 本が妥当である。

1. May score override を default-stack 側で代替する軽量 proxy を試す。
2. June-strict hybrid を正式 rollback target に昇格させるかどうかを runbook / architecture で確定する。
3. actual calendar validation を週末窓以外にも広げ、May uplift の runtime 実効範囲をさらに詰める。

## 8. 参照 artifact

- nested best summary
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_modelswitch_f1_policy_may_20240101_20240930_wf_full_nested.json`
- nested fallback summary
  - `artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_20240101_20240930_wf_full_nested.json`
- May/June serving window compare
  - `artifacts/reports/serving_smoke_compare_best_policy_may_window_vs_fallback_hybrid_window.json`
  - `artifacts/reports/serving_smoke_compare_fallback_hybrid_june_strict_window_vs_best_policy_may_window.json`
- tail weekend compare
  - `artifacts/reports/serving_smoke_compare_fallback_hybrid_june_strict_tail_weekends_vs_best_policy_may_tail_weekends.json`
  - `artifacts/reports/serving_smoke_compare_fallback_hybrid_june_strict_tail_weekends_vs_fallback_hybrid_tail_weekends.json`