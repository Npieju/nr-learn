# ROI最大化の基礎設計（実装ガイド）

最終更新: 2026-03-15

## 1. 目的
- 目的は「単発で高いROIを出すこと」ではなく、**時系列外で再現可能なROIを継続的に改善すること**。
- そのために、以下の順序を固定する。
  1. 目的関数の整合
  2. 検証設計の整合
  3. 特徴量定義の整合
  4. ハイパーパラメータ最適化（最後）
- 2026-03-07 以降の長期運用設計は [architecture_long_term.md](architecture_long_term.md) を正本とする。
- 外部 benchmark と target 値は [external_benchmark_targets.md](external_benchmark_targets.md) を正本とする。

## 2. 設計原則（幹）
1. 学習ターゲットと評価KPIは一致させる。
2. 閾値探索・戦略探索と最終評価を同一期間で行わない。
3. 特徴量名と実装意味を一致させる（例: last_3 は真のrolling 3件）。
4. ROIだけでなくベット数・ドローダウンを同時管理する。
5. 変更採用は「Out-of-Sample」でのみ判断する。

## 3. 推奨アーキテクチャ（3層）
### Layer 1: 勝率推定
- 目的: `p_true = P(win | x)` を高精度・高キャリブレーションで推定。
- 学習指標: LogLoss / Brier。
- 備考: 直接ROI回帰より分散が小さく、安定しやすい。

### Layer 2: 市場乖離推定（alpha）
- 市場確率 `p_market` をオッズから算出。
- 乖離指標の例:
  - `edge = p_true * odds - 1`
  - `delta_logit = logit(p_true) - logit(p_market)`
- 目的: 市場に対してどこに優位があるかを明示化。

### Layer 3: ベット最適化
- 目的関数: 期待対数成長（fractional Kelly）を基本に、
  - 最大賭け率
  - 最小確率
  - オッズ帯
  - ドローダウン制約
  を掛ける。
- 出力: 賭ける/賭けない + 賭け比率。

## 4. KPI体系（採用基準）
### 4.1 予測品質KPI
- LogLoss
- Brier
- Calibration Error（ECE相当）
- Benter pseudo-R²
- `ΔR² = R²_combined - R²_public`

### 4.2 投資品質KPI
- Out-of-Sample ROI
- Bet数（流動性）
- Max Drawdown
- 破綻回避（bankroll終値）

### 4.3 採用判定ルール
- ROIが高くても、Bet数が閾値未満なら採用しない。
- ROI改善があっても、DD悪化が大きい場合は採用しない。

## 5. 検証設計（必須）
- 時系列Walk-Forwardで、以下を分離:
  - 内側: 戦略パラメータ探索
  - 外側: 最終評価
- 同一期間で最適化と評価を兼ねない。
- モデル比較は同一split・同一データ範囲で実施。

## 6. 現在コードでの優先改修項目
1. [x] `features/builder.py` の履歴特徴を真のrollingへ変更。（2026-02-17対応）
2. [x] `run_evaluate.py` のWFを「内側最適化 / 外側評価」に明示分離。（nested `wf-scheme` 追加）
3. [x] ROI回帰系と確率推定系の評価フローを分岐整理（指標混在を避ける）。（2026-02-17対応）
4. [x] `horse_name` 依存の履歴キーを見直し、再同定ルールを明確化。（2026-02-17対応）
5. [x] Layer2の学習ターゲット経路を追加（`task: market_deviation`）。（2026-02-17対応）

## 7. 実装時のチェックリスト
- [x] 目的関数とKPIの整合を確認した
- [x] 内側最適化と外側評価を分離した
- [x] 変更前後で OOS ROI / Bet数 / DD を比較した
- [x] 特徴量リークの可能性を点検した（`leakage_audit` を train/evaluate レポートに追加）
- [x] 実験条件（config, rows, period）を記録した（`run_context` を train/evaluate レポートに追加）

## 8. 当面の実装順序
1. [x] 検証設計の分離（nested time-series WF）
2. [x] rolling特徴の意味修正
3. [x] 市場乖離層の明示化（評価メトリクスとして導入）
4. [x] 市場乖離層の学習ターゲット化（`market_deviation` タスク）
5. [x] ベット最適化の制約強化（DD上限）
6. [ ] その後にチューニング

## 9. 2026-03-12 時点の現状診断
- 現行 mainline の public-free benchmark は `configs/model_catboost_fundamental_enriched.yaml` + `configs/features_catboost_fundamental_enriched.yaml` で、200k tail の評価期間は `2017-06-10..2021-07-31`。
- この mainline は `top1_roi=0.7715`、`ev_top1_roi=0.4264` にとどまり、長期運用の前提である `ROI > 1` からは遠い。
- 学習目的は完全な ROI 直結ではない。分類系 CatBoost は `Logloss` / `AUC` を最適化し、ROI は evaluation layer で後段評価している。つまり現状は「ROI-first learning」ではなく「勝率推定 + ROI policy」の二段構えである。
- さらに current mainline の evaluate 実行は `wf_mode: off` で、日次サマリの `top1_roi` / `ev_top1_roi` も固定 stake の単勝1点買いが中心である。walk-forward の strategy optimization は実装済みだが、mainline の採用判定にはまだ十分使われていない。
- enriched mainline の lineages は現状ボトルネックになっている可能性が高い。`catboost_fundamental_enriched_no_lineage_win` では 200k tail で `top1_roi=0.7812`、`ev_top1_roi=0.4597` まで改善し、lineage 無効化の方が安定している。
- 実際に 100k tail の gap report では `breeder_name` / `sire_name` / `dam_name` / `damsire_name` の raw coverage が `0.12348` しかなく、force include している `breeder_last_50_win_rate`、`sire_last_100_win_rate`、`sire_last_100_avg_rank`、`damsire_last_100_win_rate`、`sire_track_distance_last_80_win_rate` がそのまま low coverage になっている。特徴量を入れていること自体より、coverage の薄い lineage を半端に混ぜていることの方が悪さをしている。
- 一方で rich + market-aware stack 側では一部の ROI>1 シグナルがある。`catboost_value_stack` は 100k tail (`2019-06-30..2021-07-31`) で `benter_kelly_roi=1.1713`、`final_bankroll=1.1088` を出しており、`catboost_value_stack_time_gpu` は 2021 年の weighted `ev_top1_roi=1.0679` を示している。
- ただしこの stack 系は `market_prob_corr=0.984..0.995` と市場に非常に近く、`benter_delta_pseudo_r2` はゼロ近傍か負である。つまり public を超える独自情報はまだ弱く、「market を強く混ぜたことで recent slice の賭け方が少し改善している」段階にとどまる。
- 時系列でも偏りがある。public-free enriched mainline は 2017-2021 の全年度で `top1_roi < 1`、特に 2020 年が弱い。逆に market-aware stack は 2021 年の EV 指標だけが `> 1` に届いており、最近年に寄せた slice では改善余地が見える。
- 2024 Q1 (`2024-01-01..2024-03-31`) の recent slice を 588 races / 8,147 rows まで拡張しても、public-free raw はまだ `ROI < 1` だった。しかも raw `top1_roi` は enriched `0.6580` より no-lineage `0.7185` の方が高く、最新年帯でも中途半端な lineage が drag になっていることを裏付けた。
- 同じ recent slice では `catboost_value_stack` が `top1_roi=0.8024`, `auc=0.8479` で依然として最強だったが、それでも raw top1 ROI は 1 に届かない。一方で `benter_ev_top1_roi=1.1714` と policy layer では利益シグナルが出ており、主ボトルネックが「最新年データ不足そのもの」から「raw model と賭け方のズレ」へ寄っていることが確認できた。
- ただし recent Q1 は 18 開催日しかなく、`wf_mode=full` でも `wf_scheme=nested` は `insufficient_data_for_nested_folds` で成立しなかった。したがって四半期 slice は nested walk-forward の正式採用判定にはまだ短い。
- `wf_scheme=single` で見た operational contrast は raw 指標と異なった。no-lineage は Kelly 最適化で `wf_test_roi=1.0056`、`wf_test_bets=132`、`wf_test_max_drawdown=0.2417` まで届いた一方、value stack は portfolio 戦略で `wf_test_roi=0.7000`、`wf_test_bets=2`、`wf_test_max_drawdown=1.0` と極端に薄い賭けしか残らなかった。つまり recent 短期運用では raw の強さだけで mainline を決めると危ない。
- 2024 H1 (`2024-01-01..2024-06-30`) まで recent append を伸ばすと 1,476 races / 20,460 rows / 44 dates となり、`wf_mode=full` + `wf_scheme=nested` が初めて 5 outer folds で成立した。四半期では足りなかったが、半期なら nested policy selection の診断に入れる。
- その H1 fixed nested WF では no-lineage が「一応は賭けられる」側だった。raw は `top1_roi=0.6921`, `ev_top1_roi=0.7207`, `auc=0.7068` と弱いままだが、nested outer test の weighted ROI は `0.6690`、総 bets は `558` で、5 folds 中 4 folds は実際に戦略を選べた。ただし `ROI > 1` には遠く、現時点では deployable ではない。
- 同じ H1 fixed nested WF では value stack の raw 優位がそのまま運用優位には変換されなかった。raw は `top1_roi=0.7769`, `auc=0.8453` と最強だが、constraint 付き nested search では 5 folds 全てが `strategy_kind=no_bet` となり、weighted test ROI は `null`、総 bets も `0` だった。つまり recent H1 では「勝ちそうに見えるモデル」と「実際に打てる policy」が完全には一致していない。
- この H1 再評価の途中で、`gate_then_roi` の探索が feasible 候補ゼロでも raw ROI の fallback を採用してしまう実装バグを修正した。現在は feasible 候補が無い fold を `strategy_kind=no_bet` / `selection_reason=no_feasible_candidate` として明示記録するため、walk-forward artifact をそのまま運用判断に使える。
- さらに 2024 Q3 (`2024-07-01..2024-09-30`) を backfill して 2024 YTD window を 2,291 races / 31,224 rows / 71 dates まで伸ばしても、no-lineage は改善しなかった。raw は `top1_roi=0.6885`, `ev_top1_roi=0.5984`, `auc=0.7142`、nested outer test の weighted ROI は `0.5576`、総 bets は `581` で、H1 より悪化している。つまり recent rows の追加だけでは public-free 側の deployability は押し上がらない。
- 同じ 2024 YTD window でも value stack の raw 優位は続く。`top1_roi=0.7863`, `auc=0.8424` と raw 指標は no-lineage を上回るが、fixed nested WF では 5 folds 全てが再び `strategy_kind=no_bet` だった。H1 の no-bet 判定はノイズではなく、Q3 を足しても変わらない構造的問題と見てよい。
- Q3 backfill 自体は 815 race IDs を zero-failure で完走し、`race_result` / `race_card` は 3,849 races で完全整合した。一方で latest-tail lineage coverage は `0.0714` まで低下しており、recent 帯では lineage 系を mainline に戻せる状態からさらに遠ざかっている。
- 2026-03-13 に feature builder へ first wave の recent-domain 特徴を追加した。具体的には `休み明け間隔`, `馬体重増減`, `距離増減`, `jockey×trainer combo history`, `course×gate bias`, `pace fit` を builder へ実装し、current CatBoost configs でも選択される状態にした。
- その後、netkeiba parser / canonical mapping を改修し、recent append (`2024-01-01..2024-09-30`) の canonical metadata を再生成した。現在は `競争条件=99.23%`, `芝・ダート区分=99.23%`, `右左回り・直線区分=96.43%`, `斤量=99.23%`, `weight=99.80%`, `東西・外国・地方区分=99.23%` まで回復しており、`race_class_score=99.23%`, `horse_class_change=72.02%`, `horse_surface_switch=72.02%`, `horse_carried_weight_change=72.02%` が recent window でも実際に立つようになった。残課題は `芝・ダート区分2` が未充足、`内・外・襷区分` が `10.85%` と疎である点である。
- この修正後データで 2024 YTD (`2024-01-01..2024-09-30`) を再評価すると、no-lineage は raw `top1_roi=0.7026`, `ev_top1_roi=0.5452`, `auc=0.7184`、fixed nested WF の weighted ROI は `0.5788`, 総 bets は `603` だった。旧 `0.5576` からは改善したが、依然として `ROI > 1` には届かない。
- 同じ修正後データで value stack を再評価すると、raw は `top1_roi=0.7854`, `ev_top1_roi=0.5126`, `auc=0.8427` で依然最強だったうえ、fixed nested WF でも全 fold `no_bet` ではなくなった。outer 5 folds のうち 2 folds は `kelly`, 3 folds は `no_bet` となり、weighted ROI は `0.7672`, 総 bets は `329` まで回復した。まだ deployable ではないが、「構造的に打てない」状態からは脱している。
- ただし、ここで「recent-domain 特徴を全 component に伝播させればさらに伸びる」という仮説は外れた。`configs/features_catboost_rich.yaml` の現行 117 features で win / alpha / roi component を再学習し、stack bundle を再構築すると、2024 YTD の value stack は raw `top1_roi=0.7751`, `ev_top1_roi=0.3804`, `auc=0.8421`、fixed nested WF の weighted ROI `0.4671`, 総 bets `153` まで悪化した。outer 5 folds のうち `kelly` は 1 fold だけで、残り 4 folds は `no_bet` に戻っている。
- same-window の raw tuning 診断では、best candidate は `alpha_weight=0`, `roi_weight=0.225`, `market_blend_weight=0.95` だった。実際に alpha を外した ROI-only diagnostic stack でも raw `top1_roi=0.7880`, `auc=0.8418`, fixed nested WF の weighted ROI `0.5856`, 総 bets `157` までは戻ったが、parser-fix 後の旧 baseline (`0.7672 / 329 bets`) は超えられていない。つまり post-retrain の主因は alpha 単独ではなく、bulk feature propagation / calibration 側にもある。
- LightGBM 混成も診断した。現行 117-feature で `LightGBM ROI` component を CPU で再学習し、`CatBoost win + LightGBM ROI` の hybrid stack を組むと、2024 YTD recent nested WF は raw `top1_roi=0.7843`, `ev_top1_roi=0.3746`, `auc=0.8418`, weighted ROI `0.6953`, 総 bets `142` だった。full CatBoost retrain (`0.4671 / 153 bets`) や CatBoost ROI-only (`0.5856 / 157 bets`) よりは明確に良いが、parser-fix 直後の旧 baseline (`0.7672 / 329 bets`) にはまだ届かなかった。
- その hybrid に narrow-grid tuning を掛けると、raw top1-optimal と raw EV-optimal が分岐した。top1-optimal (`roi_weight=0.1`, `roi_scale=3.0`, `market_blend_weight=0.97`) は raw `top1_roi=0.7894`, `ev_top1_roi=0.2926`, `auc=0.8417`, nested WF weighted ROI `0.7710`, 総 bets `232` となり、weighted ROI だけ見れば parser-fix 直後 baseline (`0.7672 / 329 bets`) を僅差で上回った。一方で EV-optimal (`roi_weight=0.4`, `roi_scale=1.0`, `market_blend_weight=0.97`) は raw `top1_roi=0.7879`, `ev_top1_roi=0.4024`, nested WF weighted ROI `0.7417`, 総 bets `225` で、raw EV 改善がそのまま operational ROI には繋がらなかった。
- `LightGBM alpha` を小さく足す first probe も実施した。`CatBoost win + LightGBM alpha + LightGBM ROI` を `alpha_weight=0.05`, `roi_weight=0.1`, `roi_scale=3.0`, `market_blend_weight=0.97` で評価すると、raw `top1_roi=0.7861`, `ev_top1_roi=0.2986`, `auc=0.8417`, nested WF weighted ROI `0.6764`, 総 bets `144` まで悪化した。したがって少なくとも現行 rich 117-feature 条件では、alpha を戻すより high-coverage subset を先に詰めるべきである。
- したがって現時点の recent YTD では、「post-retrain 系の中で最良」は `CatBoost win + LightGBM ROI` の top1-tuned hybrid である。ただし parser-fix 直後 baseline より bets は `329 -> 232` へ減っているため、mainline 交代を判断するなら weighted ROI 単独ではなく流動性との交換条件で見る必要がある。
- その後、探索の無駄を減らすために walk-forward strategy search に config-driven な `policy_search` budget override を追加した。これにより diagnostic 実験では 720 trial を総当たりせず、近傍の候補だけを評価できるようになった。
- 同じ narrowed budget で high-coverage subset (`103 features`) の `CatBoost win + LightGBM ROI` hybrid を再評価すると、raw `top1_roi=0.7941`, `ev_top1_roi=0.4303`, `auc=0.8423` は維持されたが、その後 `portfolio` の bankroll 正規化を直して nested WF を引き直すと weighted ROI は `0.8148`, 総 bets は `407` になった。しかも low-coverage force-include はゼロになっている。
- 対照的に、同じ narrowed budget で旧 `top1-tuned` 117-feature hybrid を再評価すると 5/5 folds が `no_bet` になり、nested WF weighted ROI は `null`, 総 bets `0` だった。したがって今回の改善は search budget の偶然ではなく、sparse feature を削ったことで feasible betting surface を回復できた可能性が高い。
- そのうえで subset hybrid の local tuning も 3 本だけ試し、policy fix 後に最有力だけ再評価した。`roi_weight=0.12` 単独は raw `top1_roi=0.7941`, `auc=0.8423`, nested WF weighted ROI `0.8152`, 総 bets `409` で、元の subset 設定 `0.8148 / 407 bets` を僅差で上回った。一方で旧 117-feature hybrid は corrected 後も `all no_bet` のままだった。
- さらに `scripts/run_wf_feasibility_diag.py` で corrected subset baseline の nested WF candidate gate を分解すると、fix 前に `max_drawdown` で誤って弾かれていた fold 1 (`valid=2024-04-20..2024-05-19`) の high-ROI portfolio が feasible に戻り、4 候補が通るようになった。残る fold 4 (`valid=2024-07-06..2024-08-04`) と fold 5 (`valid=2024-08-03..2024-09-01`) は 32/32 候補が `min_bets` を含んで不採用で、closest candidate でも bet 数不足が主因だった。
- ただし、その結論の次に 1 本だけ narrow liquidity 仮説も検証した。`configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity.yaml` では同じ `roi_weight=0.12` subset モデルのまま、`min_probabilities=[0.03, 0.05]`, `min_edges=[0.0, 0.01]`, `min_expected_values=[0.95, 1.0]`, `odds_max=25` に限定して nested search を緩めた。
- その結果、recent YTD nested WF は `wf_weighted_roi=0.9346`, 総 bets `700` まで改善し、fold 構成は `portfolio, kelly, kelly, portfolio, portfolio` になった。fold 4 は valid `152 bets` / test `180 bets` / test ROI `1.1972`、fold 5 は valid `141 bets` / test `95 bets` / test ROI `0.9147` で、late-summer の `no_bet` は解消した。
- その後、fold 4-5 だけを `scripts/run_wf_liquidity_probe.py` で再点検すると、late-summer の top feasible 帯は `blend_weight=0.8` と `min_expected_value=0.95` 近傍に集まっていた。そこで `blend_weight=[0.8]`, `min_probabilities=[0.04,0.05]`, `min_expected_values=[0.95,0.98]` に寄せた guardrail variant (`model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_guardrail.yaml`) も full nested で回したが、結果は `0.9047 / 703 bets` に低下した。
- guardrail では fold 1 test ROI が `0.6184` まで少し改善した一方、fold 4 が `valid_bets=107` / `test_bets=138` に縮み、late-summer 改善の重みを保てなかった。したがって late-summer 帯の問題は確かに threshold 起因でもあるが、現時点では `min_expected_value=0.95` を含む current liquidity variant のほうが実運用上はまだ強い。
- 次に立てる仮説は score shaping 一択ではなく、late-summer を壊さずに fold 1 drag をどう切り分けるかである。
- データ鮮度は依然制約だが、内容は更新された。canonical split はまだ `2021-07-31` 止まりの一方、`race_list` / cached HTML 経路で recent append は `2024-09-30` まで再構成でき、`run_evaluate.py --start-date/--end-date` で最新年 slice を切り出して比較できる段階に入った。

## 10. 次の優先順位
1. 次の主戦場は「recent-domain 特徴を全部載せること」ではない。現行 117-feature retrain は value stack の nested WF を `0.7672` から `0.4671` へ悪化させたため、次は high-coverage feature subset / feature gate を設計して、stack component へ段階的に入れるべきである。
2. alpha component は現時点では optional 扱いに下げる。same-window 診断では best raw candidate が `alpha_weight=0` で、ROI-only diagnostic stack も full stack retrain よりは明確に良かった。つまり「alpha を足せば強くなる」前提は外す必要がある。
3. LightGBM 経路は捨てる段階ではなく、むしろ subset 化と相性が良い。`CatBoost win + LightGBM ROI` の high-coverage subset hybrid は policy fix 後の corrected recent YTD nested WF で `0.8148 / 407 bets`、`roi_weight=0.12` 版で `0.8152 / 409 bets` まで来ていたが、その同じ subset モデルに narrow liquidity search を掛けると `0.9346 / 700 bets` に伸びた。一方、同 budget の旧 117-feature hybrid は `all no_bet` のままだった。したがって当面の main candidate は subset+liquidity 側で、weight 探索は `roi_weight=0.12` を固定してよい。
4. public-free benchmark では lineage を default で無条件採用しない。latest-tail coverage は `0.0714` まで落ちており、`no_lineage` を暫定 mainline 候補として維持しつつ、将来的には coverage 条件付き lineage gate に切り替える。
5. mainline 比較では `wf_mode off` の固定 stake だけでなく、walk-forward の ROI strategy を正式採用候補に含める。特に現時点の value stack は「打てる fold」と `no_bet` fold が混在しているため、raw 指標より fold 単位の実運用性を優先して見る。
6. ハイパーパラメータ調整は最後でよい、という原則自体は維持するが、今後も broad search はしない。subset 固定の `market_blend_weight` / `roi_weight` 近傍は corrected policy 下でも確認済みで、model-level 暫定 best は `roi_weight=0.12` のままでよい。そのうえで current operational best は narrow liquidity search を加えた `0.9346 / 700 bets` であり、guardrail check (`0.9047 / 703 bets`) では更新できなかった。次に動くなら weight 調整ではなく、late-summer 改善を保ったまま fold 1 drag を分離できる別仮説が必要である。
