# プロジェクト概要

## 1. 目的

`nr-learn` は、競馬予想を再現可能な機械学習パイプラインとして運用できる形に整理するプロジェクトである。

狙いは次の 3 点にある。

1. データ取得から予測・検証までを同じリポジトリで再現できるようにする。
2. 精度だけでなく、回収率と運用上の安定性まで含めて評価する。
3. 良い成績を出した設定を、説明しやすく、壊れにくいルールへ整理していく。

## 2. 何を作っているか

このプロジェクトは、次の流れを一貫して扱う。

1. レース結果やオッズなどの元データを取り込む。
2. レース前に利用可能な情報だけで特徴量を作る。
3. 勝率推定モデルと回収率補助モデルを学習する。
4. 時系列を守った検証で、過剰適合を避けながら成績を測る。
5. 購入ルールを適用し、実運用に近い形で回収率と純収益を確認する。

単に「当たるか」ではなく、「どの条件で買うと長期的に意味があるか」を重視している点が特徴である。

## 3. 現在の構成

現在の中核は次の 4 要素でできている。

- データ基盤
  - JRA 主表をベースに、netkeiba などの外部 CSV を追加できる。
- 特徴量生成
  - 過去成績、位置取り、ペース、距離適性、斤量変化などを扱う。
- モデル構成
  - CatBoost による勝率推定を主軸にし、必要に応じて LightGBM の ROI 系モデルを組み合わせる。
- 評価と運用確認
  - nested walk-forward と serving smoke を使い、検証上の成績と実運用に近い挙動を両方見る。

## 4. 現在の運用上の位置づけ

2026-03-30 時点で、現在の位置づけは次の 4 本に整理している。

| 区分 | profile | 位置づけ | 状態 |
| --- | --- | --- | --- |
| operational baseline | `current_recommended_serving_2025_latest` | 現在の既定運用 profile | formal gate `pass / promote` |
| seasonal de-risk | `current_long_horizon_serving_2025_latest` | September 系の損失圧縮を狙う保守候補 | actual-date compare で有効 |
| formal improvement candidate | `current_tighter_policy_search_candidate_2025_latest` | September difficult window 向けの analysis-first 防御候補 | formal gate `pass / promote` |
| recent-heavy retrain candidate | `current_recommended_serving_2025_recent_2018` family | true retrain 比較用の analysis-first 候補 | promotion gate `pass / promote` |

ここで重要なのは、`pass / promote` した候補が増えても、だからといって自動で baseline を差し替えていないことである。`nr-learn` では formal 通過と operational 採用を分けて扱う。また recent-heavy family では、profile 名だけの train ではなく、component を再学習した true retrain run だけを比較根拠に採る。

2026-03-27 時点の実務上の読み方は単純である。既定運用は baseline を維持し、September difficult regime だけを defensive candidate 群の比較対象にする。December tail のような control window では、候補が formal に通っていても baseline 優位を崩さない。

運用境界は次の 2 行で固定してよい。

1. September window は `current_long_horizon_serving_2025_latest` を最初の de-risk alias として参照する。
2. non-September は `current_recommended_serving_2025_latest` を既定運用に保ち、他候補は analysis-first compare に留める。

runtime 面では、`configs/data_2025_latest.yaml` に freshness-guarded primary tail cache が default で入り、current baseline smoke は faster path 上でも same-summary equivalent を確認済みである。したがって直近の queue は runtime ではなく experiment 側へ戻してよい。

experiment queue の current reading も更新された。`tighter policy search` は Rank 1 policy family のままだが、feature 側の primary line だった `class / rest / surface` interaction family は support hardening まで読み切った。`r20260330_class_rest_surface_interactions_v1` は ROI summary が強い一方で support `1/5` で止まり、`r20260330_surface_interaction_only_v1` は support `2/5` まで改善したが ROI を薄めすぎた。そこから middle-ground として作った `r20260330_surface_plus_class_layoff_interactions_v1` は formal benchmark `weighted_roi=1.1379979394080304`、`feasible_folds=3/5` で `pass / promote` に到達した。ただし actual-date compare と bankroll sweep を含む `#46` の読みでは low-frequency / concentrated shape が残り、さらに `#47` の policy-side widening probes も `3.70% -> 0.46% -> 0.00%` と suppressive 方向にしか動かなかった。したがって current queue は serving default 昇格でも widening 継続でもなく、この promoted line の role split を explicit にする段階に移っている。

この role split は次で固定する。

- formal promoted line: `r20260330_surface_plus_class_layoff_interactions_v1`
- operational default line: `current_recommended_serving_2025_latest`
- operational role: analysis-first conservative promoted candidate

したがって next feature experiment queue は、同 family の widening を続けるのではなく、Tier A 次順位の `jockey / trainer / combo` family へ移すのが自然である。

この seasonal 判断の標準文面と read order は `docs/seasonal_derisk_decision_standard.md` を正本にする。

latest 2025 の actual-date compare を再開するときは、次の 3 段で読めば十分である。

1. まず `serving_validation_guide.md` の dashboard summary JSON 一覧から、September は `long_horizon -> tighter policy -> recent-2018`、December は対応する control window を順に開く。
2. compare を回し直す必要があるときだけ、`command_reference.md` の latest 2025 compare コマンド例を同じ順序で使う。
3. 運用判断の根拠を formal support まで掘り下げる必要があるときだけ、promotion gate / evaluation summary へ降りる。

## 5. 現在の主要スコア

現時点で引き継ぎ時に押さえるべき数値は次の 2 系統である。

### 5.1 長期 benchmark の基準線

2024 年データを対象にした nested walk-forward の benchmark ladder は次のとおりである。

| 位置づけ | Weighted ROI | Bets | 意味 |
| --- | ---: | ---: | --- |
| 基準モデル | `0.5788` | `603` | 市場情報を切った基準線 |
| 改善後の高流動性候補 | `0.9346` | `700` | no-bet を大きく減らした転換点 |
| 単純な運用候補 | `0.9915` | `731` | 構成が比較的単純で、ほぼ ROI 1.0 |
| 主力候補 | `1.0073` | `713` | 2024 benchmark 上の上位候補 |

### 5.2 latest 2025 の current snapshot

2025 backfill 済み latest split では、現在の baseline と直近 candidate の formal 結果は次のとおりである。

| profile | 主要 revision | 判断 | weighted ROI / benchmark | bets |
| --- | --- | --- | ---: | ---: |
| `current_recommended_serving_2025_latest` | `r20260325_current_recommended_serving_2025_latest_benchmark_refresh` | `pass / promote` | nested WF `0.7628` | `544` |
| `current_tighter_policy_search_candidate_2025_latest` | `r20260326_tighter_policy_ratio003` | `pass / promote` | formal benchmark `1.1728` | `424` |

補足すると、tighter policy candidate は `r20260326_tighter_policy_ratio003` に加えて、threshold-only revision `r20260327_tighter_policy_ratio003_abs80` も `pass / promote` まで通っている。後者は `min_bets_abs=80` で `formal_benchmark_feasible_fold_count=5/5`、`formal_benchmark_weighted_roi=1.1042` を確認したが、serving default を即時に置き換える決定まではしていない。現時点の baseline は引き続き `current_recommended_serving_2025_latest` である。

actual-date の fresh compare でも役割ははっきりしている。2025-09 の 8 日では baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して tighter policy candidate は `9 bets / -4.3 / 0.8395` で、強い損失圧縮を示した。一方で 2025-12 tail の 8 日では baseline `45 bets / +21.8 / 1.6712` に対して `9 bets / +21.4 / 1.6032` で、profit window の top line は baseline を超えなかった。したがって現時点の位置づけは broad replacement ではなく、September difficult regime 向けの defensive candidate である。

formal support 側の境界も切り分け済みである。既存の threshold sweep では、現行 `0.03/100` は `4/5 feasible folds`、`0.03/80` は `5/5 feasible folds` だった。`80` で新たに通るのは fold 2 だけで、best feasible は既存と同じ kelly family、`98 bets / final_bankroll 0.9199 / max_drawdown 0.1098` だった。したがって `0.03/80` は別の攻め筋へ切り替える話ではなく、同じ defensive family を formal gate 上どこまで許容するかの調整とみなせる。一方で serving policy は変わらないため、operational baseline を切り替える根拠にはならない。

### 5.3 recent-heavy true retrain の current snapshot

recent-heavy split の比較では、`2018-01-01..2024-12-31` と `2020-01-01..2024-12-31` の true retrain run がともに formal に通過している。

| revision | 判断 | 主要値 |
| --- | --- | --- |
| `r20260327_recent_2018_component_retrain` | `pass / promote` | AUC `0.8432`, EV top1 ROI `0.7400`, nested WF `0.9595`, feasible folds `5/5`, bets `767` |
| `r20260327_recent_2020_component_retrain` | `pass / promote` | AUC `0.8449`, EV top1 ROI `0.7496`, nested WF `0.9218`, feasible folds `4/5`, bets `878` |

ただし、これらの run は baseline の代替として即採用するのではなく、recent-heavy learning window の効果を測る analysis-first candidate として扱う。現時点では 2018 start が formal support で優位で、2025-09 の fresh actual-date compare でも baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して `4 bets / -4.0 / 0.8557` と、recent-2020 の `8 bets / -8.0 / 0.7408` より強い de-risk を示した。一方で 2025-12 tail の fresh compare では baseline `45 bets / +21.8 / 1.6712` に対して recent-2018 が `1 bet / -1.0 / 0.9722` と明確に劣後した。したがって現時点の位置づけは broad replacement ではなく、September difficult regime 向けの recent-heavy candidate である。

ここでの current reading は、recent-heavy family を 2 本とも同格に扱わないことである。front-line の recent-heavy candidate は 2018 start とし、2020 start は AUC / EV 系の contrast を残す secondary comparison target に留める。

September difficult regime の 3 候補は、次の順で解釈すると誤読が少ない。

1. `current_long_horizon_serving_2025_latest` は baseline path を最も崩さない seasonal de-risk alias であり、実運用寄りの第一候補である。
2. `current_tighter_policy_search_candidate_2025_latest` は latest 2025 の同一 family 上で exposure を絞る defensive candidate であり、運用変更が比較的単純な第二候補である。
3. recent-2018 true retrain は September 実日付ではより強い de-risk を示すが、学習窓の再構成を伴うため current reading では第三候補の analysis-first fallback として扱う。

## 6. どこまで進んでいるか

現時点では、次の点がすでにできている。

- 学習、評価、予測、バックテストを CLI で再現できる。
- nested walk-forward で、期間分離を守った採用判定ができる。
- evaluation summary から serving ルールを再生成できる。
- representative date だけでなく、実日付の複数日 window でも serving 挙動を比較できる。
- 2025 backfill 済み latest split を使って、formal gate まで一続きで再実行できる。

つまりこのリポジトリは、単なるモデル実験置き場ではなく、研究から運用判断まで一続きで扱える段階に入っている。

## 7. いま未決定のこと

明示的な docs 導線整理は一巡完了しており、いま残っているのは定期点検だけである。

1. current reading が今後の更新でも docs 間で発散しないかを定期的に点検すること。

地方競馬データ拡張については、直近の未決定事項には置いていない。現時点の整理は、JRA latest baseline を詰める仕事とは切り離し、別 universe の将来候補として feasibility だけを持つ、というものである。

進行中の優先順位と最新の実行順は、書き捨ての会話ログではなく [roadmap.md](roadmap.md) を正本として更新する。

## 8. 引き継ぎ時の読み方

- 現在地と次の作業順を知りたい場合は [roadmap.md](roadmap.md) を読む。
- 何を良い結果とみなすかを知りたい場合は [benchmarks.md](benchmarks.md) を読む。
- 日々の進め方や revision の切り方を知りたい場合は [development_flow.md](development_flow.md) を読む。
- 実際にどの CLI をどう叩くかを知りたい場合は [command_reference.md](command_reference.md) を読む。
- 対外向け説明だけが必要な場合は [public_benchmark_snapshot.md](public_benchmark_snapshot.md) を読む。
- システムの中身やモジュール構成を知りたい場合は [system_architecture.md](system_architecture.md) を読む。
