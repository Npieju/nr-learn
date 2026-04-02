# 対外向けベンチマーク概要

## 1. この文書の役割

この文書は、`nr-learn` の現状を対外向けに説明するときに使う benchmark snapshot である。

benchmark 数字と operational role の関係そのものを説明するときは、この文書ではなく [public_benchmark_operational_reading_guide.md](public_benchmark_operational_reading_guide.md) を正本にする。

内部運用の gate 条件、fold ごとの失敗理由、候補の棄却履歴まではここに持ち込まない。そうした内部向け判断材料は [benchmarks.md](benchmarks.md)、[evaluation_guide.md](evaluation_guide.md)、[roadmap.md](roadmap.md) を正本とする。

internal reader が latest 2025 actual-date compare を再開したい場合は、この文書ではなく `serving_validation_guide.md` の dashboard summary JSON 一覧から入り、必要なときだけ `command_reference.md` と formal artifact に降りる。

local-only universe や mixed-universe compare を対外向けに説明するときも、この文書へ直接追記するのではなく、まず `artifacts/reports/local_public_snapshot_<revision>.json` または `artifacts/reports/mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` を別系統で用意する。ここは引き続き JRA latest の public snapshot を正本とする。mixed 側は当面 pointer manifest として扱い、JRA latest の数値 snapshot 自体はここで維持する。

ここでは次だけを簡潔に示す。

1. 現在の基準線
2. 直近で formal に通過した候補
3. 実運用 window で確認済みの挙動

## 2. 更新ルール

- 新しい `revision gate` が `pass / promote` で通ったら、この文書の数値を更新する。
- operational baseline が変わったら、この文書の「現在の採用位置づけ」を更新する。
- 数値の出典は、原則として `artifacts/reports/` の versioned evaluation / promotion / dashboard artifact に固定する。
- 対外向け文書では、内部 candidate の細かな棄却理由や exploratory な仮説は書かない。
- local-only / mixed の public 要約は、JRA latest と同じページへ混在させず universe 別 artifact として分離する。

## 3. 現在の snapshot

更新日: 2026-03-27

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

actual-date の fresh compare でも、この candidate は regime 依存の defensive variant と読むのが妥当だった。2025-09 の 8 日では baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して `9 bets / -4.3 / 0.8395` まで損失を圧縮した一方、2025-12 tail の 8 日では baseline `45 bets / +21.8 / 1.6712` に対して `9 bets / +21.4 / 1.6032` で、利益局面の top line は baseline を超えなかった。

### 3.4 recent-heavy retrain の比較結果

2026-03-27 には、train window を recent 寄りにした true retrain を 2 本 formal に通している。

| revision | train window | decision | AUC | EV top1 ROI | nested WF weighted test ROI | feasible folds |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `r20260327_recent_2018_component_retrain` | `2018-01-01..2024-12-31` | `pass / promote` | `0.8432` | `0.7400` | `0.9595` | `5 / 5` |
| `r20260327_recent_2020_component_retrain` | `2020-01-01..2024-12-31` | `pass / promote` | `0.8449` | `0.7496` | `0.9218` | `4 / 5` |

この結果は、recent regime を強く見た学習でも formal support を維持できることを示す。現時点では `2018` start が formal support 上位で、`2020` start は AUC / EV 系でやや優位だった。ただし、どちらも運用上の baseline を直ちに差し替えるのではなく、learning window 改善候補として保持している。

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

### 4.3 recent-heavy retrain も September では de-risk を示した

recent-heavy true retrain のうち、`2018-01-01..2024-12-31` window を使った candidate も 2025-09 の fresh compare で strong de-risk を示した。

| 指標 | Latest baseline | Recent-2018 true retrain |
| --- | ---: | ---: |
| policy bets | `32` | `4` |
| total policy net | `-27.3` | `-4.0` |
| pure path final bankroll | `0.2959` | `0.8557` |

同じ September fresh compare を recent-2020 true retrain にも広げると、`8 bets / -8.0 / 0.7408` で recent-2018 より一段弱かった。したがって recent-heavy family 内の actual-date 上位候補は 2018 start である。

このため対外向け snapshot でも、recent-heavy family の主参照は 2018 start に置く。2020 start は formal に通過しているが、current reading では補助比較用の候補であり、front-line candidate ではない。

ただし 2025-12 tail の fresh compare では、baseline `45 bets / +21.8 / 1.6712` に対して recent-2018 true retrain は `1 bet / -1.0 / 0.9722` だった。つまり recent-heavy candidate は broad replacement ではなく、September difficult window 向けの regime-specific candidate として扱うのが妥当である。

### 4.4 tighter policy candidate も September では defensive に機能した

`current_tighter_policy_search_candidate_2025_latest` についても fresh compare を行い、September difficult window では defensive variant として機能することを確認した。

| 指標 | Latest baseline | Tighter policy candidate |
| --- | ---: | ---: |
| 2025-09 policy bets | `32` | `9` |
| 2025-09 total policy net | `-27.3` | `-4.3` |
| 2025-09 pure path final bankroll | `0.2959` | `0.8395` |
| 2025-12 tail policy bets | `45` | `9` |
| 2025-12 tail total policy net | `+21.8` | `+21.4` |
| 2025-12 tail pure path final bankroll | `1.6712` | `1.6032` |

このため、tighter policy candidate は support 改善を示した formal candidate であると同時に、actual-date では September difficult window 向けの defensive option と位置づけるのが適切である。現時点では operational baseline の置換候補ではない。

対外向けの読み方としては、September difficult regime では defensive option を参照し、December tail のような control window では baseline を維持する、という 2 段だけを押さえれば十分である。

## 5. 現在の採用位置づけ

- 現在の operational baseline: `current_recommended_serving_2025_latest`
- 現在の seasonal de-risk variant: `current_long_horizon_serving_2025_latest`
- 現在の formal improvement candidate: `current_tighter_policy_search_candidate_2025_latest`
- 現在の recent-heavy defensive candidate: `current_recommended_serving_2025_recent_2018` true retrain

September difficult regime と December tail control window の読み分けは次のとおりである。

| regime | 既定の読み | defensive option |
| --- | --- | --- |
| September difficult window | baseline を基準にしつつ de-risk variant の差分を参照する | `current_long_horizon_serving_2025_latest`、`current_tighter_policy_search_candidate_2025_latest`、recent-2018 true retrain |
| December tail control window | baseline 優位を維持するかを確認する | defensive option は broad replacement の根拠に使わない |

対外向けの要約は次の 3 行で十分である。

- 既定運用は `current_recommended_serving_2025_latest` のまま維持する。
- September difficult window では `current_long_horizon_serving_2025_latest` を先頭に defensive option を参照する。
- `current_tighter_policy_search_candidate_2025_latest` と recent-2018 true retrain は formal に通過しているが、どちらも broad replacement ではない。

internal handoff では、この 3 行に `docs/seasonal_derisk_decision_standard.md` の標準結論を対応づければ十分である。

internal handoff 向けには、この 3 行を起点にして `serving_validation_guide.md` の quickstart へ戻れば十分である。

補助的な要点だけを残すと、現状は次のとおりである。

- 長期 benchmark では、基準線 `0.5788` から主力候補 `1.0073` まで改善が進んでいる。
- latest 2025 holdout でも baseline は formal gate を通過済みである。
- 直近では support 改善型の candidate も formal に通過し、latest regime に対する改善余地が確認できた。
- recent-heavy true retrain でも formal 通過は確認できており、特に `2018` start は `5 / 5` feasible folds で strong-support 側の結果を出している。
- actual-date compare でも、recent-2018 true retrain は September difficult window で baseline より損失を大きく抑え、recent-2020 true retrain よりも強く出た。
- actual-date compare でも、tighter policy candidate は September difficult window で baseline より損失を大きく抑えたが、December tail では baseline 優位を維持した。
- September difficult window の参照順は、まず seasonal de-risk alias、次に tighter policy candidate、最後に recent-2018 true retrain の analysis-first fallback とする。
- 実運用上は、September difficult window だけ defensive option を参照し、December tail のような control window では baseline を維持する。
- 一方で December tail では baseline が recent-heavy true retrain を大きく上回っており、recent-heavy は broad replacement ではない。
- 実運用に近い actual-date compare では、September のような難しい時期にだけ defensive override が意味を持つことを確認している。

## 6. 出典 artifact

- baseline formal result
	- `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- tighter policy formal result
	- `artifacts/reports/promotion_gate_r20260326_tighter_policy_ratio003.json`
	- `artifacts/reports/promotion_gate_r20260327_tighter_policy_ratio003_abs80.json`
- recent-heavy 2018 formal result
	- `artifacts/reports/promotion_gate_r20260327_recent_2018_component_retrain.json`
- September / December control compare
	- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_sep_full_month_2025_latest_profile.json`
	- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh_vs_current_tighter_policy_search_candidate_2025_latest_sep_full_month_2025_latest_vs_tighter_policy_candidate_fresh.json`
	- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh_vs_current_recommended_serving_2025_recent_2018_sep_full_month_2025_latest_vs_recent2018_true_retrain_fresh.json`
	- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_profile_vs_current_long_horizon_serving_2025_latest_dec_tail_2025_latest_profile.json`
	- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_vs_tighter_policy_candidate_fresh_vs_current_tighter_policy_search_candidate_2025_latest_dec_tail_2025_latest_vs_tighter_policy_candidate_fresh.json`
	- `artifacts/reports/dashboard/serving_compare_dashboard_current_recommended_serving_2025_latest_dec_tail_2025_latest_vs_recent2018_true_retrain_fresh_vs_current_recommended_serving_2025_recent_2018_dec_tail_2025_latest_vs_recent2018_true_retrain_fresh.json`
