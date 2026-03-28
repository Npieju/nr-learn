# 開発ロードマップ

## 1. この文書の役割

この文書は、`nr-learn` の中期的な開発計画を管理する正本である。

ここには次を残す。

1. いま何が完了しているか
2. いま何が判断済みか
3. 次にどの順番で進めるか
4. 進めるたびに何を更新するか

この文書は書き捨てのメモではない。方針や優先順位が変わったら、その都度この文書を更新する。

## 2. 更新ルール

- 新しい大きな判断をしたら、この文書の `現在地`、`実行中の優先事項`、`次の候補` を更新する。
- formal な評価や revision gate を通したら、結果を `完了済みマイルストーン` と `判断ログ` に反映する。
- 新しい作業を始める前に、この文書の優先順位と矛盾しないか確認する。
- 一時的な調査メモや試行錯誤の詳細は artifact と個別ガイドに残し、この文書には結論だけを残す。
- この文書の更新を伴わない大きな方向転換はしない。

## 2.1 データ方針の前提

- 本番 2026 に近い判断を優先するため、学習・検証では直近年帯を重く見る。
- ただしデータは有限なので、直近寄せは date-based な学習 window 制限として設計し、単純に履歴を捨てるかどうかだけで決めない。
- 古いデータは無条件に残すのではなく、recent regime を悪化させない範囲で support と特徴量安定化に寄与するかで判断する。
- 新しいデータ源を増やすときも、まず JRA latest の formal benchmark と actual-date compare を壊さないことを優先する。

## 3. 現在地

2026-03-27 時点の到達点は次のとおりである。

- netkeiba 2025 backfill は完了している。
- `configs/data_2025_latest.yaml` により、2025 を validation 末尾に含む latest split が使える。
- latest coverage snapshot は `benchmark_rerun_ready=true`。
- `_2025_latest` profile で train / evaluate / predict / backtest を呼べる。
- latest の formal 評価では `current_best_eval_2025_latest` と `current_recommended_serving_2025_latest` のトップラインが実質同一だった。
- latest の正式判断では、`current_recommended_serving_2025_latest` が matching WF を含めて `pass / promote` に到達した。
- `current_recommended_serving_2025_latest` の actual-date 運用確認として、`2025-12-28` の predict / serving smoke は正常完了した。
- その後 `2025-12` 末尾 8 日 window の serving smoke も完了し、`2025-12-06`、`2025-12-20`、`2025-12-27` では `policy_bets=1` を確認した。
- `r20260325_latest_serving_candidate_v2` の latest revision gate wrapper 実行は完了し、wrapper manifest / revision manifest / promotion report はすべて `pass / promote` で整合している。
- latest baseline の predict / serving smoke 実行例と artifact suffix 命名ルールは docs に反映済みである。
- `current_long_horizon_serving_2025_latest` の actual-date 再検証も完了し、`2025-09` 実日付では baseline に対する de-risk が確認され、`2025-12` tail では baseline と完全一致だった。
- `current_tighter_policy_search_candidate_2025_latest` は、evaluation policy の `min_bet_ratio=0.03` / `min_bets_abs=100` で formal revision gate を通過した。
- `r20260326_tighter_policy_ratio003` では revision manifest / promotion report がともに `pass / promote` で整合し、`formal_benchmark_weighted_roi=1.172842678442708`、`formal_benchmark_feasible_fold_count=4` を確認した。
- threshold-only revision として `r20260327_tighter_policy_ratio003_abs80` も完了し、既存 train artifact を再利用したまま matching WF feasibility と promotion gate まで `pass / promote` で整合した。
- 同 run では `formal_benchmark_weighted_roi=1.1042287961989103`、`formal_benchmark_feasible_fold_count=5`、`formal_benchmark_bets_total=598` を確認した。
- recent-heavy split の比較では、`value_blend` profile を profile 単位で train するだけでは component 再学習にならず、真の比較にはならないことを確認した。
- その root issue を解消した上で、`r20260327_recent_2020_component_retrain` として recent-2020 の win / roi component を実際に再学習し、matching な stack bundle、evaluation、WF feasibility、promotion gate まで完了した。
- 同 run は `pass / promote` で整合し、`AUC=0.8449`、`ev_top1_roi=0.7496`、`nested WF weighted test ROI=0.9218`、`formal_benchmark_feasible_fold_count=4` を確認した。
- さらに `r20260327_recent_2018_component_retrain` として recent-2018 の true retrain も完了し、`AUC=0.8432`、`ev_top1_roi=0.7400`、`nested WF weighted test ROI=0.9595`、`formal_benchmark_feasible_fold_count=5` を確認した。

現時点の operational baseline は `current_recommended_serving_2025_latest` とする。

一方、`current_tighter_policy_search_candidate_2025_latest` は latest 2025 regime の formal-support 改善を確認した analysis-first candidate として保持する。actual-date の fresh compare では、2025-09 の difficult window で baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して `9 bets / -4.3 / 0.8395` と強い損失圧縮を示した一方、2025-12 tail では baseline `45 bets / +21.8 / 1.6712` に対して `9 bets / +21.4 / 1.6032` と profit window の top line では届かなかった。したがって現時点では serving default へは昇格させず、September difficult regime 向けの analysis-first defensive candidate として扱う。

また、`r20260327_recent_2020_component_retrain` は recent-heavy learning window の真の再学習比較として formal に通過したが、weighted nested-WF ROI と bets total は直近の pseudo-retrain run を下回った。したがってこちらも、現時点では baseline 置換ではなく analysis-first candidate として扱う。

同様に、`r20260327_recent_2018_component_retrain` も formal に通過しており、recent-heavy family の中では 2020 start より support が強い。次の判断軸は、2018 start / 2020 start / latest baseline を actual-date compare でどう切り分けるかである。

すでに actual-date compare では、2025-09 の fresh compare で recent-2018 true retrain が baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して `4 bets / total net -4.0 / pure bankroll 0.8557`、recent-2020 true retrain が `8 bets / -8.0 / 0.7408` を示した。したがって recent-heavy family 内の actual-date 優先窓も 2018 start と読める。一方で 2025-12 tail の fresh compare では recent-2018 true retrain が baseline `45 bets / +21.8 / 1.6712` に対して `1 bet / -1.0 / 0.9722` と明確に劣後したため、現時点の位置づけは broad baseline replacement ではなく September difficult regime 向けの analysis-first de-risk candidate である。

## 4. 判断ログ

### 4.1 latest データの readiness

- root issue はデータ不足ではなく、backfill 後データを formal benchmark に載せることだった。
- latest snapshot では pedigree / owner / breeder を含む主要補助列が tail で `non_null_ratio=1.0`。
- readiness は `rerun_enriched_benchmark` で、データ再収集ではなく benchmark 再実行フェーズへ移行済み。

### 4.2 latest baseline の選定

- `current_best_eval_2025_latest` と `current_recommended_serving_2025_latest` は 120k / fast / nested で同一トップラインだった。
- 差は `score_source_count` にあり、前者が `2`、後者が `1`。
- したがって latest 環境でも、より単純な `current_recommended_serving_2025_latest` を baseline に採る。

### 4.3 latest formal gate の扱い

- 初回の revision gate block は model quality ではなく matching `wf_feasibility_diag` 不足が原因だった。
- full rows の WF feasibility は OOM kill になったため、`pre-feature-max-rows=300000` の matching WF summary を追加生成した。
- その summary は `representative` かつ `feasible_fold_count=5` を満たし、promotion gate は `pass / promote` に到達した。

### 4.4 latest operational smoke の初回確認

- `current_recommended_serving_2025_latest` で `2025-12-28` の predict は正常完了した。
- 対象は `24 races / 356 rows` で、score source は `default`、runtime policy は `sep_runtime_portfolio` だった。
- serving smoke も正常完了したが、`policy_selected_rows=0`、`policy_bets=0` だった。
- したがって current baseline は latest tail date で実行可能だが、運用判断としては「動く」ことと「bet が出る」ことをまだ分けて扱う必要がある。

### 4.5 latest tail-date window の確認

- `current_recommended_serving_2025_latest` で `2025-12-06/07/13/14/20/21/27/28` の 8 日 window を確認した。
- 全 8 日で predict / serving smoke は正常完了した。
- `policy_bets=1` が出たのは `2025-12-06`、`2025-12-20`、`2025-12-27` の 3 日だった。
- `2025-12-07`、`2025-12-13`、`2025-12-14`、`2025-12-21`、`2025-12-28` は `policy_bets=0` だった。
- したがって latest tail の zero-bet は常時ではなく、date により selection が出ることを確認済みである。

### 4.6 latest revision manifest の整合確認

- `scripts/run_netkeiba_latest_revision_gate.py` を更新し、wrapper manifest に `revision_gate_artifacts` を保持するようにした。
- `r20260325_latest_serving_candidate_v2` の実 run では wrapper manifest、`revision_gate_r20260325_latest_serving_candidate_v2.json`、`promotion_gate_r20260325_latest_serving_candidate_v2.json` がすべて `pass / promote` で一致した。
- wrapper manifest から revision manifest payload と promotion report payload を直接追えるため、latest の正式判断は wrapper artifact 一式だけで復元できる。

### 4.7 latest baseline docs / artifact 命名の固定

- `command_reference.md` と `serving_validation_guide.md` に latest baseline の predict / serving smoke 実行例を残した。
- latest の window artifact は `current_recommended_serving_2025_latest_<window_or_purpose>` を suffix 基本形とし、summary JSON も同じ slug で `--output-file` を明示する運用に揃えた。
- `run_predict.py` には suffix がない一方、`run_serving_smoke.py` の summary 既定名は profile 基準なので、window 単位の provenance は smoke 側で管理する方針を明文化した。

### 4.8 long-horizon latest の operational 判定

- `2025-09-06/07/13/14/20/21/27/28` の fresh compare では、`current_recommended_serving_2025_latest` は `policy_bets=32`、`total_policy_net=-27.3`、pure bankroll `0.2959` に対し、`current_long_horizon_serving_2025_latest` は `policy_bets=9`、`total_policy_net=-4.3`、pure bankroll `0.9996` だった。
- 差分は 8/8 日すべてで policy に現れ、score source は一致したまま September override 側が過剰 bet を抑えていた。したがって latest 実日付でも September 系 override は de-risk として有効である。
- `2025-12-06/07/13/14/20/21/27/28` の control compare では、両 profile の `differing_policy_dates=[]`、`total_policy_net=14.9`、pure bankroll `1.3691` で完全一致した。
- したがって `current_long_horizon_serving_2025_latest` は September 実日付の analysis-only 候補ではなく、seasonal de-risk 用の運用候補として残してよい。一方で non-September baseline は引き続き `current_recommended_serving_2025_latest` とする。

### 4.9 tighter policy candidate の formal 通過

- latest 2025 の悪化はデータ破損ではなく、broad betting candidate が 2025 regime で support を満たしつつ悪化していたこと、および formal gate が主に `min_bet_ratio` で詰まっていたことが原因だった。
- `current_tighter_policy_search_candidate_2025_latest` では、evaluation policy の `min_bet_ratio` だけを `0.05 -> 0.03` に下げ、`min_bets_abs=100` は維持した。
- この変更により full nested feasibility は `1 -> 4` feasible folds へ改善し、`min_feasible_folds=3` の promotion gate を通過した。
- その後の formal revision gate `r20260326_tighter_policy_ratio003` も `pass / promote` で完了した。
- 同 run の主要値は、nested WF weighted test ROI `0.9092326376053682`、nested bets total `424`、formal benchmark weighted ROI `1.172842678442708`、formal benchmark feasible folds `4` である。
- ただし operational baseline の入れ替えはまだ行わない。現時点の整理は「latest baseline は `current_recommended_serving_2025_latest` のまま維持し、tighter policy candidate は formal-support 改善が確認できた analysis-first candidate として次の比較対象に残す」である。

### 4.10 recent-heavy true retrain の初回完了

- `current_recommended_serving_2025_recent_2020` の初回比較は、data split だけが recent-heavy で component artifact は従来のままだったため、学習 window 比較としては不正確だった。
- そこで win component と roi component を同一 suffix `r20260327_recent_2020_component_retrain` で再学習し、その suffix を読む value_blend stack を再 bundle した。
- その結果、formal evaluation は `representative`、promotion gate は `pass / promote` となり、`AUC=0.8449137114185664`、`logloss=0.20107277063241344`、`top1_roi=0.8186249712577605`、`ev_top1_roi=0.7495861117498276`、`nested WF weighted test ROI=0.921753986332574`、`nested WF bets total=878` を確認した。
- matching な WF feasibility では `feasible_fold_count=4`、`formal_benchmark_weighted_roi=1.4548038057430124` だった。
- 一方で、直前の pseudo-retrain run `r20260327_current_recommended_serving_2025_recent_2020` と比べると、AUC / logloss / EV 系は改善したが、nested WF weighted test ROI は `0.9629 -> 0.9218`、bets total は `1092 -> 878` に低下した。
- したがって recent-heavy true retrain は「formal に通過した比較候補」までは到達したが、serving baseline を直ちに置き換える根拠にはまだしない。

### 4.11 recent-heavy 2018 true retrain の追加完了

- 同じ true retrain 手順を `configs/data_2025_recent_2018.yaml` に対しても実施し、`r20260327_recent_2018_component_retrain` として win / roi component 再学習、stack 再 bundle、evaluation、matching WF feasibility、promotion gate を完了した。
- formal evaluation は `representative`、promotion gate は `pass / promote` で整合し、`AUC=0.8431852285424601`、`logloss=0.201982990194456`、`top1_roi=0.8166245113819269`、`ev_top1_roi=0.7400091975166705`、`nested WF weighted test ROI=0.959452411994785`、`nested WF bets total=767` を確認した。
- matching な WF feasibility では `feasible_fold_count=5`、formal benchmark 側も `5/5` の feasible folds を満たした。
- 2020 start の true retrain と比べると、AUC / EV 系はわずかに劣る一方で、weighted nested-WF ROI は `0.9595 > 0.9218`、formal support は `5/5 > 4/5` となった。
- したがって recent-heavy family 内では、現時点の formal support 上位は 2018 start である。ただし latest baseline を置き換えるには actual-date compare がまだ必要である。

### 4.12 recent-heavy 2018 の fresh actual-date compare

- true retrain artifact を serving compare に直接載せるため、`run_serving_smoke.py` と `run_serving_profile_compare.py` に `--model-artifact-suffix` を追加した。
- この compare は `prediction-backend fresh` で実行する必要があり、`replay-existing` では canonical prediction CSV を再利用するだけで true retrain の差は出ないことも確認した。
- `2025-09-06/07/13/14/20/21/27/28` の fresh compare では、baseline `current_recommended_serving_2025_latest` が `32 bets / total net -27.3 / pure bankroll 0.2959`、recent-2018 true retrain が `4 bets / total net -4.0 / pure bankroll 0.8557` だった。
- つまり recent-2018 true retrain は September difficult regime で strong de-risk を示した。一方で `differing_policy_dates=[]` だった旧 replay compare は、canonical prediction 再利用の限界によるもので、判断根拠には使わない。

### 4.13 recent-heavy actual-date compare の優先窓確定

- 同じ `2025-09-06/07/13/14/20/21/27/28` の fresh compare を recent-2020 true retrain に対しても実施し、baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して `8 bets / total net -8.0 / pure bankroll 0.7408` を確認した。
- したがって September difficult regime における recent-heavy family の実測順位は `recent-2018 > recent-2020 > baseline` と読める。
- 一方で `2025-12-06/07/13/14/20/21/27/28` の fresh compare では、baseline が `45 bets / total net +21.8 / pure bankroll 1.6712`、recent-2018 true retrain が `1 bet / total net -1.0 / pure bankroll 0.9722` だった。
- この December control により、recent-heavy true retrain は broad replacement ではなく regime-specific candidate と扱うべきことが確認できた。現時点では 2018 start を recent-heavy family の actual-date 上位候補としつつ、baseline 置換ではなく September difficult regime 向けの analysis-first de-risk 候補として維持する。

### 4.14 tighter policy candidate の actual-date role 確定

- `current_recommended_serving_2025_latest` と `current_tighter_policy_search_candidate_2025_latest` の fresh compare を、`2025-09-06/07/13/14/20/21/27/28` と `2025-12-06/07/13/14/20/21/27/28` の 2 window で実施した。
- 2025-09 の difficult window では、baseline が `32 bets / total net -27.3 / pure bankroll 0.2959` に対して、tighter policy candidate は `9 bets / total net -4.3 / pure bankroll 0.8395` だった。
- 2025-12 tail の control window では、baseline が `45 bets / total net +21.8 / pure bankroll 1.6712`、tighter policy candidate が `9 bets / total net +21.4 / pure bankroll 1.6032` だった。
- したがって tighter policy candidate は September difficult regime では有効な defensive variant だが、broad baseline replacement の根拠はまだない。現時点の正しい位置づけは、formal-support 改善を持つ analysis-first defensive candidate である。

### 4.15 tighter policy の formal threshold frontier 確認

- `wf_threshold_sweep_current_tighter_policy_search_candidate_2025_latest.json` を基準に、`0.03/100` と `0.03/80` の frontier を確認した。
- 現行の `min_bet_ratio=0.03 / min_bets_abs=100` では `4/5 feasible folds`、`0.03 / 80` では `5/5 feasible folds` になる。
- 新たに通るのは fold 2 だけで、best feasible は既存 family と同じ `kelly blend_weight=0.8 / min_prob=0.03 / odds_max=25`、`98 bets / ROI 0.7542 / final_bankroll 0.9199 / max_drawdown 0.1098` だった。
- したがって `0.03/80` は新しい aggressive family を追加する変更ではなく、既存 defensive family を formal support 上どこまで許容するかの境界調整として扱える。一方で serving policy 自体は変わらないため、September defensive / December control という operational role はこの変更だけでは変わらない。

### 4.16 tighter policy `0.03/80` の threshold-only revision 完了

- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs80.yaml` を追加し、`min_bets_abs=80` だけを落とした threshold-only variant を formal revision に載せた。
- `value_blend` family では新 revision suffix の component model が自動では存在しないため、`run_evaluate.py` に `--model-artifact-suffix`、`run_wf_feasibility_diag.py` に同等の model artifact reuse、`run_revision_gate.py` に `--skip-train` と matching WF feasibility の自動実行を追加した。
- この flow により、`r20260327_tighter_policy_ratio003_abs80` は train を skip しつつ `r20260326_tighter_policy_ratio003` の学習済み artifact を再利用し、evaluation、WF feasibility、promotion gate まで一貫して完了した。
- promotion gate は `pass / promote` で、`formal_benchmark_feasible_fold_count=5/5`、`formal_benchmark_weighted_roi=1.1042287961989103`、`formal_benchmark_bets_total=598` だった。
- fold 2 の best feasible は既存 frontier 読みと一致し、`kelly blend_weight=0.8 / min_prob=0.03 / odds_max=25 / bets=98 / ROI=0.7542 / final_bankroll=0.9199 / max_drawdown=0.1098` だった。
- したがって `0.03/80` は frontier 上の読みだけでなく formal revision としても成立する。一方で December control で baseline 優位という operational role は変わらないため、位置づけは broad serving replacement ではなく「formal に通過した tighter defensive variant」のままとする。

## 5. 完了済みマイルストーン

### M1. 2025 backfill 完了

- race_result / race_card / pedigree を 2025 末尾まで補完。
- pedigree target と output の horse_key 集合は一致。

### M2. latest data config 整備

- `configs/data_2025_latest.yaml` を追加。
- 2025 latest split を profile から直接選べるようにした。

### M3. latest profile 導線整備

- `_2025_latest` suffix で既存 family を latest split に切り替え可能にした。
- README と command reference に入口を追記した。

### M4. latest formal evaluation 完了

- `current_best_eval_2025_latest`
- `current_recommended_serving_2025_latest`

の 2 本を比較し、simpler profile を baseline と判断した。

### M5. latest formal promotion 完了

- `current_recommended_serving_2025_latest` で latest formal gate を通過。

### M6. latest serving smoke window 確認

- `2025-12` 末尾 8 日 window で latest baseline の actual-date smoke を確認。
- zero-bet 固定ではなく、3/8 日で selected rows と `policy_bets=1` を確認。

### M7. latest revision manifest 整合整理

- `scripts/run_netkeiba_latest_revision_gate.py` が revision / promotion artifact lineage を wrapper manifest に保持するようになった。
- `r20260325_latest_serving_candidate_v2` の実 run で wrapper manifest と downstream artifacts の `pass / promote` 一致を確認した。

### M8. latest baseline artifact / docs の仕上げ

- latest baseline の predict / serving smoke 実行例を docs に追記した。
- `current_recommended_serving_2025_latest_<window_or_purpose>` を suffix 基本形とし、summary JSON は同じ slug を `--output-file` にも入れる運用で揃えた。

### M9. long-horizon latest の再検証完了

- `current_recommended_serving_2025_latest` と `current_long_horizon_serving_2025_latest` を `2025-09` 実日付 window で fresh compare し、September override 側の de-risk を確認した。
- 同じ 2 profile を `2025-12` tail window でも fresh compare し、non-September では baseline と完全一致することを確認した。

### M10. tighter policy candidate の formal revision gate 完了

- `current_tighter_policy_search_candidate_2025_latest` を `r20260326_tighter_policy_ratio003` として full train / evaluate / promotion gate まで直列実行した。
- revision manifest と promotion report はともに `pass / promote` で整合した。
- `min_bet_ratio=0.03` / `min_bets_abs=100` の組み合わせで `formal_benchmark_feasible_fold_count=4`、`formal_benchmark_weighted_roi=1.172842678442708` を確認した。

### M11. recent-heavy 2020 true retrain の formal 評価完了

- win / roi component を同一 suffix で再学習し、value_blend stack を recent-heavy component artifact に差し替えて再 bundle した。
- `r20260327_recent_2020_component_retrain` で evaluation、matching WF feasibility、promotion gate を完了し、`pass / promote` を確認した。
- 主要値は `AUC=0.8449`、`ev_top1_roi=0.7496`、`nested WF weighted test ROI=0.9218`、`formal_benchmark_feasible_fold_count=4` である。

### M12. recent-heavy 2018 true retrain の formal 評価完了

- `r20260327_recent_2018_component_retrain` で recent-2018 の true retrain compare を完了し、evaluation、matching WF feasibility、promotion gate が `pass / promote` で整合した。
- 主要値は `AUC=0.8432`、`ev_top1_roi=0.7400`、`nested WF weighted test ROI=0.9595`、`formal_benchmark_feasible_fold_count=5` である。

### M13. true retrain artifact の serving compare 導線整備

- `run_serving_smoke.py` / `run_serving_profile_compare.py` / `predict_batch.py` に model artifact suffix の override を追加した。
- これにより suffix 付き retrain artifact を actual-date compare に直接載せられるようになった。
- 同導線を使った 2025-09 の fresh compare では、recent-2018 true retrain が latest baseline に対して大きく de-risk することを確認した。

### M14. recent-heavy actual-date の優先窓確認

- 2025-09 の fresh compare を recent-2020 true retrain にも広げ、recent-heavy family 内の実測順位が `2018 > 2020` であることを確認した。
- 2025-12 tail の fresh compare では recent-2018 true retrain が baseline に明確に劣後し、recent-heavy true retrain を broad baseline replacement と見なさないことも確認した。

### M15. tighter policy candidate の actual-date role 確定

- `current_tighter_policy_search_candidate_2025_latest` の fresh actual-date compare を September difficult window と December tail control window で完了した。
- September では baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して `9 bets / total net -4.3 / pure bankroll 0.8395` と強い損失圧縮を確認した。
- 一方で December tail では baseline `45 bets / total net +21.8 / pure bankroll 1.6712` に対して `9 bets / total net +21.4 / pure bankroll 1.6032` で、baseline 優位を維持した。
- これにより、tighter policy candidate は broad replacement ではなく September difficult regime 向けの analysis-first defensive candidate として位置づけを確定した。

### M16. tighter policy の formal threshold frontier 確認

- existing threshold sweep から、`current_tighter_policy_search_candidate_2025_latest` の `0.03/100 -> 0.03/80` 変化を確認した。
- `0.03/100` は `4/5 feasible folds`、`0.03/80` は `5/5 feasible folds` で、差分は fold 2 のみだった。
- fold 2 を通す best feasible は `kelly blend_weight=0.8 / min_prob=0.03 / odds_max=25`、`98 bets / ROI 0.7542 / final_bankroll 0.9199 / max_drawdown 0.1098` で、既存 family の延長として読める。
- このため `0.03/80` は formal support を広げる候補としては defensible だが、serving policy は不変なので operational baseline の切替根拠には使わない、という整理を確定した。

### M17. tighter policy `0.03/80` の threshold-only revision 完了

- `r20260327_tighter_policy_ratio003_abs80` を train skip / model artifact reuse の threshold-only revision として完了した。
- evaluation に加えて matching `wf_feasibility_diag` も同じ reused artifact で生成し、promotion gate まで `pass / promote` で整合した。
- 主要値は `formal_benchmark_weighted_roi=1.1042287961989103`、`formal_benchmark_feasible_fold_count=5/5`、`formal_benchmark_bets_total=598` である。

### M18. regime-specific candidate 境界の docs 整理完了

- benchmark / overview / public snapshot / serving validation / command reference の role 表現を揃え、September difficult regime の候補群を同じ読み筋で再開できるようにした。
- 基本整理は `current_recommended_serving_2025_latest` を baseline に固定し、`current_long_horizon_serving_2025_latest`、`current_tighter_policy_search_candidate_2025_latest`、recent-2018 true retrain を September difficult regime 向け defensive candidate 群として参照する、という形で統一した。
- December tail のような control window では baseline 優位を維持し、formal 通過だけで broad replacement を行わないことも docs 上で明示した。

### M19. `_2025_latest` stable family 棚卸し完了

- `model_profiles.py` の自動派生仕様と docs 上の current stable family を切り分け、`current_best_eval_2025_latest`、`current_recommended_serving_2025_latest`、`current_long_horizon_serving_2025_latest`、`current_tighter_policy_search_candidate_2025_latest` の 4 本を latest 主導線として固定した。
- `current_bankroll_candidate_2025_latest`、`current_ev_candidate_2025_latest`、`current_sep_guard_candidate_2025_latest` は generated variant ではあるが current latest docs の主導線には置かないことを明文化した。
- profile 名、artifact suffix、代表 window の対応を command / scripts guide から辿れるようにした。

### M20. seasonal runtime policy の短文化完了

- `project_overview.md` と `serving_validation_guide.md` で、September は `current_long_horizon_serving_2025_latest` を最初の de-risk alias とし、non-September は `current_recommended_serving_2025_latest` を既定運用に保つ、という 2 段ルールに圧縮した。
- `current_tighter_policy_search_candidate_2025_latest` と recent-2018 true retrain は、September difficult regime では参照してよいが、non-September では analysis-first compare に留めることを短文で再確認できる形にした。
- December tail は controlled override を確認する control window として使う、という説明に統一した。

### M21. latest 2025 compare artifact 導線整理完了

- `command_reference.md` と `serving_validation_guide.md` に latest 2025 用の quick artifact map を追加し、compare 実行後にどの dashboard summary / promotion gate を最初に見ればよいかを固定した。
- long-horizon、tighter policy、recent-2018 true retrain の September / December control artifact を docs から直接辿れるようにした。
- これにより latest baseline の compare / threshold / fresh-vs-replay の再現導線は、docs 上で必要十分な入口を揃えた。

### M22. recent-heavy family の参照優先順位整理完了

- recent-heavy family 内では `2018` start を front-line candidate、`2020` start を補助比較用の secondary reference として扱う整理を docs 上で固定した。
- 根拠は、formal support では `2018` start が `5/5` feasible folds と weighted nested-WF ROI で優位、actual-date でも September difficult window で `2018 > 2020 > baseline` が確認済みである点に置いた。
- これにより current active question は、recent-heavy family 内の順位づけではなく、recent-2018 と tighter defensive variant の参照順へ絞られた。

### M23. September difficult regime の候補参照順固定

- September difficult regime の候補順は、`current_long_horizon_serving_2025_latest` を第一候補、`current_tighter_policy_search_candidate_2025_latest` を第二候補、recent-2018 true retrain を第三候補の analysis-first fallback とする整理で固定した。
- recent-2018 は September 実日付だけ見れば strongest de-risk だが、学習窓の再構成を伴うため、current operational reading では同一 latest family 上で済む tighter policy candidate を先に参照する。
- この順序により、実運用寄りの alias から順に `long_horizon -> tighter policy -> recent-2018` と辿り、December control ではいずれも broad replacement と見なさない読み筋が揃った。

### M24. public snapshot の短文化完了

- `public_benchmark_snapshot.md` の role 要約を 3 行に圧縮し、baseline と defensive option の読み分けを対外向けにも短く読める形にした。
- artifact 出典も formal result と September / December control compare の最小集合へ絞り、内部 docs のような過剰列挙を避けた。
- これにより対外向け文書でも `baseline を維持し、September だけ defensive option を参照する` という読み筋が即座に分かるようになった。

### M25. 地方競馬データ拡張の feasibility 整理完了

- 地方競馬データは JRA latest baseline を直接改善する次タスクではなく、別 universe の将来候補として扱うことを docs 上で固定した。
- ingestion、key、benchmark の 3 点を少なくとも JRA と切り分ける必要があること、混合学習を試す場合も JRA-only baseline を残すことを前提条件として明記した。
- これにより「地方拡張を今すぐやるべきか」という誤読を避け、JRA latest の残課題と future option を分離できる状態にした。

### M26. actual-date compare 再開導線の最終整理完了

- `project_overview.md` と `serving_validation_guide.md` に、latest 2025 compare を再開するときの 3-step quickstart を追加した。
- 読み順は `dashboard summary -> compare rerun -> formal support artifact` に固定し、September difficult window の参照順も `long_horizon -> tighter policy -> recent-2018` で揃えた。
- これにより latest compare の入口は overview / validation guide / command reference の 3 点で過不足なく一致した。

### M27. docs index と command 入口の quickstart 同期完了

- `docs/README.md` に latest 2025 actual-date compare 再開時の 3-step shortcut を追加し、docs index からでも同じ読み順へ入れるようにした。
- `command_reference.md` の先頭にも同じ再開順を置き、いきなり CLI 実行へ入る前に dashboard summary を先に見る運用を明示した。
- これにより overview / docs index / serving validation / command reference の 4 箇所で latest compare の再開順が一致した。

### M28. benchmark と public snapshot の再開導線同期完了

- `benchmarks.md` に latest 2025 判断再開時の 3-step 読み順を追加し、current position から compare / formal support へどう降りるかを明示した。
- `public_benchmark_snapshot.md` には internal reader 向けの但し書きを足し、actual-date compare の再開入口は snapshot ではなく `serving_validation_guide.md` であることを明示した。
- これにより latest compare の再開順は overview / docs index / benchmarks / serving validation / command reference / public snapshot で矛盾しない状態になった。

### M29. formal guide と development flow の再開順同期完了

- `evaluation_guide.md` に、latest 2025 の current reading を再開するときは formal artifact から入らず、まず actual-date role を確認してから evaluate / promotion gate へ降りる旨を追記した。
- `development_flow.md` にも、latest compare 再開時は `serving_validation_guide.md` の quickstart を先に見る運用を明記した。
- これにより actual-date role と formal support の読む順は、guide / flow / benchmark / command docs で同じになった。

### M30. artifact / script 補助資料の再開入口同期完了

- `artifact_guide.md` に、latest 2025 actual-date compare は artifact 名一覧からではなく `serving_validation_guide.md` の quickstart から再開する旨を追記した。
- `scripts_guide.md` にも、latest compare 再開時は索引から script を探し始めず、quickstart と command entrypoint を先に使う運用を明記した。
- これにより latest compare の再開順は補助資料レベルでも揃い、docs 間の入口差はほぼ解消した。

### M31. latest compare 導線監査の一巡完了

- docs 全体を横断で点検し、latest 2025 actual-date compare の再開入口が overview / index / benchmark / guide / command / helper docs で矛盾しないことを確認した。
- この結果、active priority としての「導線整理」は一巡完了とし、以後は定期的な drift 点検だけを残す。
- 次の候補は、新たな docs 入口追加ではなく future option の深掘りか、通常の保守点検になる。

### M32. 地方競馬 feasibility の設計チェックリスト具体化

- `data_extension.md` に、地方競馬を別 universe として扱うときの判定軸を `source / key / feature / benchmark / rollout` の 5 境界へ分解して追記した。
- さらに、最小チェックリストと非推奨な始め方を明記し、JRA-only baseline を壊さずに feasibility を切る順序を具体化した。
- これにより地方競馬は「将来候補」という抽象表現から、着手前に確認すべき設計項目を持つ future option へ進んだ。

### M33. 地方-only snapshot / gate artifact 方針の具体化

- `data_extension.md` に、地方-only coverage snapshot と benchmark gate manifest を JRA 既存 artifact から分離する命名方針を追記した。
- 具体例として config / snapshot / gate / revision に共通の universe slug を入れるルールを示し、JRA-only / local-only / mixed の lineage を名前で追跡できる形にした。
- `artifact_guide.md` にも同じ方針を補足し、外部データ universe 拡張時に latest JRA artifact を上書きしない原則を明記した。

### M34. 地方-only benchmark 完了条件の具体化

- `data_extension.md` に、地方-only benchmark を完了とみなす最小条件を `snapshot readiness / data integrity / feature readiness / representative evaluation / local-only gate` の 5 段で追記した。
- あわせて artifact に残す最低限の判定項目を整理し、completion bar が「JRA を上回ること」ではなく「別 universe を再現可能にすること」であると明示した。
- これにより地方競馬 future option は、artifact lineage だけでなく benchmark 完了条件まで設計レベルで持つ状態になった。

### M35. 地方-only snapshot / gate payload schema の具体化

- `data_extension.md` に、地方-only coverage snapshot と benchmark gate manifest の最小 JSON schema 例を追加し、既存 `netkeiba_*` script に寄せつつ universe 境界を明示する形へ落とした。
- 必須キーとして `universe`, `readiness`, `coverage_summary`, `integrity_summary`, `completed_step` を固定し、artifact 名だけでなく payload 本体でも universe を判別できる方針を明記した。
- `artifact_guide.md` にも同じ原則を補足し、将来の local-only / mixed CLI が返すべき payload 契約の輪郭を揃えた。

### M36. 地方-only CLI 引数契約の具体化

- `data_extension.md` に、地方-only coverage snapshot / benchmark gate が持つべき最小 CLI 引数セットを追記した。
- 方針は既存 `netkeiba_*` gate の operator experience を維持しつつ、追加は `universe`, `source-scope`, `baseline-reference`, `schema-version` に限定することである。
- `scripts_guide.md` にも同じ原則を補足し、将来 CLI を追加しても既存運用から大きく逸れないことを明示した。

### M37. 地方-only step 名と fail-fast taxonomy の具体化

- `data_extension.md` に、地方-only coverage snapshot / benchmark gate の推奨 step 名と `completed_step` の読み方を追記した。
- あわせて operator error / readiness block / execution failure の 3 層で failure taxonomy を整理し、`status`, `completed_step`, `error_code`, `recommended_action` の最小 contract を固定した。
- `scripts_guide.md` にも step 系列を補足し、将来 CLI を足すときの進捗表示と停止点の読み方を揃えた。

### M38. `netkeiba_*` snapshot / gate の universe-aware 契約実装

- `run_netkeiba_coverage_snapshot.py` に `universe`, `source-scope`, `baseline-reference`, `schema-version` を追加し、snapshot payload に `status`, `completed_step`, `coverage_summary`, `integrity_summary`, `error_code`, `recommended_action` を残せるようにした。
- `run_netkeiba_benchmark_gate.py` にも同じ 4 引数を追加し、snapshot / readiness / train / evaluate の停止点を `completed_step` と error vocabulary で読めるようにした。
- これにより local-only CLI を新設する前に、既存 `netkeiba_*` 系で universe-aware payload 契約と fail-fast 読み筋を先に検証できる状態になった。

### M39. local-only snapshot / gate 雛形の追加

- `run_netkeiba_coverage_snapshot.py` と `run_netkeiba_benchmark_gate.py` に外部 CSV path override を追加し、JRA/netkeiba 既定値を維持したまま別 universe の source path を渡せるようにした。
- `run_local_coverage_snapshot.py` と `run_local_benchmark_gate.py` を追加し、`local_nankan` を既定 universe とする artifact 名、baseline reference、source path を wrapper 側で固定した。
- `configs/data_local_nankan.yaml`、`configs/model_local_baseline.yaml`、`configs/features_local_baseline.yaml` を追加し、JRA artifact を上書きしない local-only smoke の入口を切った。

### M40. local-only integrity / feature gap / evaluation 入口の追加

- `run_local_data_source_validation.py` を追加し、`run_validate_data_sources.py` の report を `data_source_validation_local_nankan.json` へ分離して出せるようにした。
- `run_local_feature_gap_report.py` を追加し、feature gap summary / feature coverage / raw coverage を `local_nankan` 名義の artifact へ固定出力できるようにした。
- `run_local_evaluate.py` を追加し、既存 `run_evaluate.py` の versioned output を再利用しつつ、local-only 側では `evaluation_local_nankan_pointer.json` から evaluation artifact lineage を辿れるようにした。

### M41. local-only orchestration manifest の追加

- `run_local_feasibility_manifest.py` を追加し、readiness snapshot、data validation、feature gap、evaluation pointer を fail-fast で直列実行できるようにした。
- 同 script は `local_feasibility_manifest_local_nankan.json` に `completed_step`, `error_code`, `recommended_action`, 各 step command / exit_code / artifact path を残し、local-only feasibility の停止点を 1 本で追えるようにする。
- これにより local-only wrapper 群は単発 CLI の集合から、次の orchestration 単位で読める実装スケルトンへ進んだ。

### M42. local-only revision lineage の追加

- `run_local_revision_gate.py` を追加し、local benchmark gate を readiness precheck として先に通し、その後に `run_revision_gate.py` を local config と revision slug で起動できるようにした。
- 同 script は revision ごとの snapshot / benchmark manifest / wf summary / promotion report / revision manifest / evaluation pointer を `local_revision_gate_<revision>.json` からまとめて辿れるようにする。
- これにより local-only future option は、単なる feasibility manifest 群ではなく、future local revision slug を起点に artifact lineage を追える段階へ進んだ。

### M43. local public snapshot と mixed compare 命名を固定した

- `run_local_public_snapshot.py` を追加し、`local_revision_gate_<revision>.json` から `local_public_snapshot_<revision>.json` を切り出せるようにした。
- public 向けの local-only 読み順を `local_public_snapshot -> local_revision_gate -> promotion_gate -> revision_gate -> evaluation_pointer` に固定した。
- mixed-universe compare の artifact 名を `mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` 系で統一し、JRA-only / local-only / mixed の lineage を名前だけで分離できるようにした。

### M44. mixed-universe compare の pointer manifest を追加した

- `run_mixed_universe_compare.py` を追加し、left 側の local public snapshot または lineage と right 側の JRA public reference を `mixed_universe_compare_<left_universe>_vs_<right_universe>_<revision>.json` に束ねられるようにした。
- mixed compare は当面 `pointer_only` とし、`decision=separate_lineage_required` を返して promote 判定とは分離した。
- これで docs だけで先行していた mixed compare 命名規則が、最小 artifact まで揃った。

### M45. mixed-universe compare readiness precheck を追加した

- `run_mixed_universe_readiness.py` を追加し、mixed compare 前に left 側 readiness、evaluation pointer、`stability_assessment=representative`、right 側 public reference を点検できるようにした。
- readiness artifact は `mixed_universe_readiness_<left_universe>_vs_<right_universe>_<revision>.json` とし、`status=ready/not_ready` と `checks[]` から停止点を読めるようにした。
- これにより mixed compare は pointer-only であっても、比較前提が足りているかを別 manifest で機械的に確認できる段階へ進んだ。

### M46. mixed-universe comparison schema を追加した

- `run_mixed_universe_schema.py` を追加し、readiness manifest と pointer-only compare manifest を受けて comparison axes と metric rows を `mixed_universe_schema_<left_universe>_vs_<right_universe>_<revision>.json` に落とせるようにした。
- schema は当面 numeric compare を実行せず、left 側 artifact path と right 側 public reference の対応関係だけを固定する contract として扱う。
- これにより mixed compare は readiness -> pointer -> schema の 3 段で、実数値比較前の読み筋と比較軸を揃えられるようになった。

### M47. JRA public benchmark reference manifest を追加した

- `run_public_benchmark_reference.py` を追加し、JRA latest baseline の promotion / revision / evaluation artifact を `public_benchmark_reference_<reference>.json` に束ねられるようにした。
- mixed compare / readiness / schema は right 側の参照先としてこの machine-readable reference manifest を使えるようにした。
- これにより mixed compare の右側は doc 参照だけでなく JSON 正本を持ち、numeric compare へ進む土台が揃った。

### M48. mixed-universe numeric compare の最小 CLI を追加した

- `run_mixed_universe_numeric_compare.py` を追加し、mixed readiness / compare / schema と JRA public benchmark reference manifest から row 単位の compare manifest を生成できるようにした。
- row ごとに `left_value`, `right_value`, `delta_left_minus_right`, `comparison_status` を持たせ、left 側が未整備でも partial compare を止めずに欠損箇所を manifest へ残すようにした。
- これにより mixed compare は readiness -> pointer -> schema -> numeric compare の 4 段で、left/right JSON reference から最小差分行まで切り出せるようになった。

### M49. mixed-universe numeric compare の CSV 出力を追加した

- `run_mixed_universe_numeric_compare.py` に CSV 出力を追加し、row-level compare を JSON だけでなく一覧でも読めるようにした。
- numeric 行には `delta_direction` も付け、positive / negative / zero / unknown を一目で確認できるようにした。
- これにより partial compare の欠損確認と、将来 left formal metrics が埋まった後の差分レビューが軽くなった。

### M50. mixed-universe numeric summary を追加した

- `run_mixed_universe_numeric_summary.py` を追加し、numeric compare manifest から promote-safe な summary manifest を生成できるようにした。
- summary では `verdict`, `missing_left_rows`, `missing_right_rows`, numeric delta の正負行を要約し、row 全件を読まずに停止理由と差分方向を追えるようにした。
- これにより mixed compare の最上段に、人間向けの薄い判読レイヤを 1 枚追加できた。

### M51. mixed-universe numeric summary に severity と notes を追加した

- numeric summary に threshold-based な `severity` と `notes` を追加し、欠損比率や categorical difference の有無を薄い判読ルールへ落とした。
- `evidence_incomplete` では missing 比率に応じて `moderate` / `severe` を返し、numeric compare を promotion evidence とみなさない note も残すようにした。
- これにより summary は verdict だけでなく、停止強度と実務上の読み方まで 1 枚で伝えられるようになった。

### M52. mixed-universe left gap audit を追加した

- `run_mixed_universe_left_gap_audit.py` を追加し、numeric compare の `missing_left_rows` を local revision lineage と突き合わせて、必要 source artifact と command preview を row ごとに追えるようにした。
- audit summary では `rows_missing_all_sources`, `rows_with_planned_commands`, `severity`, `notes` を持たせ、left formal metrics がなぜ埋まっていないかを一段深く読めるようにした。
- これにより mixed compare の左側欠損は、summary で止まらず source gap まで manifest で追跡できるようになった。

### M53. mixed-universe left recovery plan を追加した

- `run_mixed_universe_left_recovery_plan.py` を追加し、left gap audit の command preview を重複除去した recovery plan manifest を生成できるようにした。
- plan では `step`, `command_preview`, `required_for_rows`, `artifacts` を持たせ、missing-left row の情報を operator が実行順で読める形へ畳んだ。
- これにより left 側欠損は、原因分析から実行計画まで manifest で辿れるようになった。

### M54. mixed-universe status board を追加した

- `run_mixed_universe_status_board.py` を追加し、public snapshot から left recovery plan までの mixed-universe manifest 群を 1 本の status board manifest に束ねられるようにした。
- board では `current_phase`, `next_action_source`, `recommended_action`, `phase_summaries` と summary / audit / recovery の主要 severity を持たせ、全体の現在地を一目で読めるようにした。
- これにより mixed-universe workflow は、個別 manifest を順に追わなくても status board から必要な layer へ降りられるようになった。

### M55. mixed-universe recovery wrapper を追加した

- `run_mixed_universe_recovery.py` を追加し、既存 status board を起点に local revision lineage の再実行と mixed-universe manifest 群の再生成を 1 本で回せるようにした。
- wrapper は `local_revision_gate -> local_public_snapshot -> readiness -> compare -> schema -> numeric compare -> numeric summary -> left gap audit -> left recovery plan -> status board` を既存 artifact path へ書き戻しながら実行し、partial のままでも最新停止点まで更新する。
- これにより mixed compare の left-side recovery は、plan を読む段階から実行と board 再同期まで一貫した CLI を持つようになった。

### M56. local benchmark の source preflight を追加した

- `run_netkeiba_benchmark_gate.py` に optional な source preflight を追加し、`inspect_dataset_sources` を使って raw dir / primary CSV / required table の不足を snapshot 前に `not_ready` として返せるようにした。
- `run_local_benchmark_gate.py` と `run_local_revision_gate.py` はこの preflight を使うように更新し、`data_preflight_<revision>.json` を lineage artifact に含めるようにした。
- これにより local-only recovery は、`No CSV files found ...` の深い失敗ではなく、早期の `primary_dataset_missing` と `recommended_action` で停止点を読めるようになった。

## 6. 実行中の優先事項

`current_tighter_policy_search_candidate_2025_latest` の `0.03/80` formalization は M17 で完了した。続いて seasonal / recent-heavy の運用境界整理、latest compare artifact map、actual-date compare 再開導線の同期監査、地方競馬 feasibility の設計チェックリスト・artifact 方針・benchmark 完了条件・payload schema・CLI 引数契約・step/failure taxonomy の具体化、既存 `netkeiba_*` snapshot / gate への universe-aware 契約実装、local-only snapshot / gate 雛形の追加、local-only integrity / feature gap / evaluation 入口の追加、local-only orchestration manifest の追加、および local-only revision lineage の追加まで完了した。

以後の active priority は、public / internal docs の定期点検と future option の切り分けに絞る。

### P1. docs の定期点検

目的:

- ここまで積み上げた role / artifact / candidate order の説明が、今後の更新で再び発散しないように保つ。

やること:

1. roadmap の next candidates が active priority と矛盾しないか定期的に点検する。
2. public / internal docs の current reading が新しい更新でずれていないかを巡回確認する。
3. `.vscode/` のようなローカル設定を commit 対象へ混ぜない運用を継続する。

完了条件:

- docs の current reading が少ない往復で保守できること。

## 7. 次の候補

### N1. 地方競馬データ拡張の feasibility 深掘り

- universe slug を config / artifact / revision にどう通すかを、必要なら実装前提まで下ろす。
- status board までは入ったので、next は left formal metrics が埋まった時に row 欠損がどこまで解消されるかを追う。

### N2. docs の定期点検

- public / internal docs の重複表現が増えすぎていないかを定期的に点検する。

## 8. 当面やらないこと

- crawler の追加修正
- 新しい外部データ源の拡張
- latest formal result を見ないままの broad policy rewrite

これらは現状のボトルネックではない。まずは latest baseline を運用導線へ落とし切る。

地方競馬データのような大規模拡張は選択肢としては残すが、直近の優先課題は JRA latest の学習 window と operational 判断を詰めることである。

## 9. 関連文書

- [project_overview.md](project_overview.md)
- [development_flow.md](development_flow.md)
- [benchmarks.md](benchmarks.md)
- [evaluation_guide.md](evaluation_guide.md)
- [serving_validation_guide.md](serving_validation_guide.md)
- [command_reference.md](command_reference.md)