- M103. local revision gate に top-level 停止点要約を足した
- M102. local public snapshot に top-level 停止点要約を足した
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

2026-04-05 時点の到達点は次のとおりである。

- netkeiba 2025 backfill は完了している。
- `configs/data_2025_latest.yaml` により、2025 を validation 末尾に含む latest split が使える。
- latest coverage snapshot は `benchmark_rerun_ready=true`。
- `_2025_latest` profile で train / evaluate / predict / backtest を呼べる。
- latest の formal 評価では `current_best_eval_2025_latest` と `current_recommended_serving_2025_latest` のトップラインが実質同一だった。
- latest の正式判断では、`current_recommended_serving_2025_latest` が matching WF を含めて `pass / promote` に到達した。
- `current_best_eval_2025_latest` についても、reuse-compare 前提の latest revision gate wrapper を end-to-end で再実行できる状態になった。
- `r20260405_2025latest_refresh_reuse_runtimecfg_wfprofilefix` では、`--skip-train` と historical model artifact reuse を使った formal rerun が `pass / promote` で整合し、`AUC=0.8364`、`nested WF weighted test ROI=0.9157`、`formal_benchmark_weighted_roi=0.8372`、`formal_benchmark_feasible_fold_count=5` を確認した。
- これにより latest 2025 mainline は、operational baseline と evaluation mainline reference の両方で wrapper / revision gate / promotion gate の再現導線が揃った。
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
- primary tail cache path は freshness guard と refresh automation を含めて standardize され、`configs/data_2025_latest.yaml` の default mainline に昇格した。
- runtime default smoke は `loading training table 0m02s`, total `0m15s` で、explicit alias compare との差分は `run_context.data_config` のみだった。
- runtime 完了後の experiment reentry では `tighter policy search` family を再読し、A anchor 維持・C near-par challenger・B no-op/failed side read を確認した。
- `r20260330_class_rest_surface_interactions_v1` では、true component retrain と stack rebuild までは正常完了し、formal evaluation も `auc=0.8417`、`ev_top1_roi=0.6837`、`nested WF weighted test ROI=1.1003` と強い summary を示した。
- 一方で matching WF feasibility は `1/5` feasible folds に留まり、promotion gate は `block / hold` だった。
- fold 1-4 の主な失敗理由は一貫して `min_bets` で、fold 5 のみ `portfolio` candidate が feasible になった。
- その後 support hardening を進め、`r20260330_surface_interaction_only_v1` では support が `2/5` まで改善したが、nested weighted ROI は `0.7462` に低下して `hold` だった。
- さらに middle-ground candidate `r20260330_surface_plus_class_layoff_interactions_v1` を作成し、true component retrain、stack rebuild、matching WF feasibility、promotion gate まで完了した。
- 同 run は `pass / promote` で整合し、`formal_benchmark_weighted_roi=1.1379979394080304`、`formal_benchmark_feasible_fold_count=3` を確認した。
- 続く actual-date compare では、September difficult window で `32 bets / -27.3` に対して `8 bets / -8.0`、December tail control window で `45 bets / +21.8` に対して `13 bets / +20.0` を確認した。
- bankroll sweep でも September は promoted pure path が優位だった一方、December は promoted pure path より `2025-12-06` だけ promoted を使う hybrid が優位だった。
- その後の policy-side widening probes では、`sep_date_selected_rows_kelly_candidate` が `1 / 216 races`、`portfolio_lower_blend` が `0 / 216 races` となり、promoted line の exposure widening には失敗した。
- したがって current primary line は feature family の support hardening継続でも serving default 昇格でも widening 継続でもなく、formal promoted line と operational default line の role split を explicit にする段階へ移る。
- role split の current conclusion は、formal promoted line を `r20260330_surface_plus_class_layoff_interactions_v1`、operational default line を `current_recommended_serving_2025_latest` に固定することだ。
- これにより `class / rest / surface` family は current queue 上では一段落し、次の feature reentry は `jockey / trainer / combo` family に移る。
- `jockey / trainer / combo` family の first child `r20260330_jockey_trainer_combo_style_distance_v1` は、その後 quiet-lane rerun を経て true component retrain、stack rebuild、formal revision gate まで完了した。
- 同 run は `pass / promote` で整合し、`auc=0.8431`、`nested WF weighted test ROI=0.8907`、`wf_nested_test_bets_total=516`、matching WF feasibility `3/3` を確認した。
- 一方で promotion gate の formal benchmark は `weighted_roi=0.95`、`bets_total=480` で、surface+layoff promoted line や tighter-policy promoted line より top-line は弱い。
- `#53` の actual-date compare では、September window が baseline `32 / 216 races / -27.3` に対して `5 / 216 / -3.6`、December control が baseline `45 / 264 / +21.8` に対して `30 / 264 / -9.3` だった。surface+layoff promoted line と比べても September `8 / 216 / -8.0`、December `13 / 264 / +20.0` より強い role は示さなかった。
- したがって current reading は family anchor ではなく、analysis-first promoted candidate である。
- 一方で local Nankan の root blocker は ROI 改善ではなく trust / architecture / operator trust の分離である。`#120` provenance trust gate は current alias で `pre_race=729107`, `post_race=0`, `unknown=0`, `strict_trust_ready=true` に到達し、`#121` source timing audit は fetch-timing / recoverability の別軸として維持する。これにより historical trust-ready corpus 上の `#101` formal rerun `r20260415_local_nankan_pre_race_ready_formal_v1` は完走し、current NAR benchmark reference が確立した。
- local Nankan の標準 CLI 導線として `local_nankan_recommended` は残すが、これは operator convenience alias であり、trust-ready benchmark default ではない。`r20260412_local_nankan_recommended_wf_runtime_narrow_full` の `top1_roi=0.8381912618392912`、`ev_top1_roi=1.940849373663306`、`auc=0.8775353752835744` も market-aware historical read にとどめる。
- `odds/popularity` を外した no-market refresh train `r20260413_local_nankan_no_market_refresh_v1` は旧 no-market artifact と同水準の train AUC `0.7686` に留まった。一方で current benchmark reference はその line ではなく、trust-ready historical corpus 上の `#101` formal rerunである。したがって current NAR execution order は `#120 -> #121 -> #101 reference maintenance -> #103 architecture bootstrap -> #122 future-only readiness track` と読む。
- result-ready `local_nankan_value_blend_bootstrap` は `all_safe` が raw `margin` を拾うリークを含んでいたため safe exclusion を修正して再学習したが、代表評価 `r20260412_local_nankan_value_blend_bootstrap_result_ready_eval_v1` でも `top1_roi=0.802260922700886`、`ev_top1_roi=0.3623179549852327`、`auc=0.743157455488589` と弱く、architecture 本線候補としても棄却した。

現時点の operational baseline は `current_recommended_serving_2025_latest` とする。

また、`current_recommended_serving_market_aware_prob_race_norm_2025_latest` による first market-aware probability-path candidate の representative / actual-date read も完了した。representative evaluate は `auc=0.8410248775`, `top1_roi=0.7893301105` だったが、`ev_threshold_1_0_bets=1` で stability guardrail は `probe_only` だった。さらに `2025-09-06/07/13/14/20/21/27/28` の fresh actual-date compare では baseline `34 bets / total net -13.6`、bounded sidecar `35 bets / total net -9.9` に対し candidate は `0 bets / total net 0.0` となり、compare surface 自体を失った。したがってこの first candidate は `reject` とし、JRA next action は Stage 4 bounded reintegration 再開ではなく次の architecture child issue 再定義へ戻す。

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

### 4.17 latest eval mainline の formal refresh 完了

- `current_best_eval_2025_latest` を latest revision gate wrapper で再実行すると、`value_blend` family 固有の component artifact suffix 不整合、historical runtime config 依存、`wf_feasibility_diag` の `--profile` 非互換が順に顕在化した。
- そこで wrapper に `--skip-train`、`--train-artifact-suffix`、`--evaluate-model-artifact-suffix` を通し、missing runtime config を復元し、`run_wf_feasibility_diag.py` を profile 解決対応に揃えた。
- その結果、`r20260405_2025latest_refresh_reuse_runtimecfg_wfprofilefix` は `pass / promote` で整合し、`AUC=0.8364`、`logloss=0.2048`、`top1_roi=0.7994`、`nested WF weighted test ROI=0.9157`、`formal_benchmark_weighted_roi=0.8372`、`formal_benchmark_feasible_fold_count=5/5`、`formal_benchmark_bets_total=2364` を確認した。
- この run は baseline 置換の根拠ではなく、latest 2025 の evaluation mainline reference が formal artifact まで再現可能であることを示す evidence として扱う。

### 4.18 JRA market-aware probability-path first candidate の reject

- [../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json](../artifacts/reports/evaluation_summary_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_19cd9b89087d0447_wf_off_nested.json) により、`current_recommended_serving_market_aware_prob_race_norm_2025_latest` の representative evaluate は `auc=0.8410248775`, `logloss=0.2026480271`, `top1_roi=0.7893301105` を返した。
- ただし `ev_threshold_1_0_bets=1` で stability guardrail は `probe_only` となり、representative support quality は mainline promotion judgment を支えなかった。
- さらに [../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_base_vs_cand.json](../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_base_vs_cand.json) と [../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_sidecar_vs_cand.json](../artifacts/reports/serving_smoke_compare_sep25_market_aware_prob_v1_sidecar_vs_cand.json) により、`2025-09-06/07/13/14/20/21/27/28` の actual-date compare では candidate が `8/8` 日すべて `policy_bets=0` になったことを確認した。
- 集計では baseline が `34 bets / total net -13.6`、bounded sidecar が `35 bets / total net -9.9` に対して、market-aware probability-path candidate は `0 bets / total net 0.0` だった。
- `differing_score_source_dates=[]`, `differing_policy_dates=[]` のため、問題は family 切替ではなく same compare surface 上で support が zero へ痩せたことにある。
- したがって first market-aware probability-path candidate は `reject` とし、次の JRA action は Stage 4 bounded reintegration 再開ではなく、prediction foundation か broader architecture branch の次 child issue を 1 本だけ再定義することに固定する。

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

### M37a. latest eval mainline formal refresh の再現導線整備

- `scripts/run_netkeiba_latest_revision_gate.py` に reuse-compare pass-through を追加し、`current_best_eval_2025_latest` のような `value_blend` family でも latest wrapper から train skip / artifact reuse を明示できるようにした。
- `run_wf_feasibility_diag.py` は `--profile` を受けて config / data / feature config を解決できるように揃え、revision gate との CLI 契約を一致させた。
- `r20260405_2025latest_refresh_reuse_runtimecfg_wfprofilefix` により latest eval mainline reference の revision gate / promotion gate が `pass / promote` で整合し、docs からも再開できる状態になった。

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
- `run_local_evaluate.py` を追加し、既存 `run_evaluate.py` の versioned output を再利用しつつ、local-only 側では `evaluation_local_nankan_pointer.json` から evaluation artifact lineage を辿れるようにした。historical `local_nankan` は trust blocker 時に pointer 自体へ `status=blocked_by_trust` を残し、child evaluate は起動しない。

### M41. local-only orchestration manifest の追加

- `run_local_feasibility_manifest.py` を追加し、readiness snapshot、data validation、feature gap、evaluation pointer を fail-fast で直列実行できるようにした。historical `local_nankan` の evaluation 段が trust gate で止まる場合は top-level も `status=evaluation_blocked_by_trust` で閉じる。
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

### M57. mixed left recovery に preflight blocker を繋いだ

- `run_mixed_universe_left_gap_audit.py` を更新し、local lineage に command preview が無い場合でも `data_preflight_payload` や benchmark gate 由来の blocker を gap audit に残せるようにした。
- `run_mixed_universe_left_recovery_plan.py` も更新し、`populate_primary_raw_dir` のような upstream blocker を手動 step として plan に保持するようにした。
- `run_mixed_universe_status_board.py` では gap audit blocker の `recommended_action` と `error_code` を highlights に含め、board から raw 未配置の根本原因へ直接降りられるようにした。

### M58. mixed gap audit の local lineage 解決を自動化した

- `run_mixed_universe_left_gap_audit.py` は `--left-lineage-manifest` を省略しても、numeric compare の readiness manifest から `left_lineage_manifest` と local public snapshot を辿って local revision lineage を自動解決するようにした。
- これにより `r20260328_reference_bridge` のように mixed revision 名と local public snapshot revision 名がズレるケースでも、手で lineage path を渡さず gap audit を再生成できるようになった。

### M59. mixed alias 解決を readiness / compare / schema / numeric compare へ広げた

- `run_mixed_universe_readiness.py` と `run_mixed_universe_compare.py` は、mixed revision 名から local public snapshot / local revision lineage を自動探索し、snapshot 内の `lineage_manifest` も使って left input を解決するようにした。
- `run_mixed_universe_schema.py` と `run_mixed_universe_numeric_compare.py` も、exact revision 名が無いときは同じ universe の既存 mixed manifest を fallback で拾うようにした。
- これにより `r20260328_reference_bridge` 系は readiness -> compare -> schema -> numeric compare まで既定引数だけで再生成できるようになった。

### M60. mixed alias 解決 helper を共通化した

- `src/racing_ml/common/mixed_artifacts.py` を追加し、revision prefix、latest manifest fallback、local snapshot-lineage 自動解決を共通 helper としてまとめた。
- `run_mixed_universe_readiness.py`, `run_mixed_universe_compare.py`, `run_mixed_universe_left_gap_audit.py`, `run_mixed_universe_status_board.py`, `run_mixed_universe_schema.py`, `run_mixed_universe_numeric_compare.py`, `run_mixed_universe_recovery.py` はこの helper を使うように更新し、alias 規則の drift を減らした。

### M61. local public snapshot の compare contract を resolved revision に揃えた

- `run_local_public_snapshot.py` は、lineage を読めた場合の `compare_contract` を入力 `--revision` alias ではなく resolved lineage revision と実 `public_snapshot` path から組み立てるようにした。
- これにより `reference_bridge` のように mixed revision 名と local snapshot revision 名がズレるケースでも、snapshot payload 自体が正しい left-side anchor を返すようになった。

### M62. local public snapshot の planned contract も実 output path に揃えた

- `run_local_public_snapshot.py` の dry-run planned payload でも、`compare_contract.local_only_public_snapshot` は既定ファイル名の組み立てではなく実 `--output` path を返すようにした。
- これで custom output を使う smoke / validation 時も、planned と completed で contract の path 規則がズレなくなった。

### M63. mixed manifest に resolved left revision を持たせた

- `run_mixed_universe_readiness.py` と `run_mixed_universe_compare.py` は、`revision` を mixed alias のまま維持しつつ、実際に読んだ local snapshot/lineage の revision を `resolved_left_revision` に出すようにした。
- `run_mixed_universe_status_board.py` と `run_mixed_universe_recovery.py` も同じ値を引き継ぐようにして、status board と recovery manifest から left-side の実体 revision を直接追えるようにした。

### M64. downstream mixed manifests にも revision bridge を通した

- `run_mixed_universe_schema.py`, `run_mixed_universe_numeric_compare.py`, `run_mixed_universe_numeric_summary.py`, `run_mixed_universe_left_gap_audit.py`, `run_mixed_universe_left_recovery_plan.py` も upstream manifest から `requested_revision` と `resolved_left_revision` を引き継ぐようにした。
- これにより alias revision の mixed chain は、schema 以降の downstream artifacts でも left-side の実体 revision を失わない。

### M65. status board と recovery に resolved left artifact を明示した

- `run_mixed_universe_status_board.py` は `resolved_left_source_kind` と `resolved_left_artifact` を top-level と highlights に出し、現在どの left-side artifact を読んでいるかを 1 本で判断できるようにした。
- `run_mixed_universe_recovery.py` も同じ値を保持し、recovery manifest 単体から left-side の参照元 snapshot を追えるようにした。

### M66. readiness と compare にも resolved left source 情報を揃えた

- `run_mixed_universe_readiness.py` と `run_mixed_universe_compare.py` も `resolved_left_source_kind` と `resolved_left_artifact` を top-level に持つようにした。
- これにより status board まで進まなくても、入口 manifest の時点でどの left-side artifact を参照したかを確認できる。

### M67. downstream mixed manifests にも resolved left source 情報を通した

- `run_mixed_universe_schema.py`, `run_mixed_universe_numeric_compare.py`, `run_mixed_universe_numeric_summary.py`, `run_mixed_universe_left_gap_audit.py`, `run_mixed_universe_left_recovery_plan.py` も `resolved_left_source_kind` と `resolved_left_artifact` を upstream から引き継ぐようにした。
- これにより mixed chain のどの停止点でも、left-side の実 revision だけでなく参照元 kind/path まで追えるようになった。

### M68. 要約部にも left source 情報を載せた

- `run_mixed_universe_numeric_summary.py` の `promote_safe_summary`、`run_mixed_universe_left_gap_audit.py` の `summary`、`run_mixed_universe_left_recovery_plan.py` の `summary` も `requested_revision` と `resolved_left_*` を持つようにした。
- `run_mixed_universe_status_board.py` の `phase_summaries` も同じ情報を段ごとに持つようにして、top-level を読まずに要約だけ見ても left-side の参照元を追えるようにした。

### M69. status board の public snapshot 行も bridge 情報で埋めた

- `run_mixed_universe_status_board.py` は `public_snapshot` payload 自身に `requested_revision` や `resolved_left_*` が無くても、board 文脈から `phase_summaries` の同項目を補完するようにした。
- これで status board の全 phase 行を同じキー集合で読めるようになり、`public_snapshot` 行だけ `null` になる穴をなくした。

### M70. planned status board も通常 payload と同じ読み口に揃えた

- `run_mixed_universe_status_board.py --dry-run` は `requested_revision`, `resolved_left_*`, `read_order`, `current_phase`, `phase_summaries`, `highlights` を含む planned payload を返すようにした。
- これにより manifest がまだ 1 つも無い段階でも、operator は expected phase path と不足段を通常 payload と同じ shape で確認できる。

### M71. local artifact fallback を universe-aware に締めた

- `mixed_artifacts.py` の local public snapshot / local revision lineage 解決は、revision prefix wildcard と global wildcard の両方で payload の `universe` 一致を確認するようにした。
- これにより未知 universe や将来の別 local universe で dry-run したときに、既存 `local_nankan` artifact を誤って left-side input として拾う cross-universe 汚染を防げる。

### M72. readiness planned payload も completed と同じ読み口へ寄せた

- `run_mixed_universe_readiness.py --dry-run` は left input が未生成でも `checks`, `left_summary`, `compare_command_preview` を含む planned payload を返すようにした。
- これにより mixed compare 入口の operator は、left snapshot/lineage 不足時でも completed payload と同じ shape で不足条件と次の compare コマンドを確認できる。

### M73. recovery plan planned payload も summary 付きに揃えた

- `run_mixed_universe_left_recovery_plan.py --dry-run` は gap audit 未生成でも `read_order`, `summary`, `plan_steps=[]` を含む planned payload を返すようにした。
- これにより recovery plan 入口でも、operator は completed payload と同じ shape で「まだ gap audit が必要」という状態を読める。

### M74. numeric summary planned payload も promote-safe summary 付きに揃えた

- `run_mixed_universe_numeric_summary.py --dry-run` は numeric compare 未生成でも `read_order` と `promote_safe_summary` を含む planned payload を返すようにした。
- これにより summary 入口でも、operator は completed payload と同じ shape で「まだ numeric compare が必要」という状態を読める。

### M75. gap audit planned payload も summary と blocker 付きに揃えた

- `run_mixed_universe_left_gap_audit.py --dry-run` は numeric compare / lineage 未生成でも `read_order`, `summary`, `lineage_blocker`, `gap_rows=[]` を含む planned payload を返すようにした。
- これにより gap audit 入口でも、operator は completed payload と同じ shape で「まだ compare と lineage が必要」という状態を読める。

### M76. schema planned payload も comparison contract 付きに揃えた

- `run_mixed_universe_schema.py --dry-run` は readiness / compare 未生成でも `read_order`, `comparison_axes`, `metric_rows`, `blocking_context` を含む planned payload を返すようにした。
- これにより schema 入口でも、operator は completed payload と同じ shape で予定している比較行と readiness 不足を読める。

### M77. numeric compare planned payload も summary 付きに揃えた

- `run_mixed_universe_numeric_compare.py --dry-run` は readiness / compare / schema 未生成でも `read_order`, `blocking_context`, `summary`, `row_results=[]` を含む planned payload を返すようにした。
- これにより numeric compare 入口でも、operator は completed payload と同じ shape で upstream 不足と空比較結果を読める。

### M78. pointer compare planned payload も summary と contract 付きに揃えた

- `run_mixed_universe_compare.py --dry-run` は left input 未生成でも `left_summary`, `right_summary`, `comparison_contract` を含む planned payload を返すようにした。
- これにより pointer bridge 入口でも、operator は completed payload と同じ shape で left/right 入力と compare contract を読める。

### M79. recovery planned payload に source board 現在地を写した

- `run_mixed_universe_recovery.py --dry-run` は step 一覧だけでなく、source board 由来の `recommended_action`, `refreshed_board_status`, `refreshed_current_phase`, `refreshed_next_action_source`, `refreshed_highlights` も planned payload に含めるようにした。
- これにより recovery 入口でも、operator は実行前の段階から「いまどこで止まっているか」を recovery manifest 単体で読める。

### M80. local revision lineage の dry-run も completed shape へ寄せた

- `run_local_revision_gate.py --dry-run` は top-level を `status=planned` のまま閉じ、`read_order` と `highlights` を含む planned lineage manifest を返すようにした。
- 同時に `benchmark_gate_payload`, `data_preflight_payload`, `revision_manifest_payload`, `promotion_payload`, `evaluation_pointer_payload` の planned preview も残すようにして、local raw data 未配置や revision gate 未実行の段階でも downstream artifact path と停止候補を completed payload に近い shape で読めるようにした。

### M81. local feasibility の dry-run も completed shape へ寄せた

- `run_local_feasibility_manifest.py --dry-run` は top-level を `status=planned` のまま閉じ、`read_order` と `highlights` を含む planned feasibility manifest を返すようにした。
- 同時に `snapshot_payload`, `validation_payload`, `feature_gap_payload`, `evaluation_payload` の planned preview も残すようにして、snapshot / validation / feature gap / evaluation 未実行の段階でも downstream artifact path と読み順を completed payload に近い shape で読めるようにした。

### M82. revision gate 本体の dry-run も planned 契約へ揃えた

- `run_revision_gate.py --dry-run` は `status=dry_run` ではなく `status=planned` を返すようにし、`read_order`, `current_phase`, `recommended_action`, `highlights` も持つ planned manifest に揃えた。
- step preview には train artifact suffix、evaluation manifest/summary、WF summary、promotion report の output preview も残すようにして、wrapper を介さない formal revision gate 入口でも downstream artifact を同じ読み口で追えるようにした。

### M83. latest revision gate wrapper の dry-run も planned 契約へ揃えた

- `run_netkeiba_latest_revision_gate.py --dry-run` は readiness snapshot を実行せず、top-level を `status=planned` のまま閉じる wrapper manifest を返すようにした。
- 同時に `read_order`, `current_phase`, `recommended_action`, `highlights`, readiness preview, revision gate preview, `revision_gate_artifacts` の planned path を残すようにして、2025 latest wrapper 入口でも readiness と downstream formal artifacts を completed payload に近い shape で読めるようにした。

### M84. local public snapshot の dry-run も summary preview 付きに揃えた

- `run_local_public_snapshot.py --dry-run` は `compare_contract` だけでなく `readiness`, `promotion_summary`, `benchmark_gate_summary`, `evaluation_summary`, `highlights` も含む planned payload を返すようにした。
- これにより lineage 未生成の段階でも、public-facing local bridge 入口だけで readiness と downstream mixed compare anchor を completed payload に近い shape で読めるようにした。

### M85. mixed recovery の dry-run に read_order と board preview を足した

- `run_mixed_universe_recovery.py --dry-run` は source board 由来の `refreshed_*` を写すだけでなく、recovery 自身の `read_order`, `current_phase`, `next_action_source`, `status_board_preview`, `highlights` も返すようにした。
- これにより recovery 入口でも、operator は source board の停止点だけでなく「この recovery がどの順で何を再生成するか」を planned manifest 単体で completed payload に近い shape で読めるようにした。

### M86. mixed readiness の payload に current_phase と highlights を足した

- `run_mixed_universe_readiness.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより mixed compare 前の入口でも、operator は不足 check の列挙だけでなく「いま何が blocker で、次に何をすべきか」を readiness manifest 単体で completed payload に近い shape で読めるようにした。

### M87. mixed compare の payload に current_phase と highlights を足した

- `run_mixed_universe_compare.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより pointer bridge 入口でも、operator は left/right summary だけでなく「いま compare がどの anchor を読んでいて、次に何をすべきか」を compare manifest 単体で completed payload に近い shape で読めるようにした。

### M88. status board の phase 行にも blocker 要約を写した

- `run_mixed_universe_status_board.py` は各 `phase_summaries` 行に `current_phase`, `error_code`, `highlights` も写すようにした。
- これにより board 入口でも、operator は下流 manifest を個別に開かなくても各 phase の blocker 要約を同じ一覧のまま追えるようにした。

### M89. schema の payload に current_phase と highlights を足した

- `run_mixed_universe_schema.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより schema 入口でも、operator は comparison axes だけでなく readiness blocker と次アクションを manifest 単体で completed payload に近い shape で読めるようにした。

### M90. numeric compare の payload に current_phase と highlights を足した

- `run_mixed_universe_numeric_compare.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより numeric compare 入口でも、operator は row summary だけでなく readiness/schema blocker と次アクションを manifest 単体で completed payload に近い shape で読めるようにした。

### M91. left gap audit の payload に current_phase と highlights を足した

- `run_mixed_universe_left_gap_audit.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより gap audit 入口でも、operator は missing-left row 要約だけでなく lineage blocker と次アクションを manifest 単体で completed payload に近い shape で読めるようにした。

### M92. numeric summary の payload に current_phase と highlights を足した

- `run_mixed_universe_numeric_summary.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより numeric summary 入口でも、operator は promote-safe summary だけでなく readiness/schema blocker と次アクションを manifest 単体で completed payload に近い shape で読めるようにした。

### M93. left recovery plan の payload に current_phase と highlights を足した

- `run_mixed_universe_left_recovery_plan.py` は planned / completed の両方で `current_phase` と `highlights` を返すようにした。
- これにより recovery plan 入口でも、operator は deduplicated steps だけでなく upstream blocker と次アクションを manifest 単体で completed payload に近い shape で読めるようにした。

### M94. public benchmark reference に top-level blocker summary を足した

- `run_public_benchmark_reference.py` は `read_order`, `current_phase`, `recommended_action`, `blocker_summary`, `highlights` を返すようにした。
- これにより right-side baseline 入口でも、operator は promotion / evaluation artifact を個別に開かなくても decision, stability, blocking reasons を manifest 単体で読めるようにした。

### M95. local feasibility の全状態 payload に current_phase を足した

- `run_local_feasibility_manifest.py` は dry-run だけでなく completed / failed でも `current_phase` と `highlights` を返すようにした。
- これにより local-only feasibility 入口でも、operator は snapshot / validation / feature gap / evaluation のどこで止まったかを top-level だけで読めるようにした。

### M96. serving profile compare に top-level 停止点要約を足した

- `run_serving_profile_compare.py` は compare manifest に `read_order`, `current_phase`, `recommended_action`, `highlights` を返すようにした。
- これにより serving compare 入口でも、operator は left/right smoke, compare, bankroll sweep, dashboard のどこで止まったかを manifest 単体で読めるようにした。

### M97. netkeiba benchmark gate に top-level 停止点要約を足した

- `run_netkeiba_benchmark_gate.py` は gate manifest に `read_order`, `current_phase`, `highlights` を返すようにした。
- これにより netkeiba / universe-aware benchmark gate 入口でも、operator は preflight not-ready、snapshot blocker、train/evaluate failure、completed のどこで止まったかを manifest 単体で読めるようにした。

### M98. netkeiba wait-then-cycle に top-level 停止点要約を足した

- `run_netkeiba_wait_then_cycle.py` は wait manifest に `read_order`, `current_phase`, `recommended_action`, `highlights` を返すようにした。
- これにより lock 待機、timeout、handoff backfill failure、post-cycle gate not-ready/failed のどこで止まったかを wait manifest 単体で読めるようにした。

### M99. promotion gate に top-level 停止点要約を足した

- `run_promotion_gate.py` は gate report に `read_order`, `current_phase`, `recommended_action`, `highlights` を返すようにした。
- これにより formal promotion decision 入口でも、pass/promote、block/hold、入力不備の hard failure のどれでも report 単体で現在地を読めるようにした。

### M100. serving compare dashboard summary に top-level 停止点要約を足した

- `run_serving_compare_dashboard.py` は summary JSON に `status`, `read_order`, `current_phase`, `recommended_action`, `highlights` を返すようにした。
- これにより serving validation の dashboard 入口でも、completed と入力不備の hard failure のどちらでも summary 単体で現在地を読めるようにした。

### M101. serving compare aggregate summary に top-level 停止点要約を足した

- `run_serving_compare_aggregate.py` は aggregate summary JSON に `status`, `read_order`, `current_phase`, `recommended_action`, `highlights` を返すようにした。
- これにより window 横断の serving validation 入口でも、completed と入力不備の hard failure のどちらでも aggregate summary 単体で現在地を読めるようにした。

### M102. local_nankan ID 準備の seed 入口を追加した

- `configs/crawl_local_nankan_template.yaml` を追加し、local crawler の出力先、seed file、target ごとの ID/output 契約を初版として固定した。
- `run_prepare_local_nankan_ids.py` を追加し、operator が用意した seed CSV から `race_ids.csv` と `horse_keys.csv` を生成できるようにした。
- project の `.venv` で最小 smoke を通し、`data/external/local_nankan/ids/race_ids.csv` と `horse_keys.csv` の生成導線を確認した。

### M103. local_nankan collect の planned/blocked 入口を追加した

- `src/racing_ml/data/local_nankan_collect.py` と `run_collect_local_nankan.py` を追加し、target ごとの ID 件数、output path、manifest path を planned shape で出せるようにした。
- provider 未実装の段階では blocked manifest を返し、次アクションを `implement_source_provider` として残すようにした。
- これにより Phase 0 は `prepare ids -> collect plan` まで既存 netkeiba crawler に近い骨格で進められるようになった。

### M104. local_nankan backfill の planned/blocked 入口を追加した

- `src/racing_ml/data/local_nankan_id_prep.py` へ seed ベースの ID 準備ロジックを移し、`run_prepare_local_nankan_ids.py` は薄い CLI へ整理した。
- `src/racing_ml/data/local_nankan_backfill.py` と `run_backfill_local_nankan.py` を追加し、1 cycle 分の `prepare -> collect` を backfill manifest にまとめられるようにした。
- provider 実装前でも Phase 0 は `prepare ids -> collect plan -> backfill plan` まで一通り manifest 化できる状態になった。

### M105. local_nankan primary raw 昇格の bridge を追加した

- `src/racing_ml/data/local_nankan_primary.py` と `run_materialize_local_nankan_primary.py` を追加し、external の `race_result/racecard/pedigree` から `data/local_nankan/raw` 向け primary CSV を組み立てられるようにした。
- `race_result` を必須 source、`race_card` と `pedigree` を optional enrichment として扱い、source 欠落時は `status=not_ready` と `recommended_action=populate_external_results` を返すようにした。
- これにより Phase 0 は crawler/provider 実装と独立に、benchmark preflight の `primary_dataset_missing` を解消する raw materialize 工程まで明示的に持てるようになった。

### M106. local benchmark/revision から primary raw materialize を呼べるようにした

- `run_local_benchmark_gate.py` に `--materialize-primary-before-gate` を追加し、preflight の前に `run_materialize_local_nankan_primary.py` を呼べるようにした。
- materialize が `status=not_ready` を返した場合でも wrapper は benchmark gate を続行し、最終的な blocker は従来どおり preflight / benchmark manifest に残すようにした。
- `run_local_revision_gate.py` からも同フラグと source path / materialize manifest path を渡せるようにし、lineage 側でも primary raw 昇格の試行結果を artifact と planned payload に含められるようにした。

### M107. local_nankan の race_result_keys 補助表に racecard fallback を追加した

- `configs/data_local_nankan.yaml` と smoke 用 config の `local_nankan_race_result_keys` に racecard search dir を追加し、results 側が `horse_key` 欠落で invalid schema でも racecard 側の join key から補助表を解決できるようにした。
- これにより primary raw 昇格を hook した benchmark preflight は `primary_dataset_missing` の次で止まるのではなく、実際の補助列不足か readiness 本体まで段階を進めやすくなった。

### M108. local_nankan backfill から primary raw materialize を追えるようにした

- `run_backfill_local_nankan.py` と `src/racing_ml/data/local_nankan_backfill.py` に `--materialize-after-collect` を追加し、backfill cycle ごとに `materialize_summary` を残せるようにした。
- provider 未実装で collect が blocked の段階でも、既存 external outputs から primary raw を materialize できた場合は backfill manifest を `status=partial`, `current_phase=materialized_primary_raw`, `recommended_action=run_local_preflight` として返すようにした。
- これにより Phase 0 の 4 工程は gate wrapper だけでなく backfill 入口からも 1 本の manifest で追えるようになった。

### M109. local_nankan backfill から benchmark までの handoff wrapper を追加した

- `run_local_backfill_then_benchmark.py` を追加し、backfill を `--materialize-after-collect` 付きで回した後、`current_phase=materialized_primary_raw` に到達した場合のみ local benchmark gate へ handoff する wrapper を作った。
- wrapper manifest は `read_order`, `current_phase`, `recommended_action`, `highlights` を持ち、backfill 側で止まったのか benchmark 側で止まったのかを 1 本で読めるようにした。
- smoke local_nankan config では `prepare -> collect -> materialize -> preflight -> snapshot -> benchmark` が 1 コマンドで completed まで進むことを確認した。

### M110. local revision lineage から backfill handoff を呼べるようにした

- `run_local_revision_gate.py` に `--backfill-before-benchmark` を追加し、benchmark 前段で `run_local_backfill_then_benchmark.py` を使えるようにした。
- lineage manifest は `backfill_wrapper_manifest` と `backfill_manifest` を artifacts に含め、`backfill_handoff_payload` / `backfill_summary_payload` も保持できるようにした。
- dry-run では revision lineage 側でも `prepare -> collect -> materialize -> benchmark -> revision -> evaluation_pointer` の planned shape を 1 本で確認できるようになった。

### M111. local public snapshot でも backfill handoff を要約できるようにした

- `run_local_public_snapshot.py` に `backfill_handoff_summary` の抽出を追加し、local revision lineage が `--backfill-before-benchmark` を使っている場合は public snapshot 側でも handoff 状態を要約できるようにした。
- public snapshot の `current_phase` と `recommended_action` も upstream lineage の planned / blocked / failed に追従するようにし、`lineage_planned` のような停止点を top-level から読めるようにした。
- smoke では、planned local revision lineage から生成した public snapshot が `current_phase=lineage_planned`, `recommended_action=run_local_revision_gate`, `backfill_handoff_summary.status=planned` を返すことを確認した。

### M112. mixed readiness / status board でも local handoff blocker を読めるようにした

- `run_mixed_universe_readiness.py` が left public snapshot の `backfill_handoff_summary` を `left_summary` と highlights に反映するようにした。
- `run_mixed_universe_status_board.py` は public snapshot の non-terminal `current_phase` を incomplete と見なすようにし、さらに explicit artifact path 指定も受けられるようにした。
- smoke では、planned local public snapshot を left input にすると readiness が `left_readiness_blocked` を返し、highlights に upstream handoff `status=planned, phase=planned` を出し、status board も `status=partial`, `current_phase=public_snapshot` で止まることを確認した。

### M113. mixed gap audit / recovery plan でも local handoff blocker を first step にした

- `run_mixed_universe_left_gap_audit.py` は local revision lineage に `backfill_handoff_payload` がある場合、benchmark/preflight blocker より先に upstream handoff blocker を `lineage_blocker` へ採用するようにした。
- `run_mixed_universe_left_recovery_plan.py` も、その blocker を downstream command preview より前の manual first step として並べ、top-level `recommended_action` も `run_local_backfill_then_benchmark` を返すようにした。
- synthetic smoke では gap audit が `current_phase=local_backfill_then_benchmark`, `recommended_action=run_local_backfill_then_benchmark` を返し、recovery plan も `plan_steps[0].step=run_local_backfill_then_benchmark` を返すことを確認した。

### M114. mixed recovery wrapper でも local handoff blocker を first step にした

- `run_mixed_universe_recovery.py` は source board が handoff-aware recovery plan を指している場合、local revision lineage の前に `local_backfill_then_benchmark` step を明示的に差し込むようにした。
- これにより recovery manifest 自体の highlights も、単なる lineage rerun ではなく upstream local handoff から再開することを top-level で読めるようになった。
- synthetic smoke では explicit tmp board が `recommended_action=run_local_backfill_then_benchmark` を返すとき、recovery dry-run も first step を `local_backfill_then_benchmark` に切り替えることを確認した。

## 6. 実行中の優先事項

`current_tighter_policy_search_candidate_2025_latest` の `0.03/80` formalization は M17 で完了した。続いて seasonal / recent-heavy の運用境界整理、latest compare artifact map、actual-date compare 再開導線の同期監査、地方競馬 feasibility の設計チェックリスト・artifact 方針・benchmark 完了条件・payload schema・CLI 引数契約・step/failure taxonomy の具体化、既存 `netkeiba_*` snapshot / gate への universe-aware 契約実装、local-only snapshot / gate 雛形の追加、local-only integrity / feature gap / evaluation 入口の追加、local-only orchestration manifest の追加、および local-only revision lineage の追加まで完了した。

さらに `current_best_eval_2025_latest` の latest formal refresh も、reuse-compare 前提の wrapper / revision gate / promotion gate まで end-to-end で整合した。latest mainline の formal pass は current baseline と evaluation reference の両側で再現可能になっている。

以後の active priority は、public / internal docs の定期点検に加えて、JRA mainline を prediction foundation / market deviation / execution policy に分離して compare surface を固定することに絞る。

### P1. docs の定期点検

目的:

- ここまで積み上げた role / artifact / candidate order の説明が、今後の更新で再び発散しないように保つ。

やること:

1. roadmap の next candidates が active priority と矛盾しないか定期的に点検する。
2. public / internal docs の current reading が新しい更新でずれていないかを巡回確認する。
3. `.vscode/` のようなローカル設定を commit 対象へ混ぜない運用を継続する。

完了条件:

- docs の current reading が少ない往復で保守できること。

### P2. JRA track split の baseline freeze

目的:

- current JRA mainline を compare surface として凍結し、prediction foundation / market deviation / execution policy を別 track で扱える状態へ戻す。

やること:

1. snapshot tag 候補 `strategy-baseline-20260418-pre-track-split` の anchor commit を基準に baseline freeze の commit batch を固める。
1. baseline freeze tag `strategy-baseline-20260418-pre-track-split` は `2245ed32d1b25a5a38c46501585273c81cdcf4ab` に固定済みとする。
2. [issue_library/next_issue_model_architecture_rebuild_track_split.md](issue_library/next_issue_model_architecture_rebuild_track_split.md) を親 draft として、cross-track compare の禁止境界を維持する。
3. `market_deviation` は [issue_library/next_issue_jra_market_deviation_formal_candidate.md](issue_library/next_issue_jra_market_deviation_formal_candidate.md) を first formal candidate read として保持し、mainline benchmark judgement へ直接混ぜない。
4. LightGBM alpha challenger compare `r20260418_jra_lightgbm_alpha_challenger_v1` まで完了し、current read は `CatBoost incumbent / LightGBM contrast challenger` として保持する。

完了条件:

- baseline freeze の tag / artifact pointer が確定していること。
- JRA の next compare が policy tuning ではなく track 固有 issue として読めること。

### P3. JRA market integration redesign roadmap

目的:

- 「市場をちゃんと取り入れる」設計見直しを、連想的な tweak 列ではなく段階ロードマップで進める。

operating rules:

1. JRA 本線の新 experiment は、必ずこの roadmap のどれか 1 stage に紐づける。
2. 各 run 後は `advance / hold / reject` のいずれかで stage 判定を閉じる。
3. stage をまたぐ論点を 1 issue に混ぜない。
4. policy-only tuning は Stage 3 以前に戻らない限り主線へ戻さない。

stage plan:

1. Stage 0: compare surface freeze
	- baseline tag / artifact pointer / track split rule を固定する
	- status: 完了

2. Stage 1: market signal diagnosis
	- objective: current `market_deviation` family がどの trade-off front を持つかを測る
	- success: base / corr-side / coverage-side の contrast を artifact で説明できる
	- stop: 単純 merge でも front improvement が出ない
	- status: 完了
	- current read:
	  - base challenger `r20260418_jra_lightgbm_alpha_challenger_v1`: `alpha_pred_corr=0.1831187046`, `positive_signal_rate=0.0064266667`
	  - corr-side variant `r20260418_jra_lightgbm_alpha_leaf40_v1`: `0.2079891249`, `0.0061666667`
	  - coverage-side variant `r20260418_jra_lightgbm_alpha_clip6_v1`: `0.1844760397`, `0.0069466667`
	  - merge candidate `r20260418_jra_lightgbm_alpha_leaf40_clip6_v1`: `0.1740577483`, `0.0062400000` で reject

3. Stage 2: market representation redesign
	- objective: 現 family の probe では埋まらない gap を、target / architecture / market feature routing の再設計で扱う
	- candidate themes:
	  - target redesign: `market_deviation` 単独ではなく market residual / pairwise / race-normalized target の再定義
	  - architecture redesign: classification と market branch の late fusion ではなく explicit market-aware branch を持つ構成
	  - feature routing redesign: 市場由来 signal を model input 側で独立管理する構成
	- success: 次の primary issue を 1 本に絞り、現 probe 群では埋まらない残差を明文化できる
	- stop: 1 issue に target / architecture / policy を同時に混ぜないと書けない
	- status: 完了
	- primary issue:
	  - [issue_library/next_issue_jra_market_deviation_target_redefinition.md](issue_library/next_issue_jra_market_deviation_target_redefinition.md)
	- result:
	  - diagnostic artifact `market_deviation_target_diagnostic_model_lightgbm_alpha_cpu_diag.json` で current target の clip hit を確認した
	  - race-normalized residual candidate `r20260418_jra_lightgbm_alpha_race_norm_v1` は `alpha_pred_corr=0.2232080870`, `positive_signal_rate=0.2132800000` を確認した
	  - decision: `advance`

4. Stage 3: formal candidate rebuild
	- objective: Stage 2 で選んだ設計 1 本を full artifact で formal compare する
	- entry condition: Stage 2 の primary issue が 1 本に固定されている
	- status: 完了
	- current read:
	  - current best candidate は `r20260418_jra_lightgbm_alpha_race_norm_v1`
	  - formal candidate reference は [issue_library/next_issue_jra_market_deviation_formal_candidate.md](issue_library/next_issue_jra_market_deviation_formal_candidate.md) を正本にする
	  - decision: `advance`

5. Stage 4: execution policy reintegration
	- objective: Stage 3 で signal source が固定できた後にだけ policy track へ戻す
	- entry condition: market-side candidate が `keep as candidate` 以上で維持できる
	- status: current
	- primary issue:
	  - [issue_library/next_issue_jra_market_deviation_policy_reintegration.md](issue_library/next_issue_jra_market_deviation_policy_reintegration.md)
	- current read:
	  - bounded sidecar config と profile は追加済み
	  - representative compare では sidecar `auc=0.8384159536`, `logloss=0.2037916115` と baseline `auc=0.8383868804`, `logloss=0.2039192355` はほぼ同等で、`top1_roi` は `0.7973504431 < 0.7989290990` だった
	  - September difficult window の fresh actual-date read では baseline reference `32 bets / total net -27.3 / pure bankroll 0.2959` に対して sidecar `35 bets / -4.7458333333 / 0.4352668690` を確認した
	  - `alpha_weight: 0.05 -> 0.02` の bounded follow-up も実施したが、`policy_bets=35`, `total_policy_net=-14.6`, `pure_bankroll=0.3314214738` で exposure は減らず、actual-date read は悪化したため reject した
	  - current bounded issue の decision は `hold` であり、serving default は変更しない

current next action:

1. Stage 4 bounded reintegration issue は current read を `hold` のまま凍結し、serving default を変更しない
2. 次の JRA architecture issue は [issue_library/next_issue_jra_market_deviation_market_aware_probability_path.md](issue_library/next_issue_jra_market_deviation_market_aware_probability_path.md) に固定する
3. late-fusion sidecar の blind parameter probe は追加せず、market-aware probability path を 1 measurable hypothesis として先に切る

## 7. 次の候補

### N1. 地方競馬データ拡張の feasibility 深掘り

- universe slug を config / artifact / revision にどう通すかを、必要なら実装前提まで下ろす。
- status board までは入ったので、next は left formal metrics が埋まった時に row 欠損がどこまで解消されるかを追う。
- local データ導入時の選択肢を `local-only / mixed / ensemble` の 3 段で整理し、ensemble を許す条件と rollback 境界を docs 上で固定した。
- これにより、地方データが ROI 向上へ効く可能性を見た場合でも、JRA-only 正本を維持したまま段階的に pilot を切る判断基準を持てる。
- [local_nankan_introduction_plan.md](local_nankan_introduction_plan.md) に、source bootstrap から ensemble pilot までの段階導入計画、判断ゲート、停止条件をまとめた。
- [local_nankan_crawler_design.md](local_nankan_crawler_design.md) に、local crawler の config、CLI、manifest、small backfill smoke までの設計を追加した。

### N2. docs の定期点検

- public / internal docs の重複表現が増えすぎていないかを定期的に点検する。

### N3. JRA market deviation challenger compare

- first formal candidate read `r20260418_jra_catboost_alpha_baseline_refresh_v1` は `keep as candidate` 判定まで整理済みである。
- LightGBM challenger `r20260418_jra_lightgbm_alpha_challenger_v1` も完了し、`alpha_pred_corr` は改善した一方で positive signal coverage が大きく低下したため、current read は incumbent replacement ではなく contrast challenger である。
- 次は current compare rule のまま、`higher corr / lower coverage` の trade-off をどう切るかを別 challenger issue か promotion boundary memo へ落とす。

### N4. JRA LightGBM coverage recovery

- next measurable hypothesis は [issue_library/next_issue_jra_market_deviation_lightgbm_coverage_recovery.md](issue_library/next_issue_jra_market_deviation_lightgbm_coverage_recovery.md) に固定した。
- first candidate は [issue_library/../configs/model_lightgbm_alpha_cpu_diag_leaf40.yaml](../configs/model_lightgbm_alpha_cpu_diag_leaf40.yaml) で、`min_data_in_leaf: 80 -> 40` だけを変える。
- 目的は LightGBM alpha の higher corr を大きく壊さずに positive signal coverage を回復できるかを narrow に判定することである。
- `r20260418_jra_lightgbm_alpha_leaf40_v1` の結果、`alpha_pred_corr=0.2079891249` へ改善した一方で `positive_signal_rate=0.0061666667` と coverage recovery は起きなかったため、この single-lever hypothesis は reject とする。

### N5. JRA LightGBM target-clip compression

- next measurable hypothesis は [issue_library/next_issue_jra_market_deviation_lightgbm_target_clip_compression.md](issue_library/next_issue_jra_market_deviation_lightgbm_target_clip_compression.md) に固定する。
- first candidate は [issue_library/../configs/model_lightgbm_alpha_cpu_diag_clip6.yaml](../configs/model_lightgbm_alpha_cpu_diag_clip6.yaml) で、`target_clip: 8.0 -> 6.0` だけを変える。
- 目的は negative target tail を圧縮して `positive_signal_rate` を戻せるかを narrow に判定することである。
- `r20260418_jra_lightgbm_alpha_clip6_v1` の結果、`alpha_pred_corr=0.1844760397` を維持したまま `positive_signal_rate=0.0069466667` へ小幅改善したため、この hypothesis は `keep as candidate` とする。
- ただし CatBoost incumbent の coverage には依然遠く、clip6 は current LightGBM line の coverage-side contrast variant として扱う。

### N6. JRA LightGBM trade-off merge

- next measurable hypothesis は [issue_library/next_issue_jra_market_deviation_lightgbm_tradeoff_merge.md](issue_library/next_issue_jra_market_deviation_lightgbm_tradeoff_merge.md) に固定する。
- first candidate は [issue_library/../configs/model_lightgbm_alpha_cpu_diag_leaf40_clip6.yaml](../configs/model_lightgbm_alpha_cpu_diag_leaf40_clip6.yaml) で、`min_data_in_leaf: 40` と `target_clip: 6.0` を同時に入れる。
- 目的は leaf40 の corr-side gain と clip6 の coverage-side gain を同時に取り込めるかを narrow に判定することである。
- `r20260418_jra_lightgbm_alpha_leaf40_clip6_v1` の結果、`alpha_pred_corr=0.1740577483`, `positive_signal_rate=0.0062400000` で base challenger と clip6 の両方を下回ったため、この merge hypothesis は reject とする。
- current read は Stage 1 完了であり、次は parameter probe の継ぎ足しではなく Stage 2 の market representation redesign issue を 1 本に固定する。

### N7. JRA market deviation target redesign

- Stage 2 の primary issue は [issue_library/next_issue_jra_market_deviation_target_redefinition.md](issue_library/next_issue_jra_market_deviation_target_redefinition.md) に固定する。
- 次の root-cause 仮説は parameter tuning ではなく、current `market_deviation` target shape が coverage collapse を作っていないかの検証である。
- ここでは race-normalized residual target 1 種だけを first candidate とし、target / architecture / policy を同時に混ぜない。
- diagnostic artifact `market_deviation_target_diagnostic_model_lightgbm_alpha_cpu_diag.json` では `lower_clip_rate=0.03702`, `upper_clip_rate=0.06808` を確認した。
- `r20260418_jra_lightgbm_alpha_race_norm_v1` の結果、`alpha_pred_corr=0.2232080870`, `positive_signal_rate=0.2132800000`, `ev_threshold_1_0_roi=1.1144751854` を確認した。
- current decision は `advance` とし、Stage 2 は完了とする。

### N8. JRA market deviation policy reintegration

- Stage 4 の primary issue は [issue_library/next_issue_jra_market_deviation_policy_reintegration.md](issue_library/next_issue_jra_market_deviation_policy_reintegration.md) に固定する。
- next measurable hypothesis は race-normalized residual alpha を bounded sidecar score source として reintegrate したときに、baseline probability path を壊さずに actual-date / representative compare を改善できるかである。
- ここでは baseline win model の置換や broad policy rewrite は扱わず、alpha sidecar compare 1 本だけを first candidate にする。

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
