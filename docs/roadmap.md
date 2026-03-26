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

## 3. 現在地

2026-03-26 時点の到達点は次のとおりである。

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

現時点の operational baseline は `current_recommended_serving_2025_latest` とする。

一方、`current_tighter_policy_search_candidate_2025_latest` は latest 2025 regime の formal-support 改善を確認した analysis-first candidate として保持する。現時点では serving default へは昇格させず、次の比較軸を決めるための正式通過候補として扱う。

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

## 6. 実行中の優先事項

### P1. tighter policy candidate の位置づけ確定

目的:

- formal 通過した tighter policy candidate を、analysis-first 候補のまま維持するのか、追加比較を経て serving 候補へ進めるのかを判断できる状態にする。

やること:

1. `current_recommended_serving_2025_latest` と `current_tighter_policy_search_candidate_2025_latest` の役割差を docs 上で明文化する。
2. `0.03/100` を保守的な正式閾値として維持するか、`0.03/80` まで広げて `5/5 feasible folds` を狙うかを比較する。
3. serving default を切り替えずに済む条件と、切り替えるなら必要な actual-date 比較を明文化する。

完了条件:

- tighter policy candidate を analysis-only に留めるか、次の昇格候補にするかの判断材料が roadmap 上で揃っていること。

## 7. 次の候補

### N1. latest baseline の docs 反映拡充

- `benchmarks.md` に `r20260326_tighter_policy_ratio003` の formal result を追記する。
- `project_overview.md` の現状説明に tighter policy candidate の位置づけを反映する。

### N2. latest candidate family の棚卸し

- `_2025_latest` が付く stable family を棚卸しし、operational baseline、seasonal de-risk、analysis-first formal candidate を分ける。

### N3. seasonal runtime policy の明文化

- September window では long-horizon latest を de-risk 候補として扱い、それ以外は baseline を使う運用境界を docs に残す。

## 8. 当面やらないこと

- crawler の追加修正
- 新しい外部データ源の拡張
- latest formal result を見ないままの broad policy rewrite

これらは現状のボトルネックではない。まずは latest baseline を運用導線へ落とし切る。

## 9. 関連文書

- [project_overview.md](project_overview.md)
- [development_flow.md](development_flow.md)
- [benchmarks.md](benchmarks.md)
- [evaluation_guide.md](evaluation_guide.md)
- [serving_validation_guide.md](serving_validation_guide.md)
- [command_reference.md](command_reference.md)