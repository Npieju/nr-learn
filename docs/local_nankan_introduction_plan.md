# local_nankan 導入計画

## 1. 目的

この計画は、地方データを JRA 学習へ安易に混在させず、`local_nankan` を別 universe として安全に導入するための実行順を定める。

狙いは次の 3 点である。

1. 地方データが独立 universe として再現可能かを確認する。
2. JRA-only baseline を壊さずに ROI 改善余地を見極める。
3. mixed または ensemble を検討する場合でも、rollback 可能な導線を維持する。

## 2. 非目的

この計画の初期フェーズでは次をやらない。

1. JRA-only baseline を直接置き換える。
2. local データを JRA training set へ即時マージする。
3. mixed 学習の良さだけで production 採否を決める。
4. crawler 実装前に feature や model family を広く作り替える。

## 3. 前提

既存 repo の前提は次である。

1. `local_nankan` は JRA とは別 universe として扱う。
2. primary raw は `data/local_nankan/raw`、補助 source は `data/external/local_nankan/results`、`racecard`、`pedigree` を使う。
3. 現時点では local 側に crawler はなく、readiness / gate / lineage の枠だけがある。
4. 現行の運用正本は JRA-only baseline である。

## 4. 導入方針

導入順は必ず次の 4 段に固定する。

1. source bootstrap
2. local-only benchmark
3. mixed compare
4. ensemble pilot

`mixed training` は 3 段目以降の比較対象として扱ってよいが、最初の production 候補にしない。

## 5. フェーズ計画

### Phase 0. Source Bootstrap

目的:

- local データを継続取得できる最低限の source 基盤を作る。

やること:

1. 収集対象を固定する。最低限 `race_result`、`race_card`、`pedigree` の 3 target を定義する。
2. source ごとの ID 設計を決める。`race_id`、`horse_id`、`horse_key` を JRA と別 namespace にする。
3. `local_nankan` 用 crawler config、manifest、lock、checkpoint の契約を決める。
4. raw HTML 保存先、CSV 出力先、manifest 出力先を `local_nankan` 名義で分離する。
5. 全面 backfill と日次追補の 2 系統の運用手順を分ける。

完了条件:

1. 全 target を手動または半自動で backfill できる。
2. 収集失敗時に target 単位で再開できる。
3. 出力が `data/local_nankan/raw` と `data/external/local_nankan/*` に揃う。

停止条件:

1. 利用可能な公開 source が不安定で継続取得に向かない。
2. ID 設計が定まらず canonical join の前提を作れない。

### Phase 1. Local-Only Benchmark

目的:

- 地方 universe 単独で benchmark 再現性があるかを確認する。

やること:

1. `data_preflight_local_nankan.json` で primary raw と補助 source の有無を確認する。
2. `run_local_coverage_snapshot.py` で readiness を生成する。
3. `run_local_data_source_validation.py` で key / duplicate / canonical mismatch を点検する。
4. `run_local_feature_gap_report.py` で feature coverage を確認する。
5. `run_local_evaluate.py` と `run_local_benchmark_gate.py` で local-only evaluation と gate を通す。
6. `run_local_revision_gate.py` と `run_local_public_snapshot.py` で lineage を固定する。

完了条件:

1. local-only gate が `completed` まで進む。
2. evaluation が `stability_assessment=representative` を満たす。
3. blocker がある場合でも artifact から停止理由と次アクションが読める。

停止条件:

1. source coverage 不足で benchmark 再現性が成立しない。
2. key 整合や feature coverage の欠陥が構造的で、補助 source を足しても解消しない。

### Phase 2. Mixed Compare

目的:

- JRA-only baseline を維持したまま、地方導入の差分価値を比較する。

やること:

1. `run_mixed_universe_readiness.py` で left/right 前提を確認する。
2. `run_mixed_universe_compare.py` で pointer manifest を生成する。
3. `run_mixed_universe_schema.py` で比較軸を固定する。
4. `run_mixed_universe_numeric_compare.py` と `run_mixed_universe_numeric_summary.py` で優劣と欠損を可視化する。
5. 必要なら `run_mixed_universe_left_gap_audit.py` と recovery plan で不足を埋める。

完了条件:

1. local 側の優位または劣位が numeric compare で説明できる。
2. どの regime、date bucket、course 群で差が出るかを notes に残せる。
3. JRA-only baseline の運用正本は維持される。

停止条件:

1. 差分は見えるが再現性がなく explanation を作れない。
2. local 側のメトリクスが partial のままで比較軸を埋められない。

### Phase 3. Ensemble Pilot

目的:

- JRA-only と local-only を別 lineage のまま保持しつつ、推論段で改善余地があるかを確認する。

やること:

1. ensemble strategy を限定する。初期候補は `rule-based routing`、`confidence gating`、`weighted blend` のいずれかに限る。
2. どの条件で JRA を使い、どの条件で local または blend を使うかを artifact に残す。
3. fallback 時の動作を固定する。最低限 JRA-only へ即時復帰できるようにする。
4. pilot は production lineage と分離した revision / manifest 名で評価する。

完了条件:

1. ensemble の意思決定を再現できる。
2. rollback 手順が 1 回の切り替えで実行できる。
3. JRA-only 単独より改善する条件と、改善しない条件の両方が説明できる。

停止条件:

1. local-only の優位条件を説明できない。
2. routing 条件が複雑化し、運用上の説明責任を満たせない。
3. rollback 境界が壊れる。

## 6. 実装順

最初の実装は次の順で進める。

1. local crawler 設計書を作る。
2. crawl config と CLI wrapper を追加する。
3. manifest / lock / checkpoint 契約を JRA crawler に揃える。
4. small backfill で sample dataset を作る。
5. local-only preflight から public snapshot まで通す。
6. metrics が揃ってから mixed compare を有効化する。
7. 最後に ensemble pilot の設計へ進む。

## 7. 各段の判断ゲート

次の問いに `yes` と答えられない限り、次段へ進まない。

### Gate A. crawler 実装へ進む条件

1. 地方データを別 universe として評価する価値がある。
2. source と利用条件が継続運用に耐える。

### Gate B. local-only benchmark へ進む条件

1. primary raw を継続取得できる。
2. race / horse / pedigree の最低限 join が成立する。

### Gate C. mixed compare へ進む条件

1. local-only gate が completed である。
2. representative evaluation が揃っている。

### Gate D. ensemble pilot へ進む条件

1. mixed compare で local 側の効く条件が説明できる。
2. JRA-only を維持したまま pilot する価値がある。

## 8. リスク

主要リスクは次のとおりである。

1. source 仕様の変化で crawler が壊れる。
2. local 側の key 整備が不十分で validation が恒常的に fail する。
3. JRA と地方の市場構造差が大きく、単純な mixed で悪化する。
4. ensemble の条件分岐が複雑化して運用説明が難しくなる。

## 9. 当面の次アクション

この計画に従う場合、直近の実作業は次の 3 つである。

1. local crawler 設計書を作成する。
2. target ごとの source / ID / output contract を確定する。
3. small backfill の smoke 計画を定義する。