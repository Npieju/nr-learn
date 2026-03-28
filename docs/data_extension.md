# 外部データ拡張

## 1. 目的

このプロジェクトでは、JRA 主表だけに依存せず、netkeiba などの外部データを後から安全に追加できる構成を取っている。

目的は次の 3 点である。

1. 主表を壊さずに履歴や補助列を追加する。
2. crawler と学習系を疎結合に保つ。
3. 外部データの品質を benchmark 再実行前に点検できるようにする。

## 2. 基本方針

外部データの追加は、2 種類に分けて扱う。

### 2.1 append_tables

- 履歴行を追加するための経路。
- 主に race result の増補に使う。

### 2.2 supplemental_tables

- 既存行に列を補完するための経路。
- 主に race card、pedigree、owner / breeder などの補助情報に使う。

この 2 つを `configs/data.yaml` で分けて管理することで、行追加と列補完を混同しないようにしている。

## 3. 主な外部ソース

### 3.1 race_result

- 役割:
  - 主表にないレース履歴や補助情報の追加
- 主キー:
  - `race_id + horse_id`

### 3.2 race_card

- 役割:
  - 出走時点で使える枠番、馬番、性齢、斤量、騎手、調教師などの補完
- 主キー:
  - `race_id + horse_id`

### 3.3 pedigree

- 役割:
  - 血統系特徴量の基礎情報の追加
- 主キー:
  - `horse_key`

## 4. `horse_id` と `horse_key`

このプロジェクトでは、`horse_id` と `horse_key` を分けて扱う。

- `horse_id`
  - 行結合のためのキー
- `horse_key`
  - 競走馬を継続追跡するための安定キー

実務上の流れは次のとおりである。

1. `race_result` や `race_card` から `horse_key` を補完する。
2. その後に `horse_key` を使って pedigree を結合する。
3. 履歴特徴は `horse_key` を最優先に使う。

この設計により、最近年帯でも同一馬の履歴接続を比較的安定させやすい。

## 5. 実装の入口

主なコードとスクリプトは次のとおりである。

- loader:
  - [src/racing_ml/data/dataset_loader.py](../src/racing_ml/data/dataset_loader.py)
- crawler:
  - [src/racing_ml/data/netkeiba_crawler.py](../src/racing_ml/data/netkeiba_crawler.py)
- race list:
  - [src/racing_ml/data/netkeiba_race_list.py](../src/racing_ml/data/netkeiba_race_list.py)
- ID 準備:
  - [scripts/run_prepare_netkeiba_ids.py](../scripts/run_prepare_netkeiba_ids.py)
- 収集:
  - [scripts/run_collect_netkeiba.py](../scripts/run_collect_netkeiba.py)
- backfill:
  - [scripts/run_backfill_netkeiba.py](../scripts/run_backfill_netkeiba.py)

## 6. 品質確認

外部データは、入れた後にそのまま学習へ進めない。最低限、次の確認を通す。

### 6.1 データソース整合

- [scripts/run_validate_data_sources.py](../scripts/run_validate_data_sources.py)

確認するもの:

- 必須テーブルの存在
- join key の欠落
- 重複行
- canonical 列への正規化状態

### 6.2 feature gap

- [scripts/run_feature_gap_report.py](../scripts/run_feature_gap_report.py)

確認するもの:

- force-include 特徴の low coverage
- raw 列不足による特徴未生成

### 6.3 coverage snapshot

- [scripts/run_netkeiba_coverage_snapshot.py](../scripts/run_netkeiba_coverage_snapshot.py)
- [scripts/run_netkeiba_benchmark_gate.py](../scripts/run_netkeiba_benchmark_gate.py)

確認するもの:

- crawl 状態
- readiness
- benchmark rerun の可否

## 7. 現在の注意点

### 7.1 pedigree 系はまだ疎い

recent 帯では pedigree / lineage 系の coverage がまだ薄い。列が存在していても、有効サンプルが十分とは限らない。

### 7.2 race ID 発見より quality control が主課題

最新年の race ID を起こす仕組み自体は整っているため、現在の主課題は「どの期間を優先して backfill し、どの時点で benchmark を回すか」の運用判断に移っている。

### 7.3 外部データは benchmark で判断する

新しい外部データを入れても、raw ROI の見かけだけで採用しない。採用判断は [benchmarks.md](benchmarks.md) に沿って行う。

### 7.4 recent-heavy 方針との両立が必要

本番 2026 に近い判断を重視するなら、単にデータ量を増やすだけでは不十分である。

- 学習データは recent regime を優先して切る。
- ただしデータは有限なので、古い年帯を完全に捨てるかどうかではなく、date-based な train window をどう設計するかで調整する。
- 外部データの追加も、latest holdout と actual-date compare を改善するかで評価する。

## 8. 将来拡張: 地方競馬データ

地方競馬データの大規模収集と学習利用は、将来候補として検討可能である。

ただし 2026-03-27 時点の整理では、これは JRA latest baseline を直接強化する次タスクではない。まず「JRA current baseline を詰める仕事」と「地方を別 universe として設計する仕事」を分けて扱う。

期待できる利点は次のとおりである。

- 学習行数が増える。
- 実運用機会が増える。
- 騎手、調教師、血統、馬場差などの一般化に効く可能性がある。

一方で、そのまま JRA 学習へ混ぜる前に次を切り分ける必要がある。

- レース場体系が増え、course / track 系特徴の意味が変わる。
- 頭数分布、賞金体系、開催 cadence、市場傾向が JRA と一致しない。
- race_id / horse_id / horse_key の整備方針を、JRA 系と同じ前提で流用できるとは限らない。
- JRA latest benchmark を改善したいのか、地方も含めた別運用 universe を作りたいのかを先に分ける必要がある。

設計上の前提条件は次の 4 点である。

1. ingestion は JRA と別 source として扱い、同じ readiness / benchmark gate に即接続しない。
2. key 設計は `race_id`、`horse_id`、`horse_key` を JRA と同一前提で流用しない。
3. benchmark は少なくとも一度 JRA-only baseline と別立てで持ち、混合学習を試す場合も差分比較にする。
4. current operational question は JRA latest の baseline / defensive option 整理なので、地方拡張はそれを止めない future option に留める。

feasibility を具体化するときは、少なくとも次の 5 つを別々に判定する。

1. source 境界
  - 地方データを `append_tables` / `supplemental_tables` の延長で扱えるか、それとも JRA とは別 manifest / coverage snapshot / readiness gate を持つべきか。
2. key 境界
  - `race_id`、`horse_id`、`horse_key` を JRA と同一 namespace で持てるか、prefix / source 列 / universe 列で分離すべきか。
3. feature 境界
  - course / track / purse / field-size / cadence の違いで、JRA 由来の feature がそのまま使えるか、universe-aware な feature set を持つべきか。
4. benchmark 境界
  - JRA-only benchmark、地方-only benchmark、mixed benchmark をどの順で作るか。少なくとも最初の 1 本は JRA-only baseline と完全に分離する。
5. rollout 境界
  - 目的が JRA baseline 改善なのか、地方を含む別 serving universe 追加なのかを先に固定する。両方を同時にやらない。

最小の feasibility チェックリストは次である。

- 地方 source 用の raw / interim / processed 配置を JRA と混線しない命名で切れること。
- race / horse / jockey / trainer の key を namespace 付きで保持するか、source 列で分離するかを決めること。
- `run_validate_data_sources.py` と coverage snapshot 相当で、地方 source 単体の readiness を確認できること。
- JRA-only benchmark を固定したまま、地方-only benchmark を別 artifact 系で比較できること。
- mixed 学習を試す前に、地方-only 側で最低限の feature coverage と representative evaluation を確認すること。

逆に、次の形では始めない。

- JRA の `race_id` / `horse_key` 前提をそのまま地方 CSV に当てはめる。
- JRA-only benchmark を更新せずに mixed 学習の成績だけで採否を決める。
- JRA / 地方の course 特徴や開催 cadence の違いを feature 側で吸収しないまま同一 model family へ流し込む。

したがって、現時点の推奨順序は次である。

1. JRA latest で recent-heavy split を比較する。
2. そのうえで地方競馬を別 source / 別 benchmark として取り込む設計を整理する。
3. 混合学習を試す場合も、JRA-only baseline を必ず残して差分で判断する。

設計作業をさらに分解すると、着手順は次が最小である。

1. universe 境界を config にどう表現するかを決める。
2. key namespace と canonical column を地方 source 用にどう正規化するか決める。
3. 地方-only coverage snapshot / benchmark gate の artifact 名と完了条件を決める。
4. そこで初めて地方-only ingest -> feature gap -> benchmark rerun の smoke を切る。
5. mixed 学習は最後に JRA-only baseline 差分として比較する。

地方-only coverage snapshot / benchmark gate を作るなら、artifact 契約は少なくとも次の形で分ける。

1. snapshot artifact
  - JRA 既存の `netkeiba_coverage_snapshot.json` を上書きしない。
  - 例: `artifacts/reports/local_coverage_snapshot.json` あるいは source / universe を含む `artifacts/reports/coverage_snapshot_local_nankan.json`
  - readiness は `benchmark_rerun_ready` のような同種フィールドを持ってよいが、JRA readiness と同じ manifest の一部にはしない。
2. gate manifest
  - JRA 既存の `netkeiba_benchmark_gate_manifest.json` を流用せず、地方-only の gate manifest を別名で持つ。
  - 例: `artifacts/reports/local_benchmark_gate_manifest.json`
  - gate の `status` は JRA gate と同じ語彙でよいが、`configs` と `coverage_summary` には universe 名を明示する。
3. benchmark lineage
  - JRA-only benchmark、地方-only benchmark、mixed benchmark の artifact lineage を分ける。
  - revision slug も `local_...` や `mixed_...` を含め、JRA current baseline 系の slug と衝突させない。

最小の命名ルールとしては、config / snapshot / gate / revision のすべてに同じ universe slug を入れるのが安全である。

- config 例: `configs/data_local_nankan.yaml`
- snapshot 例: `artifacts/reports/coverage_snapshot_local_nankan.json`
- gate manifest 例: `artifacts/reports/benchmark_gate_local_nankan.json`
- revision 例: `r20260328_local_nankan_baseline_smoke`

こうしておくと、JRA-only artifact と mixed artifact を後から grep したときに、source universe を名前だけで追跡できる。

要するに、地方競馬データは「今すぐ JRA 学習へ混ぜる追加データ」ではなく、「別 universe として feasibility を切る将来候補」と読むのが正しい。

## 9. この文書の読み方

- プロジェクト全体像は [project_overview.md](project_overview.md) を参照する。
- システム構造は [system_architecture.md](system_architecture.md) を参照する。