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

地方-only benchmark を「一度回せた」ではなく「最低限完了した」とみなす条件も、JRA とは別に固定しておく必要がある。最小の完了条件は次である。

1. snapshot readiness
  - 地方-only coverage snapshot が生成され、`benchmark_rerun_ready` 相当の readiness を universe 単位で返せること。
  - readiness reason は、少なくとも key 欠落、必須 source 欠落、coverage 不足を分けて読めること。
2. data integrity
  - `run_validate_data_sources.py` 相当の点検で、地方 source 単体の join key / duplicate / canonical column の破綻がないこと。
  - JRA source の整合と地方 source の整合を同じ summary に混ぜず、universe ごとに失敗理由を持つこと。
3. feature readiness
  - 地方-only feature gap report で、必須 feature の coverage 不足が把握できること。
  - ここでは「JRA と同じ feature が全部あること」ではなく、「欠けている feature を artifact で説明できること」を最低条件にする。
4. representative evaluation
  - 地方-only evaluation summary / manifest が生成され、`stability_assessment=representative` を満たすこと。
  - row 数を減らした smoke evaluate しかない段階では、benchmark 完了ではなく readiness 確認に留める。
5. local-only gate
  - 地方-only benchmark gate manifest が `completed` まで進み、evaluation と readiness snapshot の対応が取れていること。
  - mixed 学習前の最初の到達点は、JRA を含まない local-only gate の完了であって、`pass / promote` そのものではない。

mixed compare に進む前の前提条件も、この 5 段の延長で固定しておく。

1. left universe の local-only gate が完了していること。
2. left universe の evaluation pointer から `stability_assessment=representative` を読めること。
3. right 側は JRA public reference として別 artifact / doc から参照し、同一 gate manifest に混ぜないこと。
4. mixed compare の最初の artifact は promote 判定ではなく readiness / pointer manifest に留めること。

その次段の comparison schema も先に固定しておく。

1. promotion 軸では `decision` だけを比較し、baseline 置換可否と混同しない。
2. evaluation 軸では `stability_assessment`, `auc`, `top1_roi`, `ev_top1_roi`, `nested_wf_weighted_test_roi`, `nested_wf_bets_total` を候補にする。
3. support 軸では `formal_benchmark_weighted_roi`, `formal_benchmark_feasible_folds` を候補にする。
4. 最初の schema artifact は numeric compare ではなく「どの値をどこから読むか」の contract に留める。

artifact に残す最低限の判定項目も決めておくとよい。

- `universe`: `local_nankan` のような source slug
- `readiness.benchmark_rerun_ready`
- `readiness.reasons`
- `coverage_summary`: key / pedigree / owner / breeder など universe ごとの主要 coverage
- `integrity_summary`: duplicate / missing key / canonical mismatch
- `evaluation.status` と `evaluation.stability_assessment`
- `gate.status` と `gate.completed_step`

script 仕様レベルへ下ろすなら、payload schema は既存の `run_netkeiba_coverage_snapshot.py` と `run_netkeiba_benchmark_gate.py` に寄せつつ、universe 境界を明示するのが安全である。

地方-only coverage snapshot の最小 schema 例:

```json
{
  "run_context": {
    "config": "configs/data_local_nankan.yaml",
    "tail_rows": 5000,
    "universe": "local_nankan",
    "primary_source_rows_total": 123456
  },
  "coverage": {
    "latest_tail": {},
    "paired_race_subset": {}
  },
  "integrity_summary": {
    "missing_keys": {},
    "duplicate_rows": {},
    "canonical_mismatch": {}
  },
  "readiness": {
    "snapshot_consistent": true,
    "benchmark_rerun_ready": false,
    "recommended_action": "inspect_local_alignment",
    "reasons": []
  }
}
```

地方-only benchmark gate manifest の最小 schema 例:

```json
{
  "started_at": "2026-03-28T00:00:00Z",
  "finished_at": null,
  "status": "running",
  "universe": "local_nankan",
  "configs": {
    "data_config": "configs/data_local_nankan.yaml",
    "model_config": "configs/model_local_baseline.yaml",
    "feature_config": "configs/features_local_baseline.yaml"
  },
  "snapshot": {
    "label": "snapshot",
    "status": "completed",
    "exit_code": 0
  },
  "readiness": {
    "benchmark_rerun_ready": true,
    "recommended_action": "run_local_benchmark"
  },
  "coverage_summary": {},
  "integrity_summary": {},
  "train": null,
  "evaluate": null,
  "completed_step": "snapshot"
}
```

この 2 本で最低限そろえるキーは次である。

1. `universe`
   - config 名だけに依存せず、payload 自体に source universe を持たせる。
2. `run_context` または `configs`
   - data / model / feature config と tail_rows / max_rows を追えるようにする。
3. `readiness`
   - `benchmark_rerun_ready`、`recommended_action`、`reasons` を必須にする。
4. `coverage_summary` と `integrity_summary`
   - coverage 不足と key 整合を分けて読めるようにする。
5. `completed_step`
   - local gate がどこまで進んだかを、`snapshot`, `train`, `evaluate`, `completed` のような段階で読めるようにする。

JRA 既存 script と揃えてよい部分もある。

- `status`: `running / completed / failed / not_ready / interrupted`
- `started_at` / `finished_at`
- `snapshot`, `train`, `evaluate` の各 step result に `label / status / exit_code / started_at / finished_at`

一方で、地方-only schema で追加したほうがよい項目は次である。

- `universe_display_name`: 人間向けの source 名
- `source_scope`: `local_only` / `mixed`
- `baseline_reference`: 比較対象の JRA-only baseline slug
- `schema_version`: payload の互換性管理

将来 CLI を作るなら、引数契約も既存 `netkeiba_*` 系に寄せると実装負荷が低い。最小の引数セットは次でよい。

地方-only coverage snapshot 向け:

- `--data-config`
- `--tail-rows`
- `--output`
- `--universe`
- `--schema-version`
- 必要なら `--columns`, `--source-scope`, `--baseline-reference`

地方-only benchmark gate 向け:

- `--data-config`
- `--model-config`
- `--feature-config`
- `--tail-rows`
- `--snapshot-output`
- `--manifest-output`
- `--max-rows`
- `--pre-feature-max-rows`
- `--wf-mode`
- `--wf-scheme`
- `--skip-train`
- `--skip-evaluate`
- `--universe`
- `--source-scope`
- `--baseline-reference`
- `--schema-version`

引数名の設計ルールは次で固定すると分かりやすい。

1. JRA 既存 gate にある引数名は変えず、universe 固有情報だけを追加する。
2. `--universe` は artifact slug と同じ値を受ける。
3. `--source-scope` は `local_only` または `mixed` を受ける。
4. `--baseline-reference` は mixed 比較時だけ必須にし、local-only では任意にする。
5. `--schema-version` は payload 互換性を上げたときだけ明示変更する。

CLI 実装時の fail-fast 条件も先に決めておく。

- `--universe` が未指定なら fail する。
- `--source-scope=mixed` なのに `--baseline-reference` が空なら fail する。
- `--snapshot-output` / `--manifest-output` に directory を渡したら fail する。
- `--universe` と output file 名の slug が食い違う場合は warning ではなく fail-fast してよい。

この形なら、既存 `run_netkeiba_benchmark_gate.py` の operator experience を維持しつつ、local-only / mixed の違いだけを明示的に増やせる。

step 名と fail-fast taxonomy も先に固定しておくと、将来 CLI を追加するときに manifest の読み方がぶれない。

coverage snapshot 側の推奨 step 名:

1. `load_config`
2. `load_source_tables`
3. `compute_alignment`
4. `compute_coverage`
5. `write_snapshot`
6. `completed`

benchmark gate 側の推奨 step 名:

1. `init_manifest`
2. `run_snapshot`
3. `validate_readiness`
4. `run_train`
5. `run_evaluate`
6. `write_manifest`
7. `completed`

`completed_step` は最後に正常終了した段階を指すものとして扱う。たとえば readiness で止まった場合は `status=not_ready`, `completed_step=validate_readiness` まで進んだ、と読める形にする。

fail-fast の分類は、少なくとも次の 3 層に分ける。

1. operator error
  - config 不足、必須引数不足、output path 取り違え、`--source-scope=mixed` で `--baseline-reference` 不足など。
  - 表示は既存方針どおり concise な `failed: ...` でよい。
2. readiness block
  - source 欠落、key 不整合、coverage 不足により benchmark rerun 準備ができていない状態。
  - これは exception ではなく `status=not_ready` と `recommended_action` で返す。
3. execution failure
  - train / evaluate / snapshot subprocess の exit code 非ゼロ、unexpected exception、interrupt。
  - `status=failed` または `status=interrupted` を使い、step result に `exit_code` と `label` を残す。

manifest に残す failure field も最小で固定してよい。

- `status`
- `completed_step`
- `error_code`
- `error_message`
- `recommended_action`

`error_code` の語彙は、実装前提として次程度に絞ると扱いやすい。

- `missing_universe`
- `mixed_baseline_required`
- `invalid_output_path`
- `slug_mismatch`
- `snapshot_not_ready`
- `train_failed`
- `evaluate_failed`
- `interrupted`

この taxonomy を先に持っておくと、将来 local-only / mixed の gate を足したときも、`status` と `completed_step` だけで operator が停止点を読める。

逆に、地方-only benchmark の完了条件に次を混ぜない。

- mixed 学習で JRA-only baseline を上回ったこと
- serving compare で既に運用採用できること
- JRA と同一の feature set を完全再現できていること

つまり最初の completion bar は「地方 universe を別 benchmark として再現可能にしたか」であり、「JRA baseline より強いか」ではない。

要するに、地方競馬データは「今すぐ JRA 学習へ混ぜる追加データ」ではなく、「別 universe として feasibility を切る将来候補」と読むのが正しい。

## 9. この文書の読み方

- プロジェクト全体像は [project_overview.md](project_overview.md) を参照する。
- システム構造は [system_architecture.md](system_architecture.md) を参照する。