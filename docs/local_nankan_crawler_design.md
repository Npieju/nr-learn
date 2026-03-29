# local_nankan crawler 設計

## 1. 目的

この設計は、`local_nankan` 用の収集基盤を既存 `netkeiba` crawler と同じ運用感で追加するための最小仕様を定める。

狙いは次の 4 点である。

1. `prepare ids -> collect -> backfill` の 3 層を local 側にも揃える。
2. 収集進捗を manifest と lock で追えるようにする。
3. 出力先を `local_nankan` 名義へ分離し、JRA 系 artifact を汚染しない。
4. small backfill から全面 backfill へ同じ CLI 契約で拡張できるようにする。

## 2. 既存 netkeiba から再利用する契約

そのまま流用する方針は次である。

1. crawl config のトップレベルは `crawl:` とする。
2. 共通 request 設定は `base_url`, `user_agent`, `timeout_sec`, `delay_sec`, `retry_count`, `retry_backoff_sec`, `overwrite` を使う。
3. target ごとの設定は `targets.<target_name>` に集約する。
4. target ごとに `id_file`, `id_column`, `output_file`, `dedupe_on` を持つ。
5. base manifest と target manifest を分け、lock file を併用する。
6. `prepare ids`, `collect`, `backfill` はそれぞれ独立 CLI にする。
7. backfill は cycle 単位で manifest を更新し、partial completion を許す。

## 3. local 側で新たに決める点

`netkeiba` と違って最初は source 候補が複数ありえたが、初期実装では `nankankeiba.com` を正本に固定する。

1. source provider
  - 初期実装は `https://www.nankankeiba.com` を正本に固定する。
  - URL は `result/{race_id}.do`, `syousai/{race_id}.do`, `uma_info/{horse_key}.do` を直接使う。
2. race id namespace
  - `race_id` を provider 由来の raw id のまま持つか、`local_nankan:` prefix を付けるか。
3. horse id namespace
  - `horse_id` と `horse_key` を JRA と共有しない前提で持つ。
4. race discovery method
  - race list page から date 単位で収集するか、開催日一覧から組み立てるか。
5. pedigree availability
  - `uma_info` に父・母・母父があるので、初期実装では `pedigree` target を有効化する。

## 3.1 実 URL 方針

2026-03-28 時点の実装では次の URL を使う。

1. `race_result`: `https://www.nankankeiba.com/result/{race_id}.do`
2. `race_card`: `https://www.nankankeiba.com/syousai/{race_id}.do`
3. `pedigree`: `https://www.nankankeiba.com/uma_info/{horse_key}.do`

`program/{meeting_id}.do` は開催一覧の導線には使えるが、個別 race の card 取得元としては `syousai` を優先する。

## 4. 想定 target

初期 target は次の 3 本に限定する。

### 4.1 `race_result`

目的:

- レース結果、着順、人気、払戻に近い主表を作る。

最低限必要な列:

1. `date`
2. `race_id`
3. `horse_id`
4. `horse_name`
5. `finish_position`
6. `odds`
7. `popularity`
8. `jockey_id`
9. `trainer_id`

### 4.2 `race_card`

目的:

- 出走時点の card 情報を補完する。

最低限必要な列:

1. `race_id`
2. `horse_id`
3. `date`
4. `track`
5. `distance`
6. `weather`
7. `ground_condition`
8. `frame_no`
9. `gate_no`
10. `weight`

### 4.3 `pedigree`

目的:

- `horse_key` 単位の血統補助表を作る。

最低限必要な列:

1. `horse_key`
2. `sire_name`
3. `dam_name`
4. `damsire_name`

## 5. ディレクトリ契約

出力先は次で固定する。

1. raw HTML: `data/external/local_nankan/raw_html`
2. ids: `data/external/local_nankan/ids`
3. results: `data/external/local_nankan/results`
4. racecard: `data/external/local_nankan/racecard`
5. pedigree: `data/external/local_nankan/pedigree`
6. base manifest: `artifacts/reports/local_nankan_crawl_manifest.json`

`configs/data_local_nankan.yaml` が読む primary raw は引き続き `data/local_nankan/raw` とする。crawler 出力を primary raw へどう昇格させるかは別 step として扱う。

## 6. config 仕様

最初の config 名は `configs/crawl_local_nankan_template.yaml` とする。最小 shape は次である。

```yaml
crawl:
  provider: "nankankeiba"
  base_url: "https://www.nankankeiba.com"
  user_agent: "nr-learn-local-nankan-crawler/0.1"
  timeout_sec: 20
  delay_sec: 0.05
  retry_count: 3
  retry_backoff_sec: 2.0
  overwrite: false
  raw_html_dir: "data/external/local_nankan/raw_html"
  manifest_file: "artifacts/reports/local_nankan_crawl_manifest.json"
  targets:
    race_result:
      enabled: true
      id_file: "data/external/local_nankan/ids/race_ids.csv"
      id_column: "race_id"
      output_file: "data/external/local_nankan/results/local_race_result.csv"
      dedupe_on: ["race_id", "horse_id"]
    race_card:
      enabled: true
      id_file: "data/external/local_nankan/ids/race_ids.csv"
      id_column: "race_id"
      output_file: "data/external/local_nankan/racecard/local_racecard.csv"
      dedupe_on: ["race_id", "horse_id"]
    pedigree:
      enabled: true
      id_file: "data/external/local_nankan/ids/horse_keys.csv"
      id_column: "horse_key"
      output_file: "data/external/local_nankan/pedigree/local_pedigree.csv"
      dedupe_on: ["horse_key"]
```

## 7. CLI 設計

### 7.1 `run_prepare_local_nankan_ids.py`

役割:

- race ids と horse keys の候補を作る。

最低限の引数:

1. `--data-config`
2. `--crawl-config`
3. `--target`
4. `--start-date`
5. `--end-date`
6. `--date-order`
7. `--limit`
8. `--include-completed`

補足:

- 初期実装では `training_table` source に依存せず、provider の race list または開催日一覧から IDs を作る前提でよい。
- ただし将来は `race_list` と `historical_seed` の 2 source を持てるようにしておく。

### 7.2 `run_collect_local_nankan.py`

役割:

- target ごとに HTML を取得し CSV を更新する。

最低限の引数:

1. `--config`
2. `--target`
3. `--limit`
4. `--dry-run`

補足:

- 初期実装では `overwrite: true` を config に入れたときだけ raw HTML を再取得する。
- `parse-only` や provider 差し替えは、全面 backfill の必要が見えた後に追加する。

### 7.3 `run_backfill_local_nankan.py`

役割:

- IDs 準備と target crawl を cycle で回す。

最低限の引数:

1. `--data-config`
2. `--crawl-config`
3. `--start-date`
4. `--end-date`
5. `--date-order`
6. `--race-batch-size`
7. `--pedigree-batch-size`
8. `--max-cycles`
9. `--skip-race-card`
10. `--skip-pedigree`
11. `--manifest-file`

## 8. manifest 設計

base manifest は `netkeiba_crawl_manifest.json` と同じく run 全体の進捗を持つ。最小項目は次である。

1. `started_at`
2. `finished_at`
3. `status`
4. `process_id`
5. `lock_file`
6. `provider`
7. `targets`

target manifest は次を持つ。

1. `target`
2. `requested_ids`
3. `parsed_ids`
4. `rows_written`
5. `failure_count`
6. `output_file`
7. `raw_html_path`
8. `failures`

## 9. 実装方針

実装順は次で固定する。

1. config loader と path 契約を追加する。
2. base helper を provider 非依存へ切り出せるか確認する。
3. `race_result` parser を最初に実装する。
4. `race_card` parser を追加する。
5. `pedigree` を optional target として実装する。
6. IDs 準備 CLI を追加する。
7. collect CLI を追加する。
8. backfill CLI を追加する。
9. small backfill smoke を回す。

## 10. small backfill smoke

最初の smoke は全面取得にしない。最小単位は次である。

1. 連続 2 開催日または 1 週間分の race ids を取得する。
2. `race_result` を先に 20 から 50 race で試す。
3. その同一 race ids に対して `race_card` を取る。
4. horse keys が十分に取れた後に pedigree を小バッチで試す。
5. 出力 CSV を `configs/data_local_nankan.yaml` に当てて preflight を再実行する。

成功条件:

1. manifest が completed まで進む。
2. CSV が空でない。
3. `data_preflight` の primary / append / supplemental の読み口に乗る。

## 11. 未決定事項

実装前に決めるべき未決定事項は次である。

1. 正式な source provider
2. race list discovery の具体 URL
3. ID namespace の prefix 規則
4. pedigree を初期フェーズで必須にするか optional にするか
5. crawler 出力を `data/local_nankan/raw` へ昇格させる変換 step の有無

## 12. 次アクション

この設計に従う場合、直近の作業は次の 3 つである。

1. source provider を確定する。
2. `configs/crawl_local_nankan_template.yaml` の初版を作る。
3. `run_prepare_local_nankan_ids.py` の CLI から着手する。

2026-03-28 時点では 2 と 3 の最小入口を追加済みである。次の実装は `run_collect_local_nankan.py` を provider 実装へ接続し、planned / blocked ではなく実際の fetch を行えるようにする段である。

同日中に `run_backfill_local_nankan.py` の骨格も追加し、provider 実装前でも 1 cycle 分の `prepare -> collect` を planned / blocked manifest として確認できるようにした。

同日の次段として `run_materialize_local_nankan_primary.py` も追加し、crawler の external outputs から `data/local_nankan/raw` 側の primary CSV を組み立てる bridge を独立 step として扱えるようにした。これにより provider 実装とは別に、preflight の `primary_dataset_missing` を解消するための raw 昇格工程を先に固定できる。

さらに同日中に `run_backfill_local_nankan.py --materialize-after-collect` を追加し、backfill manifest 内でも `prepare -> collect -> materialize` を 1 cycle 単位で追えるようにした。provider 未実装で collect が blocked のままでも、既存 external outputs が揃っていれば `materialize_summary.status=completed` まで進められるので、backfill 入口からそのまま `run_local_preflight` へ接続できる。

その次段として `run_local_backfill_then_benchmark.py` も追加し、backfill handoff で `current_phase=materialized_primary_raw` に到達したら、そのまま local benchmark gate を起動できるようにする。これにより provider 実装前でも、smoke 用 external outputs を使って `prepare -> collect -> materialize -> preflight -> snapshot -> benchmark` を 1 本の wrapper manifest で追える。

さらに `run_local_revision_gate.py --backfill-before-benchmark` も追加し、この handoff wrapper 自体を revision lineage の benchmark 前段へ載せられるようにする。これで local-only の formal lineage でも、Phase 0 の handoff と benchmark/revision/promotion/evaluation pointer を 1 本の lineage manifest から追跡できる。