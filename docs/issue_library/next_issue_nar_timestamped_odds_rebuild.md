# Next Issue: NAR Timestamped Odds Rebuild

## Summary

`#99` の provenance audit で、current local Nankan `odds / popularity` raw は race day snapshot としては defensible ではないと分かった。

- `race_card` / `race_card_odds` raw_html は `2026-03-29 / 2026-03-30` の bulk backfill に集中
- row-level provenance 列
  - `fetch_at`
  - `odds_snapshot_at`
  - `card_snapshot_at`
  は current raw に存在しない
- current benchmark read は `market-aware backfilled benchmark` として扱うのが妥当

したがって次の corrective は、market columns を廃止することではなく、timestamped provenance を保存できる collector / raw schema へ切り替えることにある。

## Objective

local Nankan collector に timestamped provenance を追加し、future crawl から `odds / popularity` を pre-race snapshot として監査可能な形で保存する。

## Hypothesis

if collector が card HTML / odds JS の取得時刻と race post time を保存する, then NAR market-aware line は `backfilled benchmark` ではなく `timestamped pre-race benchmark` として再検証可能になる。

## In Scope

- `src/racing_ml/data/local_nankan_collect.py`
- raw CSV schema
  - `data/external/local_nankan/racecard/local_racecard.csv`
  - downstream `data/local_nankan/raw/local_nankan_race_card.csv`
- crawl manifest persistence
- provenance columns for
  - card fetch time
  - odds fetch time
  - source URL
  - fetch mode / reused cached file

## Non-Goals

- full historical recrawl をいきなり完走すること
- JRA baseline work
- NAR feature family experiments
- promotion gate redesign

## Success Metrics

- future crawl で `fetch_at / odds_snapshot_at / card_snapshot_at` が row-level に残る
- crawl manifest でも race-level provenance が読める
- pre-race / unknown / post-race の 3 区分で filtering できる
- downstream no-market / market-aware rerun の前提が整う

## Validation Plan

1. collector に provenance persistence を追加
2. small-scope local Nankan crawl smoke
3. race_card / primary materialization で provenance 列が保たれるか確認
4. `pre-race only` subset を作れるか確認

## Stop Condition

- source site から発走前 provenance を取得できない
- collector 変更だけでは `pre-race / post-race` の判定が不能
- その場合は NAR market columns exclusion を default にする別 corrective へ切り替える

## First Cut Status

first cut は実装済み。

- collector:
  - `src/racing_ml/data/local_nankan_collect.py`
  - raw HTML / odds JS ごとに sidecar manifest `*.meta.json` を保存
  - cached reuse 時は sidecar `fetched_at` を優先し、legacy cache は file mtime fallback
- row-level provenance:
  - `post_time`
  - `scheduled_post_at`
  - `card_source_url`
  - `card_fetch_mode`
  - `card_snapshot_at`
  - `card_snapshot_relation`
  - `odds_source_url`
  - `odds_fetch_mode`
  - `odds_snapshot_at`
  - `odds_snapshot_relation`
- downstream:
  - `src/racing_ml/data/local_nankan_primary.py` で primary / supplemental raw に通過

## Validation

- unit tests:
  - `PYTHONPATH=src .venv/bin/python -m unittest tests.test_local_nankan_collect tests.test_local_nankan_primary`
- compile:
  - `python -m py_compile src/racing_ml/data/local_nankan_collect.py src/racing_ml/data/local_nankan_primary.py src/racing_ml/data/local_nankan_race_list.py tests/test_local_nankan_collect.py tests/test_local_nankan_primary.py`
- temp-dir smoke:
  - [nar_timestamped_odds_rebuild_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_timestamped_odds_rebuild_smoke.log)
  - `race_card` に provenance 列出力
  - raw HTML / odds JS の sidecar meta 作成
  - `primary` に provenance 列通過

## Second Cut Status

`pre-race / unknown / post-race` を実際に読める導線も追加した。

- helper:
  - [local_nankan_provenance.py](/workspaces/nr-learn/src/racing_ml/data/local_nankan_provenance.py)
- script:
  - [run_local_nankan_provenance_audit.py](/workspaces/nr-learn/scripts/run_local_nankan_provenance_audit.py)
- outputs:
  - summary JSON
  - annotated CSV
  - strict `pre_race_only` subset CSV

strict bucket rule:

- `post_race`: card / odds のどちらかが `post_race`
- `pre_race`: card / odds の両方が `pre_race`
- `unknown`: それ以外

second-cut validation:

- unit tests:
  - `PYTHONPATH=src .venv/bin/python -m unittest tests.test_local_nankan_collect tests.test_local_nankan_primary tests.test_local_nankan_provenance`
- compile:
  - `python -m py_compile src/racing_ml/data/local_nankan_provenance.py scripts/run_local_nankan_provenance_audit.py tests/test_local_nankan_provenance.py`
- smoke:
  - [nar_timestamped_odds_rebuild_second_cut_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_timestamped_odds_rebuild_second_cut_smoke.log)
  - current backfilled fetch は `post_race_rows=1`, `pre_race_only_rows=0`
  - つまり provenance に基づく strict filtering が働く

## Third Cut Status

small-scope live recrawl で `pre_race` 行の実在を確認した。

- race list discovery:
  - `2026-04-04 .. 2026-04-10`
  - race_card target `24 races`
  - `2026-04-06`, `2026-04-07`
- live recrawl:
  - [local_nankan_timestamped_recrawl_apr06_07.log](/workspaces/nr-learn/artifacts/logs/local_nankan_timestamped_recrawl_apr06_07.log)
  - `requested_ids=24`
  - `parsed=24`
  - `failed=0`
- provenance audit on `race_card` raw:
  - [local_nankan_race_card_provenance_summary_apr06_07.json](/workspaces/nr-learn/artifacts/reports/local_nankan_race_card_provenance_summary_apr06_07.json)
  - [local_nankan_race_card_pre_race_only_apr06_07.csv](/workspaces/nr-learn/artifacts/reports/local_nankan_race_card_pre_race_only_apr06_07.csv)

confirmed read:

- `row_count=732222`
- `pre_race_only_rows=281`
- `post_race_rows=0`
- `unknown_rows=731941`
- `pre_race` rows cover `24 races`
- dates:
  - `2026-04-06`
  - `2026-04-07`

meaning:

- current backfilled benchmark は strict filter で弾ける
- future / upcoming window では actual `pre_race` snapshot row を保存できる
- timestamped recrawl path は有効

## Decision Summary

- `timestamped recrawl + provenance columns` は objective を満たした
- `#100` は close 条件を満たした
- 次段は strict `pre_race_only` subset を使った benchmark rebuild である
- execution source:
  - [next_issue_nar_pre_race_only_benchmark_rebuild.md](/workspaces/nr-learn/docs/issue_library/next_issue_nar_pre_race_only_benchmark_rebuild.md)

## Residual Risk

- 既存 backfilled raw には sidecar manifest がないため `cache_legacy` fallback が残る
- `pre_race / post_race` 判定は `scheduled_post_at` と `snapshot_at` 比較であり、発走遅延や update lag までは表現しない
- full historical recrawl と pre-race only rebuild は次段
