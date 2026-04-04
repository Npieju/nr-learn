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

## Residual Risk

- 既存 backfilled raw には sidecar manifest がないため `cache_legacy` fallback が残る
- `pre_race / post_race` 判定は `scheduled_post_at` と `snapshot_at` 比較であり、発走遅延や update lag までは表現しない
- full historical recrawl と pre-race only rebuild は次段
