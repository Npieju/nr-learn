# Next Issue: NAR Odds Provenance Audit

## Summary

local Nankan line の integrity audit で、high AUC / high ROI の大部分は `odds / popularity` 依存と確認できた。

一方で、現時点では `odds` の direct leak は未証明である。残っている主要論点は、「現在の `odds / popularity` が race card のどの時点の snapshot なのか」「発走後取得や同日後追い取得が混ざっていないか」である。

## Objective

local Nankan `odds / popularity` の provenance を監査し、現行 raw が事前 snapshot として defensible か、もしくは recrawl / timestamped rebuild が必要かを判定する。

## Hypothesis

if current local Nankan `odds / popularity` は発走前 snapshot として provenance を証明できない, then current market-aware line は benchmark reference に留め、timestamped odds recrawl を corrective issue として優先するべきである。

## Current Read

- collector は race card HTML と別に `oddsJS/<race_id>.do` を取り、`odds_tan` JS から `odds / popularity` を抽出している
- 根拠:
  - `src/racing_ml/data/local_nankan_collect.py`
  - `ODDS_TAN_BLOCK_PATTERN`
  - `parse_local_nankan_race_card_html(...)`
- raw CSV には `odds,popularity` はあるが provenance 列がない
  - `data/local_nankan/raw/local_nankan_race_card.csv`
  - `data/local_nankan/raw/local_nankan_primary.csv`
- 現行 raw header には次がない
  - `fetch_at`
  - `fetched_at`
  - `post_time`
  - `odds_snapshot_at`
- collector の `last_fetch_at` は request pacing 用の in-memory state であり、row provenance として保存されていない
- `data/external/local_nankan/raw_html/race_card_odds/` には odds JS 自体は残っている
- ただし、少なくとも現行確認範囲では crawl manifest / fetch manifest の structured JSON は見当たらない
- したがって current evidence だけでは
  - いつ取得したか
  - 発走前だったか
  - card HTML と odds JS が同時点か
  を row / race 単位で再現できない
- filesystem mtime の分布は bulk backfill を示す
  - `race_card_odds`: `33141 files @ 2026-03-29`, `29637 files @ 2026-03-30`
  - `race_card`: `33142 files @ 2026-03-29`, `29636 files @ 2026-03-30`
- サンプルでも race_id と無関係に 2026-03-29/30 mtime が付いている
  - `2025092919070101.{html,js}`
  - `2025100520100101.{html,js}`
  - `2006111319080110.{html,js}`

## Interim Conclusion

- current local Nankan `odds / popularity` raw は historical pre-race snapshot としては defensible ではない
- direct leak の証拠とはまだ言わないが、少なくとも provenance は不足している
- current benchmark read は `market-aware backfilled benchmark` として扱うべきで、pure pre-race market snapshot benchmark とは扱わない
- corrective の第一候補は `timestamped recrawl + provenance column persistence` である

## In Scope

- `src/racing_ml/data/local_nankan_collect.py`
- `src/racing_ml/data/local_nankan_primary.py`
- `data/local_nankan/raw/local_nankan_race_card.csv`
- `data/local_nankan/raw/local_nankan_primary.csv`
- local Nankan crawl manifest / raw HTML layout

## Non-Goals

- JRA baseline work
- NAR new feature experiments
- promotion gate redesign
- いきなり full recrawl を走らせること

## Success Metrics

- provenance の有無を `yes / no` で判定できる
- current raw の欠落が明文化される
- corrective action が次の 1 本に narrowed される
  - provenance column backfill
  - timestamped recrawl
  - odds exclusion sanity rerun

## Validation Plan

1. collector audit
   - `oddsJS` 取得箇所
   - race card 取得箇所
   - manifest / raw HTML 保存箇所
   - timestamp 保存の有無
2. raw schema audit
   - `race_card.csv`
   - `primary.csv`
   の header を確認する
3. timing surface audit
   - 発走時刻との比較に使える列が現行 raw / manifest にあるかを確認する
4. corrective narrowing
   - recrawl 必須か
   - row-level provenance 追加で十分か
   - no-odds rerun で代替できるか

## Stop Condition

- provenance が既に十分で、追加 corrective が不要
- もしくは provenance 不足が明確で、次 issue が timestamped recrawl に一意に narrowed される
