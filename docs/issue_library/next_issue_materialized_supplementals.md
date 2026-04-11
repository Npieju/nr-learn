# Next Issue: Materialized Supplemental Tables

## GitHub Issue

- `#16`
- <https://github.com/Npieju/nr-learn/issues/16>

## Summary

`#7` の profile-guided runtime 改善を進めた結果、残る fixed cost は policy search の内側よりも `load_training_table_tail` と supplemental table 準備側に寄っている。

特に重いのは次の 3 点である。

- `_read_csv_tail`
- `_merge_supplemental_tables`
- `corner_passing_order` の raw CSV からの展開

小さな pandas micro-opt はいくつか試したが、same-summary のまま reduced smoke floor を下げるには至らなかった。したがって次の一手は、runtime code path の微調整ではなく、supplemental input を materialize して再利用する運用へ寄せることである。

## Objective

evaluation / revision gate が毎回 raw supplemental CSV を展開し直さなくてもよいように、materialized supplemental artifact path を標準化する。

## Initial Scope

- `corner_passing_order` の pre-expanded artifact path を検討する
- 必要なら `laptime` も同じ系に載せる
- current raw-data path は fallback として残す
- join schema と current output の同値性を確認する

## Acceptance

- materialized supplemental path の docs / issue-driven standard が決まっている
- 少なくとも 1 本の supplemental table が materialized path で load できる
- reduced smoke が current safe floor を改善する

## Current Status

`corner_passing_order` を materialize する CLI と loader fallback は追加済みである。

- CLI: `scripts/run_materialize_supplemental_table.py`
- default materialized path: `data/processed/supplemental/corner_passing_order.csv`
- validation config: `configs/data_2025_latest_materialized_corner.yaml`

loader-only profile では pre-feature load が `24.0s -> 18.5s`、reduced smoke A/B でも raw `34.238/33.962s` に対して materialized `31.292/31.198s` で勝っている。したがって `corner_passing_order` の materialized path は default config に昇格しつつ、raw path fallback は残す。

## Notes

- progress 必須ルールに従い、materialization source が長時間になるなら phase / heartbeat を入れる
- この issue は `#7` の follow-up だが、optimizer micro-opt とは責務が違うため分離して進める
