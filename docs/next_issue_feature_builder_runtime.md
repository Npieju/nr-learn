# Next Issue: Feature Builder History Runtime

## GitHub Issue

- `#17`
- <https://github.com/Npieju/nr-learn/issues/17>

## Summary

`#7` と `#16` により、optimizer 側の重複 trial と supplemental load の大きな fixed cost はかなり削れた。次に profile 上で目立つのは feature builder 側の history / rolling 系である。

特に候補になっているのは次の 2 点である。

- `src/racing_ml/features/builder.py:_group_shifted_rolling_mean`
- `src/racing_ml/features/builder.py:_entity_race_shifted_rolling_mean`

つまり次の perf 本線は、data loading ではなく repeated grouped history computation の整理である。

## Objective

evaluation / challenger run の feature-build 固定コストを、feature semantics を変えずに削減する。

## Initial Scope

- `builder.py` の grouped rolling / entity-race rolling を profile-guided に見直す
- shared intermediates、reuse、cached cleaned/grouped surfaces を優先する
- feature 定義の意味変更や coverage 変更は行わない

## Acceptance

- reduced smoke または loader+build profile で repeatable な改善がある
- regression test が通る
- feature/output semantic drift がない
