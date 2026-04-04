# Next Issue: NAR Readiness Watcher Handoff

## Summary

`#105` で read-only readiness probe、`#104` で result-ready bootstrap handoff は揃った。残る gap は、人手で probe を回して `ready` を見つけたあとに handoff を起動する運用だけである。

## Objective

bounded polling で `#105` probe を監視し、`ready` 検知時に `#104` handoff を起動する watcher surface を追加する。

## Hypothesis

if a watcher can poll the read-only readiness probe and trigger the existing handoff only when readiness flips to `ready`, then NAR post-readiness resume can happen without manual polling loops.

## In Scope

- watcher script
- bounded polling / timeout
- trigger policy (`probe -> handoff`)
- watcher manifest
- minimal tests

## Non-Goals

- external notifications
- daemonization / systemd
- bootstrap logic redesign

## Success Metrics

- `not_ready` の間は bounded polling で止まる
- `ready` なら `#104` handoff を起動する
- watcher 自身は filtered CSV を直接書かない
- manifest に probe attempts と trigger result が残る

## Validation Plan

1. watcher script を追加する
2. probe-only timeout path を smoke する
3. ready trigger path は helper/unit test で確認する

## Stop Condition

- existing `#105` probe と `#104` handoff の組み合わせで状態遷移が十分に表現できない
- その場合は manual runbook 維持に留める
