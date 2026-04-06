# Next Issue: NAR Pre-Race Readiness Probe

Historical note:

- この draft は `#105` として close 条件を満たしている。
- read-only readiness probe は completed monitoring reference として扱い、この文書は historical issue source / operator reference として使う。

## Summary

`#101` handoff は result-ready race が来れば benchmark へ進めるが、current refresh は materialization と filtered outputs を伴う。external result arrival を監視するだけなら、raw を書き換えずに readiness を判定する probe が必要である。

## Objective

strict `pre_race_only` pool に `result_ready` race が入ったかどうかを、side-effect なしで一度に判定できる probe surface を作る。

## Hypothesis

if a dedicated readiness probe can read current race card / result inputs and emit a bounded manifest without writing filtered CSVs, then `#101` refresh は low-risk monitoring surface として自動化しやすくなる。

## In Scope

- probe script
- readiness summary manifest
- exit code contract
- minimal tests

## Non-Goals

- benchmark handoff の置換
- result-ready subset materialization
- bootstrap execution

## Success Metrics

- `result_ready_races=0` なら `not_ready` を返す
- `result_ready_races>0` なら `ready` を返す
- filtered CSV / primary を書かない
- `#101` refresh 用に使える manifest を残す

## Validation Plan

1. existing provenance helpers で readiness 判定を構築する
2. one-shot probe script を追加する
3. unit test と smoke を通す

## Stop Condition

- existing provenance helpers だけでは readiness 判定が benchmark handoff と一致しない
- その場合は probe を追加せず handoff refresh のみを使う
