# Next Issue: Local Nankan Baseline Formalization

Historical note:

- この draft は `#54` として baseline formalization まで完了している。
- local Nankan baseline formalization は completed NAR baseline reference として扱い、この文書は historical issue source / artifact reference として使う。
- 本文中の `status board=ready_for_benchmark` は `#54` 当時の historical read であり、current `#101/#103` blocker 判定には使わない。
- current blocker は `artifacts/reports/local_nankan_data_status_board.json` の `readiness_surfaces` を正本として読む。

## Summary

local Nankan data collection は completed で、`#54` 当時の status board も `ready_for_benchmark` になっている。

readiness artifact:

- `artifacts/reports/local_nankan_data_status_board_live.json` (`#54` 当時の historical board)
- `artifacts/reports/data_preflight_local_nankan_nar_baseline_check.json`
- `artifacts/reports/local_benchmark_gate_nar_baseline_check.json`

したがって next line は、JRA に無理に混ぜることではなく、NAR を `separate universe` として baseline formalization に進めることである。

## Objective

local Nankan baseline を formal に実行し、NAR line の first benchmark artifact と `bets / races / bet-rate` 付き read を残す。

## Current Read

- data preflight: `ready`
- local benchmark gate: `completed`
- status board (`#54` historical read): `ready_for_benchmark`
- JRA knowledge は threshold ではなく process / gating / denominator discipline として転用する

## In Scope

- local Nankan baseline run
- NAR evaluation artifact collection
- explicit denominator read
- next NAR child issue に必要な decision summary

## Non-Goals

- JRA / NAR data unification
- JRA operational anchor の変更
- new NAR feature widening before baseline read

## Success Criteria

- NAR baseline の first formal artifact が揃う
- `bets / races / bet-rate` が mandatory read として残る
- next NAR family queue を 1 本に絞れる
