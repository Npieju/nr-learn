# Next Issue: NAR Pre-Race-Only Benchmark Rebuild

## Summary

parent completion gate:

- `#123 [nar] JRA-equivalent trust completion gate`
- `#101` の role は `Stage 0 benchmark trust` を解く blocker issue であり、これ自体は NAR solved を意味しない

current status:

- completed on historical trust-ready corpus via `r20260415_local_nankan_pre_race_ready_formal_v1`
- current meaning は blocked-state 維持ではなく、Stage 0 benchmark reference の維持である

`#100` で local Nankan collector に provenance persistence を入れ、small-scope live recrawl で strict `pre_race` row の実在を確認した。

- live recrawl:
  - `2026-04-06 / 2026-04-07`
  - `24 races`
- provenance audit:
  - `pre_race_only_rows=281`
  - `post_race_rows=0`
  - `unknown_rows=731941`

この issue の corrective 自体は完了しており、strict `pre_race_only` subset materialization から formal rerun まで historical trust-ready corpus 上で一度閉じている。したがって現在の役割は、blocked issue ではなく Stage 0 benchmark reference の履歴と読み筋を保持することにある。

## Objective

strict `pre_race_only` provenance bucket だけを使った local Nankan benchmark-ready subset を構築し、current market-aware line を backfilled benchmark ではなく provenance-defensible benchmark として再評価できる導線を作る。

## Hypothesis

if strict `pre_race_only` subset だけで local Nankan raw / primary / evaluation input を再構築できる, then NAR market-aware line は provenance 不足ではなく sample/support の問題として読み直せる。

## Current Read

- `#120` trust semantics repair により、trust 判定は fetch timing ではなく `pre_race_feature_availability` basis で読む
- `#101` formal rerun `r20260415_local_nankan_pre_race_ready_formal_v1` は trust-ready historical corpus 上で completed しており、current benchmark reference は次で固定する
  - [evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](/workspaces/nr-learn/artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json)
  - [evaluation_summary_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_summary_local_nankan_baseline_wf_full_nested.json)
  - [evaluation_manifest_local_nankan_baseline_wf_full_nested.json](/workspaces/nr-learn/artifacts/reports/evaluation_manifest_local_nankan_baseline_wf_full_nested.json)
- current next action は external result arrival 待ちではなく、この benchmark reference を baseline freeze として `#103` architecture bootstrap を進めることである

historical artifact の canonical read は次である。

- board:
  - [local_nankan_data_status_board.json](/workspaces/nr-learn/artifacts/reports/local_nankan_data_status_board.json)
- primary fields:
  - `status`
  - `current_phase`
  - `recommended_action`
  - `readiness_surfaces.capture_loop`
  - `readiness_surfaces.readiness_probe`
  - `readiness_surfaces.pre_race_handoff`
  - `readiness_surfaces.bootstrap_handoff`
  - `readiness_surfaces.readiness_watcher`
  - `readiness_surfaces.followup_entrypoint`

rule:

- current benchmark judgment は board の `await_result_arrival` ではなく `#101` pointer / summary / manifest を一次参照にする
- future-only operator path を確認したいときだけ board / readiness surfaces に降りる
- detail が必要なときだけ、この文書の個別 materialize / handoff artifact に降りる

## In Scope

- `src/racing_ml/data/local_nankan_provenance.py`
- `scripts/run_local_nankan_provenance_audit.py`
- local Nankan raw / primary materialization 導線
- strict `pre_race_only` subset CSV の生成
- subset を使った small-scope benchmark / sanity rerun

## Non-Goals

- full historical timestamped recrawl
- JRA baseline work
- NAR feature family の新規 replay
- promotion gate redesign

## Success Metrics

- strict `pre_race_only` subset を raw / primary 相当の入力として再利用できる
- `unknown` / backfilled row を混ぜない benchmark-ready subset が 1 本できる
- small-scope rerun で market-aware / no-market の比較が provenance-defensible に読める
- 次の NAR benchmark read を `backfilled benchmark` ではなく `pre-race-only benchmark` として扱える

## Validation Plan

1. strict `pre_race_only` subset materialization 導線を作る
2. subset row count / race count / date range を検証する
3. subset を使った small-scope market-aware rerun を 1 本回す
4. 必要なら no-market rerun も 1 本合わせて、market dependency を provenance-defensible な universe で再確認する

## First Cut Status

subset materialization 導線は実装済み。

- script:
  - [run_materialize_local_nankan_pre_race_only.py](/workspaces/nr-learn/scripts/run_materialize_local_nankan_pre_race_only.py)
- helper:
  - [local_nankan_provenance.py](/workspaces/nr-learn/src/racing_ml/data/local_nankan_provenance.py)
- outputs:
  - [local_nankan_pre_race_only_materialize_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_only_materialize_summary.json)
  - [nar_pre_race_only_materialize_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_only_materialize_smoke.log)
  - default subset path: `data/local_nankan/raw/local_nankan_race_card_pre_race_only.csv`

confirmed read:

- `pre_race_only_rows=281`
- `pre_race_only_races=24`
- `pre_race_only_dates=['2026-04-06', '2026-04-07']`
- `result_ready_races=0`
- `pending_result_races=24`
- `ready_for_benchmark_rerun=false`

meaning:

- strict `pre_race_only` subset の materialization 自体は成立した
- ここで止まっていた blocked-state が、その後の trust semantics repair と formal rerun の前提になった
- 現在はこの subset materialization を historical benchmark reference の成立過程として読む

## Second Cut Status

result-ready subset と primary materialization までの導線も追加した。

- script:
  - [run_materialize_local_nankan_pre_race_primary.py](/workspaces/nr-learn/scripts/run_materialize_local_nankan_pre_race_primary.py)
- outputs:
  - [local_nankan_pre_race_ready_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_ready_summary.json)
  - [nar_pre_race_primary_materialize_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_primary_materialize_smoke.log)
  - default paths:
    - `data/local_nankan_pre_race_ready/raw/local_nankan_race_card_pre_race_ready.csv`
    - `data/local_nankan_pre_race_ready/raw/local_nankan_race_result_pre_race_ready.csv`
    - `data/local_nankan_pre_race_ready/raw/local_nankan_primary_pre_race_ready.csv`

confirmed read:

- `status=not_ready`
- `current_phase=await_result_arrival`
- `result_ready_races=0`
- `pending_result_races=24`
- `result_ready_rows=0`
- `pending_result_rows=281`

meaning:

- strict `pre_race_only` primary materialization 導線はできており、その後の historical rerun に再利用された
- current read では blocked-state そのものより、trust-ready corpus を固定できるようになった過程として読む

## Third Cut Status

result-ready race が出たら同じ entrypoint で benchmark gate まで進める handoff wrapper も追加した。

- script:
  - [run_local_nankan_pre_race_benchmark_handoff.py](/workspaces/nr-learn/scripts/run_local_nankan_pre_race_benchmark_handoff.py)
- outputs:
  - [local_nankan_pre_race_benchmark_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json)
  - [nar_pre_race_benchmark_handoff_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_benchmark_handoff_smoke.log)

confirmed read:

- `status=not_ready`
- `current_phase=await_result_arrival`
- `recommended_action=wait_for_result_ready_pre_race_races`

meaning:

- handoff wrapper 自体は current operator surface として残っている
- その後の historical trust-ready rerun 完了により、`#101` の remaining work は消化済みであり、current role は benchmark reference maintenance に移った

## Fourth Cut Status

handoff wrapper に bounded wait も追加した。

- wait flags:
  - `--wait-for-results`
  - `--max-wait-seconds`
  - `--poll-interval-seconds`
- smoke:
  - [nar_pre_race_benchmark_handoff_wait_smoke.log](/workspaces/nr-learn/artifacts/logs/nar_pre_race_benchmark_handoff_wait_smoke.log)

confirmed read:

- `status=not_ready`
- `attempts=1`
- `waited_seconds=9`
- `timed_out=true`

meaning:

- bounded wait contract は `#122` future-only readiness path で引き続き有効である
- ただし `#101` current read では、この blocked-state は historical transition であり現行 benchmark reference の primary meaning ではない

## Current Canonical Read

current benchmark reference read:

- pointer:
  - [evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](/workspaces/nr-learn/artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json)
- evaluate summary:
  - `auc=0.8625051675`
  - `logloss=0.2171821084`
  - `top1_roi=0.8402165859`
  - `ev_threshold_1_0_roi=2.8603751213`
  - `ev_threshold_1_2_roi=4.5726544989`
- nested WF:
  - `wf_nested_test_roi_weighted=3.9660920371`
  - `wf_nested_test_bets_total=778`
  - `wf_nested_completed=true`

historical blocked-state board read before completion:

- `status=partial`
- `current_phase=await_result_arrival`
- `recommended_action=wait_for_result_ready_pre_race_races`
- readiness probe:
  - `result_ready_races=0`
  - `pending_result_races=24`
  - `race_card_rows=562`
- pre-race handoff:
  - `status=not_ready`
- bootstrap handoff:
  - `status=not_ready`
- readiness watcher:
  - `status=not_ready`
  - `attempts=2`
  - `timed_out=true`

meaning:

- `#101` は現在 blocked ではなく completed benchmark reference である
- historical blocked-state board read は、future-only operator path と当時の transition を理解するための補助情報に留める
- `#103` は now active Stage 1 architecture bootstrap としてこの benchmark reference の後段に置く
- `#101` が完了しても、それは `#123` に対する Stage 0 exit であり、NAR completion そのものではない

## Stop Condition

- strict `pre_race_only` subset が benchmark run に必要な最小 row/race を満たさない
- downstream scripts が subset universe を前提に簡潔に再利用できず、full recrawl なしでは意味のある rerun が組めない
- その場合は benchmark rebuild を止め、provenance-aware data collection の継続を優先する
