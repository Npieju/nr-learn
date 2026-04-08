# Next Issue: local Nankan Future-Only Pre-Race Readiness Track

## Summary

parent completion gate:

- `#123 [nar] JRA-equivalent trust completion gate`
- `#122` の role は `Stage 0 benchmark trust readiness blocker resolution` であり、NAR solved を意味しない

`#121` の source timing audit により、historical result-ready local Nankan `62503 races` は current cache 上で `pre_race=0` と確定した。

- historical benchmark:
  - diagnostic only
  - trust-carrying evidence としては使わない
- remaining executable path:
  - future-only pre-race capture を継続し
  - result arrival 後に strict `pre_race_only` subset を handoff する readiness track

この文書でいう `result arrival` / `到着` は、future-only に capture 済みの strict `pre_race_only` races に対して official result rows が実データへ反映され、artifact 上で `result_ready_races>0` として観測されることを意味する。単に時刻が進むことや operator が rerun したこと自体を指してはいない。

current future-only pool は次である。

- `pre_race_only_rows=562`
- `pre_race_only_races=24`
- `result_ready_races=0`
- dates:
  - `2026-04-06`
  - `2026-04-07`

したがって next issue は、historical corrective ではなく `future-only pre-race readiness track を operator-visible かつ rerunnable に固定する` ことである。

## Objective

local Nankan の future-only pre-race capture を readiness track として固定し、result arrival 後に strict `pre_race_only` benchmark handoff を再実行できる bounded operational path を作る。

## Hypothesis

if historical corpus を diagnostic-only に降格した上で、future-only pre-race pool の capture loop / readiness probe / pre-race handoff / bootstrap handoff / readiness watcher / follow-up entrypoint を status board 上で一貫した readiness surface に揃える, then operators can accumulate trust-ready pre-race races without misreading historical ROI and can rerun `#101` only when labeled future-only support actually exists.

## In Scope

- future-only readiness phase wording の固定
- capture loop / readiness probe / pre-race handoff / bootstrap handoff / readiness watcher / follow-up entrypoint / status board の整合
- future-only pre-race pool の artifact path 固定
- result arrival 後の rerun trigger 条件整理
- queue / issue thread への handoff path 明記

## Non-Goals

- historical benchmark trust 復旧
- value-blend bootstrap `#103` 実行
- full NAR multi-region expansion
- policy retune や ROI 改善 claim

## Success Metrics

- readiness probe が `future_only_readiness_track` を返す
- readiness watcher / status board / bootstrap handoff が historical blocker と future-only capture action を矛盾なく表示する
- future-only pre-race pool と pending result races が artifact で追える
- result-ready future-only races が出現したら `#101` rerun に遷移できる

## Validation Plan

1. readiness probe / readiness watcher / handoff / board の phase/action を future-only readiness に揃える
2. source timing blocker を historical benchmark downgrade として surface に出す
3. future-only pool artifact を再生成し、`pre_race_only_rows`, `pending_result_races` を確認する
4. `#101` rerun は future-only result-ready races が現れた時だけ再開する

## Stop Condition

- future-only source horizon が増えず、result arrival 後も benchmarkable support が形成されない
- その場合は future-only readiness track も diagnostic monitoring に留め、NAR mainline advancement を止める

## Current Evidence

- `artifacts/reports/local_nankan_source_timing_audit_issue121.json`
- `artifacts/reports/local_nankan_future_only_readiness_cycle_issue122.json`
- `artifacts/reports/local_nankan_future_only_tuning_probe_issue122.json`
- `artifacts/reports/local_nankan_pre_race_capture_loop_issue122_cycle.json`
- `artifacts/reports/local_nankan_pre_race_readiness_probe_summary.json`
- `artifacts/reports/local_nankan_readiness_watcher_issue122_cycle.json`
- `artifacts/reports/local_nankan_pre_race_benchmark_handoff_issue121.json`
- `artifacts/reports/local_nankan_result_ready_bootstrap_handoff_issue122.json`
- `artifacts/reports/local_nankan_data_status_board_issue122.json`
- `artifacts/reports/local_nankan_data_status_board_issue122_cycle.json`

current operational read:

- future-only readiness cycle wrapper:
  - `status=partial`
  - `current_phase=future_only_readiness_track`
  - `recommended_action=capture_future_pre_race_rows_and_wait_for_results`
  - no-arg operator-default smoke: pass
  - run context:
    - `start_date=2026-04-07`
    - `end_date=2026-04-14`
    - `max_passes=1`
- bounded capture loop within the wrapper:
  - `status=capturing`
  - `current_phase=capturing_pre_race_pool`
  - `pre_race_only_rows=426`
  - `pending_result_races=24`

- readiness probe:
  - `status=not_ready`
  - `current_phase=future_only_readiness_track`
  - `recommended_action=capture_future_pre_race_rows_and_wait_for_results`
- readiness watcher:
  - `status=not_ready`
  - `current_phase=future_only_readiness_track`
  - `recommended_action=capture_future_pre_race_rows_and_wait_for_results`
- pre-race / bootstrap handoff:
  - `status=not_ready`
  - `current_phase=historical_source_timing_blocked`
  - `recommended_action=downgrade_historical_benchmark_to_diagnostic_only`
- status board effective readiness:
  - `benchmark_rerun_ready=false`
  - `current_phase=future_only_readiness_track`
  - reasons:
    - `readiness_probe:future_only_readiness_track`
    - `readiness_watcher:future_only_readiness_track`
    - `pre_race_handoff:historical_source_timing_blocked`
    - `bootstrap_handoff:historical_source_timing_blocked`

meaning:

- `run_local_nankan_future_only_readiness_cycle.py` は no-arg で current default operator path として使用可能
- cadence / horizon tuning probe (`h1_p1`, `h7_p1`, `h7_p2`) では unique support は全 scenario で同一だった
  - `pre_race_only_rows=426`
  - `pre_race_only_races=24`
  - `pending_result_races=24`
  - `benchmark_rerun_ready=false`
- `horizon_days=1 -> 7` でも support は増えず、`max_passes=1 -> 2` でも追加 row は増えなかった
- したがって `#122` の current judgment は `future-only operator path を維持しつつ result arrival を待つ` であり、default cadence / horizon の promote は現時点で不要である
- active operator manifests (`local_nankan_future_only_readiness_cycle_issue122.json`, `local_nankan_pre_race_capture_loop_issue122_cycle.json`) は `started_at` / `finished_at` を持つため、rerun freshness を artifact 単体で追跡できる
- capture refresh 正本の `local_nankan_pre_race_capture_loop_issue122_cycle.json` は `execution_role=pre_race_capture_refresh_loop`, `data_update_mode=capture_refresh_only`, `execution_mode=bounded_pass_loop`, `trigger_contract=direct_capture_refresh` を持つ self-describing artifact として扱う
- bounded supervisor `run_local_nankan_future_only_wait_then_cycle.py` を追加し、manual rerun を減らしつつ cycle ごとの `wrapper/status_board/capture_loop` history を artifact 化できる。これは data 更新 job ではなく、更新済み data / artifact に対する readiness 再評価 surface である
- `run_local_nankan_future_only_followup_oneshot.py` はこの capture refresh contract を upstream 条件として検証し、fresh かつ contract-valid な refresh artifact があるときだけ readiness-only oneshot follow-up を起動する
- smoke artifact: `artifacts/reports/local_nankan_future_only_wait_then_cycle_issue122_smoke.json`
- `#122` が満たすのは `result-ready strict benchmark rerun へ進む readiness surface` までであり、`#123` の completion gate を直接 close するものではない

## Decision Rule

- historical local Nankan line は引き続き diagnostic only とする
- future-only readiness track で `result_ready_races>0` が出るまで `#101` は benchmark evidence に戻さない
- `#103` は future-only pre-race result-ready subset が実在確認できるまで blocked に維持する
- cadence / horizon knob は `support growth` が artifact で確認できた時だけ変更する

## Operator Runbook

- external refresh 完了後の readiness-only follow-up は次を正本コマンドとする

```bash
PYTHONPATH=src .venv/bin/python scripts/run_local_nankan_future_only_followup_oneshot.py \
  --upstream-manifest artifacts/reports/local_nankan_pre_race_capture_loop_issue122_cycle.json \
  --max-upstream-age-seconds 7200 \
  --run-bootstrap-on-ready
```

- launch 前に freshness / contract 判定だけ確認したいときは、同じ入口に `--dry-run` を付けて `status=dry_run`, `current_phase=followup_plan_ready` を確認する

- operator は manifest を `status -> current_phase -> recommended_action -> upstream_refresh.upstream_fresh -> upstream_refresh.age_seconds -> upstream_refresh.contract_valid` の順で読む
- top-level の `observed_at` は freshness 判定に使った基準時刻そのものであり、`upstream_refresh.age_seconds` はその同時刻との差分として読む
- `status=not_ready` かつ `current_phase=await_external_refresh_completion` の場合は refresh artifact 自体が未生成なので、capture refresh 側を先に完走させる
- `status=not_ready` かつ `current_phase=invalid_upstream_refresh_contract` の場合は、contract 不一致 artifact を参照しているため、`execution_role=pre_race_capture_refresh_loop` など self-describing fields を持つ capture refresh manifest を再生成してから再実行する
- `status=not_ready` かつ `current_phase=await_fresh_external_refresh_completion` の場合は freshness 超過なので、古い artifact を再利用せず capture refresh を再実行する
- `child_launch_allowed=true` の場合だけ `run_local_nankan_future_only_wait_then_cycle.py --oneshot` が起動され、follow-up manifest 側の child command / log / wait-cycle manifest を追えば readiness-only 分岐を追跡できる
- status board を一次参照にする場合は、`readiness_surfaces.followup_entrypoint` から同じ follow-up 正本入口と dry-run/run preview をそのまま辿れる