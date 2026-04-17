# Next Issue: NAR Result-Ready Bootstrap Handoff

## Summary

`#101` で strict `pre_race_only` subset から benchmark gate まで進む handoff は実装済みで、`#103` では NAR `value_blend` bootstrap scaffold も整合済みである。

current status:

- wrapper / runtime-config orchestration surface は実装済み
- `#101` historical trust-ready formal rerun も完了済み
- current role は external result arrival 後の将来再開だけでなく、`#103` child execution に再利用できる command plan / operator surface を保持することにある

historical issue としては、result-ready race が到着したときに

- `#101` handoff rerun
- strict `pre_race_only` benchmark freeze
- `#103` win / ROI / stack / revision gate

を 1 本の再開導線として起動できない。

したがってこの practical issue 自体は実装済みであり、current read では `#101` benchmark reference と `#103` bootstrap scaffold を結ぶ reusable orchestration surface として扱う。

## Objective

result-ready strict `pre_race_only` race が到着した時点で、dedicated pre-race-ready data config を使って

1. benchmark handoff
2. NAR `value_blend` bootstrap component train
3. stack build
4. revision gate

まで resume できる orchestration surface を作る。

## Hypothesis

if result-ready strict `pre_race_only` handoff と `#103` bootstrap scaffold を 1 本の orchestration に束ねる, then NAR parity work は future-only operator path でも historical benchmark-reference maintenance でも同じ command discipline を再利用できる。

## In Scope

- dedicated `pre_race_ready` data config
- result-ready handoff wrapper
- `#103` bootstrap command plan
- optional bootstrap auto-run orchestration
- resume manifest / progress logging

## Non-Goals

- result arrival 自体を早めること
- provenance rules の再設計
- JRA baseline work
- NAR feature family の追加実験

## Success Metrics

- result-ready race が 0 件のときは bounded `not_ready` で止まる
- result-ready race があるときは benchmark-ready surface まで進める
- `#103` bootstrap 用の data config / command plan が固定される
- manual path without remembering ad hoc commands が不要になる

## Validation Plan

1. pre-race-ready dedicated data config を追加する
2. result-ready handoff wrapper に `#103` bootstrap command plan を接続する
3. `not_ready` smoke で bounded stop を確認する
4. ready artifact が揃った場合に次段 command plan が manifest に出ることを確認する

## Stop Condition

- `#101` handoff 自体が current fileset では benchmark-ready output を安定して作れない
- `#103` scaffold が dedicated pre-race-ready data config と整合しない
- その場合は automation を止め、manual runbook のみを維持する

## First Cut

result-ready 到着後の resume surface と dedicated data config を追加した。

- data config:
  - [data_local_nankan_pre_race_ready.yaml](/workspaces/nr-learn/configs/data_local_nankan_pre_race_ready.yaml)
- helper:
  - [local_nankan_bootstrap.py](/workspaces/nr-learn/src/racing_ml/data/local_nankan_bootstrap.py)
- wrapper:
  - [run_local_nankan_result_ready_bootstrap_handoff.py](/workspaces/nr-learn/scripts/run_local_nankan_result_ready_bootstrap_handoff.py)
- test:
  - [test_local_nankan_bootstrap.py](/workspaces/nr-learn/tests/test_local_nankan_bootstrap.py)

wrapper は次を固定する。

1. `#101` handoff を dedicated pre-race-ready paths で起動する
2. `not_ready` のときは bounded stop で終える
3. `#103` 用 bootstrap command plan を manifest に残す
4. `--run-bootstrap` を付けた場合だけ、win / ROI / stack / revision gate を順に実行する
5. heavy child command の stdout / stderr は `artifacts/logs/` 配下の step 別 log file に残す
6. future-only wait-cycle supervisor 経由では cycle-scoped log prefix を使い、arrival 後の handoff/bootstrap log が cycle history と 1:1 で対応する
7. future-only readiness cycle 経由では `pre_race_benchmark_handoff` / `pre_race_ready_summary` / `pre_race_ready_primary_materialize` / `pre_race_ready_benchmark_gate` も cycle-scoped report path に分離する
8. future-only wait-cycle supervisor 経由では bootstrap revision も cycle-scoped にし、runtime config と downstream model/report output の上書きを避ける
9. future-only wait-cycle supervisor 経由では filtered pre-race card/result/primary CSV も cycle-scoped にし、中間 data artifact の上書きを避ける

validation:

- `python -m py_compile scripts/run_local_nankan_result_ready_bootstrap_handoff.py src/racing_ml/data/local_nankan_bootstrap.py tests/test_local_nankan_bootstrap.py`
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_local_nankan_bootstrap`
- YAML parse:
  - `configs/data_local_nankan_pre_race_ready.yaml`
  - `configs/features_catboost_rich_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml`
  - `configs/model_catboost_win_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml`
  - `configs/model_lightgbm_roi_high_coverage_diag_local_nankan_value_blend_bootstrap.yaml`
  - `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_local_nankan_value_blend_bootstrap.yaml`
- smoke:
  - [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json)

confirmed read:

- `status=not_ready`
- `current_phase=await_result_arrival`
- `recommended_action=wait_for_result_ready_pre_race_races`
- `bootstrap_command_plan` length=`4`
- first step=`train_win_component`
- `handoff_command_result.log_file` と `bootstrap_command_plan[*].log_file` が `artifacts/logs/` を指す

meaning:

- current blocker そのものではなく、到着後の resume path と runtime-config discipline が fixed されたことが主成果である
- 実装された surface は historical trust-ready rerun 完了後も再利用できる:
  - benchmark handoff
  - bootstrap win / ROI train
  - stack build
  - revision gate

## Second Cut

bootstrap surface に revision 固有 runtime config materialization を追加した。

- helper:
  - [local_nankan_bootstrap.py](/workspaces/nr-learn/src/racing_ml/data/local_nankan_bootstrap.py)
- wrapper:
  - [run_local_nankan_result_ready_bootstrap_handoff.py](/workspaces/nr-learn/scripts/run_local_nankan_result_ready_bootstrap_handoff.py)
- test:
  - [test_local_nankan_bootstrap.py](/workspaces/nr-learn/tests/test_local_nankan_bootstrap.py)

追加したこと:

- win / ROI / stack config を `artifacts/runtime_configs/` 配下へ revision 固有名で materialize
- runtime stack config の component 参照を generated win / roi config へ差し替え
- bootstrap command plan を unsuffixed base config ではなく generated runtime config に向ける

validation:

- `python -m py_compile scripts/run_local_nankan_result_ready_bootstrap_handoff.py src/racing_ml/data/local_nankan_bootstrap.py tests/test_local_nankan_bootstrap.py`
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_local_nankan_bootstrap`
- smoke:
  - [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json)

confirmed read:

- `status=not_ready`
- `runtime_configs.win_config=artifacts/runtime_configs/model_catboost_win_local_nankan_value_blend_bootstrap_<revision>.yaml`
- `runtime_configs.roi_config=artifacts/runtime_configs/model_lightgbm_roi_local_nankan_value_blend_bootstrap_<revision>.yaml`
- `runtime_configs.stack_config=artifacts/runtime_configs/model_value_stack_local_nankan_value_blend_bootstrap_<revision>.yaml`

meaning:

- revision 固有 runtime config により rerun は artifact 単位で隔離できる
- current `#103` first execution でも、この runtime-config discipline をそのまま再利用できる

## Operator Runbook

future-only operator path の再開や wrapper-based bootstrap 実行では、この文書の wrapper を唯一の operator entrypoint として扱う。一方で historical trust-ready benchmark judgment 自体は `#101` pointer / summary / manifest を一次参照にする。

current blocker を読むときの一次参照は次である。

- board:
  - [local_nankan_data_status_board.json](/workspaces/nr-learn/artifacts/reports/local_nankan_data_status_board.json)
- read order:
  1. top-level `status`
  2. top-level `current_phase`
  3. top-level `recommended_action`
  4. `readiness_surfaces.capture_loop`
  5. `readiness_surfaces.readiness_probe`
  6. `readiness_surfaces.pre_race_handoff`
  7. `readiness_surfaces.bootstrap_handoff`
  8. `readiness_surfaces.readiness_watcher`
  9. `readiness_surfaces.followup_entrypoint`

operator rule:

- board が `status=partial`, `current_phase=await_result_arrival` の間は rerun しない
- board が `bootstrap_handoff.status=benchmark_ready` または `completed` に進んだときだけ child execution / formal read に移る
- refresh 完了直後の readiness-only 再確認は `readiness_surfaces.followup_entrypoint` の dry-run/run preview を正本入口として使う
- board だけで足りないときに限り、この文書の個別 manifest 順へ降りる

current benchmark-reference rule:

- `#101` の current benchmark judgment は board の blocked-state ではなく、[evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json](/workspaces/nr-learn/artifacts/reports/evaluation_local_nankan_r20260415_local_nankan_pre_race_ready_formal_v1_pointer.json) と versioned summary / manifest を一次参照にする
- wrapper manifest の `runtime_configs` と `bootstrap_command_plan` は、`#103` 実行時の canonical command surface として再利用してよい

default command:

```bash
cd /workspaces/nr-learn
PYTHONPATH=/workspaces/nr-learn/src /workspaces/nr-learn/.venv/bin/python \
  scripts/run_local_nankan_result_ready_bootstrap_handoff.py \
  --wait-for-results \
  --poll-interval-seconds 300 \
  --run-bootstrap
```

phase read:

1. `status=not_ready`
  - result-ready race 未到着
  - action: rerun せず blocked 維持
2. `status=benchmark_ready`
  - strict `pre_race_only` freeze 完了
  - action: manifest の `bootstrap_command_plan` と `runtime_configs` を確認して child 実行へ進める
3. `status=completed`
  - `#101 -> #103` handoff が end-to-end で完了
  - action: revision outputs を formal read して issue judgment に進む

review order:

1. [local_nankan_pre_race_ready_summary.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_ready_summary.json)
2. [local_nankan_pre_race_benchmark_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_pre_race_benchmark_handoff_manifest.json)
3. [local_nankan_result_ready_bootstrap_handoff_manifest.json](/workspaces/nr-learn/artifacts/reports/local_nankan_result_ready_bootstrap_handoff_manifest.json)
4. generated runtime configs under [artifacts/runtime_configs](/workspaces/nr-learn/artifacts/runtime_configs)

decision rule:

- `status=not_ready` は infra failure ではなく external blocker として扱う
- `status=failed` は `current_phase` を見て `#101` handoff failure か `#103` child command failure かを切り分ける
- `status=completed` のときだけ architecture parity first read に進める
- `#101` formal rerun 完了後は、board / wrapper の `not_ready` をそのまま historical benchmark blocker と誤読しない

## Queue Update Template

`#101` と `#103` が進んだ後に [github_issue_queue_current.md](/workspaces/nr-learn/docs/github_issue_queue_current.md) へ戻す summary は、次の block を起点にする。

```text
NAR completion update:

1. `#101` strict `pre_race_only` benchmark rebuild は <completed|still blocked|failed> である
2. result-ready read:
  - races=<races>
  - rows=<rows>
  - ready_for_benchmark_rerun=<true|false>
3. `#103` value-blend architecture bootstrap は <completed|benchmark_ready|blocked|failed> である
4. if completed:
  - revision=<revision>
  - weighted_roi=<weighted_roi>
  - feasible_folds=<feasible_folds>/<total_folds>
  - decision=<promote|hold|reject>
5. next NAR stage is <Stage 2 feature-family parity|keep blocked on Stage 0/1>
```

queue write rule:

- `#101` が未完了なら current blocker は引き続き Stage 0 と書く
- `#101` 完了かつ `#103` 未完了なら current blocker は Stage 1 architecture parity と書く
- `#103` まで完了したら、`docs/issue_library/next_issue_nar_class_rest_surface_replay.md` を Stage 2 future option から current next candidate へ繰り上げるかを明示する
