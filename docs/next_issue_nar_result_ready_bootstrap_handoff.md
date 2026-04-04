# Next Issue: NAR Result-Ready Bootstrap Handoff

## Summary

`#101` で strict `pre_race_only` subset から benchmark gate まで進む handoff は実装済みで、`#103` では NAR `value_blend` bootstrap scaffold も整合済みである。

ただし current surface では、result-ready race が到着しても

- `#101` handoff rerun
- strict `pre_race_only` benchmark freeze
- `#103` win / ROI / stack / revision gate

を 1 本の再開導線として起動できない。

したがって次の practical issue は、result-ready 到着時に `#101 -> #103` を再現可能な 1 entrypoint に束ねることにある。

## Objective

result-ready strict `pre_race_only` race が到着した時点で、dedicated pre-race-ready data config を使って

1. benchmark handoff
2. NAR `value_blend` bootstrap component train
3. stack build
4. revision gate

まで resume できる orchestration surface を作る。

## Hypothesis

if result-ready strict `pre_race_only` handoff と `#103` bootstrap scaffold を 1 本の orchestration に束ねる, then NAR parity work は external result arrival 後に手作業なしで immediately resume できる。

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

meaning:

- current blocker は引き続き result-ready race の未到着だけである
- ただし到着後の resume path は now fixed:
  - benchmark handoff
  - bootstrap win / ROI train
  - stack build
  - revision gate
