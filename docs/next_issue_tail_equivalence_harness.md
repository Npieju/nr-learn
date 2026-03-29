# Next Issue: Tail Loader Equivalence Harness

## Summary

`#18` で loader runtime は small safe cuts により reduced smoke `0m25s` 付近まで改善した。一方で、本命 hotspot は引き続き `src/racing_ml/data/dataset_loader.py:_read_csv_tail` であり、より大きい改善余地は aggressive な tail compaction / selection path にある。

ただし 2026-03-29 の first pass では、deque-left trimming による高速化は runtime 改善を出した一方で reduced evaluation summary drift を起こし、unsafe と判断した。

つまり次の本線は runtime そのものではなく、tail optimization を安全に試すための equivalence harness 整備である。

## Objective

tail loader の候補最適化を、selected rows / downstream summary / manifest drift なしで検証できる harness を整備する。

## Initial Scope

- old/new tail path を同一 input で比較できる test / utility を整備する
- row identity だけでなく normalized frame / key columns / downstream smoke summary まで比較できるようにする
- unsafe optimization を再度 landing すること自体は goal にしない
- downstream reduced-smoke summary compare を JSON manifest と non-zero exit で再利用できるようにする

## Non-Goals

- new feature families
- KPI 変更
- serving policy 変更
- NAR ingestion work

## Acceptance

- tail loader candidate を old/new で比較する regression path がある
- reduced smoke の summary equality まで比較できる
- future aggressive tail optimization を confidence を持って試せる
- `exact` / `canonical` / `value` gate の使い分けが docs と CLI で固定されている

## Suggested Validation

- `python -m py_compile src/racing_ml/data/dataset_loader.py tests/test_dataset_loader.py`
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_dataset_loader tests.test_policy tests.test_walk_forward tests.test_revision_gate tests.test_serving_smoke`
- old/new tail path comparison utility or test
- reduced smoke summary comparison against fixed artifact suffix pair
