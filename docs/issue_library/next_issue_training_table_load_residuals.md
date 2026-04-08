# Next Issue: Training Table Load Residuals After Tail Cuts

Historical note:

- この draft は `#26` として phase-budget attribution まで完了している。
- training-table load residual attribution は completed runtime reference として扱い、この文書は historical issue source / profiling reference として使う。

## Summary

`#21` では `_read_csv_tail` に対して small exact-safe cut を複数 landing し、tail equivalence harness でも `exact` を維持できた。

一方で reduced smoke の real run では、依然として `loading training table` が約 `0m15s` を占めている。つまり次の本命は tail core そのものではなく、`tail_training_table` 全体の残余コストを分解して減らすことである。

## Objective

`tail_training_table` の loading phase を profile し、`_read_csv_tail` 以外の residual hotspot を特定して、次の exact-safe reduction candidate を切り出す。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- tail loader / append / supplemental merge / dataset pick 周辺
- `scripts/run_evaluate.py`
- `scripts/run_summary_equivalence.py`
- existing tail equivalence harness and manifests

## Non-Goals

- `canonical` pass のみで loader を promote すること
- feature builder / policy / benchmark family changes
- NAR ingestion/readiness

## Success Criteria

- `loading training table` の残余コストを phase 単位で説明できる
- next exact-safe candidate が 1 issue 1 hypothesis に切り出せる
- if no good candidate exists, `#21` 以降の tail line は diminishing returns として明示できる

## Suggested Validation

- reduced smoke with stable artifact and model artifact suffix
- if needed, temporary local profiling around dataset load phases
- `scripts/run_tail_loader_equivalence.py` for any `_read_csv_tail` touch
- `scripts/run_summary_equivalence.py` for reduced summary checks

## Expected Outputs

- residual load breakdown
- next implementation issue or explicit stop decision
- if a candidate is tried, equivalence and summary manifests
