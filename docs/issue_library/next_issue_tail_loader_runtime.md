# Next Issue: Tail Loader Runtime Reduction

Historical note:

- この draft は `#18` として runtime reduction まで完了している。
- tail loader runtime reduction は completed runtime reference として扱い、この文書は historical issue source / implementation reference として使う。

## Summary

`#17` により feature builder 側の low-risk fixed-cost 削減は一段進み、reduced smoke の floor は `0m27s` まで下がった。2026-03-29 の profile では、次の本命 bottleneck は再び loader 側に戻っている。

特に大きいのは次の 2 点である。

- `src/racing_ml/data/dataset_loader.py:_read_csv_tail`
- `src/racing_ml/data/dataset_loader.py:load_training_table_tail`

つまり次の perf 本線は feature build ではなく、tail read / tail selection の fixed cost 削減である。

## Objective

`run_evaluate.py` と challenger flow の pre-feature load 時間を、dataset semantics を変えずに削減する。

## Initial Scope

- `_read_csv_tail` の chunk handling / tail compaction / candidate selection を profile-guided に見直す
- `load_training_table_tail` の前後で重複している sort / trim / column prep があれば削る
- current default data config と materialized supplemental path を前提に最適化する
- feature semantics、join semantics、selected row set は変えない

## Non-Goals

- new feature family の追加
- training/evaluation KPI の変更
- supplemental schema の再設計
- NAR ingestion の変更

## Acceptance

- reduced smoke または loader-only profile で repeatable な改善がある
- regression test が通る
- selected rows / summary / manifest に semantic drift がない

## Suggested Validation

- `python -m py_compile src/racing_ml/data/dataset_loader.py`
- `PYTHONPATH=src .venv/bin/python -m unittest tests.test_dataset_loader tests.test_policy tests.test_walk_forward tests.test_revision_gate tests.test_serving_smoke`
- `PYTHONPATH=src .venv/bin/python scripts/run_evaluate.py --config ... --data-config configs/data_2025_latest.yaml --artifact-suffix ... --model-artifact-suffix r20260326_tighter_policy_ratio003 --max-rows 5000 --pre-feature-max-rows 10000 --wf-mode fast --wf-scheme nested`
