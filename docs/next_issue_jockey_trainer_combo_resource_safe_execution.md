# Next Issue: Resource-Safe Execution For Jockey Trainer Combo Candidate

## Summary

`#50` の first candidate `r20260330_jockey_trainer_combo_style_distance_v1` は、feature 仮説そのものではなく execution capacity で止まっている。

current read:

- dry-run は通過済み
- win component は feature build / selection まで正常
- selected features は `109`
- clean rerun では Python exception ではなく `exit 137` の OS kill が出た
- kill point は CatBoost fit 段階で、feature config bug より resource pressure の疑いが強い

したがって next work は family hypothesis の変更ではなく、resource-safe な rerun lane を切ることである。

## Objective

`r20260330_jockey_trainer_combo_style_distance_v1` を experiment meaning を壊さずに再実行できるようにし、`#50` を formal lane に戻す。

## Current Read

- candidate config:
  - `configs/features_catboost_rich_high_coverage_diag_jockey_trainer_combo_regime_extension.yaml`
- dry-run manifest:
  - `artifacts/reports/revision_gate_r20260330_jockey_trainer_combo_style_distance_v1.json`
- clean rerun read:
  - `loading training table` 完了
  - `features built columns=165`
  - `feature selection ready features=109 categorical=37`
  - leakage audit 完了
  - fit 段階で `exit 137`

## In Scope

- resource-safe rerun rule の明文化
- `#50` 実行時の duplicate / concurrent heavy job 整理
- 必要なら reduced execution fallback の issue 分離
- `run_train.py` の fail-fast preflight で quiet heavy-job lane を要求すること

## Non-Goals

- family hypothesis の変更
- feature config の再設計
- serving role split の再議論

## Success Criteria

- `#50` の win component を 1 本だけ clean に再実行できる
- failure が feature bug か capacity かを再度曖昧にしない
- family read と execution blocker read が issue 上で分離される

## Current Implementation

- `scripts/run_train.py` は既定で concurrent heavy job を preflight で検査する
- conflict がある場合は OS kill を待たず、concise な fail-fast message で止まる
- override が必要なときだけ `--allow-concurrent-heavy-jobs` を明示する
