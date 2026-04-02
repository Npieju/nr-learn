# Next Issue: NAR Baseline Pathfixed Rerun For Policy Audit

## Summary

`#70` の first read で、baseline narrow と no-market audit の差分はかなり見えた。  
ただし baseline narrow 側の historical promotion gate は generic `wf_summary` を auto-resolve しており、fold-level exact compare には制約がある。

## Objective

baseline narrow を current path-fixed revision gate で再実行し、versioned `evaluation summary`, `wf_feasibility`, `promotion gate` を同一 tuple で揃えて、policy optimism の phase compare を exact にする。

## Hypothesis

if current baseline narrow line の formal ROI の持ち上がりは promotion / policy selection によるものなら, path-fixed rerun でも `evaluation summary -> wf_feasibility -> promotion gate` の差分として再現される。

## In Scope

- `scripts/run_local_revision_gate.py`
- `scripts/run_revision_gate.py`
- `configs/model_local_baseline_wf_runtime_narrow.yaml`
- `configs/data_local_nankan.yaml`
- `configs/features_local_baseline.yaml`

## Non-Goals

- no-market 再実行
- 新しい NAR feature family 実験
- policy correction の実装

## Success Metrics

- versioned `wf_summary` が自動生成される
- versioned `promotion_gate` が manual recovery なしで出る
- baseline narrow の phase compare を no-market と同じ粒度で比較できる

## Validation

- `python scripts/run_local_revision_gate.py ...`
- output files:
  - versioned evaluation summary
  - versioned wf feasibility summary
  - versioned promotion gate

## Stop Condition

- rerun が quiet lane で安定しない
- historical baseline と再実行 baseline の shape が大きく乖離し、別の integrity issue に切り分けが必要になる
