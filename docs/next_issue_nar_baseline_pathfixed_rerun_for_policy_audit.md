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

## Final Read

path-fixed rerun は完了し、success metrics を満たした。

- versioned evaluation summary:
  - `artifacts/reports/evaluation_summary_local_nankan_baseline_model_r20260330_local_nankan_baseline_wf_runtime_narrow_pathfix_audit_v1.json`
- versioned wf feasibility:
  - `artifacts/reports/wf_feasibility_diag_r20260330_local_nankan_baseline_wf_runtime_narrow_pathfix_audit_v1.json`
- versioned promotion gate:
  - `artifacts/reports/promotion_gate_r20260330_local_nankan_baseline_wf_runtime_narrow_pathfix_audit_v1.json`

exact read:

- evaluation:
  - `auc=0.8775363015459904`
  - `ev_top1_roi=1.940849373663306`
  - nested WF: `3/3 no_bet`
  - `wf_nested_test_bets_total=0`
- formal:
  - `formal_benchmark_weighted_roi=3.293016875212951`
  - `formal_benchmark_bets_total=4323`
  - `bets / races = 4323 / 28997 = 14.91%`
  - `wf_feasible_fold_count=3`

結論として、baseline narrow line の optimism は path-fixed rerun でも再現した。  
したがって `#71` は exact compare rerun issue として close してよい。残る本線は corrective action である。
