# Next Issue: NAR WF Summary Path Alignment

## Summary

`#64` の `jockey / trainer / combo` replay は denominator-first formal read では `pass / promote` だったが、formal run 本体の promotion gate は `wf_summary` path mismatch で block した。model judgement は recovered できた一方で、ops path は未整合のままである。

## Objective

local Nankan revision gate で、`run_wf_feasibility_diag.py` の actual output path と `run_revision_gate.py` / `run_local_revision_gate.py` が期待する `wf_summary` path を一致させ、manual promotion recovery を不要にする。

## Hypothesis

if `wf_feasibility` output naming を versioned artifact path に揃えるか、promotion gate 側で actual emitted path を確実に参照するようにする, then local Nankan formal runs are end-to-end reproducible without manual promotion recovery.

## Current Read

- `#64` recovered formal:
  - `formal_benchmark_weighted_roi=4.324456844807267`
  - `formal_benchmark_bets_total=3725`
  - `wf_feasible_fold_count=3`
  - `bet_rate=12.85%` on `28997` test races
- formal run failure:
  - `promotion_gate_r20260330_local_nankan_jockey_trainer_combo_replay_v1.json`
  - `status=failed`
  - `blocking_reasons=["[Errno 32] Broken pipe"]`
- actual usable `wf_summary`:
  - `artifacts/reports/wf_feasibility_diag_local_baseline_wf_runtime_narrow_wf_fast_nested.json`
  - `run_context.feature_config=configs/features_local_baseline_jockey_trainer_combo_replay.yaml`
  - `run_context.artifact_suffix=r20260330_local_nankan_jockey_trainer_combo_replay_v1`
- expected but missing path:
  - `artifacts/reports/wf_feasibility_diag_r20260330_local_nankan_jockey_trainer_combo_replay_v1.json`
- first cut landed:
  - `run_wf_feasibility_diag.py` now accepts `--summary-output` and `--detail-output`
  - `run_revision_gate.py` now passes versioned `wf_summary` / `.csv` paths into the child feasibility command
  - dry-run confirmation:
    - child command now includes `--summary-output artifacts/reports/wf_feasibility_diag_dryrun_nar_wf_path_alignment.json`
    - promotion command references the same path

## In Scope

- `scripts/run_wf_feasibility_diag.py`
- `scripts/run_revision_gate.py`
- `scripts/run_local_revision_gate.py`
- local Nankan revision manifest / promotion pointer alignment

## Non-Goals

- new feature family の追加
- NAR baseline の threshold 再設計
- JRA line の policy tuning

## Success Metrics

- revision gate が emitted `wf_summary` path を正しく manifest に残す
- promotion gate が manual rerun なしで `pass / hold / reject` まで完走する
- local wrapper から見た artifact path が deterministic である

## Validation

- `python -m py_compile scripts/run_wf_feasibility_diag.py scripts/run_revision_gate.py scripts/run_local_revision_gate.py`
- `PYTHONPATH=src .venv/bin/python -m unittest ...`
- local Nankan dry-run
- narrow local revision rerun 1 本で
  - versioned `wf_summary` exists
  - promotion gate auto-completes

## Stop Condition

- fix が generic non-versioned artifact consumers を壊す
- manifest path だけ直って actual file が still unstable なら reject
