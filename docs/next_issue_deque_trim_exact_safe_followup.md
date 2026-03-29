# Next Issue: Deque Trim Exact-Safe Follow-Up

## Summary

`#20` の結論として、`deque_trim` は 2026-03-29 時点では `canonical` pass / `exact` fail の candidate であり、analysis-only に据え置く。

一方で loader runtime の本命 hotspot は引き続き `_read_csv_tail` であり、`deque_trim` 系の speedup 余地自体は残っている。したがって次の issue は「canonical で通るか」ではなく、「exact-equivalent を保ったまま同等の runtime 改善を出せるか」である。

## Objective

`_read_csv_tail` の exact-safe optimization candidate を設計し、`exact` gate を通せる新しい tail path を探索する。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- `src/racing_ml/data/tail_equivalence.py`
- `scripts/run_tail_loader_equivalence.py`
- `scripts/run_summary_equivalence.py`
- `docs/tail_loader_equivalence_gate_standard.md`

## Non-Goals

- `canonical` pass のみで default replacement を認めること
- unrelated feature / policy / benchmark changes
- NAR ingestion/readiness

## Success Criteria

- new tail candidate が `exact` gate を通る
- reduced-smoke summary compare でも stable に一致する
- runtime 改善が小さくても、exact-safe path として評価可能な形で残る

## Suggested Validation

- `PYTHONPATH=src .venv/bin/python scripts/run_tail_loader_equivalence.py --raw-dir data/raw --tail-rows 10000 --left-reader current --right-reader <candidate> --manifest-file <manifest> --fail-on-diff --fail-gate exact`
- `PYTHONPATH=src .venv/bin/python scripts/run_summary_equivalence.py --left-summary <left> --right-summary <right> --manifest-file <manifest> --fail-on-diff`
- reduced smoke A/B with fixed artifact suffixes

## Expected Outputs

- exact-safe tail candidate or explicit reject decision
- equivalence manifest
- summary equivalence manifest
- if promising, measured runtime comparison
