# Next Issue: Deque Trim Promotion Decision

## Summary

`#19` で tail loader equivalence harness は整備できた。現在の `deque_trim` candidate は次の状態である。

- `exact`: fail
- `canonical`: pass
- `value`: pass

つまり value drift は見えていないが、dtype drift は残っている。現標準では `canonical` pass は analysis / optimization exploration に使えるが、promotion-ready ではない。

次の本線は、`deque_trim` を broad replacement に昇格させること自体ではなく、「analysis-only のまま据え置くか」「exact-equivalent に近づける work を切るか」を artifact ベースで決めることだ。

## Objective

`deque_trim` candidate の current status を formal に整理し、analysis-only retention / exact-safe follow-up / explicit abandon のどれに進むかを決める。

## In Scope

- `artifacts/reports/tail_loader_equivalence_deque_trim.json`
- `artifacts/reports/summary_equivalence_perf_smoke_surface_vs_materialized_corner.json`
- `scripts/run_tail_loader_equivalence.py`
- `scripts/run_summary_equivalence.py`
- `src/racing_ml/data/tail_equivalence.py`
- `docs/tail_loader_equivalence_gate_standard.md`

## Non-Goals

- `deque_trim` を即 default に昇格すること
- unrelated loader perf work
- feature / policy / benchmark family changes
- NAR ingestion/readiness

## Success Criteria

- `deque_trim` の current status を `analysis-only`, `needs exact-safe follow-up`, `abandon` のいずれかで明示できる
- next implementation issue が必要なら、その objective が 1 文で言える
- exact / canonical / value の gate 解釈が issue thread だけでなく docs と一致している

## Suggested Validation

- `PYTHONPATH=src .venv/bin/python scripts/run_tail_loader_equivalence.py --raw-dir data/raw --tail-rows 10000 --left-reader current --right-reader deque_trim --manifest-file artifacts/reports/tail_loader_equivalence_deque_trim.json --fail-on-diff --fail-gate exact`
- `PYTHONPATH=src .venv/bin/python scripts/run_tail_loader_equivalence.py --raw-dir data/raw --tail-rows 10000 --left-reader current --right-reader deque_trim --manifest-file artifacts/reports/tail_loader_equivalence_deque_trim.json --fail-on-diff --fail-gate canonical`
- `PYTHONPATH=src .venv/bin/python scripts/run_summary_equivalence.py --left-summary <left> --right-summary <right> --manifest-file <manifest> --fail-on-diff`

## Expected Outputs

- decision comment on whether `deque_trim` stays analysis-only
- optional follow-up issue for exact-safe tail optimization
- if no follow-up is warranted, explicit abandon / park decision

## Final Read

- recheck validation:
	- `PYTHONPATH=src .venv/bin/python scripts/run_tail_loader_equivalence.py --raw-dir data/raw --tail-rows 10000 --left-reader current --right-reader deque_trim --manifest-file artifacts/tmp/deque_trim_exact_recheck.json --fail-on-diff --fail-gate exact`
	- `exit_code=2`
	- raw / normalized `exact_equal=false`
	- raw / normalized `canonical_dtype_only_difference=true`
- canonical recheck:
	- `PYTHONPATH=src .venv/bin/python scripts/run_tail_loader_equivalence.py --raw-dir data/raw --tail-rows 10000 --left-reader current --right-reader deque_trim --manifest-file artifacts/tmp/deque_trim_canonical_recheck.json --fail-on-diff --fail-gate canonical`
	- `exit_code=0`
	- raw / normalized `canonical_dtype_only_difference=true`
- current drift surface:
	- value difference count `0`
	- downstream summary equivalence `exact_equal=true`
	- dtype drift は `all_null` な `レース記号/*` 列と `斤量` の `numeric_integral_equivalent` に限られる

## Decision

- `deque_trim` の current status は `analysis-only retention` とする
- `exact-safe follow-up` は新 issue に切らない
- `explicit abandon` にもせず、canonical-only optimization candidate として reference に留める

Reason:

- gate standard では promotion-ready は `exact` pass が前提であり、current recheck でも `exact` fail が再現した
- 一方で `canonical` / `value` は安定して pass し、downstream summary drift も見えていない
- したがって unsafe / valueless candidate ではないが、default replacement を正当化する根拠も足りない
- 追加の issue を切るなら generic `deque_trim` 昇格ではなく、別の exact-safe residual candidate を 1 hypothesis で切るべきである
