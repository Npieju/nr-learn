# Next Issue: Tail Training-Table Phase Budget Attribution

## Summary

`#25` では `_append_external_tables` を集中的に掘り、次の 1 本だけを accepted cut として landing できた。

- append dedupe を `drop_duplicates(..., ignore_index=True)` に寄せる
- `_sort_and_tail(..., max_rows=None)` が `RangeIndex(0..)` の frame では余分な `reset_index(drop=True)` を行わない

この cut 自体は ad hoc benchmark で append dedupe/tail section を `0.1882s -> 0.0506s` まで下げ、reduced smoke `perf_smoke_append_external_v1` でも `loading training table 0m14s`, total `0m24s`、summary drift なしで通った。

一方で、その後の候補は append subphase の micro benchmark では改善しても、reduced smoke 全体では一貫して `0m16s / 0m26s` に戻った。

- append `keep_columns/usecols` candidate: reject
- append `low_memory + dtype` exact-safe candidate: reject
- append recent-date filter fast path: reject

つまり current bottleneck は「append 単体の pure local cost」ではなく、`tail_training_table` 全体の phase interaction と variance にある可能性が高い。次の issue は append だけをさらに削るより、tail load の phase budget を end-to-end で再分解し、micro win が mainline に乗らない理由を切り分けるべきである。

## Objective

`tail_training_table` の full load path を phase budget として再計測し、micro benchmark では効くのに reduced smoke で消える候補を、phase interaction / cache effect / duplicated work の観点から説明できる状態にする。

## In Scope

- `src/racing_ml/data/dataset_loader.py`
- `scripts/run_evaluate.py`
- tail load breakdown snippets
- reduced smoke A/B methodology
- docs / issue decision logging

## Non-Goals

- 新しい model / feature / policy family の実験
- canonical drift を許容する runtime 近道
- NAR ingestion/readiness
- broad loader rewrite

## Success Criteria

- `tail_training_table` の phase budget を current mainline で再現できる
- micro win が mainline に効かない理由を 1 つ以上特定できる
- 次に切る runtime issue を `append`, `tail`, `minimum columns`, `feature build` のどこへ戻すべきかを明確にできる

## Suggested Validation

- current mainline の loader-phase timing breakdown
- reduced smoke A/B の repeated runs
- summary equivalence manifest
- 必要なら `cProfile` / targeted timing snippet

## Expected Outputs

- phase budget summary
- accepted explanation or follow-up hotspot selection
- new next-issue draft if the hotspot shifts

## Current Read

current mainline の repeated reduced smoke を `perf_smoke_phase_budget_a` / `perf_smoke_phase_budget_b` で 2 本連続実行したところ、どちらも同じ着地だった。

- `loading training table`: `0m15s`
- total elapsed: `0m25s`
- versioned summary は同値

つまり直近の `0m14s` と `0m16s` は、少なくとも 1 秒単位の run-to-run variance を含んでいる。append 単体の micro win が mainline に乗らない理由を論じる前に、この baseline variance を current issue の正本として押さえる必要がある。

この initial read により、次の focus は「append subphase の絶対値」よりも、「current mainline の phase budget を repeated smoke でどこまで再現できるか」に移った。

さらに `tail_training_table` の phase budget を current mainline で 3 回連続計測すると、次のようになった。

- run1
  - `read_tail 13.5398s`
  - `append 0.6867s`
  - `supplemental 3.2791s`
  - `minimum 1.1898s`
  - total `18.7646s`
- run2
  - `read_tail 13.5254s`
  - `append 0.6897s`
  - `supplemental 3.3483s`
  - `minimum 1.3054s`
  - total `18.9425s`
- run3
  - `read_tail 12.8691s`
  - `append 0.6892s`
  - `supplemental 3.4708s`
  - `minimum 1.2609s`
  - total `18.3422s`

この repeated read から、append は `~0.69s` でほぼ安定しており、直近で試した append-side micro cuts が full reduced smoke で効かなかったのは自然だと分かる。current mainline の dominant phase は依然 `_read_csv_tail` で、次点が `_merge_supplemental_tables` である。

したがって `#26` の結論は、append path をさらに深掘るよりも、next active issue を `_read_csv_tail` dominant phase へ戻すことが正しい、というものである。
