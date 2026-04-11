# Issue Library Index

この directory は `next_issue_*.md` の local source/reference library である。

ここを raw listing のまま読ませないために、まずこの index から入る。

## Reading Rules

- `Historical note:` が先頭にある file は、current active draft ではなく historical decision/reference として扱う。
- `Historical note:` が無い file も、自動的に active とはみなさない。まず `Current Active Local Source Drafts` を見る。
- GitHub issue thread に source-of-truth が移った後は、重複する local draft を削除候補として扱う。

## Sub-Indexes

- [jra_pruning_index.md](jra_pruning_index.md)
  - active JRA pruning source, human review package, family-level audits, staged boundary references
- [jra_operator_policy_index.md](jra_operator_policy_index.md)
  - operator ordering, seasonal fallback, policy/runtime historical references
- [nar_issue_index.md](nar_issue_index.md)
  - current NAR source drafts and NAR historical references
- [ops_runtime_index.md](ops_runtime_index.md)
  - data / loader / runtime / residual historical references

## Current Active Local Source Drafts

- JRA active source: [jra_pruning_index.md](jra_pruning_index.md)
- NAR active sources: [nar_issue_index.md](nar_issue_index.md)

## Current Human Review Package

- [../jra_pruning_staged_decision_summary_20260411.md](../jra_pruning_staged_decision_summary_20260411.md)
- [../jra_pruning_package_review_20260410.md](../jra_pruning_package_review_20260410.md)
- [jra_pruning_index.md](jra_pruning_index.md)

## Usage

- current priority を追うときは [../github_issue_queue_current.md](../github_issue_queue_current.md) から入る。
- JRA pruning は [jra_pruning_index.md](jra_pruning_index.md)、JRA operator/policy は [jra_operator_policy_index.md](jra_operator_policy_index.md)、NAR は [nar_issue_index.md](nar_issue_index.md)、ops/runtime は [ops_runtime_index.md](ops_runtime_index.md) から入る。
- raw な directory listing を直接舐めるのは最後だけにする。

## Maintenance Rule

- この index は category entrypoint が変わったときだけ更新する。
- 個別 issue doc の細部更新のたびに、ここへ平行反映しない。