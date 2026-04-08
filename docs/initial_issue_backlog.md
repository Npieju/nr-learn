# Initial Issue Backlog

## 1. Primary Epic

`EPIC: JRA long-run ROI > 1.20`

この epic の完了条件は、JRA 正本データで長期運用に耐える formal candidate 群を作り、少なくとも 1 本を `ROI > 1.20` の仮説レンジに近づける検証サイクルを安定運用できることである。

## 2. Parallel Epic

`EPIC: NAR ingestion readiness`

この epic は benchmark 直接改善ではなく、将来の universe 拡張のための準備トラックとする。

## 3. First Wave Issues

### 3.1 JRA model / policy

1. `ISSUE: Formal KPI definition for ROI>1.20`
2. `ISSUE: Current JRA baseline artifact inventory and gap audit`
3. `ISSUE: ML model development stage checklist standardization`
4. `ISSUE: Feature family ranking from existing artifacts`
5. `ISSUE: Policy family shortlist for high-ROI / controlled-drawdown experiments`
6. `ISSUE: Revision gate candidate checklist standardization`

### 3.2 Automation / process

1. `ISSUE: GitHub Project schema and automation setup`
2. `ISSUE: Artifact-to-issue linking convention`
3. `ISSUE: Benchmark update checklist template`
4. `ISSUE: PR review checklist for strategy changes`
5. `ISSUE: Progress instrumentation hardening for long-running wrappers`

### 3.3 NAR readiness

1. `ISSUE: NAR raw ingestion completion checklist`
2. `ISSUE: NAR schema parity against JRA feature inputs`
3. `ISSUE: NAR readiness gate definition`

## 4. Suggested Weekly Cadence

- 月曜: backlog grooming と issue slicing
- 平日: experiment issue 実行
- 木曜または金曜: promising candidate だけ formal gate
- 週末前: roadmap / benchmark / public snapshot の更新要否を判断

## 5. Immediate Next Step

最初に着手するべきは次の 2 本である。

1. `Formal KPI definition for ROI>1.20`
2. `Current JRA baseline artifact inventory and gap audit`

理由は、いまのリポジトリには評価資産が十分ある一方で、「ROI > 1.20 に向けて何を success とみなすか」の運用定義がまだ薄いためである。

この 2 本の直後に、`ML model development stage checklist standardization` を切ると、自動 coding の迷いがかなり減る。

issue の具体的な下書きは [issue_drafts_first_wave.md](issue_drafts_first_wave.md) を使う。

baseline inventory の初版は [jra_baseline_artifact_inventory.md](jra_baseline_artifact_inventory.md) を参照する。

feature family の初版 ranking は [feature_family_ranking.md](feature_family_ranking.md) を参照する。

policy family の初版 shortlist は [policy_family_shortlist.md](policy_family_shortlist.md) を参照する。

次の最有力 issue 下書きは [next_issue_tighter_policy_frontier.md](issue_library/next_issue_tighter_policy_frontier.md) を参照する。

formal 化前の確認には [revision_gate_candidate_checklist.md](revision_gate_candidate_checklist.md) を使う。

`tighter policy` の実行計画は [tighter_policy_frontier_execution.md](tighter_policy_frontier_execution.md) と [tighter_policy_candidate_matrix.md](tighter_policy_candidate_matrix.md) を参照する。

progress 監査と優先順位づけは [progress_coverage_audit.md](progress_coverage_audit.md) を参照する。
