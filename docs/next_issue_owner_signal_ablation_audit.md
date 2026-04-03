# Next Issue: Owner Signal Ablation Audit

## Summary

JRA current baseline には `owner_last_50_win_rate` が既に入っている。したがって owner family の次仮説は add-on ではなく、owner を外したときに formal / actual-date でどれだけ劣化するかを測る ablation である。

`feature_family_ranking.md` では owner signal は Tier C で、pedigree 全体よりは実務的だが、Tier A family ほどの優先度ではない。ここで owner を外した selective audit を 1 本作れば、current baseline に対する owner の marginal contribution を pedigree 混入なしで読める。

## Objective

`owner_last_50_win_rate` を current JRA high-coverage baseline から外した selective ablation を formal compare し、owner signal が current baseline に対して実質的な alpha source か、削ってもよい weak contributor かを判定する。

## Hypothesis

if `owner_last_50_win_rate` を除外しても evaluation / formal / actual-date の主要 read が baseline と同程度に保たれる, then owner signal は current serving family の必須要素ではなく pruning 候補になる。

## In Scope

- `owner_last_50_win_rate` を明示除外した high-coverage config 1 本
- JRA true component retrain flow
- formal compare
- September difficult window と December control window の actual-date role split

## Non-Goals

- pedigree family の再開
- owner feature の broad 拡張
- policy rewrite
- NAR work

## Candidate Definition

keep current high-coverage baseline core and exclude only:

- `owner_last_50_win_rate`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_owner_ablation.yaml`

## Success Metrics

- win / roi component の actual used feature set から `owner_last_50_win_rate` が消える
- formal compare を 1 本作れる
- actual-date read で owner の marginal contribution を baseline と比較して説明できる

## Validation Plan

1. owner ablation config を追加
2. true component retrain
3. stack rebuild
4. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`
5. September / December actual-date compare

## Stop Condition

- actual selected set から owner が外れない
- component retrain が no-op に近い
- formal support が baseline 比で明確に崩れ、analysis value も薄い
