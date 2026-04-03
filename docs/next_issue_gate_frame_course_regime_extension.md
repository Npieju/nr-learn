# Next Issue: Gate Frame Course Regime Extension

## Summary

`gate / frame / course bucket` family は JRA current baseline にすでに含まれている。

- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`

一方で、この family を primary hypothesis として切った JRA issue はまだ無い。現状は baseline 内の stable component として使われているだけで、regime-aware extension の可否は未読である。

## Objective

current baseline に残っている `gate / frame / course bucket` family を regime-aware に拡張し、support を壊さずに explanatory power か formal top-line を上積みできるかを検証する。

## Hypothesis

if `gate / frame / course bucket` family を broad add-on ではなく regime-aware extension として切る, then current baseline の coverage と support を維持したまま、course-conditioned positional bias を追加で拾える可能性がある。

## In Scope

- current baseline feature config の gate/frame/course family read
- `course_gate_bucket_*` を中心にした narrow extension candidate 設計
- feature gap / coverage read
- 1 candidate だけの formal compare 導線準備

## Non-Goals

- broad policy rewrite
- NAR work
- pedigree-heavy family の再開
- already-settled seasonal ordering の再議論

## Success Metrics

- candidate が 1 本に絞れる
- baseline family に対する additive hypothesis が文章で説明できる
- coverage risk が accept/reject できる粒度で読める

## Validation Plan

- baseline config と feature ranking から current family position を固定する
- buildability / selection risk を feature-gap で確認する
- if buildable なら narrow extension candidate を 1 本だけ formal rerun に載せる

## Expected Artifacts

- family read summary
- feature-gap summary
- chosen candidate definition
- if justified, new revision issue

## Stop Condition

- buildable でも expected gain が弱く candidate が 1 本に収束しない
- coverage / selection risk が高く、現行 baseline を壊す可能性が大きい
- existing evidence の rereadだけで feature family として優先順位を上げる理由がない
