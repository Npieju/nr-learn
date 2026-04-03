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

## Current Read

- current baseline にはすでに次が入っている
  - `gate_ratio`
  - `frame_ratio`
  - `course_gate_bucket_last_100_win_rate`
  - `course_gate_bucket_last_100_avg_rank`
- builder には `course_baseline_*` と `time_deviation` 系もある
- ただし `course_baseline_time_per_1000m` と `time_deviation` は target 近傍で、selection の default safe exclude に入っている
- したがって extension 候補は time-derived target proxy ではなく、course-conditioned regime signal に限る

## Current First Candidate

first candidate は `course_baseline_race_pace_balance_3f` 単体 add-on とする。

理由:

- `course_history_key` の shifted rolling mean から作られる course-conditioned baseline で、builder 実装済み
- `time_deviation` と違って target 近傍列ではない
- `gate / frame / course bucket` family を broad に壊さず、regime conditioning だけ足せる

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_gate_frame_course_regime_extension.yaml`

## First Gap Read

- artifact:
  - `artifacts/reports/feature_gap_summary_gate_frame_course_regime_extension_v1.json`
  - `artifacts/reports/feature_gap_feature_coverage_gate_frame_course_regime_extension_v1.csv`
- rows evaluated: `100000`
- selected feature count: `109`
- categorical feature count: `37`
- `priority_missing_raw_columns=[]`
- `missing_force_include_features=[]`
- `low_coverage_force_include_features=[]`
- focal candidate:
  - `course_baseline_race_pace_balance_3f`
  - `selected=True`
  - `present=True`
  - `non_null_ratio=0.57273`
  - `status=ok`

interpretation:

- narrow candidate としては buildable
- coverage は threshold `0.5` を上回るが、baseline core feature よりは薄い
- rerun は許容だが、high-confidence family ではなく conditional challenger として扱う

## Execution Standard

JRA current baseline は `value_blend` なので、direct revision gate train ではなく true component retrain flow を使う。

1. win component retrain
2. roi component retrain
3. stack rebuild
4. `run_revision_gate.py --skip-train --evaluate-model-artifact-suffix ...`

## Final Read

- revision:
  - `r20260403_gate_frame_course_regime_extension_v1`
- feature candidate:
  - `course_baseline_race_pace_balance_3f`
- actual used feature set:
  - win / roi / stack の全てで selected
  - stack feature count `109`

formal result:

- evaluation summary:
  - `auc=0.8426169492248933`
  - `top1_roi=0.8082087836284203`
  - `ev_top1_roi=0.8213612324672338`
  - `wf_nested_test_roi_weighted=0.7632385120350111`
  - `wf_nested_test_bets_total=457`
  - nested WF: `3/3 portfolio`
- promotion gate:
  - `status=pass`
  - `decision=promote`
  - held-out formal:
    - `weighted_roi=1.2311149102465346`
    - `bets_total=9806`
    - `feasible_fold_count=3`
    - `metric_source_counts={'test': 3}`

baseline compare:

- current recommended serving benchmark refresh:
  - `auc=0.8400959298075428`
  - `top1_roi=0.8070328660078143`
  - `ev_top1_roi=0.5568030337853367`
  - `wf_nested_test_roi_weighted=0.7628366750468021`
  - `wf_nested_test_bets_total=544`

interpretation:

- first child は formal candidate として成立
- evaluation top-line は baseline refresh を小幅に上回る
- held-out formal ROI も `1.2311` と `>1.0` を維持する
- 一方で actual-date operational role はまだ未読なので、直ちに serving default へ昇格させる段階ではない

next:

- actual-date compare と role split は別 issue で切る

## Expected Artifacts

- family read summary
- feature-gap summary
- chosen candidate definition
- if justified, new revision issue

## Stop Condition

- buildable でも expected gain が弱く candidate が 1 本に収束しない
- coverage / selection risk が高く、現行 baseline を壊す可能性が大きい
- existing evidence の rereadだけで feature family として優先順位を上げる理由がない
