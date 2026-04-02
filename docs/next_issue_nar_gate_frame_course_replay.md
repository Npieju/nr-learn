# Next Issue: NAR Gate Frame Course Replay

## Summary

NAR line は baseline formalization、class/rest replay、jockey/trainer/combo replay、`wf_summary` path alignment まで完了した。次の family は Tier A/B の次順位として、coverage が高く structural に解釈しやすい `gate / frame / course bucket bias` を narrow replay する。

## Objective

local Nankan baseline に対して、`gate / frame / course bucket` family を narrow add-on として replay し、support を大きく落とさず denominator-first formal benchmark を改善できるかを測る。

## Hypothesis

if local Nankan baseline に `gate / frame / course bucket` family の buildable high-coverage features を narrow に追加する, then current combo replay と同程度の support を維持しつつ、NAR 特有の gate / course positional bias を formal benchmark に反映できる。

## Current Read

- current best NAR baseline line:
  - `configs/model_local_baseline_wf_runtime_narrow.yaml`
  - `weighted_roi=3.6903`
  - `bets_total=3525`
  - `bet_rate=12.16%` on `28997` test races
- current best NAR promoted replay line:
  - jockey/trainer/combo replay
  - `weighted_roi=4.3245`
  - `bets_total=3725`
  - `bet_rate=12.85%` on `28997` test races
- feature ranking:
  - `gate / frame / course bucket bias` is the next recommended feature family after class/rest and jockey/trainer/combo
  - rationale: structural interpretability, likely high coverage, less raw-source risk than lineage / pace
- first gap read:
  - `feature_gap_summary_local_nankan_gate_frame_course_replay_v1.json`
  - `priority_missing_raw_columns=[]`
  - `missing_force_include_features=[]`
  - `low_coverage_force_include_features=[]`
  - candidate broad add-on 9 本のうち actual selected まで入ったのは 2 本だけ
    - `gate_ratio`
    - `frame_ratio`
  - selected feature count は baseline `13` から `15` へ増えた
  - `course_gate_bucket_last_100_*`, `course_baseline_*`, `time_deviation` は build される前提で入れたが、current local Nankan slice では selected set に残らなかった
  - therefore actual candidate は `gate_ratio + frame_ratio` の narrow replay に絞る

## In Scope

- local Nankan feature-gap read for gate/frame/course family
- narrow candidate config 1 本
- matching-tuple local revision gate
- denominator-first formal compare against current promoted NAR line

## Non-Goals

- JRA baseline replacement
- NAR CatBoost / ensemble line
- low-coverage pace / lineage family

## Success Metrics

- candidate features are actually built and selected
- formal `bets / races / bet_rate` stays in the same order as current promoted line
- weighted ROI / feasible folds beat or cleanly challenge current promoted NAR line

## Validation

- feature gap summary / coverage CSV
- `python scripts/run_local_revision_gate.py ...`
- final read with:
  - `auc`
  - `ev_top1_roi`
  - `formal_benchmark_weighted_roi`
  - `formal_benchmark_bets_total`
  - `bet_rate`

## Stop Condition

- no-op replay
- support collapses versus current promoted NAR line
- feature coverage is materially worse than combo replay

## Current First Candidate

first candidate は `gate_ratio + frame_ratio` replay とする。

理由:

- feature gap 上で actual selected set に入った gate/frame family はこの 2 本だけだった
- baseline `13 -> 15 features` なので no-op ではない
- `course_gate_bucket_*` や `course_baseline_*` を broad に抱えるより、まず buildable / selected が確認できた narrow add-on を formal compare に載せる方が defensible
