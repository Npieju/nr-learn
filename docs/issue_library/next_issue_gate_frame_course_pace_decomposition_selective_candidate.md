# Next Issue: Gate Frame Course Pace-Decomposition Selective Candidate

## Summary

`gate / frame / course` family は `#79` で `course_baseline_race_pace_balance_3f` 単体 child まで読み、formal `pass / promote`、actual-date では `analysis-first promoted candidate / compare reference` まで固定した。

ただし current read は family 自体の打ち止めを意味しない。未読の high-coverage side として、course-conditioned pace baseline の front/back 分解が残っている。

- `course_baseline_race_pace_front3f`
- `course_baseline_race_pace_back3f`

これらは `fundamental_enriched` 系 feature config には存在する一方、current JRA high-coverage line では selective child として独立に切り分けていない。

## Objective

`course_baseline_race_pace_front3f` と `course_baseline_race_pace_back3f` を current JRA high-coverage baseline に narrow add-on し、`gate / frame / course` family の second child として formal compare に載せる価値があるかを first read で判定する。

## Hypothesis

if course-conditioned pace baseline を `balance_3f` の単一軸ではなく `front3f / back3f` の 2 軸へ分解して selective に追加する, then first child より bluntness を抑えつつ course-specific race-shape bias を拾え、support を壊さずに family の second child candidate を作れる可能性がある。

## Current Read

- `feature_family_ranking.md` では `gate / frame / course bucket bias` は Tier A
- first child `r20260403_gate_frame_course_regime_extension_v1` は formal `pass / promote` だが、actual-date role は `analysis-first promoted candidate / compare reference`
- `course_baseline_race_pace_balance_3f` は既読だが、front/back 分解は独立 issue として未評価
- builder 実装を増やさず、既存列だけで fresh child hypothesis を切れる

## Candidate Definition

keep current JRA high-coverage core:

- `gate_ratio`
- `frame_ratio`
- `course_gate_bucket_last_100_win_rate`
- `course_gate_bucket_last_100_avg_rank`

add selective pair:

- `course_baseline_race_pace_front3f`
- `course_baseline_race_pace_back3f`

tracked config:

- `configs/features_catboost_rich_high_coverage_diag_gate_frame_course_pace_decomposition_selective.yaml`

## In Scope

- feature-gap / coverage read
- current baseline に対する narrow selective config 1 本
- true component retrain ready な issue body
- formal compare に進むかどうかの execution decision

## Non-Goals

- `course_baseline_race_pace_balance_3f` child の再実行
- broad gate/frame/course family の再議論
- time-derived target proxy の再導入
- NAR work
- serving policy rewrite

## Success Criteria

- focal pair の coverage / selection risk を独立に読める
- low-coverage / missing / no-op のどれかでないことを説明できる
- true component retrain に進めるか stop するかを 1 本に固定できる

## Validation Plan

1. feature-gap / coverage を読み、2 本が `present=True` かつ low-coverage でないことを確認する
2. buildable なら win / roi component retrain と stack rebuild を true component flow で行う
3. formal compare では
   - `auc`
   - `top1_roi`
   - `ev_top1_roi`
   - nested WF shape
   - held-out formal `weighted_roi`
   - `bets / races / bet_rate`
   を baseline と比較する
4. formal `pass / promote` に到達した場合だけ actual-date role split issue を次段で切る

## Stop Condition

- focal pair のどちらかが missing または low coverage
- actual used set に入らず no-op の可能性が高い
- first child `balance_3f` より説明可能性が弱く、independent child hypothesis として立てる根拠が不足する

## Why This Is Next

- `#117` で tighter policy family の same-family narrowing は close した
- current queue は「`abs90` anchor を reference に据えた別 hypothesis を 1 issue で切り直す」段階である
- existing draft library の多くは historical close 済みで、未読かつ builder-ready な JRA child hypothesis が必要
- course-conditioned pace front/back 分解は、その条件を満たす最小の fresh hypothesis である

## First Read

first read は `configs/data_2025_latest.yaml` の tail `100,000` rows を使って取得した。

- feature-gap:
   - `artifacts/reports/feature_gap_summary_gate_frame_course_pace_decomposition_selective_v1.json`
   - `artifacts/reports/feature_gap_feature_coverage_gate_frame_course_pace_decomposition_selective_v1.csv`
   - `artifacts/reports/feature_gap_raw_column_coverage_gate_frame_course_pace_decomposition_selective_v1.csv`
- rows evaluated: `100000`
- selected feature count: `109`
- categorical feature count: `37`
- `priority_missing_raw_columns=[]`
- `missing_force_include_features=[]`
- `low_coverage_force_include_features=['course_baseline_race_pace_back3f', 'course_baseline_race_pace_front3f']`
- focal pair:
   - `course_baseline_race_pace_front3f`
      - `selected=True`
      - `present=True`
      - `non_null_ratio=0.08897`
      - `status=low_coverage`
   - `course_baseline_race_pace_back3f`
      - `selected=True`
      - `present=True`
      - `non_null_ratio=0.08897`
      - `status=low_coverage`

interpretation:

- focal pair は missing ではない
- ただし current high-coverage line に対しては coverage が薄すぎる
- `balance_3f` child より sharper hypothesis には見えるが、first gate を通すだけの support がない

## Decision

- gate-frame-course pace decomposition selective candidate は reject
- `course_baseline_race_pace_front3f` / `course_baseline_race_pace_back3f` を current JRA high-coverage line の次 child に進めない
- `gate / frame / course` family は引き続き `r20260403_gate_frame_course_regime_extension_v1` を compare reference として保持し、新しい child hypothesis は別軸で切り直す