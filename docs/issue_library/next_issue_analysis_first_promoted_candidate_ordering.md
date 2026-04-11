# Next Issue: Analysis-First Promoted Candidate Ordering

## Summary

JRA では formal `pass / promote` に到達したが、actual-date role split で serving default には上げなかった line が複数並んでいる。

- `r20260330_surface_plus_class_layoff_interactions_v1`
- `r20260403_gate_frame_course_regime_extension_v1`
- `r20260403_pace_closing_fit_selective_v1`
- `r20260404_jockey_trainer_combo_closing_time_selective_v1`

個別の role split は済んでいるが、operator 向けの横断順位はまだ固定していない。

## Objective

既存の JRA `analysis-first promoted candidate` 群を formal support と actual-date role の両方で横並びに読み、operator reference ordering を 1 本に固定する。

## Hypothesis

if 既存 artifact を formal ROI / feasible folds / September difficult window / December control window で読み直す, then one line can be fixed as the first operator reference among non-default promoted lines, one line can be fixed as the second difficult-regime specialist, and the remaining lines can be demoted to compare-reference only.

## Current Read

Current promoted-but-non-default set:

1. `r20260330_surface_plus_class_layoff_interactions_v1`
   - formal `weighted_roi=1.1379979394080304`
   - `bets_total=519`
   - `feasible_fold_count=3`
   - September: `8 bets / -8.0 / pure bankroll 0.7333`
   - December: `13 bets / +20.0 / pure bankroll 1.6551`
2. `r20260403_gate_frame_course_regime_extension_v1`
   - formal `weighted_roi=1.2311149102465346`
   - `bets_total=9806`
   - `feasible_fold_count=3`
   - September: `5 bets / -3.6 / pure bankroll 0.8809`
   - December: `30 bets / -9.3 / pure bankroll 0.7514`
3. `r20260403_pace_closing_fit_selective_v1`
   - formal `weighted_roi=1.0307760253096685`
   - `bets_total=938`
   - `feasible_fold_count=3`
   - September: `3 bets / -3.0 / pure bankroll 0.892891589506173`
   - December: `0 bets / 0.0 / pure bankroll 1.0`
4. `r20260404_jockey_trainer_combo_closing_time_selective_v1`
   - formal `weighted_roi=1.2311149102465346`
   - `bets_total=9806`
   - `feasible_fold_count=3`
   - September: `5 bets / -3.6 / pure bankroll 0.8808888460219477`
   - December: `30 bets / -9.3 / pure bankroll 0.7513642270122226`

## In Scope

- 上記 4 line の formal `promotion_gate` artifact reread
- September difficult window の actual-date reread
- December control window の actual-date reread
- operator reference ordering の固定
- docs / queue / issue thread の更新

## Non-Goals

- 新しい feature family
- new retrain / rerun
- policy rewrite
- NAR work

## Success Criteria

- 4 line の operator ordering が 1 本に固定される
- first operator reference と difficult-regime specialist が分かる
- compare-reference only の line が明示される

## Validation Plan

1. `promotion_gate_*` から formal ROI / bets / feasible folds を並べる
2. role split docs / dashboard summary から September / December の `bets / total net / pure bankroll` を並べる
3. operator reading を次の 3 bucket に固定する
   - first operator reference
   - secondary difficult-regime specialist
   - compare-reference only

## Stop Condition

- actual-date evidence が相互に矛盾し、順位を 1 本に固定できない
- compare artifact が欠けていて cross-candidate read が成立しない

## Result

既存 artifact の reread だけで operator ordering は固定できた。

### Ordered Reading

1. first operator reference:
   - `r20260330_surface_plus_class_layoff_interactions_v1`
2. secondary difficult-regime specialist:
   - `r20260403_pace_closing_fit_selective_v1`
3. compare-reference only:
   - `r20260403_gate_frame_course_regime_extension_v1`
   - `r20260404_jockey_trainer_combo_closing_time_selective_v1`

### Why

`surface_plus_class_layoff` は low-exposure だが、4 line の中で唯一、September difficult window の downside control と December control window の positive carry を同時に残している。

- formal `weighted_roi=1.1379979394080304`
- September `8 bets / -8.0 / pure bankroll 0.7333`
- December `13 bets / +20.0 / pure bankroll 1.6551`

`pace_closing_fit` は formal top-line は最弱だが、September de-risk specialist としては最も clean で、December に実損を作らない。

- formal `weighted_roi=1.0307760253096685`
- September `3 bets / -3.0 / pure bankroll 0.892891589506173`
- December `0 bets / 0.0 / pure bankroll 1.0`

`gate_frame_course` と `combo_closing_time` は formal top-line は高いが、actual-date では December control window を loss に反転させる。現時点の operator reading では first reference に上げない。

- formal `weighted_roi=1.2311149102465346`
- September `5 bets / -3.6 / pure bankroll 0.8809`
- December `30 bets / -9.3 / pure bankroll 0.7514`

### Decision Summary

- `r20260330_surface_plus_class_layoff_interactions_v1` を JRA non-default promoted line の first operator reference に固定する
- `r20260403_pace_closing_fit_selective_v1` は September difficult-regime specialist として second reading に置く
- `r20260403_gate_frame_course_regime_extension_v1` と `r20260404_jockey_trainer_combo_closing_time_selective_v1` は compare-reference only に留める
- serving default と September fallback ordering は変えない

## Primary References

- `artifacts/reports/promotion_gate_r20260330_surface_plus_class_layoff_interactions_v1.json`
- `artifacts/reports/promotion_gate_r20260403_gate_frame_course_regime_extension_v1.json`
- `artifacts/reports/promotion_gate_r20260403_pace_closing_fit_selective_v1.json`
- `artifacts/reports/promotion_gate_r20260404_jockey_trainer_combo_closing_time_selective_v1.json`
- `docs/issue_library/next_issue_post_surface_plus_class_layoff_promotion.md`
- `docs/issue_library/next_issue_surface_plus_class_layoff_bet_rate_robustness.md`
- `docs/issue_library/next_issue_gate_frame_course_actual_date_role_split.md`
- `docs/issue_library/next_issue_pace_closing_fit_actual_date_role_split.md`
- `docs/issue_library/next_issue_jockey_trainer_combo_closing_time_role_split.md`
