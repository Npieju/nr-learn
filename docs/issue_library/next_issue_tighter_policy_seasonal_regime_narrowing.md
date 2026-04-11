# Next Issue: Tighter Policy Seasonal Regime Narrowing

## Summary

JRA の current policy 本線では、`current_tighter_policy_search_candidate_2025_latest` family が依然として最も formal support が強い。

- `r20260326_tighter_policy_ratio003`
  - formal `weighted_roi=1.1728`
  - feasible folds `4/5`
- `r20260327_tighter_policy_ratio003_abs80`
  - formal `weighted_roi=1.1042`
  - feasible folds `5/5`
- `r20260329_tighter_policy_ratio003_abs90`
  - strictest defensible anchor
  - formal `weighted_roi=1.1042287961989103`
  - feasible folds `5/5`

一方で、existing read は family 内 frontier の固定までで止まっている。September difficult window での defensive behavior は確認されているが、season-aware な threshold narrowing を 1 issue 1 hypothesis で formal に読む作業はまだ切られていない。

したがって次の本線 issue は、新しい feature family を増やすことではなく、既存の strongest defensive policy family に対して `seasonal regime narrowing` を narrow に試し、September downside を圧縮しつつ December control を壊さない policy 条件帯が存在するかを検証することである。

## Objective

`r20260329_tighter_policy_ratio003_abs90` anchor を起点に、September difficult regime と December control regime を分けて threshold を narrow に再探索し、operational downside を減らせる policy variant があるかを判定する。

## Hypothesis

if `tighter policy search` family の threshold を season-aware に narrow し、September difficult window と December control window を分けて読み直す, then we can preserve the family's formal support while improving difficult-window downside control without breaking control-window carry.

## In Scope

- `current_tighter_policy_search_candidate_2025_latest`
- `r20260329_tighter_policy_ratio003_abs90` anchor
- threshold axes:
  - `min_bets_abs`
  - `min_bet_ratio`
  - `min_prob`
  - `odds_max`
  - `min_expected_value`
- September broad window compare
- December control window compare
- `scripts/run_wf_threshold_sweep.py`
- `scripts/run_serving_profile_compare.py`
- `scripts/run_revision_gate.py`

## Non-Goals

- 新しい feature family の追加
- model retrain / stack rebuild
- broad baseline replacement の即時判断
- NAR work

## Success Metrics

- September difficult window で baseline 比の downside control 改善が説明できる
- December control window で baseline の positive carry を壊さない
- candidate が formal support を保ったまま 1 本に絞れる
- `promote / keep as candidate / reject` を 1 行で言える

## Validation Plan

1. existing formal / WF artifacts から September / December compare baseline を固定する
2. narrow threshold sweep を season-aware read のために再実行する
3. shortlisted candidate だけ revision gate dry-run を通す
4. 有望候補のみ formal compare を回し、September/December actual-date role を読む
5. queue / docs / issue thread に decision summary を固定する

## Validation Commands

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_sweep.py \
  --wf-summary <wf_summary> \
  --min-bet-ratio-values 0.05,0.04,0.03 \
  --min-bets-abs-values 100,90,80 \
  --min-feasible-folds 3 \
  --target-feasible-fold-counts 3,4,5 \
  --output <frontier_output> \
  --summary-csv <frontier_csv>

/workspaces/nr-learn/.venv/bin/python scripts/run_revision_gate.py \
  --config <candidate_config> \
  --data-config configs/data_2025_latest.yaml \
  --feature-config configs/features_catboost_rich_high_coverage_diag.yaml \
  --revision <revision_slug> \
  --train-artifact-suffix <revision_slug> \
  --skip-train \
  --evaluate-model-artifact-suffix r20260329_tighter_policy_ratio003_abs90 \
  --evaluate-max-rows 120000 \
  --evaluate-pre-feature-max-rows 300000 \
  --evaluate-wf-mode full \
  --evaluate-wf-scheme nested \
  --promotion-min-feasible-folds 3 \
  --dry-run

/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving_2025_latest \
  --right-config <candidate_config> \
  --window-key <september_or_december_window>
```

## Expected Artifacts

- season-aware threshold frontier summary
- candidate shortlist note
- revision gate manifest / summary for shortlisted candidate
- September / December compare artifacts
- decision summary for operational role

## Stop Condition

- September downside 改善が bet suppression だけでしか説明できない
- December control で baseline の positive carry を壊す
- strict anchor `abs90` を超える実益がなく、same-family rerun を続ける理由がなくなる

## Starting Context

- policy ranking では `tighter policy search` が Rank 1 family である
- first-wave frontier issue は close 済みで、strictest defensible anchor は `abs90`
- `current_sep_guard_candidate` は seasonal fallback として残るが、JRA 本線 policy family の mainline ではない
- current JRA active issue は空であり、再開時は 1 measurable hypothesis を先に起こす必要がある

## Proposed GitHub Issue

- `#117`
- <https://github.com/Npieju/nr-learn/issues/117>

### Title

`[experiment] Seasonal regime narrowing on tighter-policy anchor`

### Template

`Model Experiment`

### Recommended labels

- `experiment`
- `policy`
- `jra`

### Body draft

```md
Universe
JRA

Category
Policy

Objective
`r20260329_tighter_policy_ratio003_abs90` anchor を起点に、September difficult regime と December control regime を分けて threshold を narrow に再探索し、operational downside を減らせる policy variant があるかを判定する。

Hypothesis
if `tighter policy search` family の threshold を season-aware に narrow し、September difficult window と December control window を分けて読み直す, then we can preserve the family's formal support while improving difficult-window downside control without breaking control-window carry.

In-Scope Surface
- `current_tighter_policy_search_candidate_2025_latest`
- `r20260329_tighter_policy_ratio003_abs90` anchor
- threshold axes: `min_bets_abs`, `min_bet_ratio`, `min_prob`, `odds_max`, `min_expected_value`
- September broad window compare
- December control window compare
- `scripts/run_wf_threshold_sweep.py`
- `scripts/run_serving_profile_compare.py`
- `scripts/run_revision_gate.py`

Non-Goals
- 新しい feature family の追加
- model retrain / stack rebuild
- broad baseline replacement の即時判断
- NAR work

Success Metrics
- September difficult window で baseline 比の downside control 改善が説明できる
- December control window で baseline の positive carry を壊さない
- candidate が formal support を保ったまま 1 本に絞れる
- `promote / keep as candidate / reject` を 1 行で言える

Validation Plan
1. existing formal / WF artifacts から September / December compare baseline を固定する
2. narrow threshold sweep を season-aware read のために再実行する
3. shortlisted candidate だけ revision gate dry-run を通す
4. 有望候補のみ formal compare を回し、September/December actual-date role を読む
5. queue / docs / issue thread に decision summary を固定する

Stop Condition
- September downside 改善が bet suppression だけでしか説明できない
- December control で baseline の positive carry を壊す
- strict anchor `abs90` を超える実益がなく、same-family rerun を続ける理由がなくなる
```

## Primary References

- `docs/policy_family_shortlist.md`
- `docs/tighter_policy_frontier_execution.md`
- `docs/issue_library/next_issue_tighter_policy_frontier.md`
- `artifacts/reports/promotion_gate_r20260325_current_recommended_serving_2025_latest_benchmark_refresh.json`
- `artifacts/reports/promotion_gate_r20260329_tighter_policy_ratio003_abs90.json`

## First Execution Read

2026-04-06 first pass では、season-aware compare に進む前に `abs90` anchor の frontier を再確認した。

- command:
  - `/workspaces/nr-learn/.venv/bin/python scripts/run_wf_threshold_sweep.py --wf-summary artifacts/reports/wf_feasibility_diag_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_wf_full_nested.json --min-bet-ratio-values 0.05,0.04,0.03 --min-bets-abs-values 100,90,80 --min-feasible-folds 3 --target-feasible-fold-counts 3,4,5 --output artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_abs90_seasonal_regime_v1.json --summary-csv artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_abs90_seasonal_regime_v1.csv`
- output:
  - `artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_abs90_seasonal_regime_v1.json`
  - `artifacts/reports/wf_threshold_sweep_tighter_policy_ratio003_abs90_seasonal_regime_v1.csv`
- result:
  - `0.03 / 100` は `4/5 feasible folds`
  - `0.03 / 90` は `5/5 feasible folds`
  - `0.03 / 80` も `5/5 feasible folds`
  - strictest threshold passing summary は `3 folds -> 0.03/100`, `4 folds -> 0.03/100`, `5 folds -> 0.03/90`
  - blocked fold の dominant failure は一貫して `min_bets`

interpretation:

- `r20260329_tighter_policy_ratio003_abs90` は依然として strictest defensible `5/5` anchor である
- current sweep だけでは `abs90` を外して broad frontier を reopen する理由は生まれていない
- したがって next move は ratio/abs 軸の再拡張ではなく、existing September / December compare read に接続して season-aware role を narrow に判定することである

current starting baseline for seasonal read:

- 2025-09 difficult window では baseline `32 bets / total net -27.3 / pure bankroll 0.2959` に対して tighter policy candidate は `9 bets / -4.3 / 0.8395`
- 2025-12 control window では baseline `45 bets / +21.8 / 1.6712` に対して tighter policy candidate は `9 bets / +21.4 / 1.6032`

next action:

- `abs90` anchor を starting point に維持したまま、September downside control と December carry preservation を criterion にして season-aware compare 候補を narrow に読む

## Seasonal Compare Read

2026-04-06 second pass では、重い再実行を追加せず、既に生成済みの `issue117` backtest artifact を集計して season-aware compare を固定した。

- source:
  - `artifacts/reports/issue117/seasonal_compare_summary.json`
- compare basis:
  - baseline は `artifacts/reports/backtest_<date>_issue117_baseline_<date>.json` の 16 日分
  - challenger は `artifacts/reports/backtest_<date>_issue117_abs90_minprob005_<date>.json` と `artifacts/reports/backtest_<date>_issue117_abs90_odds25_<date>.json` の 16 日分
- note:
  - この read は `issue117` 用に残っていた config-direct backtest 群の相対比較である
  - `docs/project_overview.md` の baseline / tighter-policy 読みとは source が異なるため、top-line の絶対値は混ぜず、同一 source 内の relative compare だけを判断根拠に使う

September difficult window:

- baseline:
  - `33 bets / total net -20.0 / pure bankroll 0.3939 / zero-bet dates 0/8`
- `abs90_minprob005`:
  - `6 bets / total net -1.3 / pure bankroll 0.7833 / zero-bet dates 4/8`
- `abs90_odds25`:
  - `5 bets / total net -0.3 / pure bankroll 0.94 / zero-bet dates 4/8`

December control window:

- baseline:
  - `17 bets / total net -5.2 / pure bankroll 0.6941 / zero-bet dates 2/8`
- `abs90_minprob005`:
  - `4 bets / total net -4.0 / pure bankroll 0.0 / zero-bet dates 6/8`
- `abs90_odds25`:
  - `4 bets / total net -4.0 / pure bankroll 0.0 / zero-bet dates 6/8`

interpretation:

- `minprob005` と `odds25` はどちらも September downside を縮めるが、改善の大部分は exposure compression に依存している
- December control では 2 variant とも `6/8` zero-bet dates まで縮み、carry preservation を満たさない
- `odds25` は September の same-source relative compare では `minprob005` よりわずかに良いが、December profile は同一であり、strict anchor を更新する理由にならない

## Final Decision

- `r20260329_tighter_policy_ratio003_abs90` を strictest defensible anchor のまま維持する
- `abs90_minprob005` は reject する
- `abs90_odds25` も reject する
- current judgment は `season-aware narrowing failed to produce a keep-worthy near-par challenger` である

reason:

- September difficult window での net 改善は確認できるが、bet suppression が強すぎる
- December control window で carry preservation を示せない
- したがって next move は same-family threshold narrowing の継続ではなく、`abs90` anchor を reference に据えた別 hypothesis へ戻すことである