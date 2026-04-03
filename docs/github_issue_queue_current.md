# GitHub Issue Queue: Current

## 1. Purpose

この文書は、今すぐ GitHub issue に起こすべき案件を、GitHub template にそのまま貼れる粒度で整理した current queue である。

`docs/initial_issue_backlog.md` が中長期 backlog であるのに対して、こちらは直近で着手・追跡する issue の正本として使う。

## 2. Why GitHub

現在の `nr-learn` は docs と artifact が十分育ってきたため、次の段階では GitHub issue を execution source of truth にしたほうがよい。

- 進行中 / blocked / done を board 上で見分けやすい
- experiment と ops を混ぜずに追跡できる
- artifact path と decision summary を issue thread に残せる
- AI coding の着手単位を固定しやすい

## 3. Current Issue Set

### 3.0 Current Queue As Of 2026-04-03

NAR separate-universe line は、baseline narrow と `jockey_trainer_combo_replay_v1_pathfix` まで formal `pass / promote` を通した。  
その後の integrity audit と corrective で、次が確認できた。

- high AUC / high EV ROI の大部分は `odds / popularity` 依存
- no-market ablation でも policy 自体は成立
- baseline pathfix line は evaluation `3/3 no_bet + bets=0` でも formal 側で uplift していた
- `#72` で `3/3 no_bet + bets=0` line の formal promote は short-circuit 済み
- `#73` で formal benchmark は held-out test metrics に揃えた
- no-market rerun では old formal `ROI=0.8104` が held-out formal `ROI=0.7234` へ低下した
- `#74` で local Nankan promotion gate に held-out formal `weighted_roi >= 1.0` の minimal guard を入れた
- `#67` owner replay は no-op ではなかったが evaluation `3/3 no_bet + bets=0` で reject と確定した
- `#49` jockey-trainer-combo family は first child と role split まで終え、analysis-first promoted candidate で一段落した
- `#75` tighter policy frontier は existing artifact compare で再確定し、strictest defensible anchor は引き続き `abs90` で据え置きと結論した
- したがって current NAR corrective stack は `#72` no-bet short-circuit, `#73` held-out alignment, `#74` threshold realignment まで完了した

Primary completed issues:

- `#69`
- <https://github.com/Npieju/nr-learn/issues/69>
- `#70`
- <https://github.com/Npieju/nr-learn/issues/70>
- `#71`
- <https://github.com/Npieju/nr-learn/issues/71>
- `#72`
- <https://github.com/Npieju/nr-learn/issues/72>
- `#73`
- <https://github.com/Npieju/nr-learn/issues/73>
- `#74`
- <https://github.com/Npieju/nr-learn/issues/74>
- `#67`
- <https://github.com/Npieju/nr-learn/issues/67>
- `#49`
- <https://github.com/Npieju/nr-learn/issues/49>
- `#75`
- <https://github.com/Npieju/nr-learn/issues/75>

Primary next execution order:

1. JRA 本線は new hypothesis issue を切って再開する
2. `tighter policy search` family は `abs90` anchor を維持し、near-par challenger は reference 扱いに留める
3. NAR side は held-out formal benchmark を正本に維持し、新 hypothesis は必要時に別 issue で切る

Latest reread completions:

- `surface_plus_class_layoff` widening reread は close
- `sep_guard` ordering reread も existing artifact だけで close 可能

Latest reread completions:

- `shared portfolio bottleneck` narrower regime split reread も existing artifact だけで close 可能

Latest actual-date role split:

1. `#79` gate/frame/course regime extension は formal `pass / promote` まで完了した
2. `#80` actual-date role split で、September difficult window は改善したが December control window は baseline 劣後と確定した
3. したがって `r20260403_gate_frame_course_regime_extension_v1` の operational role は `analysis-first promoted candidate` に固定し、serving default には上げない
4. `current_long_horizon_serving_2025_latest` を first seasonal alias、`current_sep_guard_candidate` を second seasonal fallback として据え置く

Latest fallback compare:

1. `#81` gate/frame/course September fallback compare では、candidate は broad September と late-September の両方で `current_sep_guard_candidate` に pure bankroll で劣後した
2. broad September:
   - sep guard `9 bets / -4.3 / pure bankroll 0.9995842542264094`
   - gate-frame-course candidate `5 bets / -3.6 / pure bankroll 0.8808888460219477`
3. late-September:
   - sep guard `6 bets / -6.0 / pure bankroll 0.9857901270555528`
   - gate-frame-course candidate `3 bets / -1.6 / pure bankroll 0.9319444444444444`
4. したがって `current_sep_guard_candidate` は September seasonal fallback のまま据え置き、`r20260403_gate_frame_course_regime_extension_v1` は compare reference に留める

Current open-work expectation:

1. JRA 本線は `pace / closing-fit` family の selective replay を切る
2. `gate/frame/course` extension は broad default でも September fallback primary でも September overlay でもなく、compare reference に留める
3. `current_sep_guard_candidate` は September seasonal fallback のまま据え置く

Latest formal completion:

1. `#83` pace-closing-fit selective candidate は formal `pass / promote` まで完了した
2. first child `r20260403_pace_closing_fit_selective_v1` は
   - `auc=0.8411224406691354`
   - `top1_roi=0.7982417834980464`
   - `ev_top1_roi=0.6212479889680533`
   - `wf_nested_test_roi_weighted=0.8857142857142858`
   - `wf_nested_test_bets_total=357`
   - held-out formal `weighted_roi=1.0307760253096685`
   - `formal_benchmark_bets_total=938`
   - `bets / races / bet_rate = 938 / 6244 = 15.02%`
3. baseline refresh 比では `auc`, `ev_top1_roi`, nested WF weighted ROI は小幅に上、`top1_roi` と nested WF bet count は下
4. したがって next execution source は actual-date role split である

Primary next issue draft:

- `docs/next_issue_pace_closing_fit_actual_date_role_split.md`

Current active issue:

- none

Latest September fallback compare:

1. `#85` で `r20260403_pace_closing_fit_selective_v1` と `current_sep_guard_candidate_2025_latest` を broad September / late-September で比較した
2. broad September:
   - sep guard `9 bets / -4.3 / pure bankroll 0.9995842542264094`
   - pace-closing-fit `3 bets / -3.0 / pure bankroll 0.892891589506173`
3. late-September:
   - sep guard `6 bets / -6.0 / pure bankroll 0.9857901270555528`
   - pace-closing-fit `2 bets / -2.0 / pure bankroll 0.9184027777777779`
4. net だけ見れば pace-closing-fit は小さく見えるが、pure bankroll は両 window とも sep guard が上
5. したがって `current_sep_guard_candidate_2025_latest` は September seasonal fallback のまま据え置き、`r20260403_pace_closing_fit_selective_v1` は `analysis-first promoted candidate` に留める

Latest actual-date role split:

1. `#84` pace-closing-fit actual-date role split は September difficult window と December control window の compare で確定した
2. broad September:
   - baseline `32 bets / -27.3 / pure bankroll 0.2958869306148325`
   - pace-closing-fit candidate `3 bets / -3.0 / pure bankroll 0.892891589506173`
3. December control:
   - baseline `45 bets / +21.8 / pure bankroll 1.6711564921099862`
   - pace-closing-fit candidate `0 bets / 0.0 / pure bankroll 1.0`
4. したがって `r20260403_pace_closing_fit_selective_v1` は `analysis-first promoted candidate` に据え置き、serving default には上げない

Primary next issue draft:

- none

Latest breeder signal read:

1. `#88` で `breeder_last_50_win_rate` 単体 add-on を feature-gap で読んだ
2. result:
   - `selected=True`
   - `present=True`
   - `non_null_ratio=0.18955`
   - `status=low_coverage`
3. `low_coverage_force_include_features=['breeder_last_50_win_rate']`
4. したがって breeder 単体でも current high-coverage line には薄すぎる
5. lineage family は引き続き primary な next bet に戻さない

Latest owner signal audit:

1. `#86` で `owner_last_50_win_rate` を current JRA high-coverage baseline から外した selective ablation を formal compare した
2. win / roi component の selected features は `109 -> 108` となり、owner は actual used set から外れた
3. evaluation summary:
   - `auc=0.8449770387983483`
   - `top1_roi=0.4597201921069117`
   - `ev_top1_roi=0.8105412402909629`
   - nested WF `3/3 no_bet`
   - `wf_nested_test_bets_total=0`
4. `#72` の short-circuit により revision gate は `hold`
5. したがって owner signal は current baseline で prune しない

Latest evaluate progress fix:

1. `#87` で JRA `run_evaluate.py` の `inference complete -> leakage audit started` 無音区間を corrective した
2. `scripts/run_evaluate.py` に `[evaluate post]` progress を追加し、post-inference phase を
   - `score sources ready`
   - `summary payload ready`
   - `post-inference phases finished`
   に分解した
3. smoke log `artifacts/logs/evaluate_post_inference_progress_gap_smoke.log` で `1/3 -> 2/3 -> 3/3` を確認した
4. calibration 自体は small tail single-class で失敗したが、progress fix の確認目的は達成した

### 3.0 Current Queue After Tail Micro-Cut Exhaustion

2026-03-29 時点で、Kelly runtime family (`#10`, `#11`, `#12`, `#13`)、seasonal ordering (`#14`, `#15`)、runtime broad reduction (`#7`)、supplemental materialization (`#16`)、feature-builder runtime (`#17`) は close 済みである。loader runtime の small safe cuts を進めた `#18` も wrap-up 段階にあり、current operational anchor は引き続き `r20260329_tighter_policy_ratio003_abs90` である。

この時点の current reading は次である。`#49` の jockey-trainer-combo family では first child `#50` が `pass / promote` に到達したが、`#53` の actual-date role split read で family anchor にはならないことが確認された。したがって直近の execution queue は `#52` の NAR separate-universe baseline formalization に寄せる。

Primary active issue:

- `#60`
- <https://github.com/Npieju/nr-learn/issues/60>
- `#49`
- <https://github.com/Npieju/nr-learn/issues/49>

Primary issue draft:

- `docs/nar_model_transfer_strategy.md`
- `docs/nar_bet_denominator_standard.md`
- `docs/next_issue_local_nankan_baseline_formalization.md`
- `docs/next_issue_nar_post_formal_read.md`
- `docs/next_issue_nar_wf_runtime_followup.md`
- `docs/next_issue_nar_class_rest_surface_replay.md`
- `docs/next_issue_nar_class_rest_surface_availability_audit.md`
- `docs/next_issue_nar_selection_fix_for_buildable_replay.md`
- `docs/next_issue_nar_jockey_trainer_combo_replay.md`
- `docs/next_issue_nar_wf_summary_path_alignment.md`
- `docs/next_issue_nar_gate_frame_course_replay.md`
- `configs/model_local_baseline_wf_runtime_narrow.yaml`
- `configs/features_local_baseline_class_rest_surface_replay.yaml`
- `#60` で long-running job の live progress を repo 内 log file に tee し、VS Code から確認できる operator path を作る
- `docs/next_issue_jockey_trainer_combo_regime_extension.md`

GitHub issue:

- `#60`
- <https://github.com/Npieju/nr-learn/issues/60>
- `#49`
- <https://github.com/Npieju/nr-learn/issues/49>

Primary next execution order:

1. NAR の next family は `docs/next_issue_nar_owner_signal_replay.md` で owner signal replay を narrow に試す
2. JRA の `#49` family は promoted-but-non-anchor read として一段止め、次の widening は NAR next family formal read の後に再判断する

Side universe track:

1. `#52` で NAR を `separate universe` baseline として formalize する
2. `bets / races / bet-rate` を mandatory read にする
3. JRA の process / gating を移植し、threshold や feature priority は NAR 専用に再検証する

### 3.1 [experiment] Tighter policy search frontier refinement

GitHub issue:

- `#2`
- <https://github.com/Npieju/nr-learn/issues/2>

Template:

- `Model Experiment`

Recommended labels:

- `experiment`
- `policy`
- `jra`

Body draft:

```md
Universe
JRA

Category
Policy

Objective
`tighter policy search` family の support frontier をさらに明確化し、`ROI`、`feasible folds`、`drawdown` のバランスが最もよい policy 設定帯を整理する。新しい exotic family を増やすのではなく、既存の strongest defensive family を formal に詰める。

Hypothesis
if `tighter policy search` family の threshold frontier を `ratio`, `min_bets_abs`, `min_prob`, `odds_max`, `min_expected_value` 周辺で狭く再探索する, then we can preserve defensive behavior while improving support clarity and possibly edge toward the ROI>1.20 north-star band, while keeping drawdown and role interpretation stable.

In-Scope Surface
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_minprob005.yaml`
- `configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_tighter_policy_search_probe_ratio003_abs90_odds25.yaml`
- `scripts/run_revision_gate.py`
- `scripts/run_wf_threshold_sweep.py`
- compare / dashboard artifacts

Non-Goals
- 新しい feature family の導入
- staged family の大型追加
- broad baseline replacement の即時判断
- NAR policy への展開

Success Metrics
- `5/5` を維持する strictest anchor が `abs90` として説明できる
- drawdown / bankroll / bet volume を壊さない narrow frontier が見つかる
- next revision gate candidate を 1 本に絞れる

Eval Plan
- smoke: threshold sweep と existing compare artifact の読み直し、candidate shortlist 化
- formal: 有望候補のみ revision gate に載せ、September/December role を compare で再確認する

Validation Commands
- `python scripts/run_wf_threshold_sweep.py ...`
- `python scripts/run_revision_gate.py ... --dry-run`
- `python scripts/run_revision_gate.py ...`
- `python scripts/run_serving_profile_compare.py ...`

Expected Artifacts
- threshold frontier summary
- candidate shortlist
- revision gate artifact
- compare dashboard summary

Stop Condition
- support を増やすと drawdown / bankroll が悪化する
- role が曖昧になり baseline より説明しづらくなる
- same-family refinement より別 family 比較のほうが有望と判明する
```

Primary references:

- `docs/next_issue_tighter_policy_frontier.md`
- `docs/tighter_policy_frontier_execution.md`
- `docs/tighter_policy_candidate_matrix.md`

### 3.2 [ops] Revision gate duplicate-run prevention and artifact collision guard

GitHub issue:

- `#1`
- <https://github.com/Npieju/nr-learn/issues/1>

Template:

- `Model Experiment`

Recommended labels:

- `ops`
- `automation`
- `reliability`

Body draft:

```md
Universe
JRA

Category
Evaluation

Objective
同一 `revision` / artifact path に対する duplicate formal run を防ぎ、artifact collision や結果誤読を避ける。現在は同一 revision の `run_revision_gate.py` が並走しうるため、formal result の解釈リスクがある。

Hypothesis
if duplicate revision runs are explicitly blocked or surfaced early, then formal evaluation artifacts become easier to trust and operate, while preserving normal experiment throughput.

In-Scope Surface
- `scripts/run_revision_gate.py`
- related manifest / artifact naming rules
- 必要なら docs の execution standard

Non-Goals
- model policy の再設計
- benchmark 指標の変更
- GitHub Actions 全面導入

Success Metrics
- 同一 revision の並走が operator に明確に見える
- artifact collision を未然に防げる
- `planned / running / completed / failed` の解釈が曖昧でなくなる

Eval Plan
- smoke: same revision の duplicate invocation path を再現・検証する
- formal: collision guard と operator-facing message を docs と manifest に反映する

Validation Commands
- `python scripts/run_revision_gate.py ... --dry-run`
- `python scripts/run_revision_gate.py ...`
- `ps -af | rg 'run_revision_gate.py|run_evaluate.py'`

Expected Artifacts
- updated revision gate manifest behavior
- guardrail log messages
- docs update

Stop Condition
- 既存 operator flow を大きく壊す
- uniqueness rule が厳しすぎて legitimate rerun を阻害する
```

Why now:

- `r20260329_tighter_policy_ratio003_abs90` の formal run で、同一 revision の並走疑いが実際に発生した

### 3.3 [ops] Progress instrumentation real-run validation and message quality pass

GitHub issue:

- `#3`
- <https://github.com/Npieju/nr-learn/issues/3>

Template:

- `Model Experiment`

Recommended labels:

- `ops`
- `automation`
- `observability`

Body draft:

```md
Universe
JRA

Category
Evaluation

Objective
`scripts/run_*.py` へ入れた progress instrumentation を real run で spot check し、message quality と failure-phase logging を整える。dry-run では改善が見えているため、次は実ジョブでも stalled に見えないことを確認したい。

Hypothesis
if progress instrumentation is validated on real runs and message quality is tightened, then operators can distinguish healthy long-running execution from stalls more reliably, while keeping logs readable.

In-Scope Surface
- `scripts/run_*.py` progress instrumentation
- `docs/development_operational_cautions.md`
- `docs/progress_coverage_audit.md`

Non-Goals
- benchmark policy の変更
- NAR/JRA artifact の再定義
- unrelated logging refactor

Success Metrics
- representative real runs で start / phase / completion / failure point が読める
- message quality が coarse すぎる箇所を特定・修正できる
- operator が stuck / running を誤認しにくくなる

Eval Plan
- smoke: dry-run spot check の継続
- formal: representative real run を選んで progress と manifest exit を確認する

Validation Commands
- `python scripts/run_mixed_universe_readiness.py ...`
- `python scripts/run_mixed_universe_numeric_compare.py ...`
- `python scripts/run_revision_gate.py ...`

Expected Artifacts
- updated progress messages
- audit doc update
- spot-check notes

Stop Condition
- log volume が増えすぎて読みにくくなる
- progress 粒度の追加が処理本体より複雑になる
```

Primary references:

- `docs/development_operational_cautions.md`
- `docs/progress_coverage_audit.md`

## 4. Execution Order

今の順番は次で固定する。

1. `[ops] Revision gate duplicate-run prevention and artifact collision guard`
2. `[experiment] Tighter policy search frontier refinement`
3. `[ops] Progress instrumentation real-run validation and message quality pass`

## 5. Note

この環境では `gh` CLI が入っておらず、現在見えている GitHub 連携にも issue 作成 endpoint はない。そのため、当面はこの文書を GitHub issue の下書き正本として扱い、issue 作成は GitHub UI 側で行う前提にする。

2026-03-29 update:

- `gh` は利用可能になった
- 直近 queue は GitHub issue `#1`, `#2`, `#3` として起票済み
- 以後は原則として案件ごとに issue を先に立て、完了時に decision summary を追記して close する
