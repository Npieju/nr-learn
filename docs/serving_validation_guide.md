# Serving 検証ガイド

## 1. この文書の役割

この文書は、`nr-learn` における serving 系の確認手順をまとめたガイドである。

対象は次の 4 つである。

1. representative date の smoke
2. 複数日 window の profile compare
3. replay ベースの軽量検証
4. bankroll / dashboard / aggregate の読み方

## 2. 基本方針

- serving 検証は、runtime の挙動確認と候補の絞り込みに使う。
- 短い smoke / compare は昇格判断そのものではなく、`development_flow.md` で定義した smoke / probe の一部として扱う。
- 正式な昇格判断は、別途 `run_evaluate.py` と `run_promotion_gate.py` を通した revision 単位で行う。

## 3. 主に使う stable profile

- `current_best_eval`
  - nested evaluation 上の主力候補
- `current_recommended_serving`
  - 現時点の簡易運用候補
- `current_bankroll_candidate`
  - 強い de-risk を狙う conservative candidate
- `current_ev_candidate`
  - `current_bankroll_candidate` よりは攻める intermediate candidate
- `current_sep_guard_candidate`
  - September だけ sparse selection guard を Kelly fallback に直結する seasonal de-risk candidate

## 4. 代表日の smoke

特定日だけ runtime 挙動を確認したいときは、まず `run_serving_smoke.py` を使う。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py \
  --profile current_recommended_serving \
  --date 2024-09-14
```

built-in case を持たない profile では `--date` が必須である。

主な出力:

- `artifacts/reports/serving_smoke_<suffix>.json`
- suffix 付き prediction / backtest artifact

staged config の場合、2026-03-22 時点の smoke summary / backtest JSON には stage 名だけでなく次も入る。

- `policy_stage_traces`
- `policy_stage_fallback_reasons`

たとえば `runtime_staged_mitigation_ev_guard_probe` の `2024-08-11` replay smoke では、`kelly_fallback_1` が選ばれた理由として `portfolio_ev_only:max_expected_value_below|portfolio_lower_blend:no_selection` が保存される。これで「fallback したか」だけでなく、「どの段で何が足りず次段へ送られたか」まで artifact から追える。

## 5. 複数日 window の比較

2 候補をまとめて比較するときは `run_serving_profile_compare.py` を使う。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_profile_compare.py \
  --left-profile current_recommended_serving \
  --right-profile current_bankroll_candidate \
  --date 2024-09-16 \
  --date 2024-09-21 \
  --date 2024-09-22 \
  --date 2024-09-28 \
  --date 2024-09-29 \
  --window-label late_sep \
  --run-bankroll-sweep \
  --run-dashboard
```

この 1 run で次が揃う。

1. 左右の `serving_smoke_*.json`
2. compare JSON / CSV
3. bankroll sweep JSON / CSV
4. dashboard JSON / PNG / CSV
5. provenance 用 manifest

この provenance manifest は `serving_smoke_profile_compare_*.json` に保存され、途中 step が失敗した場合も可能な限り実行済み step と失敗位置を残す。

明示的に output path を渡す場合は、summary / compare / bankroll / dashboard の各出力とも directory ではなく file path を渡す。

2026-03-22 に `current_sep_guard_candidate` でもこの wrapper を end-to-end で確認した。

- window は `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29`
- manifest `serving_smoke_profile_compare_current_recommended_serving_late_sep_20260322_profile_vs_current_sep_guard_candidate_late_sep_20260322_profile.json` は `status=completed`, `decision=ready`
- compare は baseline `31 bets / total net -25.6` に対して candidate `10 bets / total net -10.0`
- bankroll sweep は pure path で baseline `0.2780` に対して candidate `0.9915`
- dashboard summary / PNG / CSV まで同 run で生成できることを確認した

つまり `current_sep_guard_candidate` は ad hoc config compare だけでなく、stable profile として通常の smoke -> compare -> bankroll -> dashboard 導線でも再現可能である。

ただし、同日の formal revision gate `r20260322e` では位置づけが変わらない。evaluation と matching な `wf_feasibility_diag` はともに `representative` だったが、promotion gate は `wf_feasible_fold_count=0/5` で `block/hold` になった。dominant failure reason は `min_bets`、binding source は全 fold `min_bets_abs=100` で、short-window serving compare の良さをそのまま benchmark 昇格根拠には使えない。

### 5.1 stage path の横比較

single-policy probe と staged probe を actual-date ごとに横並びで見たいときは `run_serving_stage_path_compare.py` を使う。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_stage_path_compare.py \
  --summary baseline=artifacts/reports/serving_smoke_runtime_baseline_late_sep_20260322.json \
  --summary portfolio_ev_only=artifacts/reports/serving_smoke_runtime_portfolio_ev_only_late_sep_20260322.json \
  --summary staged=artifacts/reports/serving_smoke_runtime_staged_mitigation_late_sep_20260322_v2.json \
  --summary staged_ev_guard=artifacts/reports/serving_smoke_runtime_staged_mitigation_ev_guard_late_sep_20260322.json \
  --output-json artifacts/reports/serving_stage_path_compare_late_sep_20260322.json \
  --output-csv artifacts/reports/serving_stage_path_compare_late_sep_20260322.csv
```

この report では次をまとめて確認できる。

- `shared_dates_all`
- `shared_ok_dates_all`
- `differing_policy_dates`
- `differing_stage_dates`
- `differing_stage_fallback_reason_dates`
- `differing_stage_trace_dates`
- label ごとの `policy_stage_names` 集計

staged summary に `policy_stage_traces` / `policy_stage_fallback_reasons` が入っていれば、compare 側でも label ごとの trace / reason count を集計する。

2026-03-22 の実行結果では、次が観測された。

- late-September 5 日 window は `shared_ok=5/5` で、`differing_stage_dates` は全 5 日だった
- August weekends 6 日 window は `shared_ok=6/6` で、`differing_stage_dates` は `2024-08-03`, `2024-08-10`, `2024-08-11`, `2024-08-18` だった
- fallback reason まで見ると、late-September は `differing_stage_fallback_reason_dates` も全 5 日だった
- August weekends は `differing_stage_fallback_reason_dates` が `2024-08-03`, `2024-08-11`, `2024-08-18` に絞られた

特に `staged_mitigation_ev_guard_probe` は次の形で読める。

- late-September では 5/5 日すべてで fallback reason が出ており、`kelly_fallback_1` に流れていた
- その内訳は `portfolio_ev_only:max_expected_value_below|portfolio_lower_blend:no_selection` が 4 日、`portfolio_ev_only:max_expected_value_below|portfolio_lower_blend:max_expected_value_below` が 1 日だった
- August weekends では reason を伴う fallback は 3 日で、`2024-08-03` だけが `kelly_fallback_2` まで落ち、`2024-08-11` と `2024-08-18` は `kelly_fallback_1` だった

つまり、staged probe は「runtime で読める」だけでなく、actual calendar 上でも日付単位の stage path 分岐を起こしている。ただし、その分岐自体は baseline 優位を覆す根拠にはまだなっていない。

## 6. replay ベースの軽量検証

shared feature build が重い window では、既存 prediction CSV を再利用する replay モードを使う。

### 6.1 smoke 側で replay を使う

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_smoke.py \
  --profile current_bankroll_candidate \
  --prediction-backend replay-existing \
  --date 2024-09-16 \
  --date 2024-09-21
```

### 6.2 prediction から直接 replay summary を作る

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_serving_replay_from_predictions.py \
  --config configs/model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving.yaml \
  --prediction-files artifacts/predictions/predictions_20240916.csv \
  --artifact-suffix replay_probe
```

この方法では heavy feature rebuild を避けたまま、`serving_smoke_*` 互換の summary を作れる。

### 6.3 staged EV guard threshold sweep

staged EV guard の `max_expected_value_below` が aggressive すぎるかを actual-date replay で見るときは `run_staged_ev_guard_threshold_sweep.py` を使う。

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_staged_ev_guard_threshold_sweep.py \
  --window-label late_sep_20260322 \
  --date 2024-09-16 \
  --date 2024-09-21 \
  --date 2024-09-22 \
  --date 2024-09-28 \
  --date 2024-09-29 \
  --stage1-threshold 1.08,1.06,1.04,1.02 \
  --stage2-threshold 1.03,1.01,0.99
```

この script は canonical な `artifacts/predictions/predictions_YYYYMMDD.csv` を再利用し、variant ごとに次を集計する。

- `total_policy_bets`
- `total_policy_net`
- `kelly_fallback_dates`
- `fallback_reason_dates`
- `stage_name_counts`
- `stage_trace_counts`
- `stage_fallback_reason_counts`

既定では次を出力する。

- `artifacts/reports/staged_ev_guard_threshold_sweep_<window>.json`
- `artifacts/reports/staged_ev_guard_threshold_sweep_<window>.csv`
- `artifacts/reports/staged_ev_guard_threshold_sweep_<window>_summary.csv`

2026-03-22 の replay sweep では、次が分かった。

- late-September 5 日 window では、全 12 variant が同じ `10 bets / total net -10.0` だった
- ただし stage path は変わっており、`s1=1.08,s2=1.03` では 5/5 日すべてが `kelly_fallback_1`、`s1=1.02,*` では fallback は 3/5 日まで減った
- `s1=1.08,s2=0.99` では `2024-09-28` だけ `portfolio_lower_blend:selected` になったが、top-line は改善しなかった
- August weekends 6 日 window では、`s1 in {1.08,1.06,1.04}` の全 variant が同じ `6 bets / total net -6.0` だった
- August の `s1=1.08,s2 in {1.01,0.99}` では `2024-08-03` が `kelly_fallback_2` から `portfolio_lower_blend:selected` に変わったが、top-line は不変だった
- August の `s1=1.06` では fallback reason 日付が `2024-08-11`, `2024-08-18` に絞られ、`s1=1.04` では `portfolio_ev_only:selected` が 4/6 日まで増えた
- ただし `s1=1.02` まで下げると August は全 variant で `8 bets / total net -8.0` に悪化した

実務上の読み方は単純である。

- late-September では EV guard 緩和は stage path を変えるが、今回の grid では realized net を変えなかった
- August では moderate な緩和は unnecessary fallback を減らすが、利益回復までは届かなかった
- したがって、次の論点は「EV guard を少し緩めるか」よりも、「August profitable regime を拾える selector 条件をどう設計するか」である

### 6.4 staged policy signal diagnostic

threshold 調整だけでは判断しきれないときは `run_staged_policy_signal_diagnostic.py` を使う。これは staged config を race 単位で replay し、各 stage について次を出す。

- `selected_count`
- `max_expected_value`
- `max_prob`
- `max_edge`
- `fallback`
- `fallback_reasons`
- `selected_return_units`
- `selected_net_units`
- final の `policy_stage_trace` / `policy_stage_fallback_reasons`

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_staged_policy_signal_diagnostic.py \
  --window-label aug_weekends_signal_20260322 \
  --date 2024-08-03 \
  --date 2024-08-04 \
  --date 2024-08-10 \
  --date 2024-08-11 \
  --date 2024-08-17 \
  --date 2024-08-18
```

既定の出力は次である。

- `artifacts/reports/staged_policy_signal_diagnostic_<window>.json`
- `artifacts/reports/staged_policy_signal_diagnostic_<window>.csv`
- `artifacts/reports/staged_policy_signal_diagnostic_<window>_summary.csv`

2026-03-22 の August weekends 診断では、threshold sweep より踏み込んだことが分かった。

- `2024-08-03` は `portfolio_ev_only` が 2 race、`portfolio_lower_blend` が 1 race を選んでいたが、どちらも hit せず、そのまま net は負だった
- `2024-08-11` も `portfolio_ev_only` は 2 race を選んでいたが、median `max_expected_value` は約 `1.048` で 0 hit だった
- `2024-08-18` は `portfolio_ev_only` が 4 race を選んでいたが、median `max_expected_value` は約 `1.023` で 0 hit だった
- 一方で baseline `current_recommended_serving` は同じ August window で `2024-08-17` に `7 bets / net +32.6`、`2024-08-18` に `8 bets / net +2.7` を出していた
- それにもかかわらず staged probe は `2024-08-17` で全 stage `no_selection`、`2024-08-18` でも early portfolio stages が profitable day を再現できていなかった

つまり、August の差は「EV guard が少し厳しすぎる」だけではない。少なくとも profitable day の一部では、early portfolio policy family 自体が baseline の利益局面を再現できていない。次の改善候補は threshold lowering 単体ではなく、stage policy 本体か regime-aware selector 条件である。

### 6.5 portfolio policy family sweep

early stage の policy family 自体が強すぎるかを切るときは `run_portfolio_policy_family_sweep.py` を使う。これは指定日の canonical prediction CSV を replay し、single-policy portfolio を次の grid で比較する。

- `blend_weight`
- `min_prob`
- `min_expected_value`

例:

```bash
/workspaces/nr-learn/.venv/bin/python scripts/run_portfolio_policy_family_sweep.py \
  --window-label aug_portfolio_family_20260322 \
  --reference-date 2024-08-03 \
  --date 2024-08-03 \
  --date 2024-08-04 \
  --date 2024-08-10 \
  --date 2024-08-11 \
  --date 2024-08-17 \
  --date 2024-08-18
```

2026-03-22 の August weekends sweep では、結論はかなり明快だった。

- baseline family そのものの `blend=0.8 / min_prob=0.03 / min_ev=0.95` が最良で、`34 bets / total net +20.1` を出した
- `min_ev` を `1.0` に上げた `blend=0.8 / min_prob=0.03 / min_ev=1.0` は `9 bets / total net -9.0` まで崩れた
- `2024-08-17` は `min_ev=0.95` なら `7 bets / net +32.6` だが、`min_ev=1.0` では `0 bets / net 0.0` になった
- `2024-08-18` も `min_ev=0.95` なら `8 bets / net +2.7` だが、`min_ev=1.0` では `4 bets / net -4.0` になった
- `blend=0.6` の variants は `min_ev=0.95` でも `-9.0`、`min_ev=1.0` では `-2.0` か `-1.0` に留まり、baseline family を再現できなかった

したがって、August profitable regime の primary gap は EV guard より upstream にある。具体的には、stage1 の `portfolio_ev_only` が使っている `min_expected_value=1.0` と stage2 の `blend=0.6` が、baseline August policy の利益局面を大きく削っている。

この仮説を staged runtime に戻して確かめるため、`..._staged_aug_baseline_stage1_probe.yaml` も actual-date replay で確認した。これは stage1 を baseline August family の `blend=0.8 / min_prob=0.03 / min_ev=0.95` に合わせ、fallback は no-selection 時だけ後段へ渡す構成である。

- August weekends 6 日 window では baseline と完全一致し、`34 bets / total net +20.1` だった
- 6/6 日すべてで `policy_stage_traces=["portfolio_aug_baseline:selected"]` となり、fallback reason は空だった
- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して、この probe は `38 bets / total net -32.6` まで悪化した
- late-September でも 5/5 日すべてが `portfolio_aug_baseline:selected` で、`portfolio_lower_blend` や Kelly fallback は一度も使われなかった

つまり、stage1 を baseline August family に寄せれば August profit regime 自体は回復するが、その時点で staged wrapper は実質的に single-policy baseline の複製になる。したがって、次の論点は「stage1 を baseline に寄せるかどうか」ではなく、「baseline 相当の August capture を保ったまま late-September だけを別条件で抑える guard / regime split をどう追加するか」である。

その guard 候補として、続けて `..._staged_aug_baseline_stage1_selected_rows_guard_probe.yaml` も試した。これは stage1 に `fallback_when.selected_rows_at_most: 5` を足し、選択 race が少ない日は lower-blend / Kelly fallback へ送る構成である。

- late-September 5 日 window では `31 bets / total net -25.6` から `10 bets / total net -10.0` まで強く defensive になった
- 5/5 日すべてで stage1 が `selected_rows_at_most` に引っかかり、`portfolio_lower_blend` か `kelly_fallback_1` に落ちた
- ただし August weekends 6 日 window も同時に崩れ、baseline `34 bets / total net +20.1` に対して `6 bets / total net -6.0` だった
- 特に `2024-08-17` は baseline の `7 bets / +32.6` が `0 bets / 0.0` になり、`2024-08-11` と `2024-08-18` も Kelly fallback に落ちて利益を再現できなかった

したがって、`selected_rows_at_most` は late-September の de-risk には効くが、August profitable regime を守るには blunt すぎる。次の guard は selected count 単独ではなく、date-context か別 signal と組み合わせて設計する必要がある。

この bluntness を避けるため、続けて `..._serving_sep_selected_rows_guard_candidate.yaml` も追加した。これは serving の month-aware regime split をそのまま使い、September だけ `selected_rows_at_most: 5` 付き staged policy に切り替える candidate である。

- May weekends 6 日 window は baseline と完全一致し、`6 bets / total net -6.0` だった
- August weekends 6 日 window も baseline と完全一致し、`34 bets / total net +20.1` を維持した
- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して `10 bets / total net -10.0` まで改善した
- June-August の `tail_weekends` 19 日 window でも baseline と完全一致し、`55 bets / total net -0.9`、`differing_policy_dates=[]` だった
- late-September の 5/5 日すべてで September override が選ばれ、`portfolio_aug_baseline:selected_rows_at_most` を起点に `portfolio_lower_blend` か `kelly_fallback_1` へ流れた

つまり、date-context を明示した regime-aware split にすると、`selected_rows` guard の defensive 効果だけを September に閉じ込められる。これは blunt guard 単体とは違い、May/August の既存 profitable or neutral behavior を壊さず、late-September だけを de-risk できる candidate として読むのが妥当である。

bankroll sweep でもこの読みは崩れなかった。

- late-September では total net は baseline `-25.6` に対して candidate `-10.0` まで改善したが、pure path の final bankroll は baseline `0.2780` に対して candidate `0.0` で、bankroll sweep の best path は baseline 側に残った
- May weekends は baseline / candidate ともに final bankroll `0.9906` で完全一致した
- August weekends は baseline / candidate ともに final bankroll `1.1765` で完全一致した
- `tail_weekends` も baseline / candidate ともに final bankroll `0.7432` で完全一致した

したがって、この candidate の current evidence は「September だけ net を軽くできるが、pure bankroll まで良化するわけではなく、それ以外の確認済み window では realized policy も bankroll path も baseline と一致する」である。次に広げるべきなのは guard の再調整ではなく、aggregate / dashboard でこの `net 改善 vs bankroll 悪化` の trade-off がどこまで安定して再現されるかの確認である。

この trade-off の原因を late-September の stage path まで戻して確認すると、実質的な崩壊点は `2024-09-28` の 1 日に絞られた。September candidate は 5 日中 4 日で `kelly_fallback_1` を使っていたが、`2024-09-28` だけが `portfolio_lower_blend:selected` になり、その 1 bet で pure path bankroll を `0.0` まで落としていた。

そこで次段として `..._serving_sep_selected_rows_kelly_only_candidate.yaml` も追加した。これは September override から `portfolio_lower_blend` stage を外し、`selected_rows_at_most: 5` 発火後は Kelly fallback へ直落としする variant である。

- late-September 5 日 window では bets / net がそのまま `10 bets / total net -10.0` に保たれた
- その一方で pure path final bankroll は baseline `0.2780` に対して `0.9915` まで回復し、bankroll sweep の best path も new candidate 側に移った
- May weekends 6 日、August weekends 6 日、`tail_weekends` 19 日の 3 window はすべて baseline と完全一致し、policy / net / pure bankroll の差分は 0 だった

したがって、September-only guard の実務上の論点は `selected_rows_at_most` 自体ではなく、その fallback 先の選び方にある。少なくとも現在の replay evidence では、September の sparse fallback を portfolio に戻すより Kelly に直結した方が、late-September の `net 改善` を保ったまま `bankroll 崩壊` を避けやすい。

旧 September candidate との direct compare でも、この読みはそのまま確認できた。

- late-September 5 日 window で旧 `sep_selected_rows_guard_candidate` と新 `sep_selected_rows_kelly_only_candidate` はどちらも `10 bets / total net -10.0` だった
- 一方で pure path final bankroll は旧 candidate `0.0` に対して、新 candidate は `0.9915` だった

したがって、新 variant は「同じ late-September net 改善を保ったまま bankroll 崩壊だけを取り除いた」形で、旧 September candidate をそのまま支配している。

この line に対して、September fallback の途中へ `portfolio_ev_only` を 1 段だけ戻す `..._serving_sep_selected_rows_ev_only_kelly_candidate.yaml` も replay した。意図は fold4 recovery の `min_expected_value=1.0` を serving 側で最小限に試すことだったが、結果は改善ではなかった。

- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して新 probe は `12 bets / total net -12.0` で、Kelly-only candidate の既知結果 `10 bets / -10.0` を上回れなかった
- June-August `tail_weekends` 19 日では baseline と完全一致し、`55 bets / total net -0.9`, `differing_policy_dates=[]` だった

したがって、September de-risk line の current best は依然として Kelly-only fallback であり、`portfolio_ev_only` をその前に挿すだけでは September exposure を少し戻すだけで、追加の価値は見えなかった。この probe は discard でよい。

ここで staged fallback 条件の粒度も整理した。runtime の既存 `selected_rows_at_most` は date-wide ではなく race-local に評価されるため、`top_k=1` の portfolio stage では「1 race でも選んだらほぼ発火する」guard になりやすい。September sparse day を日次件数で分けたいときに必要なのは別 key なので、runtime へ `date_selected_rows_at_most` も追加した。

この新 key で 2 本の September-only probe を replay した結果は次のとおりである。

- `..._serving_sep_date_selected_rows_kelly_candidate.yaml`: stage1 を August baseline family のままにして `date_selected_rows_at_most: 5` を掛けたが、late-September は `29 bets / total net -23.6` に留まり、tail は baseline と完全一致だった。non-fallback 日に August family がそのまま September over-exposure を戻すため、不採用。
- `..._serving_sep_baseline_date_selected_rows_kelly_candidate.yaml`: stage1 を September baseline family (`min_prob=0.05`) に揃えると late-September は `24 bets / total net -18.6` まで改善し、tail はやはり baseline と完全一致だった。ただし current Kelly-only guard の `10 bets / -10.0` には届かない。

したがって、date-wide sparse guard 自体は runtime capability として有効だが、threshold `5` の first probe ではまだ弱い。現時点の serving frontier は変わらず Kelly-only fallback であり、`date_selected_rows_at_most` は次段の threshold sweep / richer date-wide signal probe 用の土台として扱うのが妥当である。

この次段として `scripts/run_staged_date_selected_rows_sweep.py` も追加し、`..._serving_sep_baseline_date_selected_rows_kelly_candidate.yaml` の `date_selected_rows_at_most` を late-September 5 日で `1..10` まで replay sweep した。frontier は単調ではあるが、current best を更新するほどではない。

- `1` は実質 baseline と同じで `31 bets / total net -25.6`
- `5` は初回 probe と同じ `24 bets / total net -18.6`
- 最良だった `10` でも `17 bets / total net -17.0` に留まり、current Kelly-only guard の `10 bets / -10.0` には届かなかった

したがって、date-wide sparse guard の weakness は `5` という初期 threshold だけの問題ではない。少なくとも selected-row count 単独では、September loss-heavy day を十分に切り分けられていない。次に試すべきなのは threshold 追加調整より、date-wide EV / edge 集約などを含む richer guard である。

この次段の判断材料として、`scripts/run_policy_date_signal_report.py` を追加し、canonical prediction CSV と runtime policy から per-date の selected-signal artifact を直接出せるようにした。初回は baseline September policy を late-September 5 日 (`2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29`) で replay し、`artifacts/reports/policy_date_signal_report_late_sep_baseline_signal_20260322.{json,csv}` を生成した。

結果は、count-only guard の弱さを補強する一方で、単純 aggregate guard 追加にも慎重であるべきことを示している。

- `2024-09-28` は `selected_count=2` と sparse なのに、`ev_mean=1.0303`, `edge_mean=+0.0303`, `count_ev_ge_1_0=1`, `count_edge_pos=1` と aggregate が最も強く、それでも realized は `2 bets / -2.0` だった
- `2024-09-22` は `selected_count=10`, `ev_mean=0.9845`, `edge_mean=-0.0155`, `count_ev_ge_1_0=3`, `count_edge_pos=3` と特に強い aggregate ではないが、late-September の中では比較的ましな realized 日だった
- `2024-09-16` は `selected_count=11`, `ev_mean=0.9840`, `count_ev_ge_1_0=4` で `2024-09-22` にかなり近く、単純 threshold だけでは両者を素直に分離できない

したがって、`ev_mean`, `edge_mean`, `count_ev_ge_1_0`, `share_edge_pos` のような単純な date-wide summary をそのまま runtime fallback key にするのは、今の evidence では支持されない。現時点の operational reading は「count-only よりは広い signal 診断が必要だが、次に足す guard は単純 aggregate threshold ではなく、race-level composition や fallback trace を含む richer regime descriptor であるべき」というものである。

この trace 側の再確認の過程で、`scripts/run_staged_policy_signal_diagnostic.py` に seasonal staged config を読めないバグも見つかった。従来は staged policy を固定日付 `2024-01-01` で resolve していたため、September override の staged candidate を診断しようとすると default non-staged policy を拾って `serving runtime policy must be staged` で失敗していた。これを date-aware resolve に修正し、回帰テストも追加した。

修正後に `current_sep_guard_candidate` を late-September 5 日で再診断すると、simple aggregate では見えなかった非対称性が trace に現れた。

- `2024-09-28` は baseline aggregate が最も強い (`ev_mean=1.0303`, `edge_mean=+0.0303`) のに、September guard では唯一 `kelly_fallback_2` まで reach した日だった
- `2024-09-22` は stage1 selected race が 11 本と多いが `kelly_fallback_2` には reach せず、final stage は `kelly_fallback_1` 3 本だけだった
- つまり、崩れやすい日を単純 EV/edge summary だけで拾うより、`deeper fallback が必要になったか` という trace information の方が今は情報量が高い

このため、次に検討すべき richer regime descriptor は `ev_mean` や `count_ev_ge_1_0` の閾値そのものではなく、per-date の `final_stage_counts`, `fallback depth`, `stage trace composition` をまとめたものになる。新しい runtime key を足す前に、まずその集計 artifact を安定化させるのが妥当である。

この集計 artifact として、`scripts/run_staged_trace_date_report.py` も追加した。入力は `run_staged_policy_signal_diagnostic.py` の raw CSV で、per-date に `final_stage_counts`, `final_trace_counts`, `final_fallback_reason_counts`, `deepest_selected_stage_counts`, `stage2_plus_selected_race_count`, `stage3_plus_selected_race_count` を出す。ここで重要なのは、単純な `trace depth` では no-selection race まで全 stage を通るため signal が薄まることだ。実際の separator 候補は、`選択が立った最深 stage` を見る selection-aware depth にある。

late-September の `current_sep_guard_candidate` では、この selection-aware depth がようやく差を作った。

- `2024-09-28` だけが `max_selected_stage_index=3` かつ `stage3_plus_selected_race_count=1`
- `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-29` はすべて `max_selected_stage_index=2`, `stage3_plus_selected_race_count=0`
- したがって、`stage3 candidate present` は少なくとも今回の late-September 5 日では `2024-09-28` を一意に切り出す trace-aware signal になっている

これは `2024-09-28` が daily EV/edge aggregate では最も強く見えていたことと対照的である。つまり、September guard の次段設計は「日次平均 EV が強いか」ではなく、「fallback の deeper stage にまで selection candidate が立ったか」を first-class に扱うべきである。現時点の次段は runtime key 追加ではなく、この signal を他 window に展開して再現性を確認することになる。

この再現性も、artifact を人力比較せず数えられるよう `scripts/run_staged_trace_support_check.py` にまとめた。3 本の staged trace date report (`aug_ev_guard`, `late_sep_ev_guard`, `sep_guard`) を横断すると、config の最終 stage に実際の選択が立った `deepest_stage_selected_present` は 16 rows 中 4 rows、日付ユニークでは `2024-08-03`, `2024-08-10`, `2024-09-28` の 3 日だけだった。しかもその 4 rows はすべて realized `final_selected_net_units < 0` で、positive は 0 件だった。

- `aug_ev_guard` の deepest-stage selected day は `2024-08-03`, `2024-08-10`
- `late_sep_ev_guard` では `2024-09-28` だけ
- `sep_guard` でも `2024-09-28` だけ

ここで重要なのは、signal を `stage3_plus_selected` のような stage index ベースで雑に数えないことだ。3-stage config と 4-stage config を同じ意味で比べるには、「その config の deepest stage に selection が立ったか」を first-class に扱う必要がある。今回の support check はその semantics に揃えてあり、この定義なら `deepest_stage_selected_present` は small-sample ながら precision の高い high-risk subset flag と読める。

ただし、この signal は complete separator ではない。`2024-08-11`, `2024-08-18`, `2024-09-16`, `2024-09-22`, `2024-09-29` のように stage2 止まりでも non-positive な日は残る。したがって、現時点の operational reading は次のとおりである。

- `deepest_stage_selected_present` は loss-heavy subset を切り出す risk flag としては promising
- しかし単独で全ての bad day を捉える guard ではない
- よって、いまやるべきことは runtime key 化ではなく、より多い window で support を積み、必要なら stage2/3 の組み合わせ signal へ拡張すること

この recall 側の拡張も同じ artifact で数えられるようにした。`run_staged_trace_support_check.py` は row を `deepest_stage_selected` / `intermediate_stage_selected` / `stage1_only_selected` / `no_final_selection` に bucket 化して出せる。現行の 3 report / 16 rows では `intermediate_stage_selected` が 10 rows あり、これも `positive=0`, `non-positive=10` だった。日付ユニークでは `2024-08-11`, `2024-08-18`, `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-29` の 6 日で、deepest-stage signal が拾わない bad day をかなり補完している。

- `aug_ev_guard`: `intermediate_stage_selected=2`, `deepest_stage_selected=2`, `no_final_selection=2`
- `late_sep_ev_guard`: `intermediate_stage_selected=4`, `deepest_stage_selected=1`
- `sep_guard`: `intermediate_stage_selected=4`, `deepest_stage_selected=1`

したがって、現在の読みは 2 層に分けるのが妥当だ。`deepest_stage_selected_present` は precision 寄りの tail-risk flag、`intermediate_stage_selected_present` は recall を埋める broader-risk bucket である。どちらもまだ small-sample なので runtime key 化は早いが、少なくとも現行 support では `positive` day に一度も出ていないため、selection-aware trace bucket を次段の検証軸として昇格させる根拠はできた。

この仮説はその後すぐに拡張 support で再検証した。`staged_mitigation_ev_guard_probe` の staged trace を May weekends 6 日と `tail_weekends` 24 日へ追加し、5 report / 46 rows (`may_ev_guard`, `aug_ev_guard`, `tail_ev_guard`, `late_sep_ev_guard`, `sep_guard`) を `staged_trace_support_check_may_aug_tail_late_sep_depth_bucket_support_20260322.{json,csv}` にまとめ直した。

- `deepest_stage_selected` は 13 rows に増えても `negative=13`, `positive=0`
- `intermediate_stage_selected` は 20 rows に増え、`negative=18`, `positive=2`
- intermediate の positive は `tail_ev_guard` の `2024-09-07`, `2024-09-14` だった
- net severity でも差があり、`deepest_stage_selected` は mean `-0.5399`, median `-1.0`、`intermediate_stage_selected` は mean `-0.0001`, median `-0.0008` だった

report 別の bucket count は次のとおりである。

- `may_ev_guard`: `deepest=2`, `intermediate=2`, `no_final_selection=2`
- `aug_ev_guard`: `deepest=2`, `intermediate=2`, `no_final_selection=2`
- `tail_ev_guard`: `deepest=7`, `intermediate=8`, `no_final_selection=9`
- `late_sep_ev_guard`: `deepest=1`, `intermediate=4`
- `sep_guard`: `deepest=1`, `intermediate=4`

この extended support により operational reading も更新される。`deepest_stage_selected_present` は window 拡張後も precision を崩しておらず、しかも final selected net の severity まで明確に悪いので、引き続き tail-risk subset flag として扱える。一方で `intermediate_stage_selected_present` は recall 補完としては有効だが、すでに positive day を含み、loss magnitude もほぼゼロ近傍に散っているため、そのまま risk flag に昇格させるのは危険である。次段は deepest を主 signal に据えたまま、intermediate は説明用または後段の複合条件候補として扱うのが妥当である。

この reading はさらに broad-window でも再確認した。重複 window を足し合わせる代わりに、`2024-05-04..2024-09-29` の base prediction 45 race-day 全体を `full_recent_ev_guard` として replay し、`sep_guard` と並べた support artifact `staged_trace_support_check_full_recent_sep_guard_depth_bucket_support_20260322.{json,csv}` を追加した。

- `full_recent_ev_guard` 単体: `deepest_stage_selected=12`, `intermediate_stage_selected=20`, `no_final_selection=13`
- `sep_guard` を合わせた 50 rows でも `deepest_stage_selected` は `13 negative / 0 positive`
- `intermediate_stage_selected` は `21 negative / 3 positive` で、positive は `2024-06-16`, `2024-09-07`, `2024-09-14`
- severity は `deepest mean=-0.7709, median=-1.0` に対し `intermediate mean=-0.00007, median=-0.00069`

したがって、deepest-stage signal は recent calendar 全体でも precision と severity を保つ tail-loss subset flag と読める。一方で intermediate-stage signal は broad window にすると positive contamination が増えるため、説明変数としては残せても standalone guard 候補にはしない方がよい。

## 7. bankroll sweep の見方

bankroll 観点まで見たいときは `run_serving_stateful_bankroll_sweep.py` を使う。

ポイント:

- default の floor grid には `1.01` が含まれる
- pure stage path も別候補として常に評価する

つまり、threshold grid が足りず「常に candidate を使う」path を見落とすことはない。

見るべき項目:

1. `best_result.final_bankroll`
2. `best_result.selection_mode`
3. `best_result.selected_label`
4. pure path と baseline-only path の差

`selection_mode=pure_stage` なら、実質的に「常にその profile を使うのが最良だった」ことを意味する。

## 8. dashboard と aggregate

### 8.1 window ごとの dashboard

compare manifest から window 単位の可視化を作るときは `run_serving_compare_dashboard.py` を使う。

出力:

- summary JSON
- 日次 net / bankroll path の PNG
- 日別 CSV

### 8.2 複数 window の aggregate

期間横断で trade-off を見るときは `run_serving_compare_aggregate.py` を使う。

入力は次のどちらでもよい。

- dashboard summary 群
- compare manifest 群

aggregate では、window ごとの net delta / pure bankroll delta / best sweep bankroll を横断比較する。

2026-03-22 時点の aggregate JSON には、mean だけでなく window ごとの方向と分類も含める。

- `net_delta_direction`
  `right_minus_left` の net 差分の符号。`positive` / `negative` / `zero` / `unknown` を取る。
- `pure_bankroll_delta_direction`
  `right_minus_left` の pure final bankroll 差分の符号。`positive` / `negative` / `zero` / `unknown` を取る。
- `tradeoff_classification`
  net と bankroll の符号をまとめた window 単位の regime label。

実務上の解釈は次のとおりである。

- `positive_net_positive_bankroll`
  候補側が net と bankroll の両面で baseline を上回った。候補を押し上げる strongest evidence。
- `negative_net_positive_bankroll`
  候補側は bankroll 改善と引き換えに net を落としている。de-risk 候補としては有効だが、無条件の置換根拠にはならない。
- `negative_net_negative_bankroll`
  候補側が net と bankroll の両面で不利。baseline 維持を支持する counterexample として扱う。
- `positive_net_negative_bankroll`
  候補側が net は伸ばすが bankroll を悪化させる。攻め寄り候補としては意味があるが、drawdown 制御目的では注意が必要。

summary 側では `tradeoff_classification_counts` と `windows_by_tradeoff_classification` で、どの regime が何本あったかを確認する。

## 9. 候補の読み方

現時点の保守的候補の読み方は次のとおりである。

- `current_bankroll_candidate`
  strongest de-risk candidate。ベット数をかなり絞って bankroll を守る寄り。
- `current_ev_candidate`
  intermediate candidate。`current_bankroll_candidate` よりはベットするが、baseline より損失を抑えやすい。

### 9.1 直近の runtime comparison

2026-03-22 に、次の 5 日で `current_recommended_serving` と `current_best_eval` を `fresh` backend で比較した。

- `2024-09-16`
- `2024-09-21`
- `2024-09-22`
- `2024-09-28`
- `2024-09-29`

実行は [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py) で行い、compare と bankroll sweep を保存した。

観測結果は次のとおりである。

- shared dates は 5/5 で全件 `ok`
- `differing_score_source_dates=[]`
- `differing_policy_dates=[]`
- total policy bets は両者とも `31`
- mean policy ROI は両者とも `0.108`
- total policy net は両者とも `-25.6`
- bankroll sweep でも pure path / threshold grid を含めて差は出なかった

つまり、この window では `current_best_eval` の追加 score override source は runtime 上の差分を生まず、`current_recommended_serving` が simpler candidate として優先しやすい。

### 9.2 bankroll 重視候補との比較

2026-03-22 に、同じ 5 日で `current_recommended_serving` と `current_bankroll_candidate` を `fresh` backend で比較した。

- `2024-09-16`
- `2024-09-21`
- `2024-09-22`
- `2024-09-28`
- `2024-09-29`

実行は [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py) で行い、compare と bankroll sweep を保存した。

観測結果は次のとおりである。

- shared dates は 5/5 で全件 `ok`
- `differing_score_source_dates=[]`
- `differing_policy_dates` は 5/5 で、policy は全日で分岐した
- total policy bets は `current_recommended_serving=31`、`current_bankroll_candidate=1`
- mean policy ROI は comparable な日で `current_recommended_serving=0.108`、`current_bankroll_candidate=0.0`
- total policy net は `current_recommended_serving=-25.6`、`current_bankroll_candidate=-1.0`
- pure stage の bankroll path では `current_recommended_serving=0.2780` に対し、`current_bankroll_candidate=0.9583` だった
- best sweep result も `selection_mode=pure_stage` かつ `selected_label=current_bankroll_candidate` で、threshold grid を含めても bankroll 重視では `current_bankroll_candidate` が優位だった

この window では、`current_bankroll_candidate` は期待値を取りに行くよりも、ほぼノーベットでドローダウンを抑える役割が明確だった。`current_recommended_serving` は同じ score source を使いながら、より積極的にベットして損失も大きくなった。

### 9.3 EV 重視候補との比較

2026-03-22 に、同じ 5 日で `current_recommended_serving` と `current_ev_candidate` を `fresh` backend で比較した。

- `2024-09-16`
- `2024-09-21`
- `2024-09-22`
- `2024-09-28`
- `2024-09-29`

実行は [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py) で行い、compare / bankroll sweep / dashboard を保存した。

観測結果は次のとおりである。

- shared dates は 5/5 で全件 `ok`
- `differing_score_source_dates=[]`
- `differing_policy_dates` は 5/5 で、policy は全日で分岐した
- total policy bets は `current_recommended_serving=31`、`current_ev_candidate=12`
- mean policy ROI は `current_recommended_serving=0.108`、`current_ev_candidate=0.0`
- total policy net は `current_recommended_serving=-25.6`、`current_ev_candidate=-12.0`
- pure stage の bankroll path では `current_recommended_serving=0.2780` に対し、`current_ev_candidate=0.5832` だった
- best sweep result も `selection_mode=pure_stage` かつ `selected_label=current_ev_candidate` で、threshold grid を含めてもこの window では `current_ev_candidate` を全日使う path が最良だった

この window では、`current_ev_candidate` は `current_recommended_serving` よりベット数を `31 -> 12` まで減らして損失を半減させつつ、`current_bankroll_candidate` ほど極端には絞らない中間候補として機能した。late-September のこの 5 日では、役割の並びは `current_recommended_serving` が最も攻め、`current_ev_candidate` が中間、`current_bankroll_candidate` が最も防御的と読める。

### 9.4 複数 window aggregate の読み方

2026-03-22 に、`run_serving_compare_aggregate.py` で次の 2 種類の aggregate を作った。

- `current_recommended_serving` vs `current_bankroll_candidate`
- `current_recommended_serving` vs `current_ev_candidate`

入力 window は最初に次の 2 本で始め、その後 August weekend と May weekend を足して 4 本に拡張した。

- `tail_weekends`
- late-September 5 日 window
- `aug_weekends_20260322`
- `may_weekends_20260322`

aggregate artifact は次に保存した。

- `artifacts/reports/dashboard/serving_compare_aggregate_recommended_vs_bankroll_4windows_20260322.json`
- `artifacts/reports/dashboard/serving_compare_aggregate_recommended_vs_ev_4windows_20260322.json`

観測結果は次のとおりである。

- 4-window では、`current_bankroll_candidate` の pure bankroll delta は `2/4` windows で正、mean は `net -2.6500`, `bankroll +0.1520`
- 4-window では、`current_ev_candidate` の pure bankroll delta も `2/4` windows で正、mean は `net -5.6500`, `bankroll +0.0354`
- net delta が正だったのは両候補とも late-September 側だけだった
- `tail_weekends` では `current_recommended_serving` が total net で優位だったが、両候補とも pure final bankroll は改善した
- late-September では `current_bankroll_candidate` と `current_ev_candidate` の両方が、net と bankroll の両面で `current_recommended_serving` を上回った
- August weekends では逆に `current_recommended_serving` が net と bankroll の両面で両候補を上回った
- May weekends では両候補とも `current_recommended_serving` と policy / net / bankroll が一致し、差分は観測されなかった

classification で見ると、両 aggregate とも 4 本の window は次の 4 分割になった。

- late-September は `positive_net_positive_bankroll`
- `tail_weekends` は `negative_net_positive_bankroll`
- `aug_weekends_20260322` は `negative_net_negative_bankroll`
- `may_weekends_20260322` は `zero_net_zero_bankroll`

この 4 分割が示すのは、候補側の効き方が regime 依存なだけでなく、regime によっては差分自体が消えるということである。したがって、4-window mean を単独で読むのではなく、どの class が何本あるかを先に確認する。

したがって、現時点の実務的な読みは次のとおりである。

1. `current_bankroll_candidate` は drawdown の強い局面では最も defensive だが、常に優位ではない
2. `current_ev_candidate` は依然として中間候補だが、August のような好調 window では baseline を下回る
3. `current_recommended_serving` は単なる baseline ではなく、特定 window では net と bankroll の両面で最良になる
4. May のように candidate 側が baseline と同一挙動へ収束する window もあるため、「候補は常に防御的に振れる」とは限らない
5. したがって、短い serving compare の結論をそのまま一般化せず、複数 regime の window を跨いで読む必要がある

### 9.5 August weekend の反証 window

2026-03-22 に、次の 6 日で `current_recommended_serving` を `replay-existing` で再比較した。

- `2024-08-03`
- `2024-08-04`
- `2024-08-10`
- `2024-08-11`
- `2024-08-17`
- `2024-08-18`

結果は late-September と逆で、`current_recommended_serving` が両候補を上回った。

- vs `current_bankroll_candidate`: total policy net `20.1` vs `-2.0`、pure final bankroll `1.1765` vs `0.9317`
- vs `current_ev_candidate`: total policy net `20.1` vs `-9.0`、pure final bankroll `1.1765` vs `0.7482`
- 両 compare とも best sweep result は `selected_label=current_recommended_serving` だった

この window は、de-risk 候補が常に有利とは限らないことを示す反証になっている。Aug のように baseline 側が十分に利益を出せている局面では、ベット抑制そのものが機会損失になる。

つまり、比較の軸は単純な net だけではなく、次の 3 つで見る。

1. bets
2. total net
3. final bankroll

### 9.6 May weekend の zero-delta window

2026-03-22 に、次の 6 日で `current_recommended_serving` を `replay-existing` で再比較した。

- `2024-05-04`
- `2024-05-05`
- `2024-05-11`
- `2024-05-12`
- `2024-05-18`
- `2024-05-19`

結果は August や late-September とも異なり、`current_bankroll_candidate` / `current_ev_candidate` の両方が `current_recommended_serving` と完全一致した。

- 両 compare とも `differing_score_source_dates=[]`
- 両 compare とも `differing_policy_dates=[]`
- total policy bets はすべて `6`
- total policy net はすべて `-6.0`
- pure final bankroll はすべて `0.9906`
- best sweep result も baseline と同じ path を選び、`tradeoff_classification` は `zero_net_zero_bankroll` になった

この window は、「候補が defensive に効く window」と「baseline が優位な window」に加えて、「candidate を切り替えても実質差が出ない window」が存在することを示している。つまり、stable alias が違っていても、実 calendar 上では同じ policy に収束する regime がある。

### 9.7 threshold 由来の runtime candidate

2026-03-22 に、de-risk 2候補の共通 WF compare artifact から mitigation probe を組み立て、runtime-ready candidate を再生成した。

入力 chain は次のとおりである。

1. `wf_threshold_compare_current_derisk_candidates_20240601_20240929.json`
2. `wf_threshold_signature_family_compare_current_derisk_candidates_20240601_20240929.json`
3. `wf_threshold_signature_drilldown_current_derisk_candidates_20240601_20240929.json`
4. `wf_threshold_mitigation_shortlist_current_derisk_candidates_20240601_20240929.json`
5. `wf_threshold_mitigation_focus_current_derisk_candidates_20240601_20240929.json`
6. `wf_threshold_mitigation_policy_probe_current_derisk_candidates_20240601_20240929.json`
7. `generated_serving_candidates_from_mitigation_probe_current_derisk_candidates_20240601_20240929.yaml`

観測結果は次のとおりである。

- dominant blocked signature は `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95`
- blocked occurrence `84` のうち `74` では `portfolio_lower_blend` 相当の `blend_weight=0.6 / min_ev=1.0` が優位だった
- 残り `10` では `portfolio_ev_only` 相当の `blend_weight=0.8 / min_ev=1.0` が pure bankroll で優位だった
- kelly fallback はこの chain では候補化されず、`kelly_signature_count=0` だった

生成された runtime-ready candidate は次の 2 本である。

- `portfolio_lower_blend`
  - `blend_weight=0.6`
  - `min_prob=0.03`
  - `top_k=1`
  - `min_expected_value=1.0`
  - evidence count `74`
- `portfolio_ev_only`
  - `blend_weight=0.8`
  - `min_prob=0.03`
  - `top_k=1`
  - `min_expected_value=1.0`
  - evidence count `10`

同時に staged hybrid spec も書き出せるが、現行 runtime は単一 policy/date override 前提なので、そのままは load できない。したがって、この chain の実務的な読みは次のとおりである。

1. `portfolio_lower_blend` は threshold frontier を埋める runtime probe 候補として最有力
2. `portfolio_ev_only` は少数の pure-bankroll regime を拾う補助候補
3. staged hybrid は selector logic を拡張するまでは設計メモ扱いに留める

### 9.8 generated single-policy probe の actual-date validation

2026-03-22 に、上の runtime-ready candidate から生成した single-policy config を `run_serving_smoke.py` の `--config` override で actual-date probe した。

使った config は次の 2 本である。

- `model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_portfolio_lower_blend.yaml`
- `model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_portfolio_ev_only.yaml`

この probe の途中で、`run_serving_smoke.py` は explicit `--date` が profile preset に含まれていると supplied config ではなく preset case の expected policy を優先してしまうことが分かった。2024-08-10 の mismatch は runtime candidate 側の不整合ではなく、この case-selection bug に起因していたため、explicit `--date` では常に supplied config から expected policy を自動解決するよう修正した。

修正後に late-September 5 日 window と August weekends 6 日 window を再計測した結果は次のとおりである。

- late-September では baseline `current_recommended_serving` が `31 bets / total net -25.6`、`portfolio_lower_blend` は `1 bet / -1.0`、`portfolio_ev_only` は `12 bets / -12.0` だった
- つまり、strict de-risk としては両 probe とも loss は縮めたが、どちらも mean policy ROI は `0.0` に張り付き、aggressive な return recovery は示せなかった
- August weekends では baseline が `34 bets / total net +20.1`、`portfolio_lower_blend` は `2 bets / -2.0`、`portfolio_ev_only` は `9 bets / -9.0` だった
- つまり、baseline が利益を出せる regime では、single-policy probe は機会損失が大きく、late-September の defensive 改善を容易に打ち消した

実務上の読みは次のとおりである。

1. `portfolio_lower_blend` は strongest defensive probe だが、単独 stable profile に昇格できるほどの regime 汎化はまだない
2. `portfolio_ev_only` は lower-blend より bets は出るが、actual-date probe では baseline replacement の根拠にならなかった
3. 当面は `current_recommended_serving` を baseline に維持し、generated single-policy probe は threshold frontier の挙動確認用 artifact として扱う

### 9.9 staged mitigation probe の actual-date validation

2026-03-22 に、既存の staged probe config も同じ actual-date window で再確認した。

- `model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_staged_mitigation_probe.yaml`
- `model_catboost_value_stack_lgbm_roi_high_coverage_tune_roi012_liquidity_regime_hybrid_june_strict_serving_staged_mitigation_ev_guard_probe.yaml`

late-September 5 日 window では次の結果だった。

- baseline `current_recommended_serving`: `31 bets / total net -25.6`
- `staged_mitigation_probe`: `12 bets / -12.0`
- `staged_mitigation_ev_guard_probe`: `10 bets / -10.0`

August weekends 6 日 window では次の結果だった。

- baseline `current_recommended_serving`: `34 bets / total net +20.1`
- `staged_mitigation_probe`: `9 bets / -9.0`
- `staged_mitigation_ev_guard_probe`: `6 bets / -6.0`

この結果から、staged probe は runtime としては問題なく load できるが、actual-date の realized path は single-policy probe の延長に留まる。`staged_mitigation_probe` はほぼ `portfolio_ev_only` 相当の経路になり、`ev_guard` はさらに sparse にして late-September の loss を少し削る一方、August の機会損失は依然として大きい。

stage path まで見ると差はさらに明確である。

- `staged_mitigation_probe` は late-September 5 日 window の全日で `policy_stage_names=["portfolio_ev_only"]` だった
- August 側でも `staged_mitigation_probe` は `portfolio_ev_only` から実質的に離れず、stage 切替の evidence は観測されなかった
- `staged_mitigation_ev_guard_probe` は late-September では sparse なままでも stage 名は単純化しやすい一方、August weekends では `2024-08-03` に `kelly_fallback_2`、`2024-08-11` と `2024-08-18` に `kelly_fallback_1` へ落ちた

つまり、ev_guard 付き staged probe だけが実 calendar 上で fallback を発火させている。ただし、その fallback は August の profit regime を救うほど strong ではなく、結果は baseline を下回ったままだった。

実務上の読みは次のとおりである。

1. staged probe は selector/runtime の feasibility 確認としては成功している
2. ただし、current baseline を置き換える regime-robust candidate にはなっていない
3. したがって、次の優先度は generated staged spec の自動 config 化そのものより、stage fallback 条件をどう evidence-backed に設計し直すかの方にある

## 10. どこまでを serving で判断するか

serving compare で判断してよいのは、主に次である。

- runtime が壊れていないか
- policy change が actual calendar でどう効くか
- conservative candidate が bankroll をどの程度守るか

serving だけで決めないものは次である。

- 正式な昇格判断
- benchmark の代表値
- revision の確定

これらは `development_flow.md` に従って full evaluation / promotion gate で判断する。

## 11. 関連 script

- [../scripts/run_serving_smoke.py](../scripts/run_serving_smoke.py)
- [../scripts/run_serving_smoke_compare.py](../scripts/run_serving_smoke_compare.py)
- [../scripts/run_serving_profile_compare.py](../scripts/run_serving_profile_compare.py)
- [../scripts/run_serving_replay_from_predictions.py](../scripts/run_serving_replay_from_predictions.py)
- [../scripts/run_serving_stateful_bankroll_sweep.py](../scripts/run_serving_stateful_bankroll_sweep.py)
- [../scripts/run_serving_compare_dashboard.py](../scripts/run_serving_compare_dashboard.py)
- [../scripts/run_serving_compare_aggregate.py](../scripts/run_serving_compare_aggregate.py)