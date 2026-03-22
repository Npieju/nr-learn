# ベンチマークと採用基準

## 1. この文書の役割

この文書は、`nr-learn` で何を良い結果とみなすかを定義するための正本である。

日常運用の区切り方や `revision` の切り方は、[development_flow.md](development_flow.md) を参照する。
formal な evaluation 導線は、[evaluation_guide.md](evaluation_guide.md) を参照する。

ここで決めるのは次の 3 点である。

1. 何の指標で評価するか。
2. どの数値を目標とするか。
3. どの条件を満たしたら採用するか。

## 2. 基本姿勢

- 単発の高 ROI は、そのまま採用根拠にしない。
- 同一期間で最適化と評価を兼ねない。
- raw 指標だけでなく、実際に買った場合の件数と純収益まで確認する。
- より単純な構成で同じ observed behavior を再現できるなら、単純な構成を優先する。

## 3. 主要指標

### 3.1 Weighted ROI

時系列検証の各期間で得た ROI を、件数を加味してまとめた指標である。

- `1.0` を超えると、検証全体で回収率が 100% を超えたことを意味する。
- ただし bets が極端に少ない場合は採用しない。

### 3.2 Bets

実際に賭けた件数である。

- 少なすぎる ROI は再現性が低い。
- `Weighted ROI` と必ずセットで解釈する。

### 3.3 Total Net

回収総額から投資総額を引いた値である。

- mean ROI だけでは、件数差による見かけの改善を見落とすことがある。
- serving 比較では `total net` を重要指標として扱う。

### 3.4 AUC / pseudo-R²

予測の並び順品質と、市場オッズに対する追加情報の有無を見る。

- AUC は予測順位の良さを見る。
- pseudo-R² は市場に対する独自情報の強さを見る。

## 4. 外部比較の目安

Benter 系の比較では、重要なのは単純な ROI よりも `ΔR² = R²_combined - R²_public` である。

当面の目安は次のとおりとする。

- 最低到達ライン:
  - `ΔR² > 0.009`
- 強いライン:
  - `ΔR² ≈ 0.018`

また、短期サンプルで `ROI > 1.0` が出ても、それだけで長期運用の目標達成とはみなさない。

## 5. 現在の内部 benchmark ladder

2024 年データの nested walk-forward における主要候補は次のとおりである。

| 候補 | Weighted ROI | Bets | 位置づけ |
| --- | ---: | ---: | --- |
| no-lineage public-free baseline | `0.5788` | `603` | 市場情報を切った基準線 |
| liquidity high-coverage | `0.9346` | `700` | 流動性改善の転換点 |
| regime hybrid | `0.9915` | `731` | 単純な運用候補 |
| regime modelswitch f1 policy may | `1.0073` | `713` | 現在の主力候補 |

この数値だけを見ると最後の候補が最良だが、構成の複雑さも同時に増しているため、運用判断では simpler rollback 候補との比較を必ず行う。

### 5.1 直近の revision comparison

2026-03-22 時点の full revision gate では、次の 2 候補を同条件で通した。

| revision | profile | status | decision | 備考 |
| --- | --- | --- | --- | --- |
| `r20260322a` | `current_best_eval` | `pass` | `promote` | `score_source_count=2`。`may_runtime_liquidity` override を読む |
| `r20260322b` | `current_recommended_serving` | `pass` | `promote` | `score_source_count=1`。単一 source で通過 |

この 2 run では、`evaluate --max-rows 120000 --wf-mode fast --wf-scheme nested` のトップライン指標は実質同一だった。

- `top1_roi`: `0.796298...`
- `ev_top1_roi`: `0.660712...`
- `auc`: `0.834390...`

したがって、現時点の運用判断では `current_recommended_serving` を simpler candidate として優先的に扱ってよい。
ただし、両者とも promotion gate warning として `probe_only` な walk-forward slice が残るため、将来の benchmark 更新では引き続き stricter full evaluation を優先する。

### 5.2 de-risk candidate の formal gate

2026-03-22 に、de-risk 候補 2 本も formal な revision gate に載せた。

| revision | profile | status | decision | 備考 |
| --- | --- | --- | --- | --- |
| `r20260322c` | `current_bankroll_candidate` | `block` | `hold` | representative evaluate は通るが、matching WF で feasible fold `0/5` |
| `r20260322d` | `current_ev_candidate` | `block` | `hold` | representative evaluate は通るが、matching WF で feasible fold `0/5` |
| `r20260322e` | `current_sep_guard_candidate` | `block` | `hold` | representative evaluate と matching WF は通るが、feasible fold `0/5` |

この 3 候補では共通して次が確認できた。

- `evaluate` 側の `stability_assessment` は `representative`
- matching な `wf_feasibility_diag` を付けると promotion gate は `wf_feasible_fold_count=0` で block
- dominant failure reason は `min_bets`
- fold ごとの binding source は全て `min_bets_abs=100` で、ratio 側ではなかった
- `max_infeasible_bets_observed=58` に留まり、threshold 未達が明確だった

同日付の threshold sweep で、support gap も数値化した。

- `current_bankroll_candidate` と `current_ev_candidate` は `min_bets_abs=58` で初めて feasible fold が `1/5` に到達した
- `current_bankroll_candidate` と `current_ev_candidate` は `min_bets_abs=40` で feasible fold が `4/5` になり、`3/5` 条件はここで満たす
- `current_bankroll_candidate` と `current_ev_candidate` は `min_bets_abs=30` まで下げると feasible fold が `5/5` になった
- fold 別の最初の到達 threshold は `current_bankroll_candidate` と `current_ev_candidate` で共通で、fold 3 が `58`、fold 1 が `45`、fold 2 と fold 4 が `40`、fold 5 が `30` だった
- `current_sep_guard_candidate` も formal gate の blocking source 自体は同じで、`wf_feasible_fold_count=0/5`, dominant failure reason `min_bets`, `binding_min_bets_source=absolute`, `max_infeasible_bets_observed=58` だった
- 3 候補を横並びにした threshold compare では、shared `1 fold` frontier は同じ `58` だった一方、`current_sep_guard_candidate` の strictest threshold は `3 folds=45`, `5 folds=34` で、`current_bankroll_candidate` / `current_ev_candidate` の `40/30` より少し厳しかった
- ただしこれは「他の de-risk 候補との比較」であり、baseline `current_recommended_serving` との直接比較とは別である
- baseline 用 threshold sweep に `58` checkpoint を補ったうえで `current_recommended_serving` と `current_sep_guard_candidate` を直接 compare すると、strictest threshold は両者とも `1 fold=58`, `3 folds=45`, `5 folds=34` で一致した
- fold 別の最初の到達 threshold も baseline と Sep guard で一致し、`fold1=55`, `fold2=45`, `fold3=58`, `fold4=35`, `fold5=34` だった
- threshold `60` の block pattern も同一で、両者とも `4` folds が `min_bets`, `1` fold が `min_final_bankroll` で止まり、fold4 の bankroll shortfall も同じ `0.04670658682634732` だった
- shared threshold (`60/58/55/45/40/35/34`) に限ると fold compare CSV の行内容も完全一致し、best feasible / blocked signature の並び自体が recent WF 上で変わっていないことも確認できた
- 具体的には dominant blocked signature は両者とも `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95` で共通し、fold4 の recovery だけが `min_expected_value=1.0` に切り替わる点まで一致した
- したがって Sep guard の serving 改善は「baseline より重い formal frontier を受け入れている」わけではなく、baseline がもともと持っている support frontier の内側で生じている
- この direct compare から signature report / family compare / drilldown / mitigation shortlist も切り直した。blocked occurrence は `34` 件で、内訳は `min_bets=24`, `min_final_bankroll=10`、dominant family はやはり `portfolio blend=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95` だった
- 次の mitigation ranking も従来の読みを補強しており、rank 1 は `portfolio blend=0.6 / min_prob=0.03 / min_ev=1.0`、rank 3 は `portfolio blend=0.8 / min_prob=0.03 / min_ev=1.0` だった。direct compare 上では前者が 34/34 occurrence で bankroll 改善かつ lower bets、後者も 32/34 occurrence で bankroll 改善を示した
- `min_final_bankroll` の 10 件はさらに単純で、すべて baseline / Sep guard 共通の fold4 occurrence だった。threshold `60/58/55/45/40` の各点で target は同じ `63 bets / final_bankroll≈0.8533 / gap≈0.0467` の `portfolio blend=0.8 / min_ev=0.95` で止まり、recovery も毎回同じ `portfolio blend=0.8 / min_ev=1.0 / 35 bets / final_bankroll≈0.9527` だった
- したがって formal 改善の次手も Sep guard 固有の seasonal logic ではなく、shared blocked family に対する `min_ev=1.0` と lower blend の portfolio-family mitigation を baseline 共通課題として評価するのが筋である

さらに threshold compare から mitigation probe を組み立てると、runtime 候補としては次の 2 本に収束した。

- dominant blocked signature は `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=0.95` で、84 blocked occurrence を占めた
- そのうち `74/84` occurrence では `portfolio / blend_weight=0.6 / min_prob=0.03 / top_k=1 / min_ev=1.0` がより高い final bankroll を示した
- 残り `10/84` occurrence では `portfolio / blend_weight=0.8 / min_prob=0.03 / top_k=1 / min_ev=1.0` が pure bankroll で上回った
- mitigation probe からは runtime-ready candidate として `portfolio_lower_blend` と `portfolio_ev_only` の 2 本を書き出せたが、`74/10` の staged hybrid は現行 runtime の単一 policy/date override では直接表現できなかった

つまり、この 3 候補の formal block は serving 上の見え方の違いよりも、同じ `min_bets` 系 support frontier に乗っている問題として読むのが正しい。特に `current_sep_guard_candidate` は他の 2 候補より multi-fold frontier が少し厳しい一方で、baseline `current_recommended_serving` と直接比べると frontier は悪化していない。残っている promotion block は Sep override 固有の追加コストではなく、baseline から共有している support gap である。

したがって、`current_bankroll_candidate` と `current_ev_candidate` は serving 上の de-risk 候補としては有力でも、現時点では benchmark 更新や正式昇格の候補とは扱わない。`current_sep_guard_candidate` も September seasonal override としては有望だが、formal には同様に hold である。運用上はいずれも rollback / seasonal override / defensive candidate に留め、昇格判断は support を増やす別の改善が入ってから再評価する。

2026-03-22 に long-horizon default-month の書き換え仮説も明示的に棄却した。May-Sep representative replay 5 日 (`2024-05-11`, `2024-06-15`, `2024-07-27`, `2024-08-17`, `2024-09-28`) では September guard を含む month override 群が同じなので、新しい `long_horizon_default_portfolio` probe と `current_sep_guard_candidate` は完全一致した。一方で April representative 4 日 (`2024-04-06`, `2024-04-07`, `2024-04-13`, `2024-04-14`) に default-month の `portfolio blend=0.6 / min_prob=0.03 / min_ev=0.95` を当てると、baseline `current_recommended_serving` の `19 bets / total net +6.1` に対して `4 bets / +3.9` まで縮退した。これは current EV-style candidate と同じ top line であり、default-month を lower-blend / EV-style family へ寄せても long-horizon 改善にはならないことを示す。

反対に、同じ April 4 日で `current_sep_guard_candidate` は baseline と完全一致した。つまり現時点の long-horizon operational reading は「non-September は baseline のまま固定し、September だけ validated Kelly-only guard を載せる」が最も defensible である。これを stable profile として再利用しやすくするため、`current_long_horizon_serving` alias を `current_sep_guard_candidate` と同じ config に追加した。formal gate 上の status は依然 hold だが、serving 運用上の conservative default としてはこの seasonal override が現在の最善候補である。

同日の fresh actual-date headroom check で、より複雑な `current_best_eval` も long-horizon serving を上回らないことを確認した。代表 4 日 (`2024-04-07`, `2024-05-11`, `2024-08-17`, `2024-09-28`) の compare では April / May / August が実質同一、差が出たのは September `2024-09-28` だけで、`current_long_horizon_serving` は `1 bet / -1.0`、`current_best_eval` は `2 bets / -2.0` だった。つまり現時点では score model complexity を戻しても long-horizon serving の実利は増えず、改善余地は依然として shared portfolio bottleneck 側にある。

さらに available な実日付を `2024-03-31..2024-09-29` の 50 日まで広げて baseline と `current_long_horizon_serving` を replay compare しても、この読みは崩れなかった。shared compare は baseline `156 bets / total net -37.1080` に対して long-horizon `123 bets / total net -2.8014` で、`right_minus_left_total_policy_net=+34.3067`。pure-stage bankroll も baseline-only `0.1948` に対して long-horizon `0.7530` だった。`differing_policy_dates` は依然 September 10 日 (`2024-09-01`, `2024-09-07`, `2024-09-08`, `2024-09-14`, `2024-09-15`, `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29`) に限られ、March-August の 40 日は完全一致した。したがって現時点の stable long-horizon candidate は実質的に `current_long_horizon_serving` であり、追加の broad rewrite ではなく shared September bottleneck のみに exposure を絞る構成が正しい。

この September 10 日をさらに date-level signal / staged trace で点検しても、guard を局所的に緩める根拠は出なかった。まず realized net は 10/10 日すべてで baseline 以上であり、`2024-09-07` は `+12.0 -> +19.7067`、`2024-09-14` は `+13.0 -> +14.0`、`2024-09-22` も `-4.6 -> -3.0` だった。次に baseline signal 側では `2024-09-28` が `ev_mean≈1.0303`, `edge_mean≈+0.0303` と最も強いのに realized は `-2.0` で、単純な EV/edge threshold が separator にならないことを再確認した。staged trace 側でも deepest-stage selected は `2024-09-15` と `2024-09-28` にしか出ず、両日とも net-negative なので「deepest-stage を許した日だけ戻す」方向にもならない。したがって、現在の September guard は already-good day を net で傷つけず、simple date-level signal でも安全に緩められない。次の改善軸は September override の微調整ではなく shared blocked portfolio family の formal support 改善である。

## 6. serving 比較の判断基準

actual calendar の比較では、次を確認する。

1. `policy_bets`
2. `policy_roi`
3. `total return`
4. `total net`

現在の判断原則は次のとおりである。

- 追加 return があっても `total net` が悪化するなら採用しない。
- 主力候補と observed behavior が一致する単純候補があるなら、rollback ではそちらを優先する。
- representative date だけでなく、複数日 window で再確認する。

### 6.1 直近の複数 window aggregate

2026-03-22 時点で、`current_recommended_serving` を基準に次の 2 系統を複数 window aggregate で比較した。

- `current_bankroll_candidate`
- `current_ev_candidate`

入力 window は次の 4 本である。

- `tail_weekends`
- late-September 5 日 window
- `aug_weekends_20260322`
- `may_weekends_20260322`

aggregate の要点は次のとおりである。

- `current_bankroll_candidate` は pure bankroll delta が `2/4` windows で正、`1/4` は zero、4-window mean は `net -2.6500`, `bankroll +0.1520`
- `current_ev_candidate` も pure bankroll delta が `2/4` windows で正、`1/4` は zero、4-window mean は `net -5.6500`, `bankroll +0.0354`
- 両候補とも net delta が正だったのは late-September 側だけだった
- `tail_weekends` では `current_recommended_serving` が total net で優位で、候補側は pure final bankroll のみ改善した
- `aug_weekends_20260322` では `current_recommended_serving` が net と bankroll の両面で両候補を上回った
- `may_weekends_20260322` では両候補とも `current_recommended_serving` と完全一致した

aggregate JSON の `tradeoff_classification` で言い換えると、window 構成は `++`, `-+`, `--`, `00` が 1 本ずつである。

- late-September は `positive_net_positive_bankroll`
- `tail_weekends` は `negative_net_positive_bankroll`
- `aug_weekends_20260322` は `negative_net_negative_bankroll`
- `may_weekends_20260322` は `zero_net_zero_bankroll`

このため、候補側を「bankroll 改善候補」とだけ要約するのは粗すぎる。現状の evidence は、「一部 regime では効き、別 regime では baseline より悪化し、さらに差分が消える regime もある」である。

したがって、現時点の運用上の位置づけは次のとおりである。

1. `current_bankroll_candidate` は rollback / de-risk の有力候補だが、全 regime で優位とは言えない
2. `current_ev_candidate` は中間候補だが、現時点では `current_recommended_serving` を安定して上回る根拠は弱い
3. `current_recommended_serving` は baseline として維持し、候補置換は regime ごとの裏付けが増えるまで慎重に扱う

この後段で確認した `current_sep_guard_candidate` は役割が少し違う。これは broad rollback 候補ではなく、September にだけ効かせる seasonal override candidate であり、validated 4 windows では `late-September=positive_net_positive_bankroll`, `May/August/tail=zero_net_zero_bankroll` という、より局所的で素直な profile として読むべきである。

この seasonal reading は broad September 10 日 (`2024-09-01`, `2024-09-07`, `2024-09-08`, `2024-09-14`, `2024-09-15`, `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29`) でも崩れなかった。`run_serving_profile_compare.py --run-bankroll-sweep --run-dashboard` で baseline `current_recommended_serving` と stable profile `current_sep_guard_candidate` を直接比べると、shared 10 日の compare は baseline `58 bets / total net -21.6` に対して candidate `25 bets / total net +12.7067`、pure path final bankroll も baseline `0.2590` に対して candidate `1.0008` だった。つまり `current_sep_guard_candidate` は「late-September の局所当たり」ではなく、September 10 日スパンでも baseline exposure を強く削りつつ net / bankroll を同時に改善する seasonal override と読める。

さらに base prediction がある `2024-05-04..2024-09-29` の 45 race-day 全体でも同じ wrapper を回すと、差分は September だけに留まらず full recent aggregate でも残った。shared compare は baseline `136 bets / total net -42.2080` に対して candidate `103 bets / total net -7.9014` で、`right_minus_left_total_policy_net=+34.3067`。pure path final bankroll も baseline `0.1886` に対して candidate `0.7290` だった。non-September path は本来ほぼ同一なので、この差分は「September override が recent aggregate を壊さない」ではなく「recent aggregate 全体の損失を実質的に縮小する」と読むのが正しい。

重要なのは、この 45 日 compare で `differing_policy_dates` が September 10 日 (`2024-09-01`, `2024-09-07`, `2024-09-08`, `2024-09-14`, `2024-09-15`, `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29`) に限られている点である。つまり full recent aggregate の `+34.3067` 改善は「May-August を触って得た broad improvement」ではなく、「non-September を壊さず September だけ差し替えた seasonal override が recent aggregate を持ち上げた」ことを意味する。運用上はこの性質が重要で、broad serving replacement よりも controlled seasonal override としての解釈をさらに強める。

### 6.2 threshold 由来 single-policy probe の位置づけ

2026-03-22 に mitigation probe から生成した `portfolio_lower_blend` / `portfolio_ev_only` の single-policy config を actual-date smoke で再確認した。

- late-September 5 日 window では baseline `current_recommended_serving` が `31 bets / total net -25.6`、`portfolio_lower_blend` は `1 bet / -1.0`、`portfolio_ev_only` は `12 bets / -12.0`
- August weekends 6 日 window では baseline が `34 bets / total net +20.1`、`portfolio_lower_blend` は `2 bets / -2.0`、`portfolio_ev_only` は `9 bets / -9.0`
- late-September では defensive loss reduction は見えたが、August 側では baseline の利益局面を大きく取り逃した

このため、generated single-policy probe は「formal gate blockage の原因 signature を actual calendar でどう抑えるか」を観察する補助 artifact にはなるが、現時点では stable profile 登録候補ではない。運用判断は引き続き `current_recommended_serving` を baseline に置き、generated probe は selector 拡張や staged hybrid 実装の前段 evidence として扱う。

### 6.3 staged mitigation probe の位置づけ

既存の staged probe config も same-window actual-date smoke で確認したが、運用判断は変わらなかった。

- late-September では `staged_mitigation_probe` が `12 bets / total net -12.0`、`staged_mitigation_ev_guard_probe` が `10 bets / -10.0`
- August weekends では `staged_mitigation_probe` が `9 bets / -9.0`、`staged_mitigation_ev_guard_probe` が `6 bets / -6.0`
- baseline `current_recommended_serving` は同じ window で `late-September 31 bets / -25.6`、`August weekends 34 bets / +20.1`

つまり、staged probe は runtime feasibility 自体は示したが、realized calendar behavior は single-policy probe の defensiveness を少し調整した範囲に留まり、baseline replacement の根拠にはならなかった。現時点では staged も stable profile 候補ではなく、fallback 条件の再設計や selector logic の evidence 化が先に必要である。

補足すると、observed stage path もこの判断を裏づけている。

- `staged_mitigation_probe` は observed windows で実質 `portfolio_ev_only` のままで、stage diversification を生まなかった
- `staged_mitigation_ev_guard_probe` だけが August の一部日付で `kelly_fallback_1` / `kelly_fallback_2` を発火させた
- それでも August aggregate は baseline に届かなかったため、「fallback が発火すること」と「運用候補として十分に強いこと」は別問題だと分かった

加えて、staged runtime artifact には fallback reason も残すようにした。

- 例として `2024-08-11` の `staged_mitigation_ev_guard_probe` では、`portfolio_ev_only:max_expected_value_below|portfolio_lower_blend:no_selection` を経て `kelly_fallback_1` に到達したことが summary / backtest JSON の両方から確認できる
- これにより、今後の論点は単に「Kelly fallback が出たか」ではなく、「どの trigger が profitable regime を潰しているか」を actual-date artifact から直接監査することになる

同日付の `run_serving_stage_path_compare.py` でもこの差分を横比較した。

- late-September 5 日 window では `differing_stage_dates` が全 5 日だった
- August weekends 6 日 window では `differing_stage_dates` が `2024-08-03`, `2024-08-10`, `2024-08-11`, `2024-08-18` だった
- fallback reason まで入れた再比較では、late-September は `differing_stage_fallback_reason_dates` も全 5 日、August weekends は `2024-08-03`, `2024-08-11`, `2024-08-18` だった
- late-September の `staged_mitigation_ev_guard_probe` は 5/5 日すべてで `portfolio_ev_only:max_expected_value_below` を起点に Kelly fallback へ流れており、しかも 4/5 日は `portfolio_lower_blend:no_selection` で止まっていた
- August の fallback reason は 3 日だけで、`2024-08-03` のみ `portfolio_lower_blend:max_expected_value_below` まで進んだうえで `kelly_fallback_2` に落ちていた

したがって、現状の論点は「stage path が分岐しているか」ではなく、「その分岐条件が profitable regime を十分に拾えているか」である。

### 6.4 staged EV guard threshold sweep の判断

2026-03-22 に `scripts/run_staged_ev_guard_threshold_sweep.py` を追加し、`staged_mitigation_ev_guard_probe` の先頭 2 stage にある `max_expected_value_below` を actual-date replay で sweep した。

- stage1 grid: `1.08`, `1.06`, `1.04`, `1.02`
- stage2 grid: `1.03`, `1.01`, `0.99`
- late-September 5 日 window と August weekends 6 日 window の 2 本で実行

結果は、threshold 感度の解釈をかなり絞った。

- late-September では全 12 variant が `10 bets / total net -10.0` で横並びだった
- ただし fallback 頻度は `5/5 -> 4/5 -> 3/5` と段階的に減り、lower-blend selection も 1 日だけ復活した
- August では `s1 in {1.08,1.06,1.04}` がすべて `6 bets / total net -6.0` で横並びだった
- August の `s1=1.02` は全 stage2 variant で `8 bets / total net -8.0` に悪化した
- stage2 を `1.03 -> 1.01/0.99` に下げると `2024-08-03` の `kelly_fallback_2` は消えたが、window 全体の net は改善しなかった

この evidence から、今の staged bottleneck は次のように読むのが妥当である。

1. EV guard は確かに aggressive で、不要な fallback を起こしている日がある
2. ただし、今回の実 calendar では moderate な threshold 緩和だけでは baseline 置換の根拠になる profit 改善は出ない
3. さらに、stage1 を `1.02` まで下げるのは August regime では明確に悪化なので unsafe 側である

したがって、次の selector 改善は単純な threshold lowering で押し切るべきではない。候補は「August profitable regime を見分ける追加条件」か、「bankroll state / date-context を含む richer fallback 条件」である。

### 6.5 staged signal diagnostic の判断

同日、`scripts/run_staged_policy_signal_diagnostic.py` も追加し、August weekends の fallback dates を race-level signal で再確認した。

この診断は threshold sweep より重要な反証を与えた。

- `2024-08-03` と `2024-08-11` では、`portfolio_ev_only` は実際に race を選んでいるが、その選択自体が 0 hit だった
- `2024-08-18` では `portfolio_ev_only` が 4 race を選び、median `max_expected_value` は約 `1.023` だったが、やはり 0 hit だった
- `2024-08-17` はさらに厳しく、staged probe は全 stage `no_selection` だった一方、baseline `current_recommended_serving` は `7 bets / net +32.6` を出していた
- baseline は `2024-08-18` でも `8 bets / net +2.7` を出しており、staged の sparse portfolio family が profitable day を取り逃していることが分かった

この evidence により、現在の staged probe を「August は EV guard が少し厳しすぎるだけ」と解釈するのは不正確になった。より正確には、August gap は次の 2 層に分かれる。

1. 一部の race では EV guard が unnecessary fallback を起こしている
2. それとは別に、early portfolio stages 自体が profitable day を十分に再現できていない

したがって、次の selector 改善で優先すべきなのは `max_expected_value_below` の微調整ではなく、`portfolio_ev_only` / `portfolio_lower_blend` の policy family 自体を見直すこと、または baseline 側 profitable regime を stage policy に持ち込める regime-aware split を導入することである。

### 6.6 portfolio family sweep の判断

この仮説は `scripts/run_portfolio_policy_family_sweep.py` でさらに絞れた。August weekends の単純 portfolio replay sweepでは、baseline August policy family がそのまま最適だった。

- best variant は `blend=0.8 / min_prob=0.03 / min_ev=0.95` で、`34 bets / total net +20.1`
- 同じ `blend=0.8 / min_prob=0.03` でも `min_ev=1.0` に上げるだけで `9 bets / total net -9.0`
- `blend=0.8 / min_prob=0.05 / min_ev=0.95` は still positive だが `+8.6` まで縮む
- `blend=0.6` variants はすべて negative で、best でも `-1.0`

この差は profitable dates で特に大きい。

- `2024-08-17` は baseline family だと `7 bets / net +32.6`、`min_ev=1.0` に上げると `0 bets / net 0.0`
- `2024-08-18` は baseline family だと `8 bets / net +2.7`、`min_ev=1.0` だと `4 bets / net -4.0`

したがって、August で staged probe が弱い理由はもうかなり具体的である。

1. stage1 の `min_expected_value=1.0` が profitable August races を落としすぎている
2. stage2 の `blend=0.6` も recovery ではなく、むしろ profitability を削る方向に働いている
3. つまり、次の実験は threshold 調整ではなく「stage1 を baseline August portfolio family にどこまで寄せるか」で始めるべきである

### 6.7 baseline-aligned staged probe の判断

この次段として、stage1 を baseline August family へ戻した `runtime_staged_aug_baseline_stage1_probe` を actual-date replay で確認した。

- stage1 は `blend=0.8 / min_prob=0.03 / min_ev=0.95`
- stage2 以下は既存 staged probe と同じ lower-blend / Kelly fallback を残した
- fallback 条件は置かず、no-selection のときだけ次段へ進む構成にした

結果は解釈しやすい。

- August weekends 6 日 window では baseline `current_recommended_serving` と完全一致し、`34 bets / total net +20.1`
- ただし 6/6 日すべてが `portfolio_aug_baseline:selected` で、stage2 以下は一度も使われなかった
- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して、この probe は `38 bets / total net -32.6` とさらに悪化した
- late-September でも 5/5 日すべてが `portfolio_aug_baseline:selected` で、de-risk fallback は一切発火しなかった

したがって、この probe は「August gap の主因が stage1 family の strictness にある」ことは裏づけたが、運用候補にはならない。なぜなら、August を救う方法がそのまま late-September の over-exposure につながり、しかも staged wrapper 自体は何も制御していないからである。

ここからの正しい次段は、単なる staged shell の維持ではなく、次のどちらかである。

1. August 相当 regime を判別できる date-context / regime-aware split を selector に持ち込む
2. baseline-aligned stage1 の上に、late-September だけを抑える explicit fallback guard を追加する

### 6.8 selected_rows guard の判断

上の 2 に対する最初の小さい probe として、baseline-aligned stage1 に `fallback_when.selected_rows_at_most: 5` を付けた `runtime_staged_aug_baseline_stage1_selected_rows_guard_probe` も replay した。

結果は trade-off としては明快だが、採用候補にはならない。

- late-September 5 日 window では `10 bets / total net -10.0` まで縮み、baseline `31 bets / -25.6` よりかなり defensive になった
- 5/5 日すべてで `portfolio_aug_baseline:selected_rows_at_most` が発火し、最終段は `portfolio_lower_blend` か `kelly_fallback_1` だった
- しかし August weekends 6 日 window は `6 bets / total net -6.0` まで崩れ、baseline `34 bets / +20.1` を完全に失った
- 特に profitable day の `2024-08-17` は `7 bets / +32.6` から `0 bets / 0.0` へ落ちた

したがって、selected-count ベースの guard は late-September exposure を抑える効果自体はあるが、August 側では profitable regime を blunt に切り落としてしまう。運用候補として次に試すべきなのは、件数だけを見る guard ではなく、date-context か別の race-level signal を含む regime-aware 条件である。

### 6.9 September-only regime split candidate の判断

この次段として、September だけ `selected_rows_at_most: 5` 付き staged policy を使う `..._serving_sep_selected_rows_guard_candidate.yaml` を replay した。month-aware override を使って blunt guard を September に閉じ込める形である。

3 window で見ると挙動はかなり素直だった。

- May weekends 6 日 window は baseline と完全一致し、`6 bets / total net -6.0`
- August weekends 6 日 window も baseline と完全一致し、`34 bets / total net +20.1`
- late-September 5 日 window では baseline `31 bets / total net -25.6` に対して `10 bets / total net -10.0`
- June-August の `tail_weekends` 19 日 window でも baseline と完全一致し、`55 bets / total net -0.9`、`differing_policy_dates=[]`

つまり、この candidate は今までの probe と違って、September の de-risk だけをほぼ純粋に取り出せている。`selected_rows_at_most` 自体は blunt でも、date-context で適用範囲を縛れば operationally useful な serving candidate になりうる。

stateful bankroll sweep を同じ 4 window で重ねても、この読みはさらに強まった。

- late-September では total net は baseline `-25.6` に対して candidate `-10.0` まで改善したが、pure path の final bankroll は baseline `0.2780`、candidate `0.0` で、best path は baseline 側に残った
- May weekends は pure path / threshold grid ともに baseline / candidate が `0.9906` で完全一致した
- August weekends も pure path / threshold grid ともに baseline / candidate が `1.1765` で完全一致した
- `tail_weekends` も pure path / threshold grid ともに baseline / candidate が `0.7432` で完全一致した

この段階での読みは次のとおりである。

1. regime-aware split は、単独 guard や staged shell より有望である
2. current evidence の範囲では、May/August/tail を壊さず late-September の net だけは改善できている
3. ただし late-September の pure bankroll は悪化するため、この candidate は `positive_net_negative_bankroll` な trade-off 候補として扱うのが正しい
4. 次に進めるなら、この candidate を aggregate / dashboard / wider-window compare の中で再確認するのが自然である

### 6.10 September Kelly-only fallback candidate の判断

上の September-only candidate を aggregate まで伸ばすと、弱点はかなりはっきりした。late-September の pure bankroll 崩壊は `2024-09-28` に `portfolio_lower_blend:selected` が 1 回入ったことに集中しており、他の 4 日は `kelly_fallback_1` でほぼフラットだった。

このため、September override から `portfolio_lower_blend` を外し、`selected_rows_at_most: 5` 発火後は Kelly fallback に直落としする `..._serving_sep_selected_rows_kelly_only_candidate.yaml` を replay した。

4 window aggregate の結果は、前の candidate より明確に強い。

- late-September 5 日 window は bets / net が `10 bets / total net -10.0` のまま維持された
- 同じ late-September で pure path final bankroll は baseline `0.2780` に対して `0.9915` まで改善した
- May weekends、August weekends、`tail_weekends` はすべて baseline と完全一致し、net / pure bankroll とも delta `0.0` だった
- aggregate の `tradeoff_classification` は `positive_net_positive_bankroll` が late-September 1 本、残り 3 window は `zero_net_zero_bankroll` だった

この candidate の読みはかなり単純である。

1. September-only split 自体は有効だったが、問題は `selected_rows` guard ではなく fallback 先の `portfolio_lower_blend` だった
2. Kelly-only fallback に置き換えると、validated 4 windows では net を保ったまま pure bankroll も改善できた
3. 現時点では、この variant が September de-risk line の中で最初の `positive_net_positive_bankroll` 候補である

加えて、旧 `..._serving_sep_selected_rows_guard_candidate.yaml` との direct late-September compare でも dominance が確認できた。

- shared policy bets は両者とも `10`
- shared total net も両者とも `-10.0`
- しかし pure path final bankroll は旧 candidate `0.0` に対して Kelly-only candidate `0.9915`

さらに、この Kelly-only variant を stable alias `current_sep_guard_candidate` に昇格させたうえで、標準 wrapper `run_serving_profile_compare.py --run-bankroll-sweep --run-dashboard` でも late-September 5 日 window を再実行した。

- compare manifest は `status=completed`, `decision=ready`
- shared outcome は baseline `31 bets / -25.6` に対して `current_sep_guard_candidate` `10 bets / -10.0`
- pure path final bankroll も baseline `0.2780` に対して `0.9915` を再現した
- dashboard artifact まで一括生成でき、stable profile として運用導線に載せられることを確認した

したがって、September de-risk line の current frontier はもう `selected_rows guard を使うかどうか` ではない。実務上の論点は「guard 発火後に portfolio lower-blend を残す理由があるか」であり、現時点の evidence では答えは `ない` である。

この時点での実務上の位置づけはさらに明確である。`current_sep_guard_candidate` は broad な serving replacement ではなく、September sparse-selection regime にだけ効かせる stable seasonal override candidate として扱うのが妥当である。

この conclusion に対して、September fallback の途中へ `portfolio_ev_only` を 1 段だけ戻す `..._serving_sep_selected_rows_ev_only_kelly_candidate.yaml` も追加で replay した。狙いは fold4 recovery で見えた `min_expected_value=1.0` の clue を serving 側で最小限に試すことだったが、late-September 5 日 window は `12 bets / total net -12.0` に留まり、既知の Kelly-only line (`10 bets / -10.0`) を上回れなかった。一方で June-August `tail_weekends` 19 日は baseline と完全一致 (`55 bets / total net -0.9`) だったため、この probe は September にだけ余計な exposure を戻す再確認として読むのが正しい。実務上は discard でよい。

続けて runtime fallback に `date_selected_rows_at_most` を追加し、日次 sparse guard を明示的に書けるようにもした。既存の `selected_rows_at_most` は race-local なので、`top_k=1` stage では September sparse day を日次件数で切る用途には向かない。この新 key を使った最初の 2 probe は、August baseline stage1 だと late-September `29 bets / -23.6`、September baseline stage1 だと `24 bets / -18.6` で、どちらも tail は baseline と完全一致だったが current Kelly-only line (`10 bets / -10.0`) は超えられなかった。したがって、新 key は useful capability だが current best seasonal guard を更新する evidence にはまだなっていない。

さらに `scripts/run_staged_date_selected_rows_sweep.py` で September baseline stage1 variant の `date_selected_rows_at_most` を `1..10` まで sweep すると、最良の threshold は `10` で `17 bets / total net -17.0` だった。`5` の `24 bets / -18.6` よりは改善するが、still current Kelly-only line `10 bets / -10.0` を大きく下回る。運用上の読みは明快で、day-level selected-count guard は capability としては useful でも、count 単独では September de-risk の主戦略にはならない。

この読みを broad September でも崩さないか確かめるため、続けて同じ family を `2024-09-01`, `2024-09-07`, `2024-09-08`, `2024-09-14`, `2024-09-15`, `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29` の 10 race-day で再 sweep した。artifact `staged_date_selected_rows_sweep_sep_full_month_202409_probe.{json,csv}` では `date_selected_rows_at_most=8` が family 内最良で、`39 bets / total net +4.1067` まで改善した。`5` は同じ 10 日で `49 bets / -5.8933` なので、広い September では count guard の効き方自体は late-September 5 日よりかなり良い。

ただし frontier 更新には至らない。この `threshold=8` line を probe profile として標準 wrapper で `current_sep_guard_candidate` と直接比較すると、10 日共通 replay の shared outcome は現行 candidate `25 bets / total net +12.7067` に対して probe `39 bets / +4.1067` だった。つまり count-only family の broad-month best を採っても、現行の Kelly-only September guard には `net -8.6` 劣後する。したがって operational reading は変わらない。`date_selected_rows_at_most` は September sparse day を広く削る useful capability だが、standalone count guard を current frontier に戻す根拠にはならず、stable seasonal override は引き続き `current_sep_guard_candidate` を維持するのが妥当である。

この次段の判断を reusable にするため、`scripts/run_policy_date_signal_report.py` と対応テストも追加し、baseline September policy の selected rows を日単位で要約する artifact を直接出せるようにした。late-September 5 日 (`2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-28`, `2024-09-29`) の初回 report では、`selected_count`、EV/edge/prob の min/mean/median/max、`count_ev_ge_1_0`、`count_edge_pos` を並べても clean な separator は見えなかった。

- `2024-09-28` は `selected_count=2` と sparse なのに、`ev_mean=1.0303`, `edge_mean=+0.0303`, `share_ev_ge_1_0=0.5`, `share_edge_pos=0.5` と aggregate が最も強く、realized は `2 bets / net -2.0` だった
- `2024-09-22` は `selected_count=10`, `ev_mean=0.9845`, `edge_mean=-0.0155`, `share_ev_ge_1_0=0.3`, `share_edge_pos=0.3` と aggregate 上は平凡だが、late-September の中では相対的にましな realized 日だった
- `2024-09-16` も `selected_count=11`, `ev_mean=0.9840`, `share_ev_ge_1_0=0.3636` で `2024-09-22` と近く、単純 threshold では両者を分けにくい

したがって、現時点の evidence では `ev_mean >= x` や `count_ev_ge_1_0 <= y` のような単純 date-wide aggregate guard を runtime key として足す根拠は弱い。`run_policy_date_signal_report.py` は今後の検証土台として有用だが、次に見るべきなのは単純 aggregate threshold ではなく、stage trace や race-level composition まで含む richer な日次 regime signal である。

この結論を `current_sep_guard_candidate` 側の trace まで戻して確認するため、`scripts/run_staged_policy_signal_diagnostic.py` の seasonal staged override 対応も直した。従来の `_stage_policy()` は固定 `2024-01-01` で runtime policy を解決していたため、September override が staged でも diagnostic 側では default non-staged policy を掴んで失敗していた。日付ごとに staged policy を resolve するよう修正したことで、`..._serving_sep_selected_rows_kelly_only_candidate.yaml` の late-September 5 日診断 artifact を再生成できた。

この trace で見えた追加の事実は、simple aggregate より fallback depth の方が separator 候補としてまだ筋が良いことである。

- `2024-09-28` は baseline signal report 上は最も強い aggregate (`ev_mean=1.0303`, `edge_mean=+0.0303`) を持つが、September guard では 5 日中唯一 `kelly_fallback_2` まで到達した
- 同日の `kelly_fallback_1` final race は 1 本だけで、その race 自体も負けているため、強い aggregate がそのまま safe な fallback path を意味しない
- 対照的に `2024-09-22` は stage1 selected race が 11 本ありながら `kelly_fallback_2` には一度も到達せず、final は `kelly_fallback_1` 3 本に留まった

したがって、September de-risk line の次段 signal 候補としては、単純な日次 EV/edge threshold より `fallback depth` や `stage trace composition` を first-class に集計する方が合理的である。少なくとも現時点では、`2024-09-28` 型の崩れは aggregate strength ではなく deeper fallback path と結びついて見える。

この仮説を reusable artifact にするため、続けて `scripts/run_staged_trace_date_report.py` も追加した。これは `run_staged_policy_signal_diagnostic.py` の raw CSV を入力に、per-date の `final_stage_counts`、`final_trace_counts`、`final_fallback_reason_counts`、そして重要な `deepest_selected_stage_counts` / `stage3_plus_selected_race_count` を集計する小さい report である。late-September の `current_sep_guard_candidate` では、単純な trace depth ではなく「選択が立った最深 stage」を見ると差がかなり明瞭だった。

- `2024-09-28` だけが `max_selected_stage_index=3` かつ `stage3_plus_selected_race_count=1` で、`deepest_selected_stage_counts={'kelly_fallback_2': 1, 'portfolio_aug_baseline': 3}` だった
- 他の 4 日はすべて `max_selected_stage_index=2`, `stage3_plus_selected_race_count=0` に留まった
- したがって、signal として有望なのは `all-stage trace depth` そのものではなく、`deeper stage に実際の選択候補が立ったか` という selection-aware depth である

運用上の読みはさらに絞られる。`2024-09-28` 型の崩れを説明する次段 descriptor としては、`ev_mean` ではなく `stage3 candidate present` や `deepest_selected_stage` の方が明らかに筋が良い。次に設計すべきのは simple aggregate threshold ではなく、この selection-aware trace signal を複数 window へ広げて support を確認する手順である。

この support 確認もさらに整理した。`run_staged_trace_support_check.py` を追加し、3 本の staged trace date report (`aug_ev_guard`, `late_sep_ev_guard`, `sep_guard`) をまとめて deepest-stage selection の support を数えた。ここで見るべき signal は `stage3_plus_selected` のような固定閾値ではなく、config の最終 stage に実際の選択が立った `deepest_stage_selected_present` である。3 report / 16 rows でこの signal が立ったのは 4 rows、日付ユニークでは `2024-08-03`, `2024-08-10`, `2024-09-28` の 3 日だけだった。しかも 4/4 rows すべて realized `final_selected_net_units` は negative で、positive は 0 件だった。

- `aug_ev_guard` では deepest-stage selected day は `2024-08-03`, `2024-08-10`
- `late_sep_ev_guard` では `2024-09-28` だけ
- `sep_guard` でも `2024-09-28` だけ

したがって、support の読みは前段より一段強い。`deepest_stage_selected_present` は complete bad-day separator ではないが、少なくとも今ある cross-window evidence では `positive` day に一度も出ておらず、high-risk subset flag としての precision は高い。

そのうえで recall 側も整理した。`run_staged_trace_support_check.py` に selection-depth bucket (`deepest_stage_selected` / `intermediate_stage_selected` / `no_final_selection`) を追加して同じ 3 report / 16 rows を再集計すると、`intermediate_stage_selected` は 10 rows で、これも `positive` は 0 件、`non-positive` が 10/10 だった。日付ユニークでは `2024-08-11`, `2024-08-18`, `2024-09-16`, `2024-09-21`, `2024-09-22`, `2024-09-29` の 6 日で、deepest-stage selected が拾えなかった bad-day 候補の大半はここに落ちている。

- `aug_ev_guard` では `intermediate_stage_selected=2`, `deepest_stage_selected=2`, `no_final_selection=2`
- `late_sep_ev_guard` では `intermediate_stage_selected=4`, `deepest_stage_selected=1`
- `sep_guard` でも `intermediate_stage_selected=4`, `deepest_stage_selected=1`

現時点の operational reading はこうなる。`deepest_stage_selected_present` は高精度な tail-risk subset flag、`intermediate_stage_selected_present` はそれより広い recall bucket で、今回の support ではどちらも positive day には一度も出ていない。次に広げるべきのは raw stage index 閾値ではなく、この selection-aware bucket をより長い window で増やし、precision 劣化なしに support が維持されるかを確認することだ。

その確認として、同じ `staged_mitigation_ev_guard_probe` を May weekends 6 日と `tail_weekends` 24 日にも広げ、5 report / 46 rows (`may_ev_guard`, `aug_ev_guard`, `tail_ev_guard`, `late_sep_ev_guard`, `sep_guard`) の support artifact `staged_trace_support_check_may_aug_tail_late_sep_depth_bucket_support_20260322.{json,csv}` を追加した。ここで signal の差は明確だった。

- `deepest_stage_selected` は 13 rows まで増えてもなお `negative=13`, `positive=0` を維持した
- `intermediate_stage_selected` は 20 rows に増え、`negative=18`, `positive=2` になった
- intermediate の positive はどちらも `tail_ev_guard` 側の `2024-09-07`, `2024-09-14` だった
- net severity でも差があり、`deepest_stage_selected` は mean `-0.5399`, median `-1.0` に対して、`intermediate_stage_selected` は mean `-0.0001`, median `-0.0008` に留まった

report 別には次の分布だった。

- `may_ev_guard`: `deepest=2`, `intermediate=2`, `no_final_selection=2`
- `aug_ev_guard`: `deepest=2`, `intermediate=2`, `no_final_selection=2`
- `tail_ev_guard`: `deepest=7`, `intermediate=8`, `no_final_selection=9`
- `late_sep_ev_guard`: `deepest=1`, `intermediate=4`
- `sep_guard`: `deepest=1`, `intermediate=4`

したがって、読みはさらに精密になった。`deepest_stage_selected_present` は window を広げても still precision を崩しておらず、しかも net severity まで明確に悪いので、tail-risk subset flag としての支持はむしろ増えた。一方で `intermediate_stage_selected_present` は recall bucket としては有用だが、extended support ではすでに positive day (`2024-09-07`, `2024-09-14`) を含み、loss magnitude もごく薄い。つまり intermediate bucket は「bad-day を広く拾う descriptor」ではあっても、「precision-safe な risk flag」でも「tail-loss subset flag」でもない。runtime key を考えるなら deepest を主 signal に保ち、intermediate は補助説明変数として扱うのが妥当である。

さらに overlap を避けるため、window の足し合わせではなく `2024-05-04..2024-09-29` の base prediction 45 race-day を 1 本の `full_recent_ev_guard` report として再集計した。`sep_guard` を添えた 2-report support artifact `staged_trace_support_check_full_recent_sep_guard_depth_bucket_support_20260322.{json,csv}` でも結論は同じだった。

- `full_recent_ev_guard` 単体では `deepest_stage_selected=12`, `intermediate_stage_selected=20`, `no_final_selection=13`
- `sep_guard` を加えた合計 50 rows でも `deepest_stage_selected` は `13 negative / 0 positive`
- `intermediate_stage_selected` は `21 negative / 3 positive` で、positive は `2024-06-16`, `2024-09-07`, `2024-09-14`
- severity も `deepest_stage_selected mean=-0.7709, median=-1.0`、`intermediate_stage_selected mean=-0.00007, median=-0.00069` と大きく分かれた

この broad-window check によって、deepest-stage signal は「複数小 window で偶然 clean だった」ではなく、より広い recent calendar でも precision と severity を保つ tail-loss subset flag と読めるようになった。一方で intermediate-stage signal は broad window ほど positive contamination が増えるため、runtime guard 候補としてはむしろ後退した。

したがって、`stage3 candidate present` は「全ての loss day を捉える complete separator」ではないが、「deeper fallback を実際に必要とした high-risk subset」を切り出す risk flag としては支持が増えた。実務上の次段は、この signal を runtime key に即座に変換することではなく、まず他 window も含めて `stage3 present -> loss-heavy subset` の support を増やすことである。

## 7. 現在の評価方針

### 7.1 nested evaluation

本命候補の判定は、`run_evaluate.py` による nested walk-forward を基準に行う。

理由は次の 2 点である。

- 同じ期間で最適化と評価を兼ねないため。
- fold ごとの no-bet や gating failure まで記録できるため。

### 7.2 serving validation

運用候補の確認は、`run_serving_smoke.py` と `run_serving_smoke_compare.py` を基準に行う。

理由は次の 2 点である。

- summary から復元した serving ルールが、実日付でも同じように動くか確認できるため。
- nested 上の改善が、actual calendar でも意味を持つかを見られるため。

ただし、短い serving validation は smoke / probe であり、正式な採用判断は `stability_assessment=representative` と promotion gate を通した revision 単位で行う。

## 8. 採用ルール

最終的な採用判断は、次の順序で行う。

1. nested walk-forward で改善しているか。
2. bets が少なすぎないか。
3. actual calendar で total net が悪化していないか。
4. 同等挙動を再現するより単純な候補がないか。

この 4 条件を満たして初めて、本命または rollback 候補として扱う。